use std::collections::BTreeMap;
use std::fs;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use regex::Regex;
use safetensors::tensor::SafeTensors;

use manifest_core::{ManifestChunk, ModelManifest};

use crate::classify::{classify_tensor, TensorClass, TensorLocation};
use crate::io::{load_tensors, write_safetensors};
use crate::{Args, LoadedFile};

/// 核心分片流程
pub fn run(args: Args) -> Result<()> {
    // ── 校验输入目录 ──
    if !args.input.is_dir() {
        bail!("输入路径 '{}' 不是有效目录", args.input.display());
    }

    // ── 创建输出目录 ──
    fs::create_dir_all(&args.output)
        .with_context(|| format!("无法创建输出目录 {}", args.output.display()))?;

    // ─────────────────────────────────────────────────
    // 第一步：发现并 mmap 所有 .safetensors 文件
    // ─────────────────────────────────────────────────
    let mut safetensor_paths: Vec<std::path::PathBuf> = fs::read_dir(&args.input)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    safetensor_paths.sort(); // 确保多文件场景下顺序确定

    if safetensor_paths.is_empty() {
        bail!("在 {} 中未找到 .safetensors 文件", args.input.display());
    }

    println!("发现 {} 个 safetensors 文件:", safetensor_paths.len());
    for p in &safetensor_paths {
        println!("  {}", p.display());
    }

    // 内存映射所有源文件（零拷贝：Apple Silicon 统一内存下，
    // mmap 直接引用 SSD 的页缓存，无需在 CPU/GPU 间搬运数据）
    let loaded_files: Vec<LoadedFile> = safetensor_paths
        .iter()
        .map(|path| {
            let file =
                fs::File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
            // SAFETY: 文件以只读方式打开，且在处理期间保持 File 句柄存活
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("mmap 失败: {}", path.display()))?;
            Ok(LoadedFile {
                path: path.clone(),
                _file: file,
                mmap,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // ─────────────────────────────────────────────────
    // 第二步：扫描 Header，分类所有 Tensor
    // ─────────────────────────────────────────────────

    // 匹配 Qwen2.5 层级 Tensor 的正则: "model.layers.{N}.任意后缀"
    let layer_re = Regex::new(r"^model\.layers\.(\d+)\.")?;

    let mut base_tensors: Vec<TensorLocation> = Vec::new();
    let mut layer_tensors: BTreeMap<u32, Vec<TensorLocation>> = BTreeMap::new();
    let mut max_layer: u32 = 0;

    for (file_idx, loaded) in loaded_files.iter().enumerate() {
        let st = SafeTensors::deserialize(&loaded.mmap)
            .with_context(|| format!("解析 Header 失败: {}", loaded.path.display()))?;

        for name in st.names() {
            let loc = TensorLocation {
                file_idx,
                name: name.to_string(),
            };

            match classify_tensor(name, &layer_re) {
                TensorClass::Base => {
                    base_tensors.push(loc);
                }
                TensorClass::Layer(n) => {
                    if n > max_layer {
                        max_layer = n;
                    }
                    layer_tensors.entry(n).or_default().push(loc);
                }
            }
        }
    }

    let total_layers = max_layer + 1;
    println!(
        "\n分类结果: {} 个 Base Tensor, {} 层 (0..{})",
        base_tensors.len(),
        total_layers,
        max_layer
    );

    // ─────────────────────────────────────────────────
    // 第三步：写入 Base 分块
    // ─────────────────────────────────────────────────

    let mut manifest_chunks: Vec<ManifestChunk> = Vec::new();

    {
        let filename = "base.safetensors".to_string();
        let output_path = args.output.join(&filename);

        println!(
            "\n正在写入 Base 分块 ({} 个 Tensor)...",
            base_tensors.len()
        );

        let owned = load_tensors(&loaded_files, &base_tensors)?;
        let (bytes, sha256) = write_safetensors(&owned, &output_path)?;

        println!("  -> {} ({} 字节)", filename, bytes);

        manifest_chunks.push(ManifestChunk {
            id: "base".to_string(),
            filename,
            layer_start: 0,
            layer_end: 0,
            bytes,
            sha256,
            url: String::new(),
        });
    }

    // ─────────────────────────────────────────────────
    // 第四步：按层分组，写入 Layer 分块
    // ─────────────────────────────────────────────────

    let mut chunk_start: u32 = 0;

    while chunk_start < total_layers {
        let chunk_end = (chunk_start + args.layers_per_chunk).min(total_layers) - 1;
        let chunk_id = format!("layers_{}-{}", chunk_start, chunk_end);
        let filename = format!("{}.safetensors", chunk_id);
        let output_path = args.output.join(&filename);

        // 从 BTreeMap 中取出（drain）本分块涉及的所有层
        // 由于按顺序处理，每层只会出现在一个分块中
        let mut chunk_locs: Vec<TensorLocation> = Vec::new();
        let mut chunk_tensor_count = 0usize;

        for layer_idx in chunk_start..=chunk_end {
            if let Some(locs) = layer_tensors.remove(&layer_idx) {
                chunk_tensor_count += locs.len();
                chunk_locs.extend(locs);
            }
        }

        if chunk_locs.is_empty() {
            chunk_start = chunk_end + 1;
            continue;
        }

        println!(
            "正在写入分块 '{}' (层 {}-{}, {} 个 Tensor)...",
            chunk_id, chunk_start, chunk_end, chunk_tensor_count
        );

        let owned = load_tensors(&loaded_files, &chunk_locs)?;
        let (bytes, sha256) = write_safetensors(&owned, &output_path)?;

        println!("  -> {} ({} 字节)", filename, bytes);

        manifest_chunks.push(ManifestChunk {
            id: chunk_id,
            filename,
            layer_start: chunk_start,
            layer_end: chunk_end,
            bytes,
            sha256,
            url: String::new(),
        });

        chunk_start = chunk_end + 1;
    }

    // ─────────────────────────────────────────────────
    // 第五步：生成 Manifest 清单
    // ─────────────────────────────────────────────────

    let manifest = ModelManifest {
        model_id: args.model_id,
        version: "1.0.0".to_string(),
        dtype: "auto".to_string(),
        min_runnable_depth: args.layers_per_chunk,
        chunks: manifest_chunks,
    };

    manifest.validate().map_err(|e| anyhow::anyhow!(e))?;

    let manifest_path = args.output.join("manifest.json");
    let json = serde_json::to_string_pretty(&manifest)?;
    fs::write(&manifest_path, &json)?;

    println!("\n清单已写入 {}", manifest_path.display());
    println!(
        "完成！共生成 {} 个分块文件。",
        manifest.chunks.len()
    );

    Ok(())
}

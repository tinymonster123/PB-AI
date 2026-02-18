//! pb-sharder: Qwen2.5 Safetensors 模型分块工具
//!
//! 将 Qwen2.5 模型（Safetensors 格式）拆分为逻辑分块：
//! - Base 块: 包含 Embedding、Final RMSNorm、LM Head
//! - Layer 块: 按指定层数将 Transformer 层打包为独立文件
//!
//! 利用 memmap2 实现零拷贝读取，充分发挥 Apple Silicon 统一内存架构优势。

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use memmap2::Mmap;
use regex::Regex;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use sha2::{Digest, Sha256};

use manifest_core::{ManifestChunk, ModelManifest};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CLI 参数定义
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Parser, Debug)]
#[command(
    name = "pb-sharder",
    version,
    about = "将 Qwen2.5 safetensors 模型拆分为逻辑分块"
)]
struct Args {
    /// 包含源 .safetensors 文件的目录路径
    #[arg(long)]
    input: PathBuf,

    /// 输出目录（存放分块文件与 manifest.json）
    #[arg(long)]
    output: PathBuf,

    /// 模型标识 (如 "Qwen/Qwen2.5-3B")
    #[arg(long)]
    model_id: String,

    /// 每个分块包含的 Transformer 层数
    #[arg(long, default_value_t = 4)]
    layers_per_chunk: u32,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tensor 分类
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 记录某个 Tensor 在源文件中的位置
struct TensorLocation {
    /// 源文件在 loaded_files 数组中的索引
    file_idx: usize,
    /// 完整的 Tensor 名称 (如 "model.layers.3.self_attn.q_proj.weight")
    name: String,
}

/// Tensor 分类结果
enum TensorClass {
    /// 非层级 Tensor: embedding / final_norm / lm_head
    Base,
    /// 层级 Tensor，附带层索引号
    Layer(u32),
}

/// 根据 Qwen2.5 的 Tensor 命名规则进行分类
///
/// # Qwen2.5 Tensor 命名约定
///
/// ## Base（非层级）Tensor:
///   - `model.embed_tokens.weight`       — 词嵌入矩阵
///   - `model.norm.weight`               — 最终 RMSNorm（lm_head 前）
///   - `lm_head.weight`                  — 输出投影（词表 logits）
///
/// ## Layer（层级）Tensor（模式: `model.layers.{N}.{组件}`）:
///   - `model.layers.{N}.self_attn.q_proj.weight/bias`  — Query 投影（Qwen2.5 带 bias）
///   - `model.layers.{N}.self_attn.k_proj.weight/bias`  — Key 投影
///   - `model.layers.{N}.self_attn.v_proj.weight/bias`  — Value 投影
///   - `model.layers.{N}.self_attn.o_proj.weight`       — Output 投影
///   - `model.layers.{N}.mlp.gate_proj.weight`          — SwiGLU 门控
///   - `model.layers.{N}.mlp.up_proj.weight`            — SwiGLU 上投影
///   - `model.layers.{N}.mlp.down_proj.weight`          — SwiGLU 下投影
///   - `model.layers.{N}.input_layernorm.weight`        — 注意力前 RMSNorm
///   - `model.layers.{N}.post_attention_layernorm.weight` — 注意力后 RMSNorm
fn classify_tensor(name: &str, layer_re: &Regex) -> TensorClass {
    if let Some(caps) = layer_re.captures(name) {
        let layer_num: u32 = caps[1].parse().expect("层索引必须是有效的 u32");
        TensorClass::Layer(layer_num)
    } else {
        TensorClass::Base
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tensor 读取与 Safetensors 写入
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// 从 mmap 源文件中提取出来的 Tensor 完整数据（持有所有权）
struct OwnedTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// 按 TensorLocation 列表从 mmap 源文件批量加载 Tensor 数据
///
/// 内部按 file_idx 分组，避免对同一文件重复解析 Header。
/// mmap 的懒加载特性确保只有实际被访问的页面才会加载到物理内存，
/// 在 Apple Silicon 统一内存架构下尤为高效。
fn load_tensors(
    loaded_files: &[(PathBuf, File, Mmap)],
    locations: &[TensorLocation],
) -> Result<Vec<OwnedTensor>> {
    let mut result = Vec::with_capacity(locations.len());

    // 按源文件索引分组，减少重复的 Header 解析
    let mut by_file: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for loc in locations {
        by_file.entry(loc.file_idx).or_default().push(&loc.name);
    }

    for (file_idx, names) in &by_file {
        let (ref path, _, ref mmap) = loaded_files[*file_idx];
        let st = SafeTensors::deserialize(mmap)
            .with_context(|| format!("解析失败: {}", path.display()))?;

        for name in names {
            let tensor = st
                .tensor(name)
                .with_context(|| format!("Tensor '{}' 在 {} 中未找到", name, path.display()))?;
            result.push(OwnedTensor {
                name: name.to_string(),
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                // 从 mmap 区域拷贝数据到堆内存；
                // 由于按分块处理，每次只持有单个分块的数据量
                data: tensor.data().to_vec(),
            });
        }
    }

    Ok(result)
}

/// 将一组 OwnedTensor 序列化为新的 .safetensors 文件
///
/// 返回 (文件字节数, SHA-256 十六进制摘要)
fn write_safetensors(tensors: &[OwnedTensor], output_path: &Path) -> Result<(u64, String)> {
    // 构建 TensorView 引用，借用 OwnedTensor 中的数据
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|t| {
            let view =
                TensorView::new(t.dtype, t.shape.clone(), &t.data).expect("Tensor 数据格式无效");
            (t.name.as_str(), view)
        })
        .collect();

    let serialized =
        safetensors::serialize(views, &None).context("safetensors 序列化失败")?;

    // 写入前计算 SHA-256（单次遍历）
    let hash = {
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        format!("{:x}", hasher.finalize())
    };

    let size = serialized.len() as u64;

    fs::write(output_path, &serialized)
        .with_context(|| format!("写入失败: {}", output_path.display()))?;

    Ok((size, hash))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 主流程
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn main() -> Result<()> {
    let args = Args::parse();

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
    let mut safetensor_paths: Vec<PathBuf> = fs::read_dir(&args.input)?
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
    let loaded_files: Vec<(PathBuf, File, Mmap)> = safetensor_paths
        .iter()
        .map(|path| {
            let file =
                File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
            // SAFETY: 文件以只读方式打开，且在处理期间保持 File 句柄存活
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("mmap 失败: {}", path.display()))?;
            Ok((path.clone(), file, mmap))
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

    for (file_idx, (path, _, mmap)) in loaded_files.iter().enumerate() {
        let st = SafeTensors::deserialize(mmap)
            .with_context(|| format!("解析 Header 失败: {}", path.display()))?;

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

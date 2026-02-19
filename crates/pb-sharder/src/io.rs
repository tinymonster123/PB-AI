use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use sha2::{Digest, Sha256};

use crate::classify::TensorLocation;
use crate::LoadedFile;

/// 从 mmap 源文件中提取出来的 Tensor 完整数据（持有所有权）
pub struct OwnedTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// 按 TensorLocation 列表从 mmap 源文件批量加载 Tensor 数据
///
/// 内部按 file_idx 分组，避免对同一文件重复解析 Header。
/// mmap 的懒加载特性确保只有实际被访问的页面才会加载到物理内存，
/// 在 Apple Silicon 统一内存架构下尤为高效。
pub fn load_tensors(
    loaded_files: &[LoadedFile],
    locations: &[TensorLocation],
) -> Result<Vec<OwnedTensor>> {
    let mut result = Vec::with_capacity(locations.len());

    // 按源文件索引分组，减少重复的 Header 解析
    let mut by_file: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for loc in locations {
        by_file.entry(loc.file_idx).or_default().push(&loc.name);
    }

    for (file_idx, names) in &by_file {
        let loaded = &loaded_files[*file_idx];
        let st = SafeTensors::deserialize(&loaded.mmap)
            .with_context(|| format!("解析失败: {}", loaded.path.display()))?;

        for name in names {
            let tensor = st
                .tensor(name)
                .with_context(|| format!("Tensor '{}' 在 {} 中未找到", name, loaded.path.display()))?;
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
pub fn write_safetensors(tensors: &[OwnedTensor], output_path: &Path) -> Result<(u64, String)> {
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

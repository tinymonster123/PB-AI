use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use blake3;

use crate::classify::TensorLocation;
use crate::LoadedFile;

/// 从 mmap 源文件中提取出来的 Tensor 完整数据（持有所有权）
pub struct OwnedTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

pub struct LoadResult {
    pub tensors: Vec<OwnedTensor>,
    pub bytes_read: usize,
    pub timings: LoadTimings,
}

pub struct LoadTimings {
    pub deserialize_ms: u128,
    pub copy_ms: u128,
    pub total_ms: u128,
}

pub struct WriteTimings {
    pub serialize_ms: u128,
    pub hash_ms: u128,
    pub write_ms: u128,
    pub parallel_ms: u128,
    pub total_ms: u128,
}

/// 按 TensorLocation 列表从 mmap 源文件批量加载 Tensor 数据
///
/// 内部按 file_idx 分组，避免对同一文件重复解析 Header。
/// mmap 的懒加载特性确保只有实际被访问的页面才会加载到物理内存，
/// 在 Apple Silicon 统一内存架构下尤为高效。
pub fn load_tensors(
    loaded_files: &[LoadedFile],
    locations: &[TensorLocation],
) -> Result<LoadResult> {
    let total_start = Instant::now();
    let mut result = Vec::with_capacity(locations.len());
    let mut bytes_read = 0usize;
    let mut deserialize_ms = 0u128;
    let mut copy_ms = 0u128;

    // 按源文件索引分组，减少重复的 Header 解析
    let mut by_file: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for loc in locations {
        by_file.entry(loc.file_idx).or_default().push(&loc.name);
    }

    for (file_idx, names) in &by_file {
        let loaded = &loaded_files[*file_idx];
        let deserialize_start = Instant::now();
        let st = SafeTensors::deserialize(&loaded.mmap)
            .with_context(|| format!("解析失败: {}", loaded.path.display()))?;
        deserialize_ms += deserialize_start.elapsed().as_millis();

        for name in names {
            let tensor = st
                .tensor(name)
                .with_context(|| format!("Tensor '{}' 在 {} 中未找到", name, loaded.path.display()))?;
            bytes_read += tensor.data().len();

            let copy_start = Instant::now();
            let data = tensor.data().to_vec();
            copy_ms += copy_start.elapsed().as_millis();

            result.push(OwnedTensor {
                name: name.to_string(),
                dtype: tensor.dtype(),
                shape: tensor.shape().to_vec(),
                // 从 mmap 区域拷贝数据到堆内存；
                // 由于按分块处理，每次只持有单个分块的数据量
                data,
            });
        }
    }

    Ok(LoadResult {
        tensors: result,
        bytes_read,
        timings: LoadTimings {
            deserialize_ms,
            copy_ms,
            total_ms: total_start.elapsed().as_millis(),
        },
    })
}

/// 将一组 OwnedTensor 序列化为新的 .safetensors 文件
/// 返回 (文件字节数, BLAKE3 十六进制摘要, 计时信息)
pub fn write_safetensors(
    tensors: &[OwnedTensor],
    output_path: &Path,
) -> Result<(u64, String, WriteTimings)> {
    let total_start = Instant::now();
    // 构建 TensorView 引用，借用 OwnedTensor 中的数据
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|t| {
            let view =
                TensorView::new(t.dtype, t.shape.clone(), &t.data).expect("Tensor 数据格式无效");
            (t.name.as_str(), view)
        })
        .collect();

    let serialize_start = Instant::now();
    let serialized =
        safetensors::serialize(views, &None).context("safetensors 序列化失败")?;
    let serialize_ms = serialize_start.elapsed().as_millis();

    let size = serialized.len() as u64;

    let parallel_start = Instant::now();
    let (hash, hash_ms, write_ms) = std::thread::scope(|scope| {
        let serialized_ref = &serialized;
        let hash_handle = scope.spawn(move || -> Result<(String, u128)> {
            let hash_start = Instant::now();
            // 使用 BLAKE3 作为唯一哈希算法
            let hash = blake3::hash(serialized_ref).to_hex().to_string();
            let hash_ms = hash_start.elapsed().as_millis();
            Ok((hash, hash_ms))
        });

        let write_path = output_path.to_path_buf();
        let serialized_ref = &serialized;
        let write_handle = scope.spawn(move || -> Result<u128> {
            let write_start = Instant::now();
            fs::write(&write_path, serialized_ref)
                .with_context(|| format!("写入失败: {}", write_path.display()))?;
            Ok(write_start.elapsed().as_millis())
        });

        let (hash, hash_ms) = hash_handle
            .join()
            .map_err(|_| anyhow!("hash 线程异常终止"))??;
        let write_ms = write_handle
            .join()
            .map_err(|_| anyhow!("write 线程异常终止"))??;

        Ok::<(String, u128, u128), anyhow::Error>((hash, hash_ms, write_ms))
    })?;
    let parallel_ms = parallel_start.elapsed().as_millis();

    Ok((
        size,
        hash,
        WriteTimings {
            serialize_ms,
            hash_ms,
            write_ms,
            parallel_ms,
            total_ms: total_start.elapsed().as_millis(),
        },
    ))
}

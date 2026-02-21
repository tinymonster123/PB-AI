use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

pub struct ChunkPerf {
    pub id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub tensor_count: usize,
    pub bytes_read: usize,
    pub bytes_written: u64,
    pub load_deserialize_ms: u128,
    pub load_copy_ms: u128,
    pub load_total_ms: u128,
    pub serialize_ms: u128,
    pub hash_ms: u128,
    pub write_ms: u128,
    pub write_parallel_ms: u128,
    pub write_total_ms: u128,
    pub chunk_total_ms: u128,
}

pub fn format_metrics(
    files_count: usize,
    tensors_total: usize,
    base_tensors: usize,
    layer_tensors: usize,
    chunk_count: usize,
    chunk_avg_bytes: f64,
    bytes_read: usize,
    bytes_written: u64,
    scan_ms: u128,
    classify_ms: u128,
    load_deserialize_ms: u128,
    load_copy_ms: u128,
    load_ms: u128,
    serialize_ms: u128,
    hash_ms: u128,
    write_ms: u128,
    write_parallel_ms: u128,
    write_total_ms: u128,
    total_ms: u128,
    chunk_perfs: &[ChunkPerf],
) -> String {
    let mut out = format!(
        "files_count: {files_count}\n\
tensors_total: {tensors_total}\n\
base_tensors: {base_tensors}\n\
layer_tensors: {layer_tensors}\n\
chunk_count: {chunk_count}\n\
chunk_avg_bytes: {chunk_avg_bytes:.0}\n\
bytes_read: {bytes_read}\n\
bytes_written: {bytes_written}\n\
scan_ms: {scan_ms}\n\
classify_ms: {classify_ms}\n\
load_deserialize_ms: {load_deserialize_ms}\n\
load_copy_ms: {load_copy_ms}\n\
load_ms: {load_ms}\n\
serialize_ms: {serialize_ms}\n\
hash_ms: {hash_ms}\n\
write_ms: {write_ms}\n\
write_parallel_ms: {write_parallel_ms}\n\
write_total_ms: {write_total_ms}\n\
total_ms: {total_ms}"
    );

    out.push_str("\nchunk_perf_begin\n");
    for c in chunk_perfs {
        out.push_str(&format!(
            "chunk_id: {}\nlayer_start: {}\nlayer_end: {}\ntensor_count: {}\nbytes_read: {}\nbytes_written: {}\nload_deserialize_ms: {}\nload_copy_ms: {}\nload_total_ms: {}\nserialize_ms: {}\nhash_ms: {}\nwrite_ms: {}\nwrite_parallel_ms: {}\nwrite_total_ms: {}\nchunk_total_ms: {}\n---\n",
            c.id,
            c.layer_start,
            c.layer_end,
            c.tensor_count,
            c.bytes_read,
            c.bytes_written,
            c.load_deserialize_ms,
            c.load_copy_ms,
            c.load_total_ms,
            c.serialize_ms,
            c.hash_ms,
            c.write_ms,
            c.write_parallel_ms,
            c.write_total_ms,
            c.chunk_total_ms,
        ));
    }
    out.push_str("chunk_perf_end");

    out
}

pub fn write_metrics_file(metrics: &str) -> Result<PathBuf> {
    let analysis_dir = PathBuf::from("crates/pb-sharder/analysis");
    fs::create_dir_all(&analysis_dir)
        .with_context(|| format!("无法创建分析目录 {}", analysis_dir.display()))?;

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = format!("res-time-{}.txt", ts);
    let path = analysis_dir.join(filename);

    fs::write(&path, metrics)
        .with_context(|| format!("无法写入指标文件 {}", path.display()))?;

    Ok(path)
}

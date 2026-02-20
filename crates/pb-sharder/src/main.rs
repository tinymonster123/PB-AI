// 利用 memmap2 实现零拷贝读取

use std::fs::File;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use memmap2::Mmap;

mod classify;
mod io;
mod model_rules;
mod shard;
mod metrics;

// CLI 参数定义

#[derive(Parser, Debug)]
#[command(
    name = "pb-sharder",
    version,
    about = "将 safetensors 模型拆分为逻辑分块"
)]
pub struct Args {
    /// 包含源 .safetensors 文件的目录路径
    #[arg(long)]
    pub input: PathBuf,

    /// 输出目录（存放分块文件与 manifest.json）
    #[arg(long)]
    pub output: PathBuf,

    /// 模型标识 (如 "Qwen/Qwen2.5-3B")
    #[arg(long)]
    pub model_id: String,

    /// 每个分块包含的 Transformer 层数
    #[arg(long, default_value_t = 4)]
    pub layers_per_chunk: u32,
}

/// 已加载（mmap）的源文件
pub struct LoadedFile {
    pub path: PathBuf,
    pub _file: File,
    pub mmap: Mmap,
}

fn main() -> Result<()> {
    let args = Args::parse();
    shard::run(args)
}

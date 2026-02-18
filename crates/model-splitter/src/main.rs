use clap::Parser;
use manifest_core::{ManifestChunk, ModelManifest};

#[derive(Parser, Debug)]
#[command(author, version, about = "PB-AI 模型分片 CLI（占位实现）")]
struct Args {
    #[arg(long, default_value = "hf/tiny-model")]
    model_id: String,

    #[arg(long, default_value = "0.1.0")]
    version: String,

    #[arg(long, default_value = "q4")]
    dtype: String,

    #[arg(long, default_value_t = 8)]
    min_runnable_depth: u32,

    #[arg(long, default_value = "artifacts/manifest.json")]
    out: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let manifest = ModelManifest {
        model_id: args.model_id,
        version: args.version,
        dtype: args.dtype,
        min_runnable_depth: args.min_runnable_depth,
        chunks: vec![
            ManifestChunk {
                id: "embedding".to_string(),
                filename: "embedding.safetensors".to_string(),
                layer_start: 0,
                layer_end: 0,
                bytes: 123_456,
                sha256: "replace-with-real-sha256".to_string(),
                url: String::new(),
            },
            ManifestChunk {
                id: "block_0_7".to_string(),
                filename: "block_0_7.safetensors".to_string(),
                layer_start: 0,
                layer_end: 7,
                bytes: 456_789,
                sha256: "replace-with-real-sha256".to_string(),
                url: String::new(),
            },
        ],
    };

    manifest
        .validate()
        .map_err(|msg| format!("清单校验失败: {}", msg))?;

    let json = serde_json::to_string_pretty(&manifest)?;
    let out_path = std::path::Path::new(&args.out);

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(out_path, json)?;
    println!("清单已写入 {}", out_path.display());

    Ok(())
}

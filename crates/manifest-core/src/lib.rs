use serde::{Deserialize, Serialize};

/// 单个分块的元数据描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestChunk {
    /// 分块唯一标识 (如 "base", "layers_0-3")
    pub id: String,
    /// 输出文件名 (如 "base.safetensors")
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub filename: String,
    /// 起始层索引 (base 块为 0)
    pub layer_start: u32,
    /// 结束层索引 (base 块为 0)
    pub layer_end: u32,
    /// 文件大小（字节）
    pub bytes: u64,
    /// SHA-256 校验值
    pub sha256: String,
    /// 远端下载地址（由上传工具填充）
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub url: String,
}

/// 模型分块清单（描述整个模型的拆分结构）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// 模型标识 (如 "Qwen/Qwen2.5-3B")
    pub model_id: String,
    /// 清单版本号
    pub version: String,
    /// 权重数据类型 (如 "bf16", "fp16", "auto")
    pub dtype: String,
    /// 最小可运行深度（至少需要多少层才能推理）
    pub min_runnable_depth: u32,
    /// 分块列表
    pub chunks: Vec<ManifestChunk>,
}

impl ModelManifest {
    /// 校验清单完整性
    pub fn validate(&self) -> Result<(), String> {
        if self.chunks.is_empty() {
            return Err("chunks 不能为空".to_string());
        }

        if self.min_runnable_depth == 0 {
            return Err("min_runnable_depth 必须 > 0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunk() -> ManifestChunk {
        ManifestChunk {
            id: "base".to_string(),
            filename: "base.safetensors".to_string(),
            layer_start: 0,
            layer_end: 0,
            bytes: 1024,
            sha256: "abc123".to_string(),
            url: String::new(),
        }
    }

    fn sample_manifest() -> ModelManifest {
        ModelManifest {
            model_id: "Qwen/Qwen2.5-3B".to_string(),
            version: "1.0.0".to_string(),
            dtype: "auto".to_string(),
            min_runnable_depth: 4,
            chunks: vec![sample_chunk()],
        }
    }

    #[test]
    fn validate_ok() {
        assert!(sample_manifest().validate().is_ok());
    }

    #[test]
    fn validate_empty_chunks() {
        let mut m = sample_manifest();
        m.chunks.clear();
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_zero_min_runnable_depth() {
        let mut m = sample_manifest();
        m.min_runnable_depth = 0;
        assert!(m.validate().is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let manifest = sample_manifest();
        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: ModelManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_id, manifest.model_id);
        assert_eq!(deserialized.chunks.len(), 1);
        assert_eq!(deserialized.chunks[0].id, "base");
    }

    #[test]
    fn serde_skip_empty_fields() {
        let chunk = sample_chunk();
        let json = serde_json::to_string(&chunk).unwrap();
        // url is empty, should be skipped
        assert!(!json.contains("\"url\""));
    }

    #[test]
    fn serde_includes_nonempty_url() {
        let mut chunk = sample_chunk();
        chunk.url = "https://example.com/base.safetensors".to_string();
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"url\""));
    }
}

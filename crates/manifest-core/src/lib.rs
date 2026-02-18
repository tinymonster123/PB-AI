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

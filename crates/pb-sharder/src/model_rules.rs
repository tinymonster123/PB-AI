use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;

pub struct ModelRules {
    pub model_type: Option<String>,
    pub layer_re: Regex,
}

pub fn rules_from_input_dir(input_dir: &Path) -> Result<ModelRules> {
    let config_path = input_dir.join("config.json");
    let model_type = read_model_type(&config_path)?;

    let layer_re = if model_type
        .as_deref()
        .map(is_qwen_family)
        .unwrap_or(true)
    {
        Regex::new(r"^model\.layers\.(\d+)\.")?
    } else {
        // 目前仅实现 Qwen 系列规则，其它架构先回退到同一规则。
        Regex::new(r"^model\.layers\.(\d+)\.")?
    };

    Ok(ModelRules {
        model_type,
        layer_re,
    })
}

fn read_model_type(config_path: &PathBuf) -> Result<Option<String>> {
    if !config_path.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(config_path)
        .with_context(|| format!("读取 config 失败: {}", config_path.display()))?;
    let value: Value = serde_json::from_str(&raw)
        .with_context(|| format!("解析 config 失败: {}", config_path.display()))?;

    Ok(value
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string()))
}

fn is_qwen_family(model_type: &str) -> bool {
    model_type.to_ascii_lowercase().contains("qwen")
}

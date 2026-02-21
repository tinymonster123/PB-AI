use regex::Regex;

/// 记录某个 Tensor 在源文件中的位置
pub struct TensorLocation {
    /// 源文件在 loaded_files 数组中的索引
    pub file_idx: usize,
    /// 完整的 Tensor 名称 (如 "model.layers.3.self_attn.q_proj.weight")
    pub name: String,
}

/// Tensor 分类结果
pub enum TensorClass {
    /// 非层级 Tensor: embedding / final_norm / lm_head
    Base,
    /// 层级 Tensor，附带层索引号
    Layer(u32),
}

pub fn classify_tensor(name: &str, layer_re: &Regex) -> TensorClass {
    if let Some(caps) = layer_re.captures(name) {
        let layer_num: u32 = caps[1].parse().expect("层索引必须是有效的 u32");
        TensorClass::Layer(layer_num)
    } else {
        TensorClass::Base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_re() -> Regex {
        Regex::new(r"^model\.layers\.(\d+)\.").unwrap()
    }

    #[test]
    fn classify_embed_tokens_as_base() {
        let re = layer_re();
        assert!(matches!(
            classify_tensor("model.embed_tokens.weight", &re),
            TensorClass::Base
        ));
    }

    #[test]
    fn classify_final_norm_as_base() {
        let re = layer_re();
        assert!(matches!(
            classify_tensor("model.norm.weight", &re),
            TensorClass::Base
        ));
    }

    #[test]
    fn classify_lm_head_as_base() {
        let re = layer_re();
        assert!(matches!(
            classify_tensor("lm_head.weight", &re),
            TensorClass::Base
        ));
    }

    #[test]
    fn classify_layer_self_attn() {
        let re = layer_re();
        match classify_tensor("model.layers.5.self_attn.q_proj.weight", &re) {
            TensorClass::Layer(n) => assert_eq!(n, 5),
            TensorClass::Base => panic!("expected Layer"),
        }
    }

    #[test]
    fn classify_layer_mlp() {
        let re = layer_re();
        match classify_tensor("model.layers.12.mlp.gate_proj.weight", &re) {
            TensorClass::Layer(n) => assert_eq!(n, 12),
            TensorClass::Base => panic!("expected Layer"),
        }
    }

    #[test]
    fn classify_layer_layernorm() {
        let re = layer_re();
        match classify_tensor("model.layers.0.input_layernorm.weight", &re) {
            TensorClass::Layer(n) => assert_eq!(n, 0),
            TensorClass::Base => panic!("expected Layer"),
        }
    }

    #[test]
    fn classify_unknown_tensor_as_base() {
        let re = layer_re();
        assert!(matches!(
            classify_tensor("some.random.tensor.name", &re),
            TensorClass::Base
        ));
    }
}

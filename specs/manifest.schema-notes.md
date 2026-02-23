# Manifest Schema v0.2

面向"预合并 LoRA + 差分分片缓存"架构设计。每个模型变体独立一份 manifest，浏览器通过比较 shard hash 实现跨变体缓存复用。

## 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `version` | string | ✅ | schema 版本，当前 `"0.2"` |
| `model_id` | string | ✅ | 模型标识，如 `"Qwen/Qwen2.5-0.5B-Instruct"` |
| `variant` | string | ✅ | 变体名，如 `"base"`, `"lora-code"`, `"lora-chat"` |
| `framework` | string | ✅ | 推理框架，当前固定 `"onnxruntime-web"` |
| `dtype` | string | ✅ | 量化类型，如 `"int8"`, `"q4f16"`, `"fp16"` |
| `total_layers` | int | ✅ | Transformer 层总数 |
| `shards` | Shard[] | ✅ | 分片列表 |

## Shard 字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | ✅ | 分片唯一 ID，如 `"embed"`, `"layer_0"`, `"lm_head"` |
| `kind` | enum | ✅ | 分片类型：`"embed"` \| `"layer"` \| `"lm_head"` |
| `filename` | string | ✅ | 文件名（相对于 manifest 所在目录） |
| `bytes` | int | ✅ | 文件字节大小 |
| `hash` | string | ✅ | 格式 `"blake3:<hex>"`，用于完整性校验和跨变体去重 |
| `layer_range` | [int, int] | 仅 layer | 层范围 [start, end]，含两端 |

## 浏览器差分缓存流程

```
1. 首次加载变体 A:
   GET manifest.json → 解析 shards → 全部下载 → IndexedDB 缓存 (key: hash)

2. 切换到变体 B:
   GET manifest.json → 对比已缓存 hash →
     embed  hash 相同 → 跳过 ✓
     lm_head hash 相同 → 跳过 ✓
     layer_0 hash 不同 → 下载 ↓
     layer_1 hash 不同 → 下载 ↓
     ...
   → 重建 InferenceSession
```

`kind` 字段的意义：
- `embed` / `lm_head`：LoRA 不影响这些权重，跨变体 hash 一定相同，浏览器可预判跳过比较
- `layer`：LoRA 合并后权重不同，需要逐个比较 hash

## 示例 manifest.json

```json
{
  "version": "0.2",
  "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
  "variant": "lora-code",
  "framework": "onnxruntime-web",
  "dtype": "q4f16",
  "total_layers": 24,
  "shards": [
    {
      "id": "embed",
      "kind": "embed",
      "filename": "model.onnx_data_embed",
      "bytes": 31457280,
      "hash": "blake3:a1b2c3d4..."
    },
    {
      "id": "layer_0",
      "kind": "layer",
      "filename": "model.onnx_data_0",
      "bytes": 18874368,
      "hash": "blake3:e5f6a7b8...",
      "layer_range": [0, 0]
    },
    {
      "id": "layer_1",
      "kind": "layer",
      "filename": "model.onnx_data_1",
      "bytes": 18874368,
      "hash": "blake3:c9d0e1f2...",
      "layer_range": [1, 1]
    },
    {
      "id": "lm_head",
      "kind": "lm_head",
      "filename": "model.onnx_data_lm_head",
      "bytes": 31457280,
      "hash": "blake3:12345678..."
    }
  ]
}
```

## 与 v0.1 的变更

| v0.1 | v0.2 | 原因 |
|------|------|------|
| `chunks[]` | `shards[]` | 与工具名 "sharder" 一致 |
| `sha256` | `hash` (带算法前缀) | 当前用 blake3，保留扩展性 |
| `layerStart` / `layerEnd` | `layer_range: [start, end]` | 更紧凑，仅 layer 类型需要 |
| 无 | `kind` | 浏览器差分缓存的核心判据 |
| 无 | `variant` | 标识预合并的 LoRA 变体 |
| 无 | `framework` | 明确推理运行时 |
| `minRunnableDepth` | `total_layers` | 语义更清晰 |

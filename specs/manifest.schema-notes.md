# Manifest 草案 v0.1（MVP）

字段约定：

- `modelId`: 模型标识，如 `hf/Qwen2.5-0.5B-Instruct`
- `version`: 分发版本号
- `dtype`: 量化或精度标识，如 `q4` / `fp16`
- `minRunnableDepth`: 最小可运行层深
- `chunks[]`: 按层范围划分的权重块元信息

`chunks[]` 结构：

- `id`: chunk 唯一 ID
- `layerStart`, `layerEnd`: 层范围（含边界）
- `bytes`: chunk 字节大小
- `sha256`: 完整性校验
- `url`: 下载地址

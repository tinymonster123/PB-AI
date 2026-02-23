"""将分类后的 initializer 写入编号的 external data 文件。

split_base=True (默认) 输出结构:
  model.onnx                  — 图定义（无内联权重）
  model.onnx_data_embed       — embed_tokens + norm 权重
  model.onnx_data_0           — layer 0
  model.onnx_data_1           — layer 1
  ...
  model.onnx_data_lm_head     — lm_head 权重

split_base=False 输出结构 (兼容旧模式):
  model.onnx          — 图定义
  model.onnx_data_0   — 所有 base 权重 (embed + norm + lm_head)
  model.onnx_data_1   — layers 0..N-1
  ...
"""

from pathlib import Path

import blake3
import onnx
from onnx import TensorProto

from ..parser.classify import ClassifyResult
from .manifest import Shard, ShardKind


def _tensor_raw_bytes(tensor: TensorProto) -> bytes:
    """提取 tensor 的原始字节数据。"""
    if tensor.raw_data:
        return tensor.raw_data
    arr = onnx.numpy_helper.to_array(tensor)
    return arr.tobytes()


def _write_data_file(
    tensors: list[TensorProto],
    data_path: Path,
) -> int:
    """将一组 tensor 写入单个 external data 文件。

    同时更新每个 tensor 的 external_data 引用（文件名、偏移量、长度）。

    Returns:
        写入的总字节数
    """
    offset = 0
    data_filename = data_path.name

    with open(data_path, "wb") as f:
        for tensor in tensors:
            raw = _tensor_raw_bytes(tensor)
            length = len(raw)
            f.write(raw)

            # 清除内联数据
            tensor.raw_data = b""
            tensor.ClearField("float_data")
            tensor.ClearField("int32_data")
            tensor.ClearField("int64_data")
            tensor.ClearField("double_data")

            # 设置 external data 引用
            tensor.data_location = TensorProto.EXTERNAL
            del tensor.external_data[:]
            tensor.external_data.add(key="location", value=data_filename)
            tensor.external_data.add(key="offset", value=str(offset))
            tensor.external_data.add(key="length", value=str(length))

            offset += length

    return offset


def _blake3_file(path: Path) -> str:
    """计算文件的 BLAKE3 哈希。"""
    hasher = blake3.blake3()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_shard(
    tensors: list[TensorProto],
    shard_id: str,
    kind: ShardKind,
    data_path: Path,
    label: str,
    layer_range: tuple[int, int] | None = None,
) -> Shard | None:
    """写入一个分片并返回 Shard，无 tensor 时返回 None。"""
    if not tensors:
        return None

    total_bytes = _write_data_file(tensors, data_path)
    file_hash = _blake3_file(data_path)
    print(f"  {data_path.name}: {label} ({total_bytes / 1024 / 1024:.1f} MB)")

    return Shard(
        id=shard_id,
        kind=kind,
        filename=data_path.name,
        bytes=total_bytes,
        hash=file_hash,
        layer_range=layer_range,
    )


def write_shards(
    model: onnx.ModelProto,
    classify_result: ClassifyResult,
    output_dir: Path,
    layers_per_chunk: int,
    split_base: bool = True,
) -> list[Shard]:
    """将模型写为精简 ONNX + 编号 external data 文件。

    Args:
        model: 加载的 ONNX 模型
        classify_result: 分类结果
        output_dir: 输出目录
        layers_per_chunk: 每个分片的层数
        split_base: 是否将 base 拆分为 embed/lm_head 独立分片

    Returns:
        Shard 列表（用于生成 manifest.json）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Shard] = []
    data_idx = 0

    if split_base:
        # --- embed 分片 (embed_tokens + norm) ---
        embed_tensors = classify_result.embed + classify_result.norm
        shard = _write_shard(
            embed_tensors, "embed", "embed",
            output_dir / "model.onnx_data_embed", "embed + norm",
        )
        if shard:
            shards.append(shard)

        # --- layer 分片 ---
        if classify_result.max_layer < 0:
            print("警告: 未找到任何层级 tensor")
        else:
            total_layers = classify_result.max_layer + 1
            for group_start in range(0, total_layers, layers_per_chunk):
                group_end = min(group_start + layers_per_chunk, total_layers) - 1
                data_path = output_dir / f"model.onnx_data_{group_start}"

                group_tensors: list[TensorProto] = []
                for layer_idx in range(group_start, group_end + 1):
                    group_tensors.extend(classify_result.layers.get(layer_idx, []))

                if not group_tensors:
                    continue

                if group_start == group_end:
                    shard_id = f"layer_{group_start}"
                    label = f"layer {group_start}"
                else:
                    shard_id = f"layers_{group_start}-{group_end}"
                    label = f"layers {group_start}-{group_end}"

                shard = _write_shard(
                    group_tensors, shard_id, "layer", data_path, label,
                    layer_range=(group_start, group_end),
                )
                if shard:
                    shards.append(shard)

        # --- lm_head 分片 ---
        shard = _write_shard(
            classify_result.lm_head, "lm_head", "lm_head",
            output_dir / "model.onnx_data_lm_head", "lm_head",
        )
        if shard:
            shards.append(shard)

    else:
        # --- 旧模式: 所有 base 权重合并为一个分片 ---
        all_base = classify_result.embed + classify_result.norm + classify_result.lm_head
        if all_base:
            shard = _write_shard(
                all_base, "embed", "embed",
                output_dir / f"model.onnx_data_{data_idx}", "base (embed + norm + lm_head)",
            )
            if shard:
                shards.append(shard)
            data_idx += 1

        # --- 按层分组 ---
        if classify_result.max_layer < 0:
            print("警告: 未找到任何层级 tensor")
        else:
            total_layers = classify_result.max_layer + 1
            for group_start in range(0, total_layers, layers_per_chunk):
                group_end = min(group_start + layers_per_chunk, total_layers) - 1
                data_path = output_dir / f"model.onnx_data_{data_idx}"

                group_tensors: list[TensorProto] = []
                for layer_idx in range(group_start, group_end + 1):
                    group_tensors.extend(classify_result.layers.get(layer_idx, []))

                if not group_tensors:
                    continue

                shard_id = f"layers_{group_start}-{group_end}"
                shard = _write_shard(
                    group_tensors, shard_id, "layer", data_path,
                    f"layers {group_start}-{group_end}",
                    layer_range=(group_start, group_end),
                )
                if shard:
                    shards.append(shard)
                data_idx += 1

    # --- 保存精简 model.onnx ---
    model_path = output_dir / "model.onnx"
    onnx.save(model, str(model_path))
    model_size = model_path.stat().st_size
    print(f"  model.onnx: graph only ({model_size / 1024 / 1024:.1f} MB)")

    return shards

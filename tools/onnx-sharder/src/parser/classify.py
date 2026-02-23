"""按层分类 ONNX initializer。

支持两种命名模式:
1. 原始命名: model.layers.N.* (safetensors 风格)
2. 量化命名: onnx::MatMul_XXXX (INT8 量化后的命名)
   通过图节点名 /model/layers.N/ 反向追踪到层索引

Base 权重细分为:
- embed: model.embed_tokens.* 相关权重
- lm_head: /lm_head/ 节点相关权重
- norm: model.norm.* (最终 RMSNorm，很小)
"""

import re
from dataclasses import dataclass, field
from typing import Literal

import onnx

# 匹配 initializer 名称中的层索引
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
# 匹配图节点名称中的层索引
NODE_LAYER_RE = re.compile(r"/model/layers\.(\d+)/")
# 匹配 embed_tokens
EMBED_RE = re.compile(r"^model\.embed_tokens\.")
# 匹配最终 norm
NORM_RE = re.compile(r"^model\.norm\.")


@dataclass
class ClassifyResult:
    """分类结果：embed/lm_head/norm 和按层分组的权重。"""

    embed: list[onnx.TensorProto] = field(default_factory=list)
    lm_head: list[onnx.TensorProto] = field(default_factory=list)
    norm: list[onnx.TensorProto] = field(default_factory=list)
    layers: dict[int, list[onnx.TensorProto]] = field(default_factory=dict)
    max_layer: int = -1


def _build_init_to_layer_map(graph: onnx.GraphProto) -> dict[str, int]:
    """通过图节点名称，将 initializer 名称映射到层索引。

    INT8 量化后权重名变为 onnx::MatMul_XXXX，但图节点名保留了
    /model/layers.N/ 结构，可以反向追踪。
    """
    init_to_layer: dict[str, int] = {}
    for node in graph.node:
        m = NODE_LAYER_RE.search(node.name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        for inp in node.input:
            if inp not in init_to_layer:
                init_to_layer[inp] = layer_idx
    return init_to_layer


def _build_init_to_lm_head(graph: onnx.GraphProto) -> set[str]:
    """通过图节点名称，找出属于 /lm_head/ 的 initializer。

    量化后 lm_head 权重名变为 onnx::MatMul_XXXX，
    通过节点名 /lm_head/ 反向追踪。
    """
    lm_head_inits: set[str] = set()
    for node in graph.node:
        if "/lm_head/" in node.name:
            for inp in node.input:
                lm_head_inits.add(inp)
    return lm_head_inits


def classify_tensor(
    name: str,
    init_to_layer: dict[str, int],
    lm_head_inits: set[str],
) -> tuple[Literal["embed", "lm_head", "norm", "layer"], int | None]:
    """将 tensor 名称分类为 embed/lm_head/norm/layer(N)。

    优先用 initializer 名称匹配，fallback 到图节点追踪。
    """
    # 方式 1: 直接从名称匹配层 (model.layers.N.*)
    m = LAYER_RE.match(name)
    if m:
        return "layer", int(m.group(1))

    # 方式 2: 通过图节点反向追踪层 (onnx::MatMul_XXXX)
    if name in init_to_layer:
        return "layer", init_to_layer[name]

    # embed_tokens
    if EMBED_RE.match(name):
        return "embed", None

    # 最终 norm
    if NORM_RE.match(name):
        return "norm", None

    # lm_head (通过图节点追踪)
    if name in lm_head_inits:
        return "lm_head", None

    # 未匹配的 base 权重默认归入 embed
    return "embed", None


def classify_initializers(
    initializers: list[onnx.TensorProto],
    graph: onnx.GraphProto,
) -> ClassifyResult:
    """将所有 initializer 按层分类。"""
    init_to_layer = _build_init_to_layer_map(graph)
    lm_head_inits = _build_init_to_lm_head(graph)
    result = ClassifyResult()

    for tensor in initializers:
        kind, layer_idx = classify_tensor(tensor.name, init_to_layer, lm_head_inits)
        if kind == "layer":
            result.layers.setdefault(layer_idx, []).append(tensor)
            result.max_layer = max(result.max_layer, layer_idx)
        elif kind == "embed":
            result.embed.append(tensor)
        elif kind == "lm_head":
            result.lm_head.append(tensor)
        elif kind == "norm":
            result.norm.append(tensor)

    return result


def print_summary(result: ClassifyResult) -> None:
    """打印分类摘要表。"""

    def _group_size(tensors: list[onnx.TensorProto]) -> int:
        return sum(len(t.raw_data) if t.raw_data else 0 for t in tensors)

    embed_total = _group_size(result.embed)
    lm_head_total = _group_size(result.lm_head)
    norm_total = _group_size(result.norm)
    layer_total = sum(_group_size(ts) for ts in result.layers.values())

    print(f"\n{'='*60}")
    print(f"{'分类摘要':^60}")
    print(f"{'='*60}")

    print(f"  Embed tensors: {len(result.embed)} ({embed_total / 1024 / 1024:.1f} MB)")
    for tensor in result.embed:
        size = len(tensor.raw_data) if tensor.raw_data else 0
        if size > 0.01 * 1024 * 1024:
            print(f"    - {tensor.name} ({size / 1024 / 1024:.2f} MB)")

    print(f"  LM Head tensors: {len(result.lm_head)} ({lm_head_total / 1024 / 1024:.1f} MB)")
    for tensor in result.lm_head:
        size = len(tensor.raw_data) if tensor.raw_data else 0
        if size > 0.01 * 1024 / 1024:
            print(f"    - {tensor.name} ({size / 1024 / 1024:.2f} MB)")

    print(f"  Norm tensors: {len(result.norm)} ({norm_total / 1024 / 1024:.1f} MB)")
    for tensor in result.norm:
        size = len(tensor.raw_data) if tensor.raw_data else 0
        if size > 0.01 * 1024 * 1024:
            print(f"    - {tensor.name} ({size / 1024 / 1024:.2f} MB)")

    print(f"\n  Layer tensors: {sum(len(v) for v in result.layers.values())} "
          f"across {len(result.layers)} layers (0..{result.max_layer}), "
          f"total {layer_total / 1024 / 1024:.1f} MB")

    for layer_idx in sorted(result.layers.keys()):
        tensors = result.layers[layer_idx]
        total = sum(len(t.raw_data) if t.raw_data else 0 for t in tensors)
        print(f"    Layer {layer_idx:3d}: {len(tensors):3d} tensors, {total / 1024 / 1024:.1f} MB")
    print(f"{'='*60}\n")

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

import onnx
from onnx import TensorProto


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "models" / "tinyllama-1.1b-chat-int8"


def _pick_model_path() -> List[Path]:
    candidates = [
        OUTPUT_DIR / "model_quantized.onnx",
        OUTPUT_DIR / "onnx_export" / "model.onnx",
    ]
    paths = []
    for path in candidates:
        if path.exists():
            paths.append(path)
    if paths:
        return paths

    raise FileNotFoundError(
        "未找到 ONNX 文件，请先运行量化。已检查: "
        + ", ".join(str(p) for p in candidates)
    )


def _dtype_name(dtype_id: int) -> str:
    try:
        return TensorProto.DataType.Name(dtype_id)
    except Exception:
        return str(dtype_id)


def inspect_onnx(model_paths: List[Path]) -> None:
    quant_ops = [
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "QLinearMatMul",
        "MatMulInteger",
        "QLinearConv",
    ]

    for model_path in model_paths:
        try:
            model = onnx.load(str(model_path), load_external_data=True)
        except Exception as exc:
            print(f"无法加载 ONNX 文件: {model_path} -> {exc}")
            continue

        init_dtype_counter = Counter(t.data_type for t in model.graph.initializer)
        op_counter = Counter(node.op_type for node in model.graph.node)

        print("" + "=" * 10)
        print(f"ONNX 文件: {model_path}")
        print(f"initializer 数量: {len(model.graph.initializer)}")
        print("\n=== Initializer dtype 分布 ===")
        for dtype_id, count in sorted(init_dtype_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"{_dtype_name(dtype_id):>16}: {count}")

        print("\n=== 量化相关算子 ===")
        has_quant_op = False
        for op in quant_ops:
            count = op_counter.get(op, 0)
            if count > 0:
                has_quant_op = True
                print(f"{op:>20}: {count}")

        if not has_quant_op:
            print("未检测到常见量化算子（可能是未量化模型）")

        int8_inits = init_dtype_counter.get(TensorProto.INT8, 0)
        uint8_inits = init_dtype_counter.get(TensorProto.UINT8, 0)
        if int8_inits + uint8_inits > 0 or has_quant_op:
            print("\n结论: 检测到量化特征（INT8/UINT8 initializer 或量化算子）。")
        else:
            print("\n结论: 未检测到明显 INT8 量化特征。")


def main() -> None:
    model_path = _pick_model_path()
    inspect_onnx(model_path)


if __name__ == "__main__":
    main()

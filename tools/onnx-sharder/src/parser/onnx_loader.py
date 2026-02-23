"""ONNX protobuf 加载与 initializer 提取。"""

from pathlib import Path

import onnx


def load_onnx_model(path: Path) -> onnx.ModelProto:
    """加载 ONNX 模型（含 external data）。

    使用 load_external_data=True 确保所有权重数据都加载到内存中，
    即使原模型使用了 external data 格式。
    """
    model_path = str(path)
    model = onnx.load(model_path, load_external_data=True)
    print(f"已加载模型: {path.name}")
    print(f"  IR version: {model.ir_version}")
    opsets = [f'{o.domain or "ai.onnx"}:{o.version}' for o in model.opset_import]
    print(f"  Opset: {opsets}")
    print(f"  Graph nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")
    return model

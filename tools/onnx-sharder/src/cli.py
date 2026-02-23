"""argparse CLI for onnx-sharder。"""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 ONNX 模型按 Transformer 层切分为多个 external data 文件",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 ONNX 模型路径 (如 model_quantized.onnx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="模型标识 (如 Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        help="变体名称，用于标识预合并的 LoRA (默认: base)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="int8",
        help="量化类型 (如 int8, q4f16, fp16，默认: int8)",
    )
    parser.add_argument(
        "--layers-per-chunk",
        type=int,
        default=1,
        help="每个分片包含的 Transformer 层数 (默认: 1)",
    )
    parser.add_argument(
        "--split-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将 base 权重拆分为 embed/lm_head 独立分片 (默认: 开启，--no-split-base 关闭)",
    )
    parser.add_argument(
        "--copy-tokenizer",
        type=Path,
        default=None,
        help="从指定目录复制 tokenizer 文件",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llama",
        help="模型架构类型 (默认: llama)",
    )
    return parser.parse_args()

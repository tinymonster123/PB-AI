"""生成 transformers.js 兼容的 config.json 并复制 tokenizer 文件。"""

import json
import shutil
from pathlib import Path


def generate_config(
    output_dir: Path,
    num_data_files: int,
    model_type: str = "llama",
) -> Path:
    """生成含 transformers.js_config 的 config.json。

    Args:
        output_dir: 输出目录
        num_data_files: external data 文件数量
        model_type: 模型架构类型
    """
    config = {
        "transformers.js_config": {
            "model_type": model_type,
            "use_external_data_format": num_data_files,
        }
    }

    path = output_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"已写入 {path}")
    return path


def copy_tokenizer(src_dir: Path, output_dir: Path) -> list[Path]:
    """从原模型目录复制 tokenizer 文件。"""
    copied = []
    for name in ("tokenizer.json", "tokenizer_config.json"):
        src = src_dir / name
        if src.exists():
            dst = output_dir / name
            shutil.copy2(src, dst)
            copied.append(dst)
            print(f"已复制 {name}")
        else:
            print(f"警告: 未找到 {src}")
    return copied

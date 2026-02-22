from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "tinyllama-1.1b-chat"
# 输出到 pb-ai 根目录下的 models/
OUTPUT_DIR = PROJECT_ROOT / "models" / "tinyllama-1.1b-chat-int8"


def quantize():
    if not LOCAL_MODEL_DIR.exists():
        raise FileNotFoundError(
            "未找到本地模型目录。请先执行: python src/load_models.py"
        )

    onnx_dir = OUTPUT_DIR / "onnx_export"

    # 导出 ONNX 模型（直接使用本地已下载模型）
    print(f"正在从本地模型导出 ONNX: {LOCAL_MODEL_DIR}")
    model = ORTModelForCausalLM.from_pretrained(str(LOCAL_MODEL_DIR), export=True)
    model.save_pretrained(onnx_dir)

    # INT8 动态量化
    print("正在进行 INT8 量化 ...")
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=False,
        per_channel=False,
    )
    quantizer.quantize(save_dir=OUTPUT_DIR, quantization_config=qconfig)

    # 复制 tokenizer 和配置文件到输出目录
    print("正在复制 tokenizer 和配置文件 ...")
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
    tokenizer.save_pretrained(OUTPUT_DIR)

    for cfg_file in onnx_dir.glob("*.json"):
        if cfg_file.name not in ("quantize_config.json",):
            dest = OUTPUT_DIR / cfg_file.name
            if not dest.exists():
                dest.write_text(cfg_file.read_text())

    print(f"量化完成！输出目录: {OUTPUT_DIR}")

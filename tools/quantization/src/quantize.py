from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# 输出到 pb-ai 根目录下的 models/
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "models" / "tinyllama-1.1b-chat-int8"


def quantize():
    onnx_dir = OUTPUT_DIR / "onnx_export"

    # 导出 ONNX 模型
    print(f"正在导出模型 {MODEL_ID} 到 ONNX ...")
    model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(OUTPUT_DIR)

    for cfg_file in onnx_dir.glob("*.json"):
        if cfg_file.name not in ("quantize_config.json",):
            dest = OUTPUT_DIR / cfg_file.name
            if not dest.exists():
                dest.write_text(cfg_file.read_text())

    print(f"量化完成！输出目录: {OUTPUT_DIR}")

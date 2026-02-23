import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

# pb-ai 根目录下 models/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "tinyllama-1.1b-chat"


def has_local_tinyllama(local_dir: Path = LOCAL_MODEL_DIR) -> bool:
    if not local_dir.exists() or not local_dir.is_dir():
        return False

    required_files = (
        "config.json",
        "tokenizer_config.json",
    )
    has_required = all((local_dir / name).exists() for name in required_files)
    has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
    return has_required and has_weights


def download_tinyllama(local_dir: Path = LOCAL_MODEL_DIR) -> Path:
    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"使用镜像源: {os.environ['HF_ENDPOINT']}")
    print(f"正在下载模型 {MODEL_ID} 到: {local_dir}")

    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"下载完成！本地模型目录: {local_dir}")
    print(f"snapshot 缓存路径: {snapshot_path}")
    return local_dir


def ensure_tinyllama_local(local_dir: Path = LOCAL_MODEL_DIR) -> Path:
    if has_local_tinyllama(local_dir):
        print(f"检测到本地模型，跳过下载: {local_dir}")
        return local_dir

    return download_tinyllama(local_dir)
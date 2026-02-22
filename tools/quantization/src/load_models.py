import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

# pb-ai 根目录下 models/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "tinyllama-1.1b-chat"


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
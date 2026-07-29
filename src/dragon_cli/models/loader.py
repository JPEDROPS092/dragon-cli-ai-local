"""Model download and loading (huggingface_hub + llama-cpp-python)."""

import os
from pathlib import Path

from ..config import C_GREEN, C_RESET, C_YELLOW, MODELS_DIR


def download_model(model: dict) -> Path:
    from huggingface_hub import hf_hub_download
    MODELS_DIR.mkdir(exist_ok=True)
    destination = MODELS_DIR / model["filename"]
    if destination.exists():
        print(f"{C_GREEN}[ok]{C_RESET} Model already saved at {destination}")
        return destination
    print(f"{C_YELLOW}[download]{C_RESET} Downloading {model['filename']} "
          f"(a few GB, only happens once)...")
    path = hf_hub_download(repo_id=model["repo_id"], filename=model["filename"], local_dir=str(MODELS_DIR))
    print(f"{C_GREEN}[ok]{C_RESET} Saved to {path}")
    return Path(path)


def load_llm(path: Path, ctx: int):
    from llama_cpp import Llama
    threads = max(1, (os.cpu_count() or 4) - 2)
    print(f"{C_YELLOW}[load]{C_RESET} Loading model (threads={threads}, ctx={ctx})...")
    return Llama(model_path=str(path), n_ctx=ctx, n_threads=threads, n_gpu_layers=0, verbose=False)

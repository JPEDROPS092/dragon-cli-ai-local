"""Model download and loading (huggingface_hub + llama-cpp-python)."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

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


def load_llm(
    path: Path,
    ctx: int,
    model_info: Optional[Dict[str, Any]] = None,
    hardware_profile: Optional[Dict[str, Any]] = None,
):
    """Load LLM with optimized runtime parameters.
    
    Args:
        path: Path to the GGUF model file
        ctx: Context window size (fallback if not auto-detected)
        model_info: Optional model metadata for optimization
        hardware_profile: Optional hardware profile for optimization
        
    Returns:
        Loaded Llama instance
    """
    from llama_cpp import Llama
    
    # Try to use runtime configuration for optimal params
    try:
        from ..runtime import RuntimeConfig
        config = RuntimeConfig(hardware_profile=hardware_profile, model_info=model_info)
        params = config.get_runtime_params()
        
        # Override context if explicitly provided
        if ctx:
            params["n_ctx"] = ctx
        
        threads = params["n_threads"]
        gpu_layers = params["n_gpu_layers"]
        batch_size = params["n_batch"]
        
        exec_mode = config.decide_execution_mode()
        mode_msg = f"mode={exec_mode.upper()}"
        
    except ImportError:
        # Fallback to simple configuration
        threads = max(1, (os.cpu_count() or 4) - 2)
        gpu_layers = 0
        batch_size = 512
        mode_msg = "mode=CPU (fallback)"
    
    print(f"{C_YELLOW}[load]{C_RESET} Loading model (threads={threads}, ctx={params.get('n_ctx', ctx)}, {mode_msg})...")
    
    return Llama(
        model_path=str(path),
        n_ctx=params.get("n_ctx", ctx),
        n_threads=threads,
        n_gpu_layers=gpu_layers,
        n_batch=batch_size,
        verbose=False,
    )

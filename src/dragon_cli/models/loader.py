"""Model download and loading (huggingface_hub + llama-cpp-python)."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .._cuda import preload_cuda_runtime as _preload_cuda_runtime
from ..config import C_GREEN, C_RED, C_RESET, C_YELLOW, MODELS_DIR



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
    gpu_layers: Optional[int] = None,
    force_cpu: bool = False,
    force_gpu: bool = False,
):
    """Load LLM with optimized runtime parameters.

    GPU offload precedence (first match wins):
      1. ``gpu_layers`` — explicit layer count (0 = CPU, -1/<0 = all layers);
      2. ``force_cpu``  — disables the GPU entirely;
      3. ``force_gpu``  — offloads all layers when a GPU is present;
      4. otherwise      — auto, from the hardware-aware ``RuntimeConfig``.

    Args:
        path: Path to the GGUF model file
        ctx: Context window size (fallback if not auto-detected)
        model_info: Optional model metadata for optimization
        hardware_profile: Optional hardware profile for optimization
        gpu_layers: Explicit number of layers to offload to GPU (None = auto)
        force_cpu: Force CPU-only execution (ignore any GPU)
        force_gpu: Force GPU offload of all layers

    Returns:
        Loaded Llama instance
    """
    _preload_cuda_runtime()
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
        batch_size = params["n_batch"]
        gpu_available = config.hardware.get("gpu_available", False)

        # Resolve the number of GPU layers according to the precedence above.
        if force_cpu:
            n_gpu_layers = 0
            exec_mode = "CPU"
        elif gpu_layers is not None:
            n_gpu_layers = gpu_layers
            exec_mode = "CPU" if gpu_layers == 0 else "GPU"
        elif force_gpu:
            if gpu_available:
                n_gpu_layers = -1  # llama.cpp convention: -1 == all layers
                exec_mode = "GPU"
            else:
                print(f"{C_YELLOW}[gpu]{C_RESET} --gpu requested but no GPU was "
                      f"detected; falling back to CPU.")
                n_gpu_layers = 0
                exec_mode = "CPU"
        else:
            n_gpu_layers = params["n_gpu_layers"]
            exec_mode = config.decide_execution_mode().upper()

        mode_msg = f"mode={exec_mode} n_gpu_layers={n_gpu_layers}"

    except ImportError:
        # Fallback to simple configuration
        threads = max(1, (os.cpu_count() or 4) - 2)
        n_gpu_layers = 0
        batch_size = 512
        exec_mode = "CPU"
        mode_msg = "mode=CPU (fallback)"

    print(f"{C_YELLOW}[load]{C_RESET} Loading model (threads={threads}, "
          f"ctx={params.get('n_ctx', ctx)}, {mode_msg})...")

    # Full GPU offload can run out of VRAM on small cards (the whole model +
    # the display's memory). Rather than crash, try the requested layer count,
    # then progressively fewer GPU layers, finally falling back to CPU.
    total_layers = int((model_info or {}).get("n_layers", 32))
    if n_gpu_layers == 0:
        candidates = [0]
    elif n_gpu_layers > 0:
        candidates = []
        for n in (n_gpu_layers, n_gpu_layers // 2, 0):
            if n not in candidates:
                candidates.append(n)
    else:  # -1 / "all layers"
        candidates = []
        for frac in (1.0, 0.66, 0.5, 0.33, 0.0):
            n = int(round(total_layers * frac))
            if n not in candidates:
                candidates.append(n)

    n_ctx = params.get("n_ctx", ctx)
    llm = None
    last_err = None
    for ng in candidates:
        try:
            llm = Llama(
                model_path=str(path),
                n_ctx=n_ctx,
                n_threads=threads,
                n_gpu_layers=ng,
                n_batch=batch_size,
                verbose=False,
            )
            used_layers = ng
            break
        except Exception as exc:  # noqa: BLE001 - ggml surfaces OOM as ValueError
            last_err = exc
            llm = None
            if ng == 0:
                break
            print(f"{C_RED}[gpu]{C_RESET} Offloading {ng} layers failed (likely "
                  f"out of VRAM). Retrying with fewer layers...")
            # A failed constructor can strand VRAM until GC runs; reclaim it
            # before the next attempt so retries don't cascade into failure.
            import gc
            gc.collect()

    if llm is None:
        # Even CPU (0 layers) failed — this is a real error (corrupt file, etc.).
        raise last_err

    if used_layers > 0 and exec_mode != "CPU":
        exec_mode = "GPU"
        if used_layers < total_layers:
            print(f"{C_YELLOW}[gpu]{C_RESET} Running with {used_layers}/{total_layers} "
                  f"layers on GPU (partial offload).")
    else:
        exec_mode = "CPU"
        print(f"{C_YELLOW}[gpu]{C_RESET} GPU offload unavailable; running on CPU.")

    # Expose the chosen backend so the UI (chat banner, server) can display it.
    llm.exec_mode = exec_mode
    llm.n_gpu_layers = used_layers
    return llm

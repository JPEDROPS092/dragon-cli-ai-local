"""Catalog of verified GGUF models.

Each entry carries the metadata the runtime needs to choose a sensible GPU
offload without crashing on small VRAM:

    ram_min_gb   — system RAM (GB) required to run the model
    vram_min_gb  — GPU VRAM (GB) required to offload ALL layers (full GPU mode)
    n_layers     — transformer block count, used to compute partial offload and
                   to back off gracefully when full offload runs out of VRAM
    ctx          — default context window
"""

from ..config import C_DIM, C_GREEN, C_RED, C_RESET, MODELS_DIR

MODEL_CATALOG = [
    {
        "name": "Qwen2.5-1.5B-Instruct (Q4_K_M) — tiny, GPU-friendly",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "ram_min_gb": 3, "vram_min_gb": 1.5, "n_layers": 28, "ctx": 8192,
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-1.5B (Q4_K_M) — light, for testing",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        "ram_min_gb": 3, "vram_min_gb": 1.5, "n_layers": 28, "ctx": 4096,
    },
    {
        "name": "Llama-3.2-3B-Instruct (Q4_K_M) — small, GPU-friendly",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "ram_min_gb": 4, "vram_min_gb": 2.5, "n_layers": 28, "ctx": 8192,
    },
    {
        "name": "Qwen2.5-3B-Instruct (Q4_K_M) — fast, good multilingual quality",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "ram_min_gb": 4, "vram_min_gb": 2.5, "n_layers": 36, "ctx": 8192,
    },
    {
        "name": "Qwen2.5-7B-Instruct (Q4_K_M) — general use, great for tools",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "ram_min_gb": 8, "vram_min_gb": 5.0, "n_layers": 28, "ctx": 8192,
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-7B (Q4_K_M) — reasoning/code",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "ram_min_gb": 8, "vram_min_gb": 5.0, "n_layers": 28, "ctx": 8192,
    },
    {
        "name": "Meta-Llama-3.1-8B-Instruct (Q4_K_M) — general use",
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "ram_min_gb": 9, "vram_min_gb": 5.5, "n_layers": 32, "ctx": 8192,
    },
    {
        "name": "Qwen2.5-14B-Instruct (Q4_K_M) — heavy, best quality",
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "ram_min_gb": 14, "vram_min_gb": 9.0, "n_layers": 48, "ctx": 8192,
    },
]


def available_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return -1


def list_catalog() -> None:
    ram = available_ram_gb()
    if ram > 0:
        print(f"{C_DIM}Available RAM: {ram:.1f} GB{C_RESET}")
    for i, m in enumerate(MODEL_CATALOG, 1):
        downloaded = (MODELS_DIR / m["filename"]).exists()
        tag = f"{C_GREEN}[downloaded]{C_RESET}" if downloaded else f"{C_DIM}[cloud]{C_RESET}"
        warning = f" {C_RED}[!] RAM may be too low right now{C_RESET}" if 0 < ram < m["ram_min_gb"] else ""
        gpu = m.get("vram_min_gb")
        gpu_tag = f" {C_DIM}(GPU full ~{gpu:.0f}GB){C_RESET}" if gpu else ""
        print(f"  {i}. {m['name']}  {tag}{gpu_tag}{warning}")

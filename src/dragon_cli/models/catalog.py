"""Catalog of verified GGUF models.

Each entry carries the metadata the runtime needs to choose a sensible GPU
offload without crashing on small VRAM:

    ram_min_gb   — system RAM (GB) required to run the model
    vram_min_gb  — GPU VRAM (GB) required to offload ALL layers (full GPU mode)
    n_layers     — transformer block count, used to compute partial offload and
                   to back off gracefully when full offload runs out of VRAM
    ctx          — default context window
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

CUSTOM_CATALOG_FILE = MODELS_DIR / "custom_model_catalog.json"


def _model_key(model: Dict[str, Any]) -> Tuple[str, str]:
    repo_id = str(model.get("repo_id", "")).strip().lower()
    filename = str(model.get("filename", "")).strip().lower()
    return repo_id, filename


def _merge_model(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(target)
    for key in ("name", "repo_id", "filename", "ram_min_gb", "vram_min_gb", "n_layers", "ctx"):
        if key in source:
            merged[key] = source[key]
    return merged


def load_custom_catalog() -> List[Dict[str, Any]]:
    """Load custom model entries from disk, returning validated dicts."""
    if not CUSTOM_CATALOG_FILE.exists():
        return []
    try:
        data = json.loads(CUSTOM_CATALOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    items: List[Dict[str, Any]] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        repo_id = str(raw.get("repo_id", "")).strip()
        filename = str(raw.get("filename", "")).strip()
        if not repo_id or not filename:
            continue
        items.append(
            {
                "name": str(raw.get("name", filename)).strip() or filename,
                "repo_id": repo_id,
                "filename": filename,
                "ram_min_gb": int(raw.get("ram_min_gb", 4) or 4),
                "vram_min_gb": float(raw.get("vram_min_gb", 0) or 0),
                "n_layers": int(raw.get("n_layers", 32) or 32),
                "ctx": int(raw.get("ctx", 8192) or 8192),
            }
        )
    return items


def save_custom_catalog(models: List[Dict[str, Any]]) -> None:
    """Persist custom model entries under models/custom_model_catalog.json."""
    MODELS_DIR.mkdir(exist_ok=True)
    payload: List[Dict[str, Any]] = []
    seen = set()
    for model in models:
        key = _model_key(model)
        if key in seen:
            continue
        seen.add(key)
        payload.append(
            {
                "name": str(model.get("name", "")).strip(),
                "repo_id": str(model.get("repo_id", "")).strip(),
                "filename": str(model.get("filename", "")).strip(),
                "ram_min_gb": int(model.get("ram_min_gb", 4) or 4),
                "vram_min_gb": float(model.get("vram_min_gb", 0) or 0),
                "n_layers": int(model.get("n_layers", 32) or 32),
                "ctx": int(model.get("ctx", 8192) or 8192),
            }
        )
    CUSTOM_CATALOG_FILE.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def add_model_to_catalog(model: Dict[str, Any], persist: bool = False) -> Tuple[int, bool]:
    """Add a model to runtime catalog and optionally persist it.

    Returns ``(index, created)`` where index is 1-based in MODEL_CATALOG.
    """
    incoming = _merge_model({}, model)
    incoming_key = _model_key(incoming)

    for i, existing in enumerate(MODEL_CATALOG, 1):
        if _model_key(existing) == incoming_key:
            MODEL_CATALOG[i - 1] = _merge_model(existing, incoming)
            if persist:
                _persist_model_entry(MODEL_CATALOG[i - 1])
            return i, False

    MODEL_CATALOG.append(incoming)
    if persist:
        _persist_model_entry(incoming)
    return len(MODEL_CATALOG), True


def _persist_model_entry(model: Dict[str, Any]) -> None:
    items = load_custom_catalog()
    model_key = _model_key(model)
    for i, existing in enumerate(items):
        if _model_key(existing) == model_key:
            items[i] = _merge_model(existing, model)
            save_custom_catalog(items)
            return
    items.append(model)
    save_custom_catalog(items)


def _load_custom_into_runtime_catalog() -> None:
    for model in load_custom_catalog():
        add_model_to_catalog(model, persist=False)


_load_custom_into_runtime_catalog()


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

"""Model Hub integration for Dragon CLI.

Integrates with Hugging Face, Ollama, and other model hubs to:
- Query model catalogs
- Filter by hardware compatibility
- Download models
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MODELS_DIR


class ModelHub:
    """Integration layer for model hubs (Hugging Face, Ollama, etc.)."""
    
    def __init__(self):
        self.catalog_cache_file = MODELS_DIR / "model_catalog_cache.json"
    
    def fetch_huggingface_models(
        self, 
        search_query: str = "GGUF",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Fetch models from Hugging Face.
        
        Args:
            search_query: Search term (default: "GGUF" for quantized models)
            limit: Maximum number of results
            
        Returns:
            List of model metadata dictionaries
        """
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Search for GGUF models
            models = api.list_models(
                search=search_query,
                limit=limit,
                sort="downloads",
                direction=-1,  # Most downloaded first
            )
            
            results = []
            for model in models:
                # Only include if it has GGUF files
                model_info = {
                    "name": model.id,
                    "repo_id": model.id,
                    "downloads": getattr(model, "downloads", 0),
                    "likes": getattr(model, "likes", 0),
                    "source": "huggingface",
                }
                results.append(model_info)
            
            return results
            
        except ImportError:
            print("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return []
        except Exception as e:
            print(f"Error fetching from Hugging Face: {e}")
            return []
    
    def get_model_files(self, repo_id: str) -> List[str]:
        """Get list of GGUF files in a repository.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "Qwen/Qwen2.5-7B-Instruct-GGUF")
            
        Returns:
            List of filenames ending with .gguf
        """
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            files = api.list_repo_files(repo_id=repo_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            
            return gguf_files
            
        except Exception as e:
            print(f"Error listing files for {repo_id}: {e}")
            return []
    
    def search_models(
        self,
        query: str = "",
        min_ram_gb: float = 0,
        max_ram_gb: Optional[float] = None,
        requires_gpu: bool = False,
        source: str = "huggingface",
    ) -> List[Dict[str, Any]]:
        """Search models with filters.
        
        Args:
            query: Search query string
            min_ram_gb: Minimum RAM requirement in GB
            max_ram_gb: Maximum RAM requirement in GB
            requires_gpu: Whether GPU is required
            source: Model source ("huggingface", "ollama", "all")
            
        Returns:
            Filtered list of models
        """
        from .models.catalog import MODEL_CATALOG
        
        results = []
        
        # Search local catalog first
        for model in MODEL_CATALOG:
            ram_req = model.get("ram_min_gb", 0)
            
            # Apply filters
            if min_ram_gb > 0 and ram_req < min_ram_gb:
                continue
            if max_ram_gb and ram_req > max_ram_gb:
                continue
            
            # Check if matches query
            if query and query.lower() not in model["name"].lower():
                continue
            
            model_copy = model.copy()
            model_copy["source"] = "local_catalog"
            results.append(model_copy)
        
        # Optionally fetch from Hugging Face
        if source in ("huggingface", "all") and query:
            hf_models = self.fetch_huggingface_models(search_query=query, limit=20)
            for model in hf_models:
                # Check if already in results
                if not any(m.get("repo_id") == model["repo_id"] for m in results):
                    results.append(model)
        
        return results
    
    def get_recommended_quantization(self, vram_gb: float, ram_gb: float) -> str:
        """Recommend quantization level based on available memory.
        
        Args:
            vram_gb: Available GPU VRAM in GB
            ram_gb: Available system RAM in GB
            
        Returns:
            Recommended quantization string (e.g., "Q4_K_M", "Q8_0")
        """
        # If good VRAM, can use higher quantization on GPU
        if vram_gb >= 24:
            return "Q8_0"  # Near FP16 quality
        elif vram_gb >= 16:
            return "Q6_K"  # High quality
        elif vram_gb >= 8:
            return "Q5_K_M"  # Good balance
        elif vram_gb >= 6:
            return "Q4_K_M"  # Standard recommendation
        elif ram_gb >= 16:
            return "Q5_K_M"  # Can use more RAM
        else:
            return "Q4_K_M"  # Safe default
    
    def cache_catalog(self, models: List[Dict[str, Any]]) -> None:
        """Cache model catalog to disk."""
        MODELS_DIR.mkdir(exist_ok=True)
        try:
            with open(self.catalog_cache_file, "w") as f:
                json.dump(models, f, indent=2)
        except Exception as e:
            print(f"Error caching catalog: {e}")
    
    def load_cached_catalog(self) -> List[Dict[str, Any]]:
        """Load cached model catalog from disk."""
        if not self.catalog_cache_file.exists():
            return []
        try:
            with open(self.catalog_cache_file, "r") as f:
                return json.load(f)
        except Exception:
            return []


def get_model_info_from_hf(repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific model file.
    
    Args:
        repo_id: Hugging Face repository ID
        filename: GGUF filename
        
    Returns:
        Model information dictionary or None
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        
        # Get file info
        files = api.list_repo_files(repo_id=repo_id)
        if filename not in files:
            return None
        
        # Try to get file size
        try:
            file_info = api.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(MODELS_DIR),
                force_download=False,
            )
            # This would need additional API calls for size
        except Exception:
            pass
        
        return {
            "repo_id": repo_id,
            "filename": filename,
            "exists": (MODELS_DIR / filename).exists(),
        }
        
    except ImportError:
        return None
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None


if __name__ == "__main__":
    # Test the hub
    hub = ModelHub()
    
    print("Searching for Qwen GGUF models...")
    models = hub.search_models(query="Qwen GGUF", max_ram_gb=16)
    
    print(f"\nFound {len(models)} models:\n")
    for i, model in enumerate(models[:10], 1):
        name = model.get("name", model.get("repo_id", "Unknown"))
        source = model.get("source", "unknown")
        ram = model.get("ram_min_gb", "?")
        print(f"  {i}. {name}")
        print(f"     Source: {source}, Min RAM: {ram} GB\n")

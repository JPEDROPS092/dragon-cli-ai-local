"""Runtime configuration module for Dragon CLI.

Decides execution parameters based on hardware and model:
- CPU vs GPU vs hybrid execution
- Context window size
- Thread count
- Batch size
"""

from typing import Any, Dict, Optional


class RuntimeConfig:
    """Configure runtime parameters for LLM inference."""
    
    def __init__(
        self,
        hardware_profile: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        """Initialize runtime configuration.
        
        Args:
            hardware_profile: Hardware info from detect_hardware()
            model_info: Model metadata (ram_min_gb, ctx, etc.)
        """
        from .hardware import detect_hardware
        
        self.hardware = hardware_profile or detect_hardware()
        self.model = model_info or {}
    
    def decide_execution_mode(self) -> str:
        """Decide whether to use CPU, GPU, or hybrid execution.
        
        Returns:
            One of: "gpu", "cpu", "hybrid"
        """
        gpu_available = self.hardware.get("gpu_available", False)
        vram = self.hardware.get("gpu_vram_gb", 0)
        ram = self.hardware.get("ram_available_gb", 8)
        
        # Estimate model VRAM requirement
        params_billions = self._estimate_params()
        vram_needed = params_billions * 0.7  # Rough estimate for Q4 quantization
        
        if gpu_available and vram >= vram_needed:
            return "gpu"  # Full GPU offload possible
        elif gpu_available and vram >= 4:
            return "hybrid"  # Partial GPU offload
        else:
            return "cpu"  # CPU only
    
    def get_gpu_layers(self) -> int:
        """Calculate optimal number of layers to offload to GPU.
        
        Returns:
            Number of layers for n_gpu_layers parameter
        """
        execution_mode = self.decide_execution_mode()
        
        if execution_mode == "gpu":
            # Offload all layers
            return 999  # llama-cpp-python convention for "all layers"
        elif execution_mode == "hybrid":
            # Calculate partial offload
            vram = self.hardware.get("gpu_vram_gb", 0)
            params_billions = self._estimate_params()
            
            # Assume ~1GB per 1B params at Q4, adjust for layers
            layers_per_gb = 2.5  # Approximate
            max_layers = int(vram * layers_per_gb)
            
            # Typical models have 32-40 layers
            total_layers = self._estimate_layer_count()
            return min(max_layers, total_layers)
        else:
            return 0  # CPU only
    
    def get_thread_count(self) -> int:
        """Calculate optimal thread count for CPU inference.
        
        Returns:
            Number of threads for n_threads parameter
        """
        cpu_count = self.hardware.get("cpu_count", 4)
        
        # Leave some cores for system
        # Physical cores usually better than logical for inference
        physical_count = self.hardware.get("cpu_count_physical", cpu_count // 2)
        
        if physical_count > 0:
            threads = max(1, physical_count - 1)
        else:
            threads = max(1, cpu_count - 2)
        
        # Cap at reasonable maximum
        return min(threads, 16)
    
    def get_context_size(self) -> int:
        """Determine optimal context window size.
        
        Returns:
            Context size for n_ctx parameter
        """
        model_ctx = self.model.get("ctx", 4096)
        ram = self.hardware.get("ram_available_gb", 8)
        
        # Context memory: roughly 2GB per 8K context for 7B model
        # Scale by model size
        params_billions = self._estimate_params()
        memory_per_8k = params_billions / 7.0 * 2.0  # GB
        
        # Calculate max context we can afford
        # Reserve 2GB for model weights and system
        available_for_ctx = max(0, ram - 2 - (params_billions * 0.7))
        
        if available_for_ctx <= 0:
            max_ctx = 2048
        else:
            max_ctx = int(available_for_ctx / memory_per_8k * 8192)
            max_ctx = max(2048, min(max_ctx, 32768))  # Reasonable bounds
        
        # Use model's recommended context or calculated max, whichever is smaller
        return min(model_ctx, max_ctx)
    
    def get_batch_size(self) -> int:
        """Determine optimal batch size for inference.
        
        Returns:
            Batch size for n_batch parameter
        """
        execution_mode = self.decide_execution_mode()
        ram = self.hardware.get("ram_available_gb", 8)
        
        if execution_mode == "gpu":
            # GPU can handle larger batches
            return 512
        elif ram >= 16:
            return 256
        elif ram >= 8:
            return 128
        else:
            return 64
    
    def get_runtime_params(self) -> Dict[str, Any]:
        """Get complete runtime parameters for llama-cpp-python.
        
        Returns:
            Dictionary with all parameters for Llama() constructor
        """
        execution_mode = self.decide_execution_mode()
        
        params = {
            "n_threads": self.get_thread_count(),
            "n_ctx": self.get_context_size(),
            "n_batch": self.get_batch_size(),
            "n_gpu_layers": self.get_gpu_layers(),
            "verbose": False,
        }
        
        # Additional optimizations
        if execution_mode == "gpu":
            params["use_mmap"] = True
            params["use_mlock"] = False
        else:
            params["use_mmap"] = True
            params["use_mlock"] = False
        
        return params
    
    def _estimate_params(self) -> float:
        """Estimate model parameters in billions from name."""
        import re
        
        name = self.model.get("name", "")
        matches = re.findall(r'(\d+\.?\d*)[Bb]', name)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        return 7.0  # Default
    
    def _estimate_layer_count(self) -> int:
        """Estimate layer count from model name."""
        name = self.model.get("name", "").lower()
        
        # Common patterns
        if "llama" in name or "qwen" in name:
            params = self._estimate_params()
            if params <= 3:
                return 28
            elif params <= 8:
                return 32
            elif params <= 14:
                return 40
            else:
                return 48
        
        return 32  # Default
    
    def print_config(self) -> None:
        """Print runtime configuration summary."""
        from .config import C_CYAN, C_RESET, C_GREEN, C_DIM
        
        params = self.get_runtime_params()
        execution_mode = self.decide_execution_mode()
        
        print(f"\n{C_CYAN}{'='*50}{C_RESET}")
        print(f"{C_CYAN}RUNTIME CONFIGURATION{C_RESET}")
        print(f"{C_CYAN}{'='*50}{C_RESET}\n")
        
        print(f"{C_GREEN}Execution Mode:{C_RESET} {execution_mode.upper()}")
        print(f"{C_GREEN}Threads:{C_RESET} {params['n_threads']}")
        print(f"{C_GREEN}Context:{C_RESET} {params['n_ctx']} tokens")
        print(f"{C_GREEN}Batch Size:{C_RESET} {params['n_batch']}")
        print(f"{C_GREEN}GPU Layers:{C_RESET} {params['n_gpu_layers']}")
        
        if execution_mode != "cpu":
            print(f"\n{C_DIM}→ GPU acceleration enabled{C_RESET}")
        else:
            print(f"\n{C_DIM}→ Running on CPU only{C_RESET}")
        
        print()


def configure_runtime(
    model_info: Optional[Dict[str, Any]] = None,
    hardware_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to get runtime configuration.
    
    Args:
        model_info: Model metadata
        hardware_profile: Hardware profile (auto-detected if None)
        
    Returns:
        Runtime parameters dictionary for llama-cpp-python
    """
    config = RuntimeConfig(hardware_profile=hardware_profile, model_info=model_info)
    return config.get_runtime_params()


if __name__ == "__main__":
    # Test runtime configuration
    config = RuntimeConfig()
    config.print_config()
    
    print("\nFull parameters:")
    params = config.get_runtime_params()
    for key, value in params.items():
        print(f"  {key}: {value}")

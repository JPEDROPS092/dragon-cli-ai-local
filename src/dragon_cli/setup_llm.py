"""Cross-platform installer for llama-cpp-python.

Compiling llama-cpp-python from source requires a C/C++ toolchain (Visual
Studio Build Tools on Windows, gcc/build-essential on Linux, Xcode command
line tools on macOS). This detects the current OS and GPU hardware, then
installs the appropriate prebuilt wheel (CPU, CUDA, or ROCm) for optimal
performance.
"""

import platform
import subprocess
import sys


def detect_gpu():
    """Detect available GPU and return GPU type."""
    gpu_type = "cpu"  # default
    
    # Try NVIDIA CUDA detection
    try:
        import ctypes
        cuda_lib_names = ["cuda", "nvcuda", "libcuda.so", "libcuda.so.1"]
        
        for lib_name in cuda_lib_names:
            try:
                cuda_lib = ctypes.CDLL(lib_name)
                device_count = ctypes.c_int()
                result = cuda_lib.cuDeviceGetCount(ctypes.byref(device_count))
                
                if result == 0 and device_count.value > 0:
                    gpu_type = "cuda"
                    print(f"[dragon-cli] Detected {device_count.value} NVIDIA GPU(s)")
                    
                    # Get CUDA version if possible
                    try:
                        cuda_version = ctypes.c_int()
                        cuda_lib.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
                        major = cuda_version.value // 1000
                        minor = (cuda_version.value % 1000) // 10
                        print(f"[dragon-cli] CUDA version: {major}.{minor}")
                    except Exception:
                        pass
                    
                    return gpu_type
            except OSError:
                continue
    except Exception:
        pass
    
    # Try PyTorch for more reliable GPU detection
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_type = "cuda"
            device_count = torch.cuda.device_count()
            print(f"[dragon-cli] Detected {device_count} NVIDIA GPU(s) via PyTorch")
            
            for i in range(min(device_count, 3)):  # Show first 3 GPUs
                gpu_name = torch.cuda.get_device_name(i)
                vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"[dragon-cli]   GPU {i}: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            
            return gpu_type
            
    except ImportError:
        pass
    
    # Try AMD ROCm detection (Linux only)
    if platform.system() == "Linux":
        try:
            import subprocess
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "Agent" in result.stdout:
                gpu_type = "rocm"
                print("[dragon-cli] Detected AMD GPU(s) via ROCm")
                return gpu_type
        except Exception:
            pass
        
        # Check for rocm-smi
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_type = "rocm"
                print("[dragon-cli] Detected AMD GPU(s) via rocm-smi")
                return gpu_type
        except Exception:
            pass
    
    # Try Apple Silicon detection
    if platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and "Apple" in result.stdout:
                gpu_type = "mps"  # Metal Performance Shaders
                print("[dragon-cli] Detected Apple Silicon GPU (Metal/MPS)")
                return gpu_type
        except Exception:
            pass
    
    print("[dragon-cli] No dedicated GPU detected, using CPU-only mode")
    return gpu_type


def get_install_command(gpu_type: str):
    """Return the appropriate pip install command based on GPU type."""
    
    if gpu_type == "cuda":
        # Install with CUDA support (llama-cpp-python will auto-detect CUDA)
        # We'll install with CMAKE_ARGS to enable CUDA
        print("[dragon-cli] Installing llama-cpp-python with CUDA support...")
        print("[dragon-cli] This may take a few minutes to compile...")
        
        # Set environment variables for CUDA build
        env = {
            **subprocess.os.environ,
            "CMAKE_ARGS": "-DGGML_CUDA=on",
            "FORCE_CMAKE": "1",
        }
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "llama-cpp-python", "--no-cache-dir"
        ], env=env)
        
        return result
        
    elif gpu_type == "rocm":
        # Install with ROCm support
        print("[dragon-cli] Installing llama-cpp-python with ROCm support...")
        print("[dragon-cli] This may take a few minutes to compile...")
        
        env = {
            **subprocess.os.environ,
            "CMAKE_ARGS": "-DGGML_HIPBLAS=on",
            "FORCE_CMAKE": "1",
        }
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "llama-cpp-python", "--no-cache-dir"
        ], env=env)
        
        return result
        
    elif gpu_type == "mps":
        # Apple Silicon - use Metal acceleration
        print("[dragon-cli] Installing llama-cpp-python with Metal (MPS) support...")
        print("[dragon-cli] This may take a few minutes to compile...")
        
        env = {
            **subprocess.os.environ,
            "CMAKE_ARGS": "-DGGML_METAL=on",
            "FORCE_CMAKE": "1",
        }
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "llama-cpp-python", "--no-cache-dir"
        ], env=env)
        
        return result
        
    else:
        # CPU-only installation using prebuilt wheels
        WHEEL_INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
        print(f"[dragon-cli] Installing llama-cpp-python from prebuilt CPU wheels ({WHEEL_INDEX_URL})...")
        
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "llama-cpp-python",
            "--prefer-binary", "--extra-index-url", WHEEL_INDEX_URL,
        ])
        
        return result


TROUBLESHOOTING = {
    "Windows": (
        "Installation failed. Make sure you have Visual Studio Build Tools\n"
        "with C++ workload installed:\n"
        "https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
        "Also ensure CMake is installed and in PATH."
    ),
    "Linux": (
        "Installation failed. Install build tools and CMake:\n"
        "  Ubuntu/Debian: sudo apt install build-essential cmake\n"
        "  Fedora/RHEL: sudo dnf install gcc gcc-c++ make cmake\n"
        "\nFor GPU support, also install:\n"
        "  NVIDIA CUDA: https://developer.nvidia.com/cuda-downloads\n"
        "  AMD ROCm: https://rocm.docs.amd.com/"
    ),
    "Darwin": (
        "Installation failed. Install Xcode command line tools:\n"
        "  xcode-select --install\n"
        "\nFor Apple Silicon, ensure you're using Python from python.org or Homebrew."
    ),
}


def main() -> None:
    system = platform.system()
    print(f"[dragon-cli] Detected {system} ({platform.machine()}), Python {platform.python_version()}")
    print()
    
    # Detect GPU first
    gpu_type = detect_gpu()
    print()
    
    # Install appropriate version
    result = get_install_command(gpu_type)
    
    if result.returncode == 0:
        print()
        print("[dragon-cli] ✓ llama-cpp-python installed successfully!")
        
        if gpu_type != "cpu":
            print(f"[dragon-cli] GPU acceleration enabled ({gpu_type.upper()})")
            print("[dragon-cli] Models will automatically use GPU for faster inference")
        else:
            print("[dragon-cli] Running in CPU-only mode")
            print("[dragon-cli] For faster inference, consider adding an NVIDIA or AMD GPU")
        
        return
    
    print("\n[dragon-cli] ✗ Installation failed.")
    print(TROUBLESHOOTING.get(
        system,
        "Install a C/C++ toolchain and CMake for your platform, then retry: "
        "pip install llama-cpp-python",
    ))
    sys.exit(1)


if __name__ == "__main__":
    main()

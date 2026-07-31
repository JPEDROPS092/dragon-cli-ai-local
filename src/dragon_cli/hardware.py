"""Hardware detection module for Dragon CLI.

Detects CPU, RAM, GPU, storage, and operating system to calculate
execution capacity for LLM models.
"""

import platform
import shutil
from typing import Any, Dict, Optional


def get_os_info() -> str:
    """Return operating system information."""
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    
    if system == "Windows":
        # Get more detailed Windows version
        win_ver = platform.win32_ver()
        return f"{system} {release} ({win_ver[0]}) {machine}"
    elif system == "Darwin":
        return f"macOS {release} ({machine})"
    else:
        # Linux or other
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        pretty_name = line.strip().split("=", 1)[1].strip('"')
                        return f"{pretty_name} {machine}"
        except FileNotFoundError:
            pass
        return f"{system} {release} ({machine})"


def get_cpu_info() -> Dict[str, Any]:
    """Return CPU information."""
    import os
    
    cpu_count = os.cpu_count() or 0
    # Physical count estimation (usually half of logical for hyperthreading)
    cpu_count_physical = max(1, cpu_count // 2) if cpu_count > 0 else 0
    
    # Try to get CPU model/name
    cpu_model = "Unknown"
    
    if platform.system() == "Windows":
        try:
            import ctypes
            # Use wmi or registry for Windows CPU info
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                cpu_model = lines[1].strip()
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                cpu_model = result.stdout.strip()
        except Exception:
            pass
    else:
        # Linux
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except FileNotFoundError:
            pass
    
    return {
        "cpu_count": cpu_count,
        "cpu_count_physical": cpu_count_physical,
        "cpu_model": cpu_model,
    }


def get_ram_info() -> Dict[str, float]:
    """Return RAM information in GB."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total_gb = vm.total / (1024 ** 3)
        ram_available_gb = vm.available / (1024 ** 3)
        ram_used_gb = vm.used / (1024 ** 3)
        ram_percent = vm.percent
    except ImportError:
        # Fallback without psutil
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    mem_info = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value_str = parts[1].strip().split()[0]
                            try:
                                mem_info[key] = int(value_str)
                            except ValueError:
                                pass
                    
                    # Values are in kB
                    ram_total_gb = mem_info.get("MemTotal", 0) / (1024 ** 2)
                    ram_free_gb = mem_info.get("MemFree", 0) / (1024 ** 2)
                    ram_available_gb = mem_info.get("MemAvailable", ram_free_gb) / (1024 ** 2)
                    ram_used_gb = ram_total_gb - ram_free_gb
                    ram_percent = (ram_used_gb / ram_total_gb * 100) if ram_total_gb > 0 else 0
            except FileNotFoundError:
                ram_total_gb = -1
                ram_available_gb = -1
                ram_used_gb = -1
                ram_percent = -1
        else:
            ram_total_gb = -1
            ram_available_gb = -1
            ram_used_gb = -1
            ram_percent = -1
    
    return {
        "ram_total_gb": round(ram_total_gb, 2),
        "ram_available_gb": round(ram_available_gb, 2),
        "ram_used_gb": round(ram_used_gb, 2),
        "ram_percent": round(ram_percent, 2),
    }


def get_gpu_info() -> Dict[str, Any]:
    """Return GPU information with graceful fallback."""
    gpu_name = "None detected"
    gpu_vram_gb = 0.0
    gpu_available = False
    
    # Try NVIDIA CUDA
    try:
        import ctypes
        # Try to load CUDA library
        cuda_lib_names = ["cuda", "nvcuda", "libcuda.so", "libcuda.so.1"]
        cuda_lib = None
        
        for lib_name in cuda_lib_names:
            try:
                cuda_lib = ctypes.CDLL(lib_name)
                break
            except OSError:
                continue
        
        if cuda_lib is not None:
            # Check CUDA device count
            device_count = ctypes.c_int()
            result = cuda_lib.cuDeviceGetCount(ctypes.byref(device_count))
            
            if result == 0 and device_count.value > 0:
                gpu_available = True
                # Get first device name
                device = ctypes.c_int()
                cuda_lib.cuDeviceGet(ctypes.byref(device), 0)
                
                name_buffer = ctypes.create_string_buffer(256)
                cuda_lib.cuDeviceGetName(name_buffer, 256, device)
                gpu_name = name_buffer.value.decode("utf-8")
                
                # Get memory info
                free_mem = ctypes.c_size_t()
                total_mem = ctypes.c_size_t()
                cuda_lib.cuMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
                
                gpu_vram_gb = total_mem.value / (1024 ** 3)
                
    except Exception:
        pass
    
    # If no CUDA, try pytorch for GPU detection
    if not gpu_available:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except ImportError:
            pass
    
    # Try Apple Silicon (Metal/MPS)
    if not gpu_available and platform.system() == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                output = result.stdout
                if "Apple" in output:
                    gpu_available = True
                    # Extract GPU name
                    for line in output.split("\n"):
                        if "Chipset Model" in line:
                            gpu_name = line.split(":", 1)[1].strip()
                            break
                    # Apple Silicon shares unified memory
                    ram_info = get_ram_info()
                    gpu_vram_gb = ram_info.get("ram_total_gb", 0) * 0.75  # Estimate 75% available for GPU
        except Exception:
            pass
    
    return {
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(gpu_vram_gb, 2),
        "gpu_available": gpu_available,
    }


def get_disk_info(path: str = ".") -> Dict[str, float]:
    """Return disk space information for the given path."""
    try:
        import psutil
        usage = psutil.disk_usage(path)
        disk_total_gb = usage.total / (1024 ** 3)
        disk_free_gb = usage.free / (1024 ** 3)
        disk_used_gb = usage.used / (1024 ** 3)
        disk_percent = usage.percent
    except ImportError:
        # Fallback using shutil
        total, used, free = shutil.disk_usage(path)
        disk_total_gb = total / (1024 ** 3)
        disk_free_gb = free / (1024 ** 3)
        disk_used_gb = used / (1024 ** 3)
        disk_percent = (used / total * 100) if total > 0 else 0
    
    return {
        "disk_total_gb": round(disk_total_gb, 2),
        "disk_free_gb": round(disk_free_gb, 2),
        "disk_used_gb": round(disk_used_gb, 2),
        "disk_percent": round(disk_percent, 2),
    }


def detect_hardware() -> Dict[str, Any]:
    """Detect all hardware information and return a complete profile.
    
    Returns:
        Dictionary containing:
        - os: Operating system string
        - cpu_count: Number of logical CPU cores
        - cpu_model: CPU model name
        - ram_total_gb: Total RAM in GB
        - ram_available_gb: Available RAM in GB
        - gpu_name: GPU name if available
        - gpu_vram_gb: GPU VRAM in GB
        - disk_free_gb: Free disk space in GB
    """
    os_info = get_os_info()
    cpu_info = get_cpu_info()
    ram_info = get_ram_info()
    gpu_info = get_gpu_info()
    disk_info = get_disk_info()
    
    return {
        "os": os_info,
        "cpu_count": cpu_info["cpu_count"],
        "cpu_model": cpu_info["cpu_model"],
        "ram_total_gb": ram_info["ram_total_gb"],
        "ram_available_gb": ram_info["ram_available_gb"],
        "gpu_name": gpu_info["gpu_name"],
        "gpu_vram_gb": gpu_info["gpu_vram_gb"],
        "gpu_available": gpu_info["gpu_available"],
        "disk_free_gb": disk_info["disk_free_gb"],
    }


def print_hardware_profile(profile: Optional[Dict[str, Any]] = None) -> None:
    """Print hardware profile in a formatted way."""
    from .config import C_CYAN, C_RESET, C_DIM, C_GREEN, C_YELLOW, C_RED
    
    if profile is None:
        profile = detect_hardware()
    
    print(f"\n{C_CYAN}{'='*60}{C_RESET}")
    print(f"{C_CYAN}HARDWARE PROFILE{C_RESET}")
    print(f"{C_CYAN}{'='*60}{C_RESET}\n")
    
    print(f"{C_GREEN}Operating System:{C_RESET}")
    print(f"  {profile['os']}\n")
    
    print(f"{C_GREEN}CPU:{C_RESET}")
    print(f"  Model: {profile['cpu_model']}")
    print(f"  Cores: {profile['cpu_count']} logical\n")
    
    print(f"{C_GREEN}Memory (RAM):{C_RESET}")
    if profile['ram_total_gb'] > 0:
        print(f"  Total: {profile['ram_total_gb']:.2f} GB")
        print(f"  Available: {profile['ram_available_gb']:.2f} GB")
        print(f"  Usage: {profile.get('ram_percent', 0):.1f}%\n")
    else:
        print(f"  {C_DIM}Unable to detect RAM{C_RESET}\n")
    
    print(f"{C_GREEN}GPU:{C_RESET}")
    if profile['gpu_available']:
        print(f"  Name: {profile['gpu_name']}")
        print(f"  VRAM: {profile['gpu_vram_gb']:.2f} GB{C_GREEN} [GPU acceleration possible]{C_RESET}\n")
    else:
        print(f"  {C_DIM}No dedicated GPU detected (CPU-only mode){C_RESET}\n")
    
    print(f"{C_GREEN}Storage:{C_RESET}")
    print(f"  Free Space: {profile['disk_free_gb']:.2f} GB\n")
    
    # Capacity assessment
    print(f"{C_CYAN}{'='*60}{C_RESET}")
    print(f"{C_CYAN}EXECUTION CAPACITY ASSESSMENT{C_RESET}")
    print(f"{C_CYAN}{'='*60}{C_RESET}\n")
    
    ram = profile['ram_available_gb']
    vram = profile['gpu_vram_gb']
    
    if profile['gpu_available'] and vram >= 6:
        print(f"{C_GREEN}✓ GPU Acceleration:{C_RESET} Available with {vram:.1f} GB VRAM")
        print(f"  → Can run larger models with GPU offloading\n")
    elif profile['gpu_available']:
        print(f"{C_YELLOW}⚠ GPU Limited:{C_RESET} Only {vram:.1f} GB VRAM available")
        print(f"  → Small models or partial GPU offloading recommended\n")
    else:
        print(f"{C_DIM}○ CPU-Only Mode:{C_RESET} No GPU detected")
        print(f"  → All computation will use CPU\n")
    
    if ram < 4:
        print(f"{C_RED}✗ Memory Critical:{C_RESET} Only {ram:.1f} GB available")
        print(f"  → Very limited model options (≤1.5B parameters)\n")
    elif ram < 8:
        print(f"{C_YELLOW}⚠ Memory Limited:{C_RESET} {ram:.1f} GB available")
        print(f"  → Models up to 7B (quantized) should work\n")
    else:
        print(f"{C_GREEN}✓ Memory Adequate:{C_RESET} {ram:.1f} GB available")
        print(f"  → Can run models up to 14B+ (quantized)\n")


if __name__ == "__main__":
    print_hardware_profile()

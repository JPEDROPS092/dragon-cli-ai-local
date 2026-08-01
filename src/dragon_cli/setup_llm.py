"""Cross-platform installer for llama-cpp-python.

Compiling llama-cpp-python from source requires a C/C++ toolchain (Visual
Studio Build Tools on Windows, gcc/build-essential on Linux, Xcode command
line tools on macOS). This detects the current OS and GPU hardware, then
installs the appropriate prebuilt wheel (CPU, CUDA, or ROCm) for optimal
performance.
"""

import os
import platform
import shutil
import subprocess
import sys


WHEEL_INDEX_URLS = {
    "cpu": "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    "cuda": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    "rocm": "https://abetlen.github.io/llama-cpp-python/whl/rocm",
    "mps": "https://abetlen.github.io/llama-cpp-python/whl/metal",
}


def detect_gpu():
    """Detect available GPU and return GPU type.

    Prefer system commands such as nvidia-smi and rocm-smi because they work
    even in stripped-down virtual environments where torch or CUDA libraries
    may not be installed yet.
    """
    gpu_type = "cpu"

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
            if first_line:
                gpu_name = first_line.split(",")[0].strip()
                print(f"[dragon-cli] Detected NVIDIA GPU via nvidia-smi: {gpu_name}")
                gpu_type = "cuda"
                return gpu_type

    if platform.system() == "Linux":
        rocminfo = shutil.which("rocminfo")
        if rocminfo:
            result = subprocess.run(
                [rocminfo],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and "Agent" in result.stdout:
                print("[dragon-cli] Detected AMD GPU via ROCm")
                return "rocm"

        rocm_smi = shutil.which("rocm-smi")
        if rocm_smi:
            result = subprocess.run(
                [rocm_smi, "--showproductname"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                print("[dragon-cli] Detected AMD GPU via rocm-smi")
                return "rocm"

    if platform.system() == "Darwin":
        system_profiler = shutil.which("system_profiler")
        if system_profiler:
            result = subprocess.run(
                [system_profiler, "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and "Apple" in result.stdout:
                print("[dragon-cli] Detected Apple Silicon GPU via system_profiler")
                return "mps"

    print("[dragon-cli] No dedicated GPU detected, using CPU-only mode")
    return gpu_type


def prompt_backend_choice(gpu_type: str) -> str:
    """Ask the user in English which backend to use.

    CPU remains the default backend for safety, even if a GPU is detected.
    """
    if gpu_type == "cpu":
        print("[dragon-cli] No dedicated GPU detected. Defaulting to CPU backend.")
        return "cpu"

    print("[dragon-cli] GPU detected. Do you want to use a GPU-enabled wheel?")
    print("[dragon-cli] Default backend: CPU")
    print("[dragon-cli] Available backends: cpu, cuda, rocm, mps")

    if not sys.stdin.isatty():
        print("[dragon-cli] Non-interactive shell detected. Keeping CPU as the default backend.")
        return "cpu"

    answer = input("[dragon-cli] Choose backend [cpu]: ").strip().lower()
    if answer in {"cpu", "cuda", "rocm", "mps"}:
        return answer
    if answer == "":
        return "cpu"

    print("[dragon-cli] Invalid choice. Keeping CPU as the default backend.")
    return "cpu"


def get_install_command(backend: str):
    """Install llama-cpp-python using a prebuilt wheel backend.

    CPU is the safe default; GPU backends can be selected later by the user.
    """
    index_url = WHEEL_INDEX_URLS.get(backend, WHEEL_INDEX_URLS["cpu"])
    print(f"[dragon-cli] Installing llama-cpp-python from prebuilt {backend.upper()} wheels ({index_url})...")

    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "llama-cpp-python",
        "--prefer-binary", "--extra-index-url", index_url,
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

    gpu_type = detect_gpu()
    print()

    backend = prompt_backend_choice(gpu_type)
    print()

    result = get_install_command(backend)

    if result.returncode == 0:
        print()
        print("[dragon-cli] ✓ llama-cpp-python installed successfully!")

        if backend != "cpu":
            print(f"[dragon-cli] GPU backend selected ({backend.upper()})")
            print("[dragon-cli] GPU-enabled wheel installed successfully")
        else:
            print("[dragon-cli] CPU backend selected by default")
            print("[dragon-cli] You can choose a GPU wheel later if needed")

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

"""Cross-platform installer for llama-cpp-python.

Compiling llama-cpp-python from source requires a C/C++ toolchain (Visual
Studio Build Tools on Windows, gcc/build-essential on Linux, Xcode command
line tools on macOS). This detects the current OS and installs a prebuilt
CPU wheel instead, so most users never need a compiler at all.
"""

import platform
import subprocess
import sys

WHEEL_INDEX_URL = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

TROUBLESHOOTING = {
    "Windows": (
        "No prebuilt wheel worked for your Windows/Python combination.\n"
        "Install the 'Desktop development with C++' workload via the Visual\n"
        "Studio Installer (https://visualstudio.microsoft.com/visual-cpp-build-tools/),\n"
        "plus CMake, then retry: pip install llama-cpp-python"
    ),
    "Linux": (
        "No prebuilt wheel worked for your Linux/Python combination.\n"
        "Install a C/C++ toolchain and CMake, e.g. on Debian/Ubuntu:\n"
        "    sudo apt install build-essential cmake\n"
        "then retry: pip install llama-cpp-python"
    ),
    "Darwin": (
        "No prebuilt wheel worked for your macOS/Python combination.\n"
        "Install the Xcode command line tools:\n"
        "    xcode-select --install\n"
        "then retry: pip install llama-cpp-python"
    ),
}


def main() -> None:
    system = platform.system()
    print(f"[dragon-cli] Detected {system} ({platform.machine()}), Python {platform.python_version()}")
    print(f"[dragon-cli] Installing llama-cpp-python from prebuilt CPU wheels ({WHEEL_INDEX_URL})...")

    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "llama-cpp-python",
        "--prefer-binary", "--extra-index-url", WHEEL_INDEX_URL,
    ])
    if result.returncode == 0:
        print("[dragon-cli] llama-cpp-python installed successfully.")
        return

    print("\n[dragon-cli] Prebuilt install failed.")
    print(TROUBLESHOOTING.get(
        system,
        "Install a C/C++ toolchain and CMake for your platform, then retry: "
        "pip install llama-cpp-python",
    ))
    sys.exit(1)


if __name__ == "__main__":
    main()

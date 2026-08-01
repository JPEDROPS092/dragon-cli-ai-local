"""Preload CUDA runtime shared libraries shipped via pip wheels.

The GPU-enabled llama-cpp-python wheel links against libcudart/libcublas/
libnvrtc, which the pip ``nvidia-*-cu12`` packages install under
``site-packages/nvidia/*/lib`` (not on the system library path). Loading them
here by absolute path with ``RTLD_GLOBAL`` lets libllama.so resolve them
without LD_LIBRARY_PATH. No-op when the libs are already resolvable or absent
(e.g. the CPU-only wheel).

Kept dependency-free (stdlib only) so it can run at package import time
without risking circular imports.
"""

import ctypes
import glob
import os
import sys

_CUDA_SONAMES = (
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libnvrtc.so.12",
)


def preload_cuda_runtime() -> None:
    loaded: set[str] = set()
    for entry in sys.path:
        nvidia_dir = os.path.join(entry, "nvidia")
        if not entry or not os.path.isdir(nvidia_dir):
            continue
        for soname in _CUDA_SONAMES:
            if soname in loaded:
                continue
            for match in glob.glob(os.path.join(nvidia_dir, "*", "lib", soname)):
                try:
                    ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)
                    loaded.add(soname)
                    break
                except OSError:
                    continue

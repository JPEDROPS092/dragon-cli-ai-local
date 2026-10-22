__version__ = "0.2.1"

from ._cuda import preload_cuda_runtime

preload_cuda_runtime()

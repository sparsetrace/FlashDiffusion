"""
setup.py
========
Builds flash_diffusion_cuda extension.

Usage
-----
  # CPU only
  pip install -e .

  # GPU (auto-detects SM)
  FLASHDIFFUSION_BUILD_CUDA=1 pip install -e ".[cuda]"

  # specific arch
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0" pip install -e ".[cuda]"
"""
import os, sys, glob
from setuptools import setup

def find_cuda_includes():
    """
    Find CUDA headers in order of preference:
    1. $CUDA_HOME/include  (system CUDA)
    2. nvidia pip packages (e.g. nvidia-cuda-runtime-cu12)
    3. /usr/local/cuda/include
    """
    candidates = []

    # system CUDA
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "include"))

    # nvidia pip packages — covers cusparse.h, cublas.h etc
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for base in [
        f"/usr/local/lib/{py}/site-packages/nvidia",
        f"/usr/lib/{py}/site-packages/nvidia",
    ]:
        candidates += glob.glob(f"{base}/*/include")

    # fallback
    candidates += ["/usr/local/cuda/include", "/usr/cuda/include"]

    # return only paths that exist and contain cuda_runtime.h
    return [p for p in candidates
            if os.path.isdir(p) and
            any(os.path.exists(os.path.join(p, h))
                for h in ["cuda_runtime.h", "cusparse.h", "cublas.h"])]


BUILD_CUDA = os.environ.get("FLASHDIFFUSION_BUILD_CUDA", "0") == "1"

ext_modules = []
cmdclass    = {}

if BUILD_CUDA:
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        cuda_includes = find_cuda_includes()
        print(f"[setup.py] CUDA include dirs: {cuda_includes}")

        # CUTLASS (optional — only needed for future TiledMMA version)
        cutlass_path = os.environ.get("CUTLASS_PATH", "")
        if cutlass_path:
            cuda_includes = [cutlass_path] + cuda_includes

        # arch flags
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if not arch_list:
            # auto-detect from current GPU
            try:
                major, minor = torch.cuda.get_device_capability()
                arch_list = f"{major}.{minor}"
            except Exception:
                arch_list = "8.0"
        print(f"[setup.py] CUDA arch: {arch_list}")

        gencode = []
        for a in arch_list.replace("+PTX", "").split():
            sm = a.strip().replace(".", "")
            gencode += [f"-gencode=arch=compute_{sm},code=sm_{sm}"]

        nvcc_flags = [
            "-std=c++17", "-O3", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ] + gencode

        ext_modules = [CUDAExtension(
            name    = "flash_diffusion_cuda",
            sources = ["src/flashdiffusion/csrc/flash_diffusion_sm80.cu"],
            include_dirs     = cuda_includes,
            extra_compile_args = {
                "cxx":  ["-std=c++17", "-O3"],
                "nvcc": nvcc_flags,
            },
        )]
        cmdclass = {"build_ext": BuildExtension}
        print("[setup.py] CUDA extension configured.")

    except ImportError as e:
        print(f"[setup.py] skipping CUDA extension: {e}")

setup(
    name             = "flashdiffusion",
    version          = "0.1.0",
    description      = "Tiled memory-efficient diffusion maps eigensolver",
    package_dir      = {"": "src"},
    packages         = ["flashdiffusion"],
    python_requires  = ">=3.10",
    install_requires = ["numpy>=1.24", "scipy>=1.10"],
    extras_require   = {
        "cuda": ["torch>=2.1"],
        "dev":  ["pytest", "matplotlib"],
    },
    ext_modules = ext_modules,
    cmdclass    = cmdclass,
)

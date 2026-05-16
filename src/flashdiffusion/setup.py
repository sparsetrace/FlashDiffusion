"""
setup.py
========
Builds the flash_diffusion_cuda extension for SM80 (A100 / Ampere).

Usage
-----
  # CPU-only (numpy backend, no compilation):
  pip install -e .

  # With SM80 CUDA kernel:
  TORCH_CUDA_ARCH_LIST="8.0" pip install -e ".[cuda]"

  # For SM80 + SM90 + SM120 (all at once):
  TORCH_CUDA_ARCH_LIST="8.0 9.0 12.0+PTX" pip install -e ".[cuda]"

SM target notes
---------------
  SM80  : A100, A10, A30  (mma.sync, cp.async)
  SM90  : H100, H200      (wgmma, TMA)  -- future kernel
  SM120 : RTX 5090/5080   (mma.sync extended, TMA)  -- future kernel
  SM100 : B100/B200        (UMMA, TMEM, TMA)  -- future kernel

For SM80 only (what we have now), set TORCH_CUDA_ARCH_LIST="8.0".
The kernel also runs on SM90/SM120 via PTX JIT fallback without recompilation,
but with suboptimal performance — compile natively for each target in production.
"""

import os
from setuptools import setup, find_packages

# Check if CUDA build is requested
BUILD_CUDA = os.environ.get("FLASHDIFFUSION_BUILD_CUDA", "0") == "1"

ext_modules = []
cmdclass = {}

if BUILD_CUDA:
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
        import torch

        # Detect SM from environment or fall back to sm_80
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0")
        print(f"[setup.py] Building CUDA extension for arch: {arch_list}")

        # Map arch strings to nvcc flags
        def arch_to_gencode(arch: str) -> list[str]:
            """Convert '8.0' -> ['-gencode', 'arch=compute_80,code=sm_80']"""
            flags = []
            for a in arch.split():
                a = a.strip().replace("+PTX", "")
                sm = a.replace(".", "")
                flags += [
                    "-gencode", f"arch=compute_{sm},code=sm_{sm}",
                ]
            # Add PTX for forward compat if requested
            if "+PTX" in arch:
                last_sm = arch_list.split()[-1].replace("+PTX","").replace(".","")
                flags += ["-gencode", f"arch=compute_{last_sm},code=compute_{last_sm}"]
            return flags

        nvcc_flags = [
            "-std=c++17",
            "-O3",
            "--use_fast_math",           # fast exp — matches our fast_exp()
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ] + arch_to_gencode(arch_list)

        # CUTLASS include path — set CUTLASS_PATH or we assume it's installed
        cutlass_include = os.environ.get(
            "CUTLASS_PATH",
            os.path.join(os.path.dirname(__file__), "third_party", "cutlass", "include")
        )

        ext_modules = [
            CUDAExtension(
                name  = "flash_diffusion_cuda",
                sources = [
                    "flashdiffusion/csrc/flash_diffusion_sm80.cu",
                ],
                include_dirs = [
                    cutlass_include,
                ],
                extra_compile_args = {
                    "cxx":  ["-std=c++17", "-O3"],
                    "nvcc": nvcc_flags,
                },
                # No torch autograd — pure forward kernel
                # Setting this removes the _cuda_graphs overhead
                extra_link_args = [],
            )
        ]
        cmdclass = {"build_ext": BuildExtension}
        print("[setup.py] CUDA extension configured.")

    except ImportError:
        print("[setup.py] torch not found — skipping CUDA extension.")

setup(
    name             = "flashdiffusion",
    version          = "0.1.0",
    description      = "Tiled memory-efficient diffusion maps eigensolver",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = ["numpy>=1.24", "scipy>=1.10"],
    extras_require   = {
        "cuda": ["torch>=2.1"],
        "dev":  ["pytest", "matplotlib"],
    },
    ext_modules = ext_modules,
    cmdclass    = cmdclass,
)

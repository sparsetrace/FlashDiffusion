"""
setup.py
========
Builds all FlashDiffusion CUDA extensions.

  CPU only:
    pip install -e .

  GPU — auto-detects SM:
    FLASHDIFFUSION_BUILD_CUDA=1 pip install -e ".[cuda]"

  GPU — specific arch:
    FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0 12.0" pip install -e ".[cuda]"
"""

import os, sys, glob
from setuptools import setup

def find_cuda_includes():
    py = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "include"))
    for base in [
        f"/usr/local/lib/{py}/site-packages/nvidia",
        f"/usr/lib/{py}/site-packages/nvidia",
    ]:
        candidates += glob.glob(f"{base}/*/include")
    candidates += ["/usr/local/cuda/include"]
    return [p for p in candidates
            if os.path.isdir(p) and
            any(os.path.exists(os.path.join(p, h))
                for h in ["cuda_runtime.h", "cusparse.h", "cublas.h"])]


# SM → source file mapping
_SM_SOURCES = {
    "80":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",
    "86":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",   # RTX 3090 etc
    "89":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",   # RTX 4090
    "90":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",   # H100 fallback until sm90 written
    "120": "src/flashdiffusion/csrc/flash_diffusion_sm120.cu",
    "121": "src/flashdiffusion/csrc/flash_diffusion_sm120.cu",  # DGX Spark
    "100": "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",   # B200 fallback until sm100 written
    "103": "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",   # B300 fallback
}

# SM → module name (pybind11 PYBIND11_MODULE name must match)
_SM_MODULE = {
    "80":  "flash_diffusion_cuda",
    "86":  "flash_diffusion_cuda",
    "89":  "flash_diffusion_cuda",
    "90":  "flash_diffusion_cuda",
    "120": "flash_diffusion_sm120",
    "121": "flash_diffusion_sm120",
    "100": "flash_diffusion_cuda",
    "103": "flash_diffusion_cuda",
}

BUILD_CUDA = os.environ.get("FLASHDIFFUSION_BUILD_CUDA", "0") == "1"

ext_modules = []
cmdclass    = {}

if BUILD_CUDA:
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        cuda_includes = find_cuda_includes()
        print(f"[setup.py] CUDA includes: {cuda_includes}")

        # auto-detect arch from GPU if not set
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if not arch_list:
            try:
                major, minor = torch.cuda.get_device_capability()
                arch_list = f"{major}.{minor}"
            except Exception:
                arch_list = "8.0"
        print(f"[setup.py] building for arch: {arch_list}")

        # group archs by module name to avoid duplicate extensions
        module_to_archs: dict[str, list[str]] = {}
        module_to_source: dict[str, str]      = {}
        for a in arch_list.replace("+PTX", "").split():
            sm = a.strip().replace(".", "")
            mod = _SM_MODULE.get(sm, "flash_diffusion_cuda")
            src = _SM_SOURCES.get(sm, _SM_SOURCES["80"])
            module_to_archs.setdefault(mod, []).append(sm)
            module_to_source[mod] = src  # last one wins (same file anyway)

        for mod, sms in module_to_archs.items():
            gencode = [f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in sms]
            nvcc_flags = [
                "-std=c++17", "-O3", "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ] + gencode

            src = module_to_source[mod]
            if not os.path.exists(src):
                print(f"[setup.py] skipping {mod} — {src} not found")
                continue

            print(f"[setup.py] building {mod} from {src} for SM{sms}")
            ext_modules.append(CUDAExtension(
                name    = mod,
                sources = [src],
                include_dirs = cuda_includes,
                extra_compile_args = {
                    "cxx":  ["-std=c++17", "-O3"],
                    "nvcc": nvcc_flags,
                },
            ))

        cmdclass = {"build_ext": BuildExtension}
        print(f"[setup.py] {len(ext_modules)} extension(s) configured.")

    except ImportError as e:
        print(f"[setup.py] skipping CUDA: {e}")

setup(
    name             = "flashdiffusion",
    version          = "0.1.0",
    description      = "Tiled memory-efficient diffusion maps eigensolver",
    package_dir      = {"": "src"},
    packages         = ["flashdiffusion"],
    python_requires  = ">=3.10",
    install_requires = ["numpy>=1.24", "scipy>=1.10"],
    extras_require   = {
        "cuda": ["torch>=2.1", "ninja"],
        "dev":  ["pytest>=7.0", "matplotlib"],
    },
    ext_modules = ext_modules,
    cmdclass    = cmdclass,
)

"""
setup.py
========
Builds FlashDiffusion CUDA extensions.

Kernels in csrc/:
  flash_diffusion_sm80.cu   — SM80 scalar (A100, RTX 3090/4090, H100 fallback)
  flash_diffusion_sm90.cu   — SM90 wgmma+TMA (H100 native, requires CUTLASS)
  flash_diffusion_sm100.cu  — SM100 WMMA tensor cores (B200/B300)
  flash_diffusion_sm120.cu  — SM120 128x128 tiles (RTX 5090 / PRO 6000)

Usage
-----
  # CPU only:
  pip install -e .

  # GPU — auto-detects SM from current GPU:
  FLASHDIFFUSION_BUILD_CUDA=1 pip install -e ".[cuda]"

  # specific arch:
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0"  pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0"  pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="10.0" pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0" pip install -e ".[cuda]"

  # multiple arches:
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0 12.0" pip install -e ".[cuda]"

SM90 note
---------
  Requires CUTLASS >= 3.5.1 headers for wgmma atoms.
  Set CUTLASS_PATH env var, or add as git submodule:
    git submodule add https://github.com/NVIDIA/cutlass third_party/cutlass
    git submodule update --init --depth=1

SM100 note
----------
  Uses WMMA (nvcuda::wmma) tensor cores — no CUTLASS needed.
  Uses regular expf, so --use_fast_math is deliberately omitted.
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
    candidates += ["/usr/local/cuda/include", "/usr/cuda/include"]
    return [p for p in candidates
            if os.path.isdir(p) and
            any(os.path.exists(os.path.join(p, h))
                for h in ["cuda_runtime.h", "cusparse.h", "cublas.h"])]


def find_cutlass_include():
    env = os.environ.get("CUTLASS_PATH", "")
    if env and os.path.isdir(env):
        return env
    sub = os.path.join(os.path.dirname(__file__),
                       "third_party", "cutlass", "include")
    if os.path.isdir(sub):
        return sub
    if os.path.isdir("/tmp/cutlass/include"):
        return "/tmp/cutlass/include"
    return None


# ---------------------------------------------------------------------------
# SM routing tables
# ---------------------------------------------------------------------------

_SM_SOURCES = {
    "80":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",
    "86":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",
    "89":  "src/flashdiffusion/csrc/flash_diffusion_sm80.cu",
    "90":  "src/flashdiffusion/csrc/flash_diffusion_sm90.cu",
    "100": "src/flashdiffusion/csrc/flash_diffusion_sm100.cu",
    "103": "src/flashdiffusion/csrc/flash_diffusion_sm100.cu",
    "120": "src/flashdiffusion/csrc/flash_diffusion_sm120.cu",
    "121": "src/flashdiffusion/csrc/flash_diffusion_sm120.cu",
}

_SM_MODULE = {
    "80":  "flash_diffusion_cuda",
    "86":  "flash_diffusion_cuda",
    "89":  "flash_diffusion_cuda",
    "90":  "flash_diffusion_sm90",
    "100": "flash_diffusion_sm100",
    "103": "flash_diffusion_sm100",
    "120": "flash_diffusion_sm120",
    "121": "flash_diffusion_sm120",
}

_SM_ARCH = {
    "80":  "sm_80",
    "86":  "sm_86",
    "89":  "sm_89",
    "90":  "sm_90a",   # 'a' required for wgmma
    "100": "sm_100a",  # 'a' for full SM100 features
    "103": "sm_100a",  # SM103 uses same arch as SM100
    "120": "sm_120",
    "121": "sm_121",
}

# SM100 uses regular expf — --use_fast_math changes semantics, omit it
_SM_NO_FAST_MATH = {"100", "103"}

# SM90 needs CUTLASS headers
_SM_NEEDS_CUTLASS = {"90"}


def _gencode(sm):
    arch    = _SM_ARCH.get(sm, f"sm_{sm}")
    compute = arch.replace("sm_", "compute_")
    return f"-gencode=arch={compute},code={arch}"


# ---------------------------------------------------------------------------
# Extension build
# ---------------------------------------------------------------------------

BUILD_CUDA = os.environ.get("FLASHDIFFUSION_BUILD_CUDA", "0") == "1"
ext_modules = []
cmdclass    = {}

if BUILD_CUDA:
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        cuda_includes   = find_cuda_includes()
        cutlass_include = find_cutlass_include()

        print(f"[setup.py] CUDA includes : {cuda_includes}")
        print(f"[setup.py] CUTLASS       : {cutlass_include or 'NOT FOUND'}")

        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if not arch_list:
            try:
                major, minor = torch.cuda.get_device_capability()
                arch_list = f"{major}.{minor}"
                print(f"[setup.py] auto-detected: SM{major*10+minor} "
                      f"({torch.cuda.get_device_name(0)})")
            except Exception:
                arch_list = "8.0"
                print("[setup.py] no GPU found, defaulting to 8.0")

        sms = [a.strip().replace(".", "").replace("+PTX", "")
               for a in arch_list.split() if a.strip()]

        groups: dict[tuple[str, str], list[str]] = {}
        for sm in sms:
            mod = _SM_MODULE.get(sm, "flash_diffusion_cuda")
            src = _SM_SOURCES.get(sm, _SM_SOURCES["80"])
            groups.setdefault((mod, src), []).append(sm)

        for (mod, src), sm_list in groups.items():
            if not os.path.exists(src):
                print(f"[setup.py] SKIP {mod}: {src} not found in repo")
                continue

            needs_cutlass = any(sm in _SM_NEEDS_CUTLASS for sm in sm_list)
            if needs_cutlass and cutlass_include is None:
                print(f"[setup.py] SKIP {mod} (SM90): CUTLASS not found.")
                continue

            no_fast_math = any(sm in _SM_NO_FAST_MATH for sm in sm_list)

            includes = cuda_includes[:]
            if needs_cutlass:
                includes = [cutlass_include] + includes

            nvcc_extra = []
            if needs_cutlass:
                nvcc_extra += [
                    "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1",
                    "-DCUTE_ARCH_MMA_SM90A_ENABLED=1",
                ]

            nvcc_flags = [
                "-std=c++17", "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ]
            if not no_fast_math:
                nvcc_flags.append("--use_fast_math")

            nvcc_flags += [_gencode(sm) for sm in sm_list]
            nvcc_flags += nvcc_extra

            print(f"[setup.py] building {mod}  SM{sm_list}  "
                  f"arch={[_SM_ARCH.get(s, s) for s in sm_list]}  "
                  f"fast_math={not no_fast_math}")

            ext_modules.append(CUDAExtension(
                name    = mod,
                sources = [src],
                include_dirs = includes,
                extra_compile_args = {
                    "cxx":  ["-std=c++17", "-O3"],
                    "nvcc": nvcc_flags,
                },
            ))

        cmdclass = {"build_ext": BuildExtension}
        print(f"[setup.py] {len(ext_modules)} extension(s) configured.")

    except ImportError as e:
        print(f"[setup.py] torch not available, skipping CUDA: {e}")
    except Exception as e:
        print(f"[setup.py] CUDA setup error: {e}")
        raise


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

setup(
    name             = "flashdiffusion",
    version          = "0.1.0",
    description      = "Tiled memory-efficient diffusion maps eigensolver",
    author           = "Julio Candanedo",
    author_email     = "julio@sparsetrace.ai",
    license          = "MIT",
    package_dir      = {"": "src"},
    packages         = ["flashdiffusion"],
    python_requires  = ">=3.10",
    install_requires = ["numpy>=1.24", "scipy>=1.10"],
    extras_require   = {
        "cuda":     ["torch>=2.1", "ninja"],
        "examples": ["matplotlib>=3.7", "scikit-learn>=1.3", "jupyter>=1.0"],
        "dev":      ["pytest>=7.0", "ruff>=0.3.0", "black>=24.0"],
    },
    ext_modules = ext_modules,
    cmdclass    = cmdclass,
)

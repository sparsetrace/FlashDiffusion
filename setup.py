"""
setup.py
========
Builds FlashDiffusion CUDA extensions.

Kernels in csrc/:
  flash_diffusion_sm80.cu   — SM80 scalar       (A100, RTX 3090/4090, fallback)
  flash_diffusion_sm90.cu   — SM90 wgmma+TMA    (H100 native, requires CUTLASS)
  flash_diffusion_sm100.cu  — SM100 UMMA+TMEM   (B200/B300 native, requires CUTLASS)
  flash_diffusion_sm120.cu  — SM120 128x128 tile (RTX 5090 / RTX PRO 6000)

Usage
-----
  # CPU only:
  pip install -e .

  # GPU — auto-detects SM from current GPU:
  FLASHDIFFUSION_BUILD_CUDA=1 pip install -e ".[cuda]"

  # specific arch:
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0"  pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="10.0" pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0" pip install -e ".[cuda]"

  # multiple arches:
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0 10.0 12.0" pip install -e ".[cuda]"

CUTLASS note
------------
  SM90 and SM100 kernels require CUTLASS headers.
  Set CUTLASS_PATH env var, or add as git submodule:
    git submodule add https://github.com/NVIDIA/cutlass third_party/cutlass
    git submodule update --init --depth=1
  SM90 requires CUTLASS >= 3.5.1
  SM100 requires CUTLASS >= 3.7.0  (for TmemAllocator + umma_arrive/commit/wait)
"""

import os, sys, glob
from setuptools import setup


def find_cuda_includes():
    """Find CUDA headers from nvidia pip packages or system CUDA."""
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


def find_cutlass_include(min_version=None):
    """
    Find CUTLASS headers.
    Checks CUTLASS_PATH env var, then third_party/cutlass, then /tmp/cutlass*.
    min_version: '3.5.1' or '3.7.0' — checks version.h if provided.
    """
    candidates = []

    env = os.environ.get("CUTLASS_PATH", "")
    if env:
        candidates.append(env)

    sub = os.path.join(os.path.dirname(__file__),
                       "third_party", "cutlass", "include")
    if os.path.isdir(sub):
        candidates.append(sub)

    # notebook builds clone to /tmp/cutlass or /tmp/cutlass37 etc.
    for p in sorted(glob.glob("/tmp/cutlass*/include"), reverse=True):
        candidates.append(p)

    for path in candidates:
        if not os.path.isdir(path):
            continue
        if min_version is None:
            return path
        # check version
        ver_file = os.path.join(path, "cutlass", "version.h")
        if os.path.exists(ver_file):
            with open(ver_file) as f:
                content = f.read()
            # parse CUTLASS_VERSION_MAJOR/MINOR/PATCH
            import re
            major = re.search(r'CUTLASS_VERSION_MAJOR\s+(\d+)', content)
            minor = re.search(r'CUTLASS_VERSION_MINOR\s+(\d+)', content)
            patch = re.search(r'CUTLASS_VERSION_PATCH\s+(\d+)', content)
            if major and minor and patch:
                found = (int(major.group(1)),
                         int(minor.group(1)),
                         int(patch.group(1)))
                need  = tuple(int(x) for x in min_version.split('.'))
                if found >= need:
                    return path
                else:
                    print(f"[setup.py] CUTLASS at {path} is "
                          f"{'.'.join(str(x) for x in found)}, "
                          f"need >= {min_version} — skipping")
        else:
            return path   # no version.h, assume ok

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
    "90":  "sm_90a",    # 'a' required for wgmma
    "100": "sm_100a",   # 'a' required for UMMA/TMEM
    "103": "sm_100a",   # SM103 uses same arch as SM100
    "120": "sm_120",
    "121": "sm_121",
}

# CUTLASS requirements per SM
# SM90 needs CUTLASS >= 3.5.1 (wgmma)
# SM100 needs CUTLASS >= 3.7.0 (UMMA + TmemAllocator)
_SM_CUTLASS_MIN = {
    "90":  "3.5.1",
    "100": "3.7.0",
    "103": "3.7.0",
}

# Extra nvcc defines per SM
_SM_NVCC_DEFINES = {
    "90":  ["-DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1",
            "-DCUTE_ARCH_MMA_SM90A_ENABLED=1"],
    "100": ["-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "-DCUTE_ARCH_MMA_SM100A_ENABLED=1"],
    "103": ["-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
            "-DCUTE_ARCH_MMA_SM100A_ENABLED=1"],
}


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

        cuda_includes = find_cuda_includes()
        print(f"[setup.py] CUDA includes: {cuda_includes}")

        # resolve arch list
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

        # normalise: "8.0 10.0+PTX" → ["80", "100"]
        sms = [a.strip().replace(".", "").replace("+PTX", "")
               for a in arch_list.split() if a.strip()]

        # group by (module, source)
        groups: dict[tuple[str, str], list[str]] = {}
        for sm in sms:
            mod = _SM_MODULE.get(sm, "flash_diffusion_cuda")
            src = _SM_SOURCES.get(sm, _SM_SOURCES["80"])
            groups.setdefault((mod, src), []).append(sm)

        for (mod, src), sm_list in groups.items():
            if not os.path.exists(src):
                print(f"[setup.py] SKIP {mod}: {src} not found")
                continue

            # check CUTLASS requirements
            includes = cuda_includes[:]
            nvcc_extra = []
            skip = False

            for sm in sm_list:
                if sm in _SM_CUTLASS_MIN:
                    min_ver = _SM_CUTLASS_MIN[sm]
                    cutlass = find_cutlass_include(min_ver)
                    if cutlass is None:
                        print(f"[setup.py] SKIP {mod} (SM{sm}): "
                              f"CUTLASS >= {min_ver} not found. "
                              f"Set CUTLASS_PATH or add submodule.")
                        skip = True
                        break
                    if cutlass not in includes:
                        includes = [cutlass] + includes
                        print(f"[setup.py] CUTLASS for SM{sm}: {cutlass}")
                    nvcc_extra += _SM_NVCC_DEFINES.get(sm, [])

            if skip:
                continue

            nvcc_flags = [
                "-std=c++17", "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ] + [_gencode(sm) for sm in sm_list] + list(set(nvcc_extra))

            print(f"[setup.py] building {mod}  SM{sm_list}  "
                  f"arch={[_SM_ARCH.get(s,s) for s in sm_list]}")

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

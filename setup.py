"""
setup.py
========
Builds FlashDiffusion CUDA extensions.

Kernels in csrc/:
  flash_diffusion_sm80.cu   — SM80 scalar       (A100, fallback)
  flash_diffusion_sm90.cu   — SM90 wgmma+TMA    (H100, requires CUTLASS >= 3.5)
  flash_diffusion_sm100.cu  — SM100 WMMA         (B200/B300)
  flash_diffusion_sm120.cu  — SM120 128×128 tile (RTX 5090 / RTX PRO 6000)

Usage
-----
  # CPU only:
  pip install -e .

  # GPU — auto-detects SM, auto-clones CUTLASS if needed:
  FLASHDIFFUSION_BUILD_CUDA=1 pip install -e ".[cuda]"

  # specific arch:
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0"  pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="10.0" pip install -e ".[cuda]"
  FLASHDIFFUSION_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST="12.0" pip install -e ".[cuda]"

CUTLASS
-------
  SM90 and SM100 kernels need CUTLASS headers.
  setup.py will auto-clone CUTLASS main into third_party/cutlass if not found.
  To skip auto-clone, set CUTLASS_PATH to an existing CUTLASS include dir.
  To use a git submodule instead:
    git submodule add https://github.com/NVIDIA/cutlass third_party/cutlass
    git submodule update --init --depth=1
"""

import os, sys, glob, subprocess
from setuptools import setup

# ---------------------------------------------------------------------------
# Helper: find CUDA headers
# ---------------------------------------------------------------------------

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
    return [p for p in candidates if os.path.isdir(p)]


# ---------------------------------------------------------------------------
# Helper: find or auto-clone CUTLASS
# ---------------------------------------------------------------------------

def _has_sm100_headers(inc):
    return os.path.exists(
        os.path.join(inc, "cute", "arch", "mma_sm100_umma.hpp"))

def _has_sm90_headers(inc):
    return os.path.exists(
        os.path.join(inc, "cute", "tensor.hpp"))

def find_cutlass_include():
    """
    Find CUTLASS headers. Check order:
      1. CUTLASS_PATH env var
      2. third_party/cutlass submodule (in this repo)
      3. /tmp/cutlass* (notebook session clones)
      4. Auto-clone CUTLASS main into third_party/cutlass
    Returns path to include/ dir, or None if all fail.
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    # 1. env var
    env = os.environ.get("CUTLASS_PATH", "")
    if env:
        candidates.append(env if env.endswith("include") else
                          os.path.join(env, "include"))

    # 2. git submodule
    sub = os.path.join(repo_root, "third_party", "cutlass", "include")
    candidates.append(sub)

    # 3. /tmp session clones — prefer cutlass_main (has SM100 headers)
    for p in sorted(glob.glob("/tmp/cutlass*/include"), reverse=True):
        candidates.append(p)

    for inc in candidates:
        if os.path.isdir(inc) and _has_sm90_headers(inc):
            return inc

    return None   # trigger auto-clone below


def ensure_cutlass():
    """
    Return CUTLASS include path, auto-cloning if necessary.
    Clones into third_party/cutlass (persists across sessions).
    """
    inc = find_cutlass_include()
    if inc:
        return inc

    # auto-clone CUTLASS main into third_party/cutlass
    repo_root = os.path.dirname(os.path.abspath(__file__))
    target    = os.path.join(repo_root, "third_party", "cutlass")
    inc       = os.path.join(target, "include")

    print("[setup.py] CUTLASS not found — cloning main branch into "
          "third_party/cutlass ...")
    print("[setup.py] (This happens once; subsequent installs reuse the clone.)")

    os.makedirs(os.path.dirname(target), exist_ok=True)
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/NVIDIA/cutlass", target
    ])
    print(f"[setup.py] CUTLASS cloned to {target}")
    return inc


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
    "100": "sm_100a",  # 'a' required for UMMA
    "103": "sm_100a",
    "120": "sm_120",
    "121": "sm_121",
}

# SMs that need CUTLASS headers
_SM_NEEDS_CUTLASS = {"90", "100", "103"}

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
    compute = arch.replace("sm_", "compute_").rstrip("a")  # compute_100a → compute_100
    return f"-gencode=arch={compute},code={arch}"


# ---------------------------------------------------------------------------
# Extension build
# ---------------------------------------------------------------------------

BUILD_CUDA  = os.environ.get("FLASHDIFFUSION_BUILD_CUDA", "0") == "1"
ext_modules = []
cmdclass    = {}

if BUILD_CUDA:
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

        cuda_includes = find_cuda_includes()
        print(f"[setup.py] CUDA includes: {len(cuda_includes)} paths found")

        # resolve arch list
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        if not arch_list:
            try:
                major, minor = torch.cuda.get_device_capability()
                arch_list = f"{major}.{minor}"
                print(f"[setup.py] Auto-detected: SM{major*10+minor} "
                      f"({torch.cuda.get_device_name(0)})")
            except Exception:
                arch_list = "8.0"
                print("[setup.py] No GPU found, defaulting to SM80")

        # normalise: "8.0 10.0+PTX" → ["80", "100"]
        sms = [a.strip().replace(".", "").replace("+PTX", "")
               for a in arch_list.split() if a.strip()]

        # check if any SM needs CUTLASS
        needs_cutlass_globally = any(sm in _SM_NEEDS_CUTLASS for sm in sms)
        cutlass_inc = None
        if needs_cutlass_globally:
            try:
                cutlass_inc = ensure_cutlass()
                print(f"[setup.py] CUTLASS: {cutlass_inc}")
                if _has_sm100_headers(cutlass_inc):
                    print("[setup.py] CUTLASS has SM100 UMMA headers ✓")
                else:
                    print("[setup.py] CUTLASS missing SM100 headers "
                          "(SM90 only, no tcgen05.mma support)")
            except Exception as e:
                print(f"[setup.py] WARNING: CUTLASS unavailable ({e}) — "
                      f"SM90/SM100 kernels will be skipped")

        # group by (module, source)
        groups: dict[tuple[str, str], list[str]] = {}
        for sm in sms:
            mod = _SM_MODULE.get(sm, "flash_diffusion_cuda")
            src = _SM_SOURCES.get(sm, _SM_SOURCES["80"])
            groups.setdefault((mod, src), []).append(sm)

        for (mod, src), sm_list in groups.items():
            if not os.path.exists(src):
                print(f"[setup.py] SKIP {mod}: {src} not found in repo")
                continue

            # check CUTLASS requirement
            needs_cutlass = any(sm in _SM_NEEDS_CUTLASS for sm in sm_list)
            if needs_cutlass and cutlass_inc is None:
                print(f"[setup.py] SKIP {mod} (SM{sm_list}): "
                      f"CUTLASS unavailable")
                continue

            # build include list
            includes = cuda_includes[:]
            if needs_cutlass:
                includes = [cutlass_inc] + includes

            # per-SM nvcc defines (deduplicated)
            nvcc_extra = list({
                d
                for sm in sm_list
                for d in _SM_NVCC_DEFINES.get(sm, [])
            })

            nvcc_flags = [
                "-std=c++17", "-O3", "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ] + [_gencode(sm) for sm in sm_list] + nvcc_extra

            print(f"[setup.py] Building {mod}  SM{sm_list}  "
                  f"arch={[_SM_ARCH.get(s, s) for s in sm_list]}")

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

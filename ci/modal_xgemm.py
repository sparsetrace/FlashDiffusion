# ci/modal_xgemm.py
"""Run the xgemm SM80 JAX FFI tests on a Modal A10G or A100 GPU.

From the FlashDiffusion repository root:
    python -m pip install modal
    modal setup
    modal run ci/modal_xgemm.py                 # A10G (sm_86)
    modal run ci/modal_xgemm.py --gpu A100      # A100 (sm_80)

The Modal Image uses a CUDA *development* base image (nvcc included),
installs CUDA-enabled JAX, copies this monorepo, builds xgemm._sm80, and
finally runs pytest on the provisioned GPU. No CUDA compiler is needed
on the GitHub Actions machine.
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/FlashDiffusion"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install("jax[cuda12]", "pytest>=8", "setuptools>=68", "wheel")
    .add_local_dir(
        str(ROOT),
        remote_path=REMOTE,
        copy=True,
        ignore=[
            ".git/**", ".pytest_cache/**", "**/__pycache__/**", "**/*.pyc",
            ".venv/**", "**/*.egg-info/**", "build/**", "dist/**",
            "xgemm/_sm80*.so", "**/.DS_Store",
        ],
    )
    .run_commands(
        f"python -m pip install --no-deps -e {REMOTE}",
        f"python {REMOTE}/tools/build_sm80.py",
    )
)

app = modal.App("flashdiffusion-xgemm-sm80-tests")


@app.function(image=image, gpu="A10G", timeout=900)
def run_xgemm_tests() -> None:
    import os
    import subprocess
    import sys

    # Don't quietly fall back to CPU if the CUDA JAX plugin failed to load.
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["XGEMM_TEST_CUDA"] = "1"

    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(["nvcc", "--version"], check=True)

    import jax
    from xgemm import _ffi

    devices = jax.devices("gpu")
    assert devices, "CUDA-enabled JAX found no GPU"
    assert jax.default_backend() == "gpu", "JAX selected a non-GPU backend"
    assert _ffi.available(), "xgemm._sm80 not built/importable"
    print("[xgemm] JAX devices:", devices, flush=True)
    print("[xgemm] CUDA extension loaded:", _ffi.available(), flush=True)

    subprocess.run(
        [
            sys.executable, "-m", "pytest", "-vv", "-rs",
            "tests/test_xgemm.py", "tests/test_xgemm_ffi_gpu.py",
        ],
        cwd=REMOTE,
        env=os.environ.copy(),
        check=True,
    )


@app.local_entrypoint()
def main(gpu: str = "A10G") -> None:
    gpu = gpu.upper()
    if gpu not in {"A10G", "A100"}:
        raise ValueError("Only A10G (sm_86) and A100 (sm_80) are supported")
    print(f"Running xgemm SM80 FFI correctness tests on Modal {gpu}")
    run_xgemm_tests.with_options(gpu=gpu).remote()

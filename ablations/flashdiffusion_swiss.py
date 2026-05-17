#!/usr/bin/env python3
"""
flashdiffusion_swiss.py
=======================

Swiss-roll benchmark suite for FlashDiffusion / DiffusionMap.

Experiment 1 focuses only on the low-dimensional Swiss roll:
  1. Dense-reference correctness check for prepass + matvec.
  2. N-scaling benchmark with repeats and error bars.
  3. Beta sweep at fixed N, probing the rank-one -> identity regime.
  4. One embedding visualization.

Default Coifman-Lafon alpha is 1.0.

Example
-------
python flashdiffusion_swiss.py \
    --repo /path/to/FlashDiffusion \
    --outdir results_swiss \
    --n-values 50000 100000 200000 500000 1000000 \
    --repeats 3 \
    --beta-n 200000 \
    --alpha 1.0

For a quick smoke test:
python flashdiffusion_swiss.py --quick --outdir results_quick
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Environment / backend
# -----------------------------------------------------------------------------

def add_repo_to_path(repo: Optional[str]) -> None:
    """Add repo/src or repo to sys.path if provided."""
    if repo is None:
        return

    repo_path = Path(repo).expanduser().resolve()
    candidates = [repo_path / "src", repo_path]

    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_flashdiffusion_backend() -> Tuple[Any, Any, Any]:
    """
    Import DiffusionMap and CUDA backend.

    Returns
    -------
    DiffusionMap : class
    fk           : flashdiffusion.kernel module
    ext          : selected CUDA extension module, or None
    """
    import flashdiffusion.kernel as fk
    importlib.reload(fk)

    dmap_mod = importlib.import_module("flashdiffusion.dmap")
    importlib.reload(dmap_mod)
    DiffusionMap = dmap_mod.DiffusionMap

    ext = fk._get_cuda_backend()
    sm = fk._cuda_sm() if torch.cuda.is_available() else None

    print(f"[backend] torch cuda available: {torch.cuda.is_available()}")
    print(f"[backend] cuda module: {ext}")
    print(f"[backend] SM: {sm} -> {getattr(fk, '_SM_BACKEND', {}).get(sm)}")

    return DiffusionMap, fk, ext


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# Swiss roll generation and beta estimation
# -----------------------------------------------------------------------------

def make_swiss_roll(
    N: int,
    device: str = "cuda",
    seed: int = 42,
    noise: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a normalized Swiss roll directly on GPU.

    Returns
    -------
    X : (N, 3) torch.float32
        Normalized ambient coordinates.
    t : (N,) torch.float32
        Latent roll coordinate.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    t = torch.rand(N, device=device, generator=g) * (4.5 - 1.5) * torch.pi + 1.5 * torch.pi
    h = torch.rand(N, device=device, generator=g) * 10.0

    X = torch.stack([t * torch.cos(t), h, t * torch.sin(t)], dim=1)
    X = X + torch.randn(X.shape, device=device, generator=g) * noise
    X = (X - X.mean(0)) / X.std(0)

    return X.float(), t.float()


def estimate_beta(
    X: torch.Tensor,
    k: int = 10,
    K: int = 5000,
    seed: int = 123,
) -> float:
    """
    Estimate beta as inverse mean k-NN squared distance on a subsample.
    """
    N = X.shape[0]
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)

    idx = torch.randperm(N, device=X.device, generator=g)[: min(K, N)]
    Xs = X[idx].float()

    sq = (Xs * Xs).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)).clamp(min=0.0)

    # Drop self-distance.
    knn = d2.topk(k + 1, dim=1, largest=False).values[:, 1:]
    return 1.0 / knn.mean().item()


# -----------------------------------------------------------------------------
# Correctness checks
# -----------------------------------------------------------------------------

def correctness_check(
    ext: Any,
    N: int = 300,
    d: int = 3,
    beta: float = 1.0,
    seed: int = 0,
    atol: float = 5e-2,
) -> Dict[str, Any]:
    """
    Compare CUDA prepass/matvec against dense NumPy reference.
    """
    if ext is None:
        print("[correctness] no CUDA extension found; skipping correctness check")
        return {"skipped": True}

    print(f"[correctness] N={N} d={d} beta={beta}")

    rng = np.random.default_rng(seed)
    X_np = rng.standard_normal((N, d), dtype=np.float32)

    X_fp = torch.from_numpy(X_np).half().cuda()
    sq = ext.compute_sq_norms(X_fp)

    w = torch.ones(N, dtype=torch.float32, device="cuda")
    cuda_sync()
    D = ext.prepass(X_fp, w, sq, float(beta))
    cuda_sync()

    X64 = X_np.astype(np.float64)
    sq64 = (X64 * X64).sum(1)
    d2 = np.maximum(sq64[:, None] + sq64[None, :] - 2.0 * (X64 @ X64.T), 0.0)
    K = np.exp(-beta * d2)

    D_ref = K.sum(1)
    prepass_max_abs = float(np.max(np.abs(D.detach().cpu().numpy() - D_ref)))
    prepass_rel_l2 = float(
        np.linalg.norm(D.detach().cpu().numpy() - D_ref) / np.linalg.norm(D_ref)
    )

    v = torch.randn(N, dtype=torch.float32, device="cuda")
    rscale = torch.ones(N, dtype=torch.float32, device="cuda")

    cuda_sync()
    out = ext.matvec(X_fp, v, rscale, sq, float(beta))
    cuda_sync()

    v_np = v.detach().cpu().numpy()
    out_ref = K @ v_np

    matvec_max_abs = float(np.max(np.abs(out.detach().cpu().numpy() - out_ref)))
    matvec_rel_l2 = float(
        np.linalg.norm(out.detach().cpu().numpy() - out_ref) / np.linalg.norm(out_ref)
    )

    result = {
        "skipped": False,
        "N": N,
        "d": d,
        "beta": beta,
        "prepass_max_abs": prepass_max_abs,
        "prepass_rel_l2": prepass_rel_l2,
        "matvec_max_abs": matvec_max_abs,
        "matvec_rel_l2": matvec_rel_l2,
        "prepass_pass": prepass_max_abs < atol,
        "matvec_pass": matvec_max_abs < atol,
        "atol": atol,
    }

    print(
        f"[correctness] prepass max abs={prepass_max_abs:.4g}, rel l2={prepass_rel_l2:.4g} "
        f"{'PASS' if result['prepass_pass'] else 'FAIL'}"
    )
    print(
        f"[correctness] matvec  max abs={matvec_max_abs:.4g}, rel l2={matvec_rel_l2:.4g} "
        f"{'PASS' if result['matvec_pass'] else 'FAIL'}"
    )

    return result


# -----------------------------------------------------------------------------
# DiffusionMap run wrapper
# -----------------------------------------------------------------------------

def run_dmap_swiss(
    DiffusionMap: Any,
    N: int,
    alpha: float = 1.0,
    n_components: int = 8,
    beta: Optional[float] = None,
    beta_scale: float = 1.0 / 8.0,
    beta_k: int = 10,
    beta_subsample: int = 5000,
    seed: int = 42,
    noise: float = 0.1,
    maxiter: Optional[int] = None,
    tol: float = 0.0,
    keep_arrays: bool = False,
) -> Dict[str, Any]:
    """
    Generate Swiss roll, run DiffusionMap, and collect runtime/geometry metrics.

    The matvec counter wraps dm.matvec so the Lanczos iteration count is measured
    independent of eigensolver internals.
    """
    print(f"[run] N={N:,} seed={seed} alpha={alpha}")

    X_gpu, t_gpu = make_swiss_roll(N, seed=seed, noise=noise)

    if beta is None:
        beta0 = estimate_beta(X_gpu, k=beta_k, K=beta_subsample, seed=seed + 999)
        beta = beta0 * beta_scale
    else:
        beta0 = np.nan

    X_np = X_gpu.cpu().numpy().astype(np.float32)
    t_np = t_gpu.cpu().numpy().astype(np.float32)

    print(f"[run] beta={beta:.8g} beta0={beta0} beta_scale={beta_scale}")

    dm = DiffusionMap(beta=float(beta), alpha=float(alpha), n_components=int(n_components), tile=512)

    cuda_sync()
    t0 = time.perf_counter()
    dm.fit(X_np)
    cuda_sync()
    fit_time = time.perf_counter() - t0

    # Wrap matvec to count eigensolver calls.
    matvec_counter = {"n": 0}
    orig_matvec = dm.matvec

    def counted_matvec(v: np.ndarray) -> np.ndarray:
        matvec_counter["n"] += 1
        return orig_matvec(v)

    dm.matvec = counted_matvec  # type: ignore[method-assign]

    cuda_sync()
    t0 = time.perf_counter()
    coords = dm.transform(t=1, tol=tol, maxiter=maxiter)
    cuda_sync()
    eig_time = time.perf_counter() - t0

    n_matvec = int(matvec_counter["n"])
    matvec_time = eig_time / n_matvec if n_matvec > 0 else np.nan

    # Geometry metrics.
    n_corr = min(5, coords.shape[1])
    corrs = [
        abs(np.corrcoef(t_np, coords[:, j])[0, 1])
        for j in range(n_corr)
        if np.std(coords[:, j]) > 0
    ]
    best_corr_t = float(np.max(corrs)) if len(corrs) else np.nan

    # Oracle linear R^2 from first m diffusion coordinates to latent t.
    # Subsample for large N to keep the least-squares cheap and deterministic.
    m = min(5, coords.shape[1])
    if N > 200_000:
        rng = np.random.default_rng(seed + 2027)
        idx = rng.choice(N, size=200_000, replace=False)
        C = coords[idx, :m]
        y = t_np[idx]
    else:
        C = coords[:, :m]
        y = t_np

    A = np.c_[C, np.ones(C.shape[0], dtype=C.dtype)]
    coef, *_ = np.linalg.lstsq(A.astype(np.float64), y.astype(np.float64), rcond=None)
    pred = A.astype(np.float64) @ coef
    denom = np.sum((y - y.mean()) ** 2)
    oracle_r2_t = float(1.0 - np.sum((y - pred) ** 2) / denom) if denom > 0 else np.nan

    ev = np.asarray(dm.eigenvalues_, dtype=np.float64)
    spectral_gap_12 = float(ev[0] - ev[1]) if len(ev) > 1 else np.nan
    spectral_gap_23 = float(ev[1] - ev[2]) if len(ev) > 2 else np.nan

    row = {
        "N": int(N),
        "seed": int(seed),
        "alpha": float(alpha),
        "n_components": int(n_components),
        "beta": float(beta),
        "beta0": float(beta0) if np.isfinite(beta0) else np.nan,
        "beta_scale": float(beta_scale),
        "fit_time": float(fit_time),
        "eig_time": float(eig_time),
        "total_time": float(fit_time + eig_time),
        "n_matvec": int(n_matvec),
        "matvec_time": float(matvec_time),
        "best_corr_t": float(best_corr_t),
        "oracle_r2_t": float(oracle_r2_t),
        "spectral_gap_12": float(spectral_gap_12),
        "spectral_gap_23": float(spectral_gap_23),
    }

    for j in range(min(8, len(ev))):
        row[f"lambda{j+1}"] = float(ev[j])

    print(
        f"[run] fit={fit_time:.2f}s eig={eig_time:.2f}s total={fit_time+eig_time:.2f}s "
        f"matvecs={n_matvec} matvec_time={matvec_time:.3f}s "
        f"best_corr={best_corr_t:.3f} R2={oracle_r2_t:.3f}"
    )

    if keep_arrays:
        row["_X_np"] = X_np
        row["_t_np"] = t_np
        row["_coords"] = coords
        row["_eigenvalues"] = ev

    # Help memory pressure between sweeps.
    del X_gpu, t_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

def benchmark_N_scaling(
    DiffusionMap: Any,
    N_values: Iterable[int],
    repeats: int,
    alpha: float,
    beta_scale: float,
    n_components: int,
    beta_k: int,
    beta_subsample: int,
    noise: float,
    maxiter: Optional[int],
    tol: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for N in N_values:
        for r in range(repeats):
            seed = 1000 + 37 * r + int(N % 997)
            print(f"\n=== N-scaling: N={N:,} repeat={r+1}/{repeats} ===")
            row = run_dmap_swiss(
                DiffusionMap=DiffusionMap,
                N=int(N),
                alpha=alpha,
                n_components=n_components,
                beta=None,
                beta_scale=beta_scale,
                beta_k=beta_k,
                beta_subsample=beta_subsample,
                seed=seed,
                noise=noise,
                maxiter=maxiter,
                tol=tol,
                keep_arrays=False,
            )
            row["experiment"] = "N_scaling"
            row["repeat"] = r
            rows.append(row)

    return pd.DataFrame(rows)


def benchmark_beta_sweep(
    DiffusionMap: Any,
    N: int,
    beta_factors: Iterable[float],
    repeats: int,
    alpha: float,
    n_components: int,
    beta_k: int,
    beta_subsample: int,
    noise: float,
    maxiter: Optional[int],
    tol: float,
) -> pd.DataFrame:
    # Estimate a single beta0 from a fixed dataset so beta_factor means one thing.
    X_gpu, _ = make_swiss_roll(N, seed=424242, noise=noise)
    beta0 = estimate_beta(X_gpu, k=beta_k, K=beta_subsample, seed=777)
    del X_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[beta sweep] fixed N={N:,}, beta0={beta0:.8g}")

    rows: List[Dict[str, Any]] = []

    for factor in beta_factors:
        beta = float(beta0 * factor)

        for r in range(repeats):
            seed = 2000 + 53 * r
            print(
                f"\n=== beta-sweep: N={N:,} beta/beta0={factor:g} "
                f"beta={beta:.8g} repeat={r+1}/{repeats} ==="
            )

            row = run_dmap_swiss(
                DiffusionMap=DiffusionMap,
                N=int(N),
                alpha=alpha,
                n_components=n_components,
                beta=beta,
                beta_scale=float(factor),
                beta_k=beta_k,
                beta_subsample=beta_subsample,
                seed=seed,
                noise=noise,
                maxiter=maxiter,
                tol=tol,
                keep_arrays=False,
            )
            row["experiment"] = "beta_sweep"
            row["repeat"] = r
            row["beta_factor"] = float(factor)
            row["beta0_fixed"] = float(beta0)
            rows.append(row)

    return pd.DataFrame(rows)


def run_visualization_case(
    DiffusionMap: Any,
    N: int,
    alpha: float,
    beta_scale: float,
    n_components: int,
    beta_k: int,
    beta_subsample: int,
    noise: float,
    seed: int,
    maxiter: Optional[int],
    tol: float,
) -> Dict[str, Any]:
    print(f"\n=== visualization run: N={N:,} ===")
    return run_dmap_swiss(
        DiffusionMap=DiffusionMap,
        N=N,
        alpha=alpha,
        n_components=n_components,
        beta=None,
        beta_scale=beta_scale,
        beta_k=beta_k,
        beta_subsample=beta_subsample,
        seed=seed,
        noise=noise,
        maxiter=maxiter,
        tol=tol,
        keep_arrays=True,
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _mean_std(df: pd.DataFrame, group: str, cols: List[str]) -> pd.DataFrame:
    agg = df.groupby(group)[cols].agg(["mean", "std"]).reset_index()
    return agg


def plot_N_scaling(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    g = df.groupby("N").agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for col, label in [
        ("fit_time", "precompute"),
        ("eig_time", "Lanczos"),
        ("total_time", "total"),
    ]:
        x = g.index.values.astype(float)
        y = g[(col, "mean")].values
        e = g[(col, "std")].fillna(0).values
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N samples")
    ax.set_ylabel("time (s)")
    ax.set_title("Swiss roll scaling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    x = g.index.values.astype(float)
    y = g[("matvec_time", "mean")].values
    e = g[("matvec_time", "std")].fillna(0).values
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label="time per matvec")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N samples")
    ax.set_ylabel("time per matvec (s)")
    ax.set_title("Implicit dense-kernel matvec")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = outdir / "swiss_roll_N_scaling.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_beta_sweep(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    g = df.groupby("beta_factor").agg(["mean", "std"])
    x = g.index.values.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time and matvec count.
    ax1 = axes[0]
    y = g[("eig_time", "mean")].values
    e = g[("eig_time", "std")].fillna(0).values
    ax1.errorbar(x, y, yerr=e, marker="o", capsize=3, label="Lanczos time")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\beta/\beta_0$")
    ax1.set_ylabel("Lanczos time (s)")
    ax1.grid(True, which="both", alpha=0.3)

    ax1b = ax1.twinx()
    y2 = g[("n_matvec", "mean")].values
    e2 = g[("n_matvec", "std")].fillna(0).values
    ax1b.errorbar(x, y2, yerr=e2, marker="s", capsize=3, label="# matvecs")
    ax1b.set_ylabel("matvec calls")
    ax1.set_title("Bandwidth controls eigensolver cost")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Geometry and spectrum.
    ax2 = axes[1]
    y = g[("oracle_r2_t", "mean")].values
    e = g[("oracle_r2_t", "std")].fillna(0).values
    ax2.errorbar(x, y, yerr=e, marker="o", capsize=3, label=r"oracle $R^2(t)$")

    y_corr = g[("best_corr_t", "mean")].values
    e_corr = g[("best_corr_t", "std")].fillna(0).values
    ax2.errorbar(x, y_corr, yerr=e_corr, marker="s", capsize=3, label="best |corr(t, DCj)|")

    if ("spectral_gap_12", "mean") in g.columns:
        y_gap = g[("spectral_gap_12", "mean")].values
        ax2.plot(x, y_gap, marker="^", linestyle="--", label=r"$\lambda_1-\lambda_2$")

    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\beta/\beta_0$")
    ax2.set_ylabel("quality / spectral gap")
    ax2.set_title("Rank-one to localized-kernel regime")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    path = outdir / "swiss_roll_beta_sweep.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)

    # Optional eigenvalue spectrum plot.
    lambda_cols = [c for c in df.columns if c.startswith("lambda")]
    lambda_cols = sorted(lambda_cols, key=lambda s: int(s.replace("lambda", "")))
    if lambda_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        for col in lambda_cols[:8]:
            y = g[(col, "mean")].values
            ax.plot(x, y, marker="o", label=col.replace("lambda", r"$\lambda_") + "$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta/\beta_0$")
        ax.set_ylabel("eigenvalue")
        ax.set_title("Top diffusion eigenvalues vs bandwidth")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        path = outdir / "swiss_roll_beta_eigenvalues.png"
        fig.savefig(path, dpi=200)
        print(f"[plot] wrote {path}")
        plt.close(fig)


def plot_embedding(case: Dict[str, Any], outdir: Path, n_plot: int = 5000) -> None:
    if "_X_np" not in case:
        return

    X_np = case["_X_np"]
    t_np = case["_t_np"]
    coords = case["_coords"]
    N = int(case["N"])

    rng = np.random.default_rng(12345)
    idx = rng.choice(N, size=min(n_plot, N), replace=False)

    fig = plt.figure(figsize=(15, 4))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(
        X_np[idx, 0],
        X_np[idx, 1],
        X_np[idx, 2],
        c=t_np[idx],
        cmap="plasma",
        s=1,
    )
    ax.set_title(f"Input 3D, N={N:,}")

    for p, (a, b) in enumerate([(0, 1), (1, 2)], start=2):
        ax = fig.add_subplot(1, 3, p)
        sc = ax.scatter(
            coords[idx, a],
            coords[idx, b],
            c=t_np[idx],
            cmap="plasma",
            s=1,
        )
        ax.set_xlabel(f"DC{a+1}")
        ax.set_ylabel(f"DC{b+1}")
        ax.set_title(f"DC{a+1} vs DC{b+1}")

    plt.colorbar(sc, ax=fig.axes[-1], label="latent t")
    plt.suptitle(
        f"FlashDiffusion Swiss roll, N={N:,}, alpha={case['alpha']}, beta={case['beta']:.4g}, "
        f"precompute={case['fit_time']:.1f}s, Lanczos={case['eig_time']:.1f}s",
        y=1.02,
        fontsize=10,
    )
    plt.tight_layout()

    path = outdir / f"swiss_roll_embedding_N{N}.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swiss-roll benchmark suite for FlashDiffusion.")

    p.add_argument("--repo", type=str, default=None, help="Path to FlashDiffusion repo; repo/src is added to sys.path.")
    p.add_argument("--outdir", type=str, default="results_swiss", help="Output directory.")

    p.add_argument("--alpha", type=float, default=1.0, help="Coifman-Lafon alpha. Default: 1.0")
    p.add_argument("--n-components", type=int, default=8, help="Number of eigenvectors including trivial.")
    p.add_argument("--tol", type=float, default=0.0, help="Eigensolver tolerance passed to transform().")
    p.add_argument("--maxiter", type=int, default=None, help="Eigensolver maxiter passed to transform().")
    p.add_argument("--noise", type=float, default=0.1, help="Swiss-roll Gaussian noise.")
    p.add_argument("--beta-k", type=int, default=10, help="k for kNN beta estimate.")
    p.add_argument("--beta-subsample", type=int, default=5000, help="Subsample K for beta estimate.")
    p.add_argument("--beta-scale", type=float, default=1.0/8.0, help="Default beta = beta0 * beta_scale for N scaling.")

    p.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 200_000, 500_000, 1_000_000],
        help="N values for scaling sweep.",
    )
    p.add_argument("--repeats", type=int, default=3, help="Repeats per N and per beta factor.")

    p.add_argument("--beta-n", type=int, default=200_000, help="Fixed N for beta sweep.")
    p.add_argument(
        "--beta-factors",
        type=float,
        nargs="+",
        default=[1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16],
        help="Multipliers of beta0 for beta sweep.",
    )

    p.add_argument("--viz-n", type=int, default=200_000, help="N for embedding visualization run.")
    p.add_argument("--viz-points", type=int, default=5000, help="Number of points in embedding plot.")

    p.add_argument("--skip-correctness", action="store_true", help="Skip dense-reference correctness check.")
    p.add_argument("--skip-n-scaling", action="store_true", help="Skip N-scaling benchmark.")
    p.add_argument("--skip-beta-sweep", action="store_true", help="Skip beta sweep benchmark.")
    p.add_argument("--skip-viz", action="store_true", help="Skip embedding visualization run.")

    p.add_argument("--quick", action="store_true", help="Quick smoke-test settings.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        args.n_values = [5_000, 10_000, 20_000]
        args.beta_n = 20_000
        args.viz_n = 20_000
        args.repeats = 1
        args.beta_factors = [1/32, 1/8, 1/2, 2, 8]
        args.beta_subsample = 2000
        print("[quick] using quick smoke-test settings")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    add_repo_to_path(args.repo)
    DiffusionMap, fk, ext = load_flashdiffusion_backend()

    metadata = {
        "alpha": args.alpha,
        "n_components": args.n_components,
        "tol": args.tol,
        "maxiter": args.maxiter,
        "noise": args.noise,
        "beta_k": args.beta_k,
        "beta_subsample": args.beta_subsample,
        "beta_scale": args.beta_scale,
        "n_values": args.n_values,
        "repeats": args.repeats,
        "beta_n": args.beta_n,
        "beta_factors": args.beta_factors,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repo": args.repo,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[main] wrote {outdir / 'metadata.json'}")

    if not args.skip_correctness:
        corr = correctness_check(ext=ext, N=300, d=3, beta=1.0)
        pd.DataFrame([corr]).to_csv(outdir / "correctness.csv", index=False)

    if not args.skip_n_scaling:
        df_N = benchmark_N_scaling(
            DiffusionMap=DiffusionMap,
            N_values=args.n_values,
            repeats=args.repeats,
            alpha=args.alpha,
            beta_scale=args.beta_scale,
            n_components=args.n_components,
            beta_k=args.beta_k,
            beta_subsample=args.beta_subsample,
            noise=args.noise,
            maxiter=args.maxiter,
            tol=args.tol,
        )
        df_N.to_csv(outdir / "swiss_roll_N_scaling.csv", index=False)
        print(f"[main] wrote {outdir / 'swiss_roll_N_scaling.csv'}")
        plot_N_scaling(df_N, outdir)

    if not args.skip_beta_sweep:
        df_beta = benchmark_beta_sweep(
            DiffusionMap=DiffusionMap,
            N=args.beta_n,
            beta_factors=args.beta_factors,
            repeats=args.repeats,
            alpha=args.alpha,
            n_components=args.n_components,
            beta_k=args.beta_k,
            beta_subsample=args.beta_subsample,
            noise=args.noise,
            maxiter=args.maxiter,
            tol=args.tol,
        )
        df_beta.to_csv(outdir / "swiss_roll_beta_sweep.csv", index=False)
        print(f"[main] wrote {outdir / 'swiss_roll_beta_sweep.csv'}")
        plot_beta_sweep(df_beta, outdir)

    if not args.skip_viz:
        case = run_visualization_case(
            DiffusionMap=DiffusionMap,
            N=args.viz_n,
            alpha=args.alpha,
            beta_scale=args.beta_scale,
            n_components=args.n_components,
            beta_k=args.beta_k,
            beta_subsample=args.beta_subsample,
            noise=args.noise,
            seed=314159,
            maxiter=args.maxiter,
            tol=args.tol,
        )
        # Save arrays for possible reuse, but compressed.
        np.savez_compressed(
            outdir / f"swiss_roll_embedding_N{args.viz_n}.npz",
            X=case["_X_np"],
            t=case["_t_np"],
            coords=case["_coords"],
            eigenvalues=case["_eigenvalues"],
            beta=case["beta"],
            alpha=case["alpha"],
        )
        plot_embedding(case, outdir, n_plot=args.viz_points)

    print(f"[done] outputs in {outdir}")


if __name__ == "__main__":
    main()

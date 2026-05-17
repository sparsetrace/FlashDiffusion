#!/usr/bin/env python3
"""
flashdiffusion_emnist.py
========================

EMNIST benchmark suite for FlashDiffusion / DiffusionMap.

Experiment 2 focuses on high-dimensional real data:
  1. EMNIST ByMerge / ByClass / Balanced loading from torchvision.
  2. N-scaling at fixed raw-pixel dimension d=784.
  3. 5-NN accuracy in diffusion-coordinate space vs number of DCs.
  4. Matvec-only microbenchmark after precompute.
  5. Backend tagging for SM80/SM100/SM120 comparisons.

Default:
  split       = bymerge
  include_test = True   # gives the full largest EMNIST variants
  alpha       = 1.0
  n_components = 20

Install deps if needed:
  pip install torch torchvision scikit-learn pandas matplotlib

Example full-ish run:
  python flashdiffusion_emnist.py \
      --repo /path/to/FlashDiffusion \
      --outdir results_emnist_bymerge_sm100 \
      --split bymerge \
      --include-test \
      --n-values 50000 100000 200000 400000 814255 \
      --repeats 3 \
      --n-components 20 \
      --backend-label B200-sm100

Quick smoke test:
  python flashdiffusion_emnist.py --quick --outdir results_emnist_quick

Notes
-----
This script intentionally keeps d fixed at 784 raw pixels. No PCA sweep.
To compare GPU backends, run this same script under each compiled backend
and compare the resulting CSVs. Use --backend-label to tag runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Environment / backend
# -----------------------------------------------------------------------------

def add_repo_to_path(repo: Optional[str]) -> None:
    if repo is None:
        return
    repo_path = Path(repo).expanduser().resolve()
    for p in [repo_path / "src", repo_path]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_flashdiffusion_backend() -> Tuple[Any, Any, Any, Dict[str, Any]]:
    import flashdiffusion.kernel as fk
    importlib.reload(fk)

    dmap_mod = importlib.import_module("flashdiffusion.dmap")
    importlib.reload(dmap_mod)
    DiffusionMap = dmap_mod.DiffusionMap

    ext = fk._get_cuda_backend()
    sm = fk._cuda_sm() if torch.cuda.is_available() else None
    backend_name = getattr(fk, "_SM_BACKEND", {}).get(sm)

    meta = {
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "sm": sm,
        "flashdiffusion_backend": str(backend_name),
        "cuda_ext": str(ext),
    }

    print(f"[backend] torch cuda available: {meta['torch_cuda_available']}")
    print(f"[backend] cuda device: {meta['cuda_device']}")
    print(f"[backend] cuda module: {ext}")
    print(f"[backend] SM: {sm} -> {backend_name}")

    return DiffusionMap, fk, ext, meta


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------------------------------------------------------
# EMNIST loading
# -----------------------------------------------------------------------------

def load_emnist(
    root: str,
    split: str = "bymerge",
    include_test: bool = True,
    download: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load EMNIST from torchvision.

    Parameters
    ----------
    split:
        One of: balanced, bymerge, byclass, digits, letters, mnist.
    include_test:
        If True, concatenate train and test sets. This yields the full EMNIST
        ByMerge/ByClass size of 814,255 images.

    Returns
    -------
    X : float32 array, shape (N, 784), values in [0,1]
    y : int64 array, shape (N,)
    meta : dict
    """
    try:
        from torchvision.datasets import EMNIST
    except Exception as e:
        raise RuntimeError(
            "torchvision is required. Install with: pip install torchvision"
        ) from e

    root = str(Path(root).expanduser())

    print(f"[data] loading EMNIST split={split!r}, include_test={include_test}")
    ds_train = EMNIST(root=root, split=split, train=True, download=download)
    X_train = ds_train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = np.asarray(ds_train.targets, dtype=np.int64)

    if include_test:
        ds_test = EMNIST(root=root, split=split, train=False, download=download)
        X_test = ds_test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_test = np.asarray(ds_test.targets, dtype=np.int64)

        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        split_id = np.concatenate(
            [
                np.zeros(len(y_train), dtype=np.int8),
                np.ones(len(y_test), dtype=np.int8),
            ]
        )
    else:
        X = X_train
        y = y_train
        split_id = np.zeros(len(y_train), dtype=np.int8)

    classes = np.unique(y)
    meta = {
        "split": split,
        "include_test": include_test,
        "N": int(X.shape[0]),
        "d": int(X.shape[1]),
        "n_classes": int(len(classes)),
        "classes": classes.tolist(),
        "train_count": int(len(y_train)),
        "test_count": int(np.sum(split_id == 1)),
    }

    print(
        f"[data] X={X.shape} dtype={X.dtype} classes={meta['n_classes']} "
        f"train={meta['train_count']} test={meta['test_count']}"
    )

    return X, y, meta


def subsample_emnist(
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    seed: int,
    stratified: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample EMNIST. For paper runs, stratified=True is preferable so all
    classes remain represented at smaller N.
    """
    total = X.shape[0]
    if N >= total:
        idx = np.arange(total)
        return X, y, idx

    rng = np.random.default_rng(seed)

    if not stratified:
        idx = rng.choice(total, size=N, replace=False)
        return X[idx], y[idx], idx

    idx_parts: List[np.ndarray] = []
    classes, counts = np.unique(y, return_counts=True)

    # Proportional allocation with at least one per class when possible.
    raw = counts / counts.sum() * N
    alloc = np.floor(raw).astype(int)

    # Ensure non-empty classes if N >= number of classes.
    if N >= len(classes):
        alloc = np.maximum(alloc, 1)

    # Adjust to exact N.
    diff = N - int(alloc.sum())
    frac_order = np.argsort(raw - np.floor(raw))[::-1]

    if diff > 0:
        for j in frac_order[:diff]:
            alloc[j] += 1
    elif diff < 0:
        for j in frac_order[::-1]:
            if diff == 0:
                break
            if alloc[j] > 1:
                alloc[j] -= 1
                diff += 1

    for c, n_c in zip(classes, alloc):
        c_idx = np.flatnonzero(y == c)
        take = min(int(n_c), len(c_idx))
        idx_parts.append(rng.choice(c_idx, size=take, replace=False))

    idx = np.concatenate(idx_parts)
    if len(idx) > N:
        idx = rng.choice(idx, size=N, replace=False)
    elif len(idx) < N:
        rest = np.setdiff1d(np.arange(total), idx, assume_unique=False)
        extra = rng.choice(rest, size=N - len(idx), replace=False)
        idx = np.concatenate([idx, extra])

    rng.shuffle(idx)
    return X[idx], y[idx], idx


# -----------------------------------------------------------------------------
# Beta estimation
# -----------------------------------------------------------------------------

def estimate_beta_numpy(
    X: np.ndarray,
    k: int = 10,
    K: int = 5000,
    seed: int = 123,
    batch: Optional[int] = None,
) -> float:
    """
    Estimate beta = 1 / mean kNN squared distance on a random subsample.

    Uses a dense K x K distance matrix. K=5000 is ~100 MB float32 for d2,
    and usually fine on CPU RAM.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    K_eff = min(K, N)
    idx = rng.choice(N, size=K_eff, replace=False)
    Xs = X[idx].astype(np.float32, copy=False)

    sq = np.sum(Xs * Xs, axis=1, dtype=np.float32)
    gram = Xs @ Xs.T
    d2 = sq[:, None] + sq[None, :] - 2.0 * gram
    np.maximum(d2, 0.0, out=d2)

    vals = np.partition(d2, kth=k, axis=1)[:, 1 : k + 1]
    mean_knn = float(vals.mean())
    beta = 1.0 / mean_knn
    print(f"[beta] K={K_eff} k={k} mean_knn_d2={mean_knn:.6g} beta={beta:.8g}")
    return beta


# -----------------------------------------------------------------------------
# DiffusionMap run wrapper
# -----------------------------------------------------------------------------

def run_dmap_emnist(
    DiffusionMap: Any,
    X: np.ndarray,
    y: np.ndarray,
    N: int,
    alpha: float = 1.0,
    n_components: int = 20,
    beta: Optional[float] = None,
    beta_k: int = 10,
    beta_subsample: int = 5000,
    seed: int = 0,
    stratified: bool = True,
    maxiter: Optional[int] = None,
    tol: float = 0.0,
    matvec_bench: int = 10,
    keep_arrays: bool = False,
) -> Dict[str, Any]:
    """
    Subsample EMNIST, fit DiffusionMap, run eigensolver, and collect metrics.
    """
    Xn, yn, idx = subsample_emnist(X, y, N=N, seed=seed, stratified=stratified)

    if beta is None:
        beta = estimate_beta_numpy(Xn, k=beta_k, K=beta_subsample, seed=seed + 999)

    print(f"[run] N={Xn.shape[0]:,} d={Xn.shape[1]} beta={beta:.8g} alpha={alpha}")

    dm = DiffusionMap(beta=float(beta), alpha=float(alpha), n_components=int(n_components))

    cuda_sync()
    t0 = time.perf_counter()
    dm.fit(Xn)
    cuda_sync()
    fit_time = time.perf_counter() - t0

    # Matvec microbenchmark after precompute, before Lanczos.
    matvec_bench_time = np.nan
    matvec_bench_each = np.nan
    if matvec_bench and matvec_bench > 0:
        rng = np.random.default_rng(seed + 2024)
        vecs = [rng.standard_normal(Xn.shape[0]).astype(np.float32) for _ in range(matvec_bench)]

        # Warmup.
        _ = dm.matvec(vecs[0])
        cuda_sync()

        t0 = time.perf_counter()
        for v in vecs:
            _ = dm.matvec(v)
        cuda_sync()
        matvec_bench_time = time.perf_counter() - t0
        matvec_bench_each = matvec_bench_time / matvec_bench
        print(f"[matvec bench] {matvec_bench} matvecs in {matvec_bench_time:.3f}s; each={matvec_bench_each:.4f}s")

    # Count matvecs during eigensolver.
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
    matvec_time_lanczos = eig_time / n_matvec if n_matvec > 0 else np.nan

    ev = np.asarray(dm.eigenvalues_, dtype=np.float64)

    row: Dict[str, Any] = {
        "N": int(Xn.shape[0]),
        "d": int(Xn.shape[1]),
        "seed": int(seed),
        "alpha": float(alpha),
        "n_components": int(n_components),
        "beta": float(beta),
        "fit_time": float(fit_time),
        "eig_time": float(eig_time),
        "total_time": float(fit_time + eig_time),
        "n_matvec": int(n_matvec),
        "matvec_time_lanczos": float(matvec_time_lanczos),
        "matvec_bench_n": int(matvec_bench),
        "matvec_bench_time": float(matvec_bench_time),
        "matvec_bench_each": float(matvec_bench_each),
        "n_classes_present": int(len(np.unique(yn))),
    }

    for j in range(min(32, len(ev))):
        row[f"lambda{j+1}"] = float(ev[j])

    print(
        f"[run] fit={fit_time:.2f}s eig={eig_time:.2f}s total={fit_time+eig_time:.2f}s "
        f"eig_matvecs={n_matvec} eig_matvec_time={matvec_time_lanczos:.4f}s"
    )
    print(f"[run] eigenvalues[:8] = {ev[:8]}")

    if keep_arrays:
        row["_X"] = Xn
        row["_y"] = yn
        row["_coords"] = coords
        row["_idx"] = idx
        row["_eigenvalues"] = ev

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


# -----------------------------------------------------------------------------
# KNN accuracy in diffusion coordinates
# -----------------------------------------------------------------------------

def knn_accuracy_sweep(
    coords: np.ndarray,
    y: np.ndarray,
    coord_dims: Sequence[int],
    n_neighbors: int = 5,
    train_size: int = 100_000,
    eval_size: int = 20_000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Estimate kNN accuracy in diffusion-coordinate space.

    For large N, exact all-vs-all scoring is expensive. This function fits on a
    random training subset and scores on a disjoint random evaluation subset.

    If N is small enough, it still uses disjoint train/eval splits unless the
    requested sizes exceed N.
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except Exception as e:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn") from e

    N = coords.shape[0]
    rng = np.random.default_rng(seed)

    perm = rng.permutation(N)
    n_train = min(train_size, max(1, N // 2))
    n_eval = min(eval_size, max(1, N - n_train))

    train_idx = perm[:n_train]
    eval_idx = perm[n_train : n_train + n_eval]

    if len(eval_idx) == 0:
        # Fallback for tiny smoke tests.
        train_idx = perm
        eval_idx = perm

    rows: List[Dict[str, Any]] = []

    for m in coord_dims:
        m_eff = int(min(m, coords.shape[1]))
        if m_eff <= 0:
            continue

        clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        t0 = time.perf_counter()
        clf.fit(coords[train_idx, :m_eff], y[train_idx])
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        acc = float(clf.score(coords[eval_idx, :m_eff], y[eval_idx]))
        score_time = time.perf_counter() - t0

        rows.append(
            {
                "coord_dim": m_eff,
                "knn_k": int(n_neighbors),
                "train_size": int(len(train_idx)),
                "eval_size": int(len(eval_idx)),
                "accuracy": acc,
                "fit_time": fit_time,
                "score_time": score_time,
            }
        )

        print(
            f"[knn] m={m_eff:>3} acc={acc:.4f} "
            f"train={len(train_idx)} eval={len(eval_idx)} score_time={score_time:.2f}s"
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

def benchmark_N_scaling(
    DiffusionMap: Any,
    X: np.ndarray,
    y: np.ndarray,
    N_values: Iterable[int],
    repeats: int,
    alpha: float,
    n_components: int,
    beta_k: int,
    beta_subsample: int,
    stratified: bool,
    maxiter: Optional[int],
    tol: float,
    matvec_bench: int,
    coord_dims: Sequence[int],
    knn_train_size: int,
    knn_eval_size: int,
    knn_k: int,
    backend_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    knn_rows: List[pd.DataFrame] = []

    last_case: Optional[Dict[str, Any]] = None

    for N in N_values:
        for r in range(repeats):
            seed = 10_000 + 101 * r + int(N % 997)
            print(f"\n=== EMNIST N-scaling: N={N:,} repeat={r+1}/{repeats} ===")

            keep_arrays = (r == 0)  # use first repeat for kNN and plotting.
            row = run_dmap_emnist(
                DiffusionMap=DiffusionMap,
                X=X,
                y=y,
                N=int(N),
                alpha=alpha,
                n_components=n_components,
                beta=None,
                beta_k=beta_k,
                beta_subsample=beta_subsample,
                seed=seed,
                stratified=stratified,
                maxiter=maxiter,
                tol=tol,
                matvec_bench=matvec_bench,
                keep_arrays=keep_arrays,
            )

            row["experiment"] = "N_scaling"
            row["repeat"] = r
            row["backend_label"] = backend_label
            run_rows.append({k: v for k, v in row.items() if not k.startswith("_")})

            if keep_arrays:
                last_case = row
                df_knn = knn_accuracy_sweep(
                    coords=row["_coords"],
                    y=row["_y"],
                    coord_dims=coord_dims,
                    n_neighbors=knn_k,
                    train_size=knn_train_size,
                    eval_size=knn_eval_size,
                    seed=seed + 404,
                )
                df_knn["N"] = int(N)
                df_knn["repeat"] = r
                df_knn["experiment"] = "component_sweep"
                df_knn["backend_label"] = backend_label
                df_knn["n_components"] = int(n_components)
                df_knn["alpha"] = float(alpha)
                df_knn["beta"] = float(row["beta"])
                knn_rows.append(df_knn)

            # Drop arrays after use.
            for key in ["_X", "_y", "_coords", "_idx", "_eigenvalues"]:
                row.pop(key, None)

    df_runs = pd.DataFrame(run_rows)
    df_knn_all = pd.concat(knn_rows, ignore_index=True) if knn_rows else pd.DataFrame()

    return df_runs, df_knn_all, last_case


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

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
    ax.set_xlabel("N images")
    ax.set_ylabel("time (s)")
    ax.set_title("EMNIST scaling at d=784")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    x = g.index.values.astype(float)
    if ("matvec_bench_each", "mean") in g.columns:
        y = g[("matvec_bench_each", "mean")].values
        e = g[("matvec_bench_each", "std")].fillna(0).values
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, label="matvec microbench")

    y2 = g[("matvec_time_lanczos", "mean")].values
    e2 = g[("matvec_time_lanczos", "std")].fillna(0).values
    ax.errorbar(x, y2, yerr=e2, marker="s", capsize=3, label="Lanczos avg")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N images")
    ax.set_ylabel("time per matvec (s)")
    ax.set_title("Implicit dense-kernel matvec")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    path = outdir / "emnist_N_scaling.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_knn_components(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for N, sub in sorted(df.groupby("N")):
        sub = sub.sort_values("coord_dim")
        ax.plot(
            sub["coord_dim"],
            sub["accuracy"],
            marker="o",
            label=f"N={int(N):,}",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of diffusion coordinates")
    ax.set_ylabel("5-NN accuracy")
    ax.set_title("EMNIST label structure in diffusion-coordinate space")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = outdir / "emnist_knn_vs_components.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_eigenvalues(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return

    lambda_cols = [c for c in df.columns if c.startswith("lambda")]
    if not lambda_cols:
        return
    lambda_cols = sorted(lambda_cols, key=lambda s: int(s.replace("lambda", "")))

    # Plot one representative spectrum per N, using repeat 0 if available.
    fig, ax = plt.subplots(figsize=(7, 4))

    for N, sub in sorted(df.groupby("N")):
        sub0 = sub.sort_values("repeat").iloc[0]
        vals = [sub0[c] for c in lambda_cols[:20] if c in sub0 and np.isfinite(sub0[c])]
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=f"N={int(N):,}")

    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Top EMNIST diffusion eigenvalues")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = outdir / "emnist_eigenvalues.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


def plot_embedding(case: Optional[Dict[str, Any]], outdir: Path, n_plot: int = 10_000) -> None:
    if case is None or "_coords" not in case:
        return

    coords = case["_coords"]
    y = case["_y"]
    N = coords.shape[0]

    rng = np.random.default_rng(12345)
    idx = rng.choice(N, size=min(n_plot, N), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (a, b) in zip(axes, [(0, 1), (1, 2)]):
        sc = ax.scatter(coords[idx, a], coords[idx, b], c=y[idx], cmap="tab20", s=2, alpha=0.8)
        ax.set_xlabel(f"DC{a+1}")
        ax.set_ylabel(f"DC{b+1}")
        ax.set_title(f"DC{a+1} vs DC{b+1}")

    fig.suptitle(
        f"EMNIST diffusion coordinates, N={N:,}, beta={case['beta']:.4g}, alpha={case['alpha']}",
        y=1.02,
    )
    fig.tight_layout()
    path = outdir / f"emnist_embedding_N{N}.png"
    fig.savefig(path, dpi=200)
    print(f"[plot] wrote {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EMNIST benchmark suite for FlashDiffusion.")

    p.add_argument("--repo", type=str, default=None, help="Path to FlashDiffusion repo; repo/src is added to sys.path.")
    p.add_argument("--outdir", type=str, default="results_emnist", help="Output directory.")

    p.add_argument("--data-root", type=str, default="/tmp/emnist", help="torchvision EMNIST root.")
    p.add_argument(
        "--split",
        type=str,
        default="bymerge",
        choices=["balanced", "bymerge", "byclass", "digits", "letters", "mnist"],
        help="EMNIST split.",
    )
    p.add_argument("--include-test", action="store_true", help="Concatenate train and test sets.")
    p.add_argument("--train-only", action="store_true", help="Use train set only, overrides --include-test.")
    p.add_argument("--no-download", action="store_true", help="Do not download EMNIST.")

    p.add_argument("--alpha", type=float, default=1.0, help="Coifman-Lafon alpha. Default: 1.0")
    p.add_argument("--n-components", type=int, default=20, help="Eigenvectors including trivial.")
    p.add_argument("--tol", type=float, default=0.0, help="Eigensolver tolerance passed to transform().")
    p.add_argument("--maxiter", type=int, default=None, help="Eigensolver maxiter passed to transform().")

    p.add_argument("--beta-k", type=int, default=10, help="k for kNN beta estimate.")
    p.add_argument("--beta-subsample", type=int, default=5000, help="Subsample size for beta estimate.")

    p.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[50_000, 100_000, 200_000, 400_000, 814_255],
        help="N values for scaling sweep. Values larger than dataset size are clipped/skipped.",
    )
    p.add_argument("--repeats", type=int, default=3, help="Repeats per N.")
    p.add_argument("--no-stratified", action="store_true", help="Disable stratified subsampling.")

    p.add_argument(
        "--coord-dims",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64],
        help="Diffusion-coordinate dimensions for 5-NN accuracy sweep.",
    )
    p.add_argument("--knn-k", type=int, default=5, help="k for kNN accuracy.")
    p.add_argument("--knn-train-size", type=int, default=100_000, help="Max training subset for kNN.")
    p.add_argument("--knn-eval-size", type=int, default=20_000, help="Max eval subset for kNN.")
    p.add_argument("--matvec-bench", type=int, default=10, help="Number of random matvecs after precompute.")
    p.add_argument("--plot-points", type=int, default=10_000, help="Points in DC scatter plot.")
    p.add_argument("--backend-label", type=str, default="", help="Tag for backend comparisons, e.g. B200-sm100.")

    p.add_argument("--skip-n-scaling", action="store_true", help="Skip N-scaling benchmark.")
    p.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")

    p.add_argument("--quick", action="store_true", help="Quick smoke-test settings.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_only:
        include_test = False
    else:
        # For this benchmark, full dataset is more useful by default if requested.
        include_test = bool(args.include_test)

    if args.quick:
        args.split = "balanced"
        include_test = False
        args.n_values = [5_000, 10_000]
        args.repeats = 1
        args.beta_subsample = 1000
        args.n_components = min(args.n_components, 12)
        args.coord_dims = [2, 4, 8]
        args.knn_train_size = 5000
        args.knn_eval_size = 2000
        args.matvec_bench = 3
        print("[quick] using quick smoke-test settings")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    add_repo_to_path(args.repo)
    DiffusionMap, fk, ext, backend_meta = load_flashdiffusion_backend()

    X, y, data_meta = load_emnist(
        root=args.data_root,
        split=args.split,
        include_test=include_test,
        download=not args.no_download,
    )

    # Clip N values to available dataset size and unique-sort.
    N_total = X.shape[0]
    N_values = sorted(set(int(min(N, N_total)) for N in args.n_values if int(min(N, N_total)) > 0))

    metadata = {
        "backend": backend_meta,
        "data": data_meta,
        "alpha": args.alpha,
        "n_components": args.n_components,
        "tol": args.tol,
        "maxiter": args.maxiter,
        "beta_k": args.beta_k,
        "beta_subsample": args.beta_subsample,
        "N_values": N_values,
        "repeats": args.repeats,
        "coord_dims": args.coord_dims,
        "knn_k": args.knn_k,
        "knn_train_size": args.knn_train_size,
        "knn_eval_size": args.knn_eval_size,
        "matvec_bench": args.matvec_bench,
        "backend_label": args.backend_label,
        "torch_version": torch.__version__,
        "repo": args.repo,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[main] wrote {outdir / 'metadata.json'}")

    if not args.skip_n_scaling:
        df_runs, df_knn, last_case = benchmark_N_scaling(
            DiffusionMap=DiffusionMap,
            X=X,
            y=y,
            N_values=N_values,
            repeats=args.repeats,
            alpha=args.alpha,
            n_components=args.n_components,
            beta_k=args.beta_k,
            beta_subsample=args.beta_subsample,
            stratified=not args.no_stratified,
            maxiter=args.maxiter,
            tol=args.tol,
            matvec_bench=args.matvec_bench,
            coord_dims=args.coord_dims,
            knn_train_size=args.knn_train_size,
            knn_eval_size=args.knn_eval_size,
            knn_k=args.knn_k,
            backend_label=args.backend_label,
        )

        df_runs.to_csv(outdir / "emnist_N_scaling.csv", index=False)
        df_knn.to_csv(outdir / "emnist_knn_components.csv", index=False)
        print(f"[main] wrote {outdir / 'emnist_N_scaling.csv'}")
        print(f"[main] wrote {outdir / 'emnist_knn_components.csv'}")

        if last_case is not None:
            # Save only the final representative embedding arrays. For full 814k this can be large.
            np.savez_compressed(
                outdir / f"emnist_embedding_N{last_case['N']}.npz",
                coords=last_case["_coords"],
                y=last_case["_y"],
                eigenvalues=last_case["_eigenvalues"],
                beta=last_case["beta"],
                alpha=last_case["alpha"],
            )
            print(f"[main] wrote {outdir / ('emnist_embedding_N' + str(last_case['N']) + '.npz')}")

        if not args.skip_plots:
            plot_N_scaling(df_runs, outdir)
            plot_knn_components(df_knn, outdir)
            plot_eigenvalues(df_runs, outdir)
            plot_embedding(last_case, outdir, n_plot=args.plot_points)

    print(f"[done] outputs in {outdir}")


if __name__ == "__main__":
    main()

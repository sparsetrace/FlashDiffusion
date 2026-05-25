# JFlashDiffusion Project TODO

This TODO captures the current plan for finishing JFlashDiffusion: a backend-generic, GPU-first diffusion-map solver for SM80, SM90, SM100/SM103, and SM120, with exact correctness paths first and performance optimizations layered in after.

## 0. Project goals

### Main goals

- Make diffusion maps run correctly and fast on GPUs using JAX + CUDA FFI.
- Support multiple GPU backends:
  - SM80 / SM86 / SM89
  - SM90
  - SM100 / SM103
  - SM120 / SM121
- Keep the notebook and public API backend-generic.
- Use block-IRLM / restarted block Lanczos because the core operation is a block kernel matmat.
- Exploit GPU-friendly RHS block sizes:
  - SM100 / B200: prefer RHS multiple 16.
  - SM120 / RTX-6000 Blackwell: prefer RHS multiple 8 with current CUDA kernel.
- Support multi-GPU row-sharded execution.
- Build toward continuation methods in beta and alpha.

### Performance targets

| Backend | Target |
|---|---:|
| RTX-6000 Torch baseline | about 175 s for 1M Swiss roll |
| RTX-6000 JAX SM120 | first goal: functional and comparable; later optimize |
| 1×B200 JAX SM100 | about 60 s or better |
| 2×B200 JAX SM100 | about 30 s or better |
| 8×B200 JAX SM100 | aspirational 5–10 s |

Current B200 results suggest the 2×B200 target is realistic if the solver can reduce the operator-call count toward the old 32-matvec baseline.

---

## 1. Current status

### Working / encouraging

- SM100 / B200 path is functional.
- 2×B200 scaling is promising.
  - Observed: about 49.7 s for 1M Swiss roll, 64 block matmats, converged.
  - Estimated if matmat count drops to 32: about 25–30 s on 2×B200.
- Single B200 path works.
  - Observed: about 95 s for 1M Swiss roll with RTX-like settings.
- Small Swiss roll tests pass against SciPy.
- Multi-GPU row-sharded path can produce excellent subspace agreement.
- Backend-generic API direction is clear:
  - `kernels.precompute(...)`
  - `kernels.matmat(state, U)`
  - `kernels.matvec(state, u)`
  - `kernels.backend_info()`

### Main issues

- SM120 / RTX-6000 JAX path is very slow with current defaults.
- `flashdiffusion.py` still tends to default to `k_search=16`, `bs=16`, which is good for SM100 but bad for current SM120.
- SM120 CUDA matmat has `BLOCK_S = 8`; `bs=16` causes two full all-pairs launches per logical matmat.
- Old Torch precompute timing was probably asynchronous or underreported, but old Torch full fit time around 175 s is a valid baseline.
- Need a more robust backend testbed.

---

## 2. Backend API standard

Every backend module should expose the same public interface.

### Public API

Each backend module should expose:

```python
precompute(...)
matvec(state, u)
matmat(state, U)
backend_info()
```

Generic notebooks should never call:

```python
precompute_sm100(...)
precompute_sm120(...)
```

except in backend-specific debugging.

### Internal symbols

Each `smXXX.py` should expose:

```python
_BACKEND_NAME
_SUPPORTED_SMS

_TARGET_SQ_NORMS
_TARGET_MATVEC
_TARGET_MATMAT

_SQNORMS_TARGET
_MATVEC_TARGET
_MATMAT_TARGET

_RHS_COL_MULTIPLE
_PREFERRED_BLOCK_SIZE
_MAX_EFFICIENT_BLOCK_SIZE
_FEATURE_DIM_MULTIPLE
_ROW_MULTIPLE
_V_INPUT_DTYPE

_ensure_ffi_registered()

_sqnorms_fn
_matvec_fn
_matmat_fn
```

Important: `_matmat_fn` must point to the raw signature:

```python
_matmat_fn(Q_scaled_or_K, Q, V_pre)
```

not the public stateful function:

```python
matmat(state, U)
```

---

## 3. Backend metadata

### SM100 / B200

Expected metadata:

```python
_BACKEND_NAME = "sm100"
_SUPPORTED_SMS = (100, 103)

_RHS_COL_MULTIPLE = 16
_PREFERRED_BLOCK_SIZE = 16
_MAX_EFFICIENT_BLOCK_SIZE = 64

_FEATURE_DIM_MULTIPLE = 64
_ROW_MULTIPLE = 128
_V_INPUT_DTYPE = jnp.bfloat16
```

Notes:

- Pads feature dimension to 64.
- Pads rows to multiples of 128.
- Pads RHS columns to multiples of 16.
- `block_size=16` is natural.

### SM120 / RTX-6000 Blackwell

Expected metadata:

```python
_BACKEND_NAME = "sm120"
_SUPPORTED_SMS = (120, 121)

_RHS_COL_MULTIPLE = 8
_PREFERRED_BLOCK_SIZE = 8
_MAX_EFFICIENT_BLOCK_SIZE = 8

_FEATURE_DIM_MULTIPLE = 1
_ROW_MULTIPLE = 1
_V_INPUT_DTYPE = jnp.float32
```

Notes:

- Current SM120 CUDA matmat uses `BLOCK_S = 8`.
- For `bs=16`, it launches the full matmat twice and recomputes dot/exp tiles twice.
- Do not default SM120 to `bs=16` until the CUDA kernel is improved.

### SM80 / SM90

Initial target:

```text
SM80/86/89 -> sm80
SM90       -> sm90 if present, else sm80 fallback compiled for sm_90
```

Long-term target:

```text
sm80.py + csrc/jax_flashdiffusion_sm80.cu
sm90.py + csrc/jax_flashdiffusion_sm90.cu
```

---

## 4. Backend router: `kernels/__init__.py`

### Goal

Select exactly one active backend and expose a stable public API.

### Desired routing table

```text
SM80/86/89 -> sm80
SM90       -> sm90 if present, else sm80 fallback
SM100      -> sm100
SM103      -> sm103 if present, else sm100 fallback
SM120/121  -> sm120
```

### Required router functions

```python
detect_sm()
backend_for_sm(sm)
load_backend(sm=None)
backend_info()
ensure_backend_registered(verbose=False)

precompute(...)
matvec(state, u)
matmat(state, U)
```

### Notebook usage after router

```python
from flashdiffusion import kernels

print(kernels.backend_info())

state = kernels.precompute(X, β=beta, α=alpha)
Y = kernels.matmat(state, U)
y = kernels.matvec(state, u)
```

---

## 5. Solver defaults in `flashdiffusion.py`

### Problem

Current backend-independent auto-block policy behaves like:

```text
k=5 -> k_search=16, bs=16, ncv=48, keep=32
```

This is good for SM100 and bad for SM120.

### Desired backend-aware behavior

`flashdiffusion.py` should read the active backend metadata and default to:

```text
SM100 -> internal_k_multiple=16, default bs=16
SM120 -> internal_k_multiple=8,  default bs=8
Fallback -> old/default behavior
```

Manual overrides must remain possible.

### Manual config examples

Old Torch-like comparison:

```python
flashdiffusion(
    ...,
    block_size=5,
    internal_k_multiple=None,
    ncv=40,
    keep=15,
)
```

SM120-native current CUDA path:

```python
flashdiffusion(
    ...,
    block_size=8,
    internal_k_multiple=8,
    max_auto_block_size=8,
    ncv=40,
    keep=16,
)
```

SM100 / B200 path:

```python
flashdiffusion(
    ...,
    block_size=16,
    internal_k_multiple=16,
    ncv=64,
    keep=32,
)
```

---

## 6. SM120 performance investigation

### Current diagnosis

The current SM120 JAX 1M run is slow because it still used:

```text
k_search=16, bs=16, ncv=48, keep=32
```

Current `jax_flashdiffusion_sm120.cu` has:

```cpp
static constexpr int BLOCK_S = 8;
```

and loops over RHS columns:

```cpp
for (int s0 = 0; s0 < b; s0 += BLOCK_S) {
    matmat_kernel_sm120<<<...>>>(...)
}
```

Therefore, `bs=16` causes two full all-pairs launches per logical matmat.

### Immediate SM120 tasks

- [ ] Ensure `sm120.py` exposes:
  - [ ] `_RHS_COL_MULTIPLE = 8`
  - [ ] `_PREFERRED_BLOCK_SIZE = 8`
  - [ ] `_MAX_EFFICIENT_BLOCK_SIZE = 8`
- [ ] Ensure `kernels.backend_info()` returns those values.
- [ ] Ensure `flashdiffusion.py` uses those values.
- [ ] Stop running full 1M SM120 tests with `bs=16` unless intentionally benchmarking the bad path.

### Raw SM120 matmat sweep

Run before full IRLM:

```python
for b in [1, 5, 8, 9, 16, 24, 32]:
    benchmark kernels.matmat(state, U_b)
```

Expected current pattern:

```text
b=5  ≈ b=8
b=16 ≈ 2 × b=8
b=32 ≈ 4 × b=8
```

If confirmed, the current bottleneck is the `BLOCK_S=8` chunking.

### Precompute breakdown

Instrument SM120 precompute:

```text
sq_norms time
D rowsum time
D_alpha rowsum time
```

Expected for N=1M:

```text
sq_norms: small
D rowsum: about 2–3 s
D_alpha rowsum: about 2–3 s
total: about 5–6 s
```

### Future SM120 CUDA improvements

- [ ] Test `BLOCK_S = 16`.
- [ ] Watch register pressure and occupancy.
- [ ] Redesign matmat to compute dot/exp tile once and accumulate more RHS columns.
- [ ] Consider porting old Torch distance-form kernel more directly to JAX FFI.

---

## 7. SM100 / B200 performance plan

### Current meaningful results

Observed 1M Swiss roll:

```text
Old RTX-6000 Torch: about 174.82 s, about 32 matvecs
1×B200 JAX:         about 95.24 s, 72 matmats, converged
2×B200 JAX:         about 49.68 s, 64 matmats, converged
```

Per operator estimate:

```text
1×B200: about 1.32 s per block matmat
2×B200: about 0.78 s per block matmat
```

If 2×B200 reaches 32 matmats:

```text
32 × 0.78 s ≈ 25 s
```

plus precompute.

### B200 solver configs to test

- [ ] `bs=16, ncv=48, keep=24`
- [ ] `bs=16, ncv=64, keep=32`
- [ ] `bs=16, ncv=80, keep=48`

Track:

```text
cycles
matmat calls
formula residuals
true residuals
subspace agreement
wall time
```

### B200 target

- [ ] Reduce 1M Swiss roll matmat count toward 32.
- [ ] Keep 2×B200 per-matmat time near or below 0.8 s.
- [ ] Reach about 30 s total on 2×B200.

---

## 8. Block-IRLM / restarted Lanczos design

### Why block-IRLM is the right inner solver

- The expensive primitive is `A @ V`, not scalar operations.
- GPUs prefer block RHS matmats.
- SM100 especially wants RHS multiples of 16.
- Diffusion-map spectra often have clusters, so block methods are more stable than single-vector methods.

### Important behavior

If user asks for `k=5`, backend may internally search a wider space:

```text
SM100: k_search=16
SM120: k_search=8 or 16 depending config
```

But the solver should:

```text
search with hardware-friendly block size
converge only requested k modes
return only requested k modes
```

Never pad Krylov vectors with zero columns. Extra columns must be real search directions.

### Needed feature

- [ ] Add `return_search_space=True`.

This should return the internal search/Ritz subspace, not only the requested `k` eigenvectors. It is needed for continuation.

---

## 9. Beta continuation

### Goal

Avoid cold-starting every beta.

Use:

```text
β0 < β1 < ... < βtarget
```

At each step:

1. Precompute at new beta.
2. Use previous eigenspace as a warm start.
3. Rayleigh-Ritz rotate old subspace under the new operator.
4. Run short IRLM cleanup.

### Why this helps

Even across spectral “snap” regions, the nearby beta subspace is usually much closer than random initialization, especially if tracking `k_search=16` or `32`.

### Snap detection

Track:

```text
eigengaps
subspace principal angles
Ritz residual under new beta
subspace derivative estimate
embedding shape metrics
```

Warning signs:

```text
small λ_k - λ_{k+1}
large subspace rotation
large derivative norm
sudden mode swaps
```

### Beta-gradient step proposal

Approximate:

```text
B ≈ (A(β+h)V - A(β)V) / h
B_perp = (I - VVᵀ)B
```

Then choose:

```text
Δβ ≈ safety * target_residual / ||B_perp||
```

Accept/reject using Rayleigh-Ritz residuals.

---

## 10. Alpha continuation and alpha=0 warm start

### Observation

For `α=0`, only the first degree normalization is needed.

This makes `α=0` a cheaper geometric base problem.

### Strategy

1. Solve `α=0`.
2. Compute target-α potential.
3. Use `α=0` search subspace as a trial space.
4. Rayleigh-Ritz rotate under target `α`.
5. Run short IRLM cleanup if needed.

### Important correction

Do not assume alpha only changes eigenvalues. The diagonal potential can rotate eigenvectors. Treat `α=0` as a warm-start subspace, not an exact formula.

---

## 11. Normalization optimization

### Current exact diffusion map

```text
W      = Gaussian affinity
D_CL   = W 1
P      = D_CL^{-α} W D_CL^{-α}
D      = P 1
A      = D^{-1/2} P D^{-1/2}
```

Current exact precompute:

```text
Pass 1: D_CL
Pass 2: D
```

### Exact future optimization

Fuse the second normalization into the first block apply:

```text
P [1, V] -> [D, P V]
```

This removes a standalone preprocessing pass for `D = P1`.

### Dependency caveat

The exact symmetric apply needs input-side `D_j^{-1/2}`. Therefore, the first pass can compute `D`, but exact symmetric applies need `D` available afterward.

### Possible generalized eigenproblem

Solve:

```text
P u = λ D u
```

using weighted inner product:

```text
<u, v>_D = uᵀ D v
```

This may avoid explicitly wrapping every apply with `D^{-1/2}`.

---

## 12. Validation tests

### Dense small tests

For `N=2048`, compare against SciPy dense reference.

Checks:

```text
eigenvalue relative error
orthogonality
true residual
subspace match for clustered modes
```

Do not rely only on individual eigenvector correlation.

### Subspace comparison

For clustered modes:

```python
s = svd(Q_ref.T @ Q_flash)
```

Use singular values near 1 as the pass criterion.

### Backend smoke test

For each backend:

- [ ] Detect backend.
- [ ] Build/register FFI.
- [ ] Precompute small Swiss roll.
- [ ] Check trivial mode residual.
- [ ] Raw matmat shape test for `b = 1, 5, 8, 16, 17, 32`.
- [ ] Small flashdiffusion solve.
- [ ] Optional sharded solve if `n_gpus > 1`.

### Raw matmat benchmark

For each backend and size:

```text
N=2048
N=8192
N=1M
```

Benchmark:

```text
b=1, 5, 8, 16, 32
single-GPU
multi-GPU if available
```

Record:

```text
backend
SM
N
d
b
time per matmat
effective chunk count
```

---

## 13. Notebook cleanup

### Environment/install cell

Use:

```python
import flashdiffusion
from flashdiffusion import kernels

print(kernels.backend_info())
kernels.ensure_backend_registered(verbose=True)
```

### Precompute cell

Use:

```python
state = kernels.precompute(X, β=beta, α=alpha)
```

### Matmat calls

Use:

```python
kernels.matmat(state, U)
```

### Visualization

For diffusion coordinates, convert symmetric eigenvectors:

```python
psi = evecs / np.sqrt(D_alpha)[:, None]
```

Do not judge latent plots from raw symmetric eigenvectors alone.

---

## 14. Immediate next tasks

1. Patch `flashdiffusion.py` to read backend metadata.
   - SM100 defaults to `bs=16`.
   - SM120 defaults to `bs=8`.

2. Run SM120 raw matmat sweep:
   - `b = 1, 5, 8, 9, 16, 24, 32`.

3. Rerun RTX-6000 1M Swiss roll with SM120-native settings:
   - `block_size=8`
   - `internal_k_multiple=8`
   - `max_auto_block_size=8`
   - `ncv=40`
   - `keep=16`

4. Rerun RTX-6000 apples-to-apples comparison:
   - `block_size=5`
   - `internal_k_multiple=None`
   - `ncv=40`
   - `keep=15`

5. For B200, run:
   - `bs=16, ncv=64, keep=32`
   - `bs=16, ncv=80, keep=48`

6. Add `return_search_space=True`.

7. Prototype beta continuation using Rayleigh-Ritz warm starts.

8. Prototype alpha=0 warm start.

---

## 15. Production readiness checklist

### Must have

- [ ] Backend router selects correct backend on SM80, SM90, SM100, SM103, SM120.
- [ ] Generic API works:
  - [ ] `kernels.precompute`
  - [ ] `kernels.matmat`
  - [ ] `kernels.matvec`
  - [ ] `kernels.backend_info`
- [ ] `flashdiffusion.py` uses backend metadata for default block sizing.
- [ ] SM120 defaults to `bs=8`.
- [ ] SM100 defaults to `bs=16`.
- [ ] Small dense SciPy tests pass.
- [ ] Multi-GPU sharded tests pass on at least 2 GPUs.
- [ ] Raw matmat sweep exists and is easy to run.
- [ ] 1M Swiss roll benchmark notebook is backend-generic.

### Should have

- [ ] `return_search_space=True`.
- [ ] Beta-continuation prototype.
- [ ] Alpha=0 warm-start prototype.
- [ ] Fused second-normalization prototype:
  - [ ] `P [1, V] -> [D, P V]`
- [ ] SM120 CUDA experiment with `BLOCK_S=16`.

### Nice to have

- [ ] SM90-specific backend.
- [ ] SM103-specific backend.
- [ ] Pallas/Mosaic prototype for Blackwell FA-style normalized matmat.
- [ ] Generalized eigenproblem variant using weighted inner product.
- [ ] Nyström/landmark path for approximate 1M+ workflows.

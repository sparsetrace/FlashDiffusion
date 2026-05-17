# SM100 UMMA v7 Development Notes
## For Paper 2 (JCTC) — FlashDiffusion on Membrane Proteins

**Date:** May 2026  
**Status:** v6 correct, v7 (full TMEM epilogue) deferred  
**Goal:** eliminate scalar SMEM epilogue, read dot products directly from TMEM

---

## What v6 achieves (current state)

- tcgen05.mma fires correctly on B200 via `gemm(tiled_mma, tCrA, tCrB, tCtAcc)`
- TMEM accumulator holds correct Xi @ Xj^T dot products
- Epilogue recomputes dots from SMEM scalarly (redundant, slow)
- Timing: 0.883s precompute vs 0.228s WMMA — 4× slower due to scalar epilogue
- Eigenvalues and DC1 corr=0.978 match WMMA exactly — numerically correct

---

## What v7 needs to do

Replace the scalar SMEM epilogue with TMEM→RMEM readout:

```cpp
// After gemm(tiled_mma, tCrA, tCrB, tCtAcc):

// 1. Create tiled copy for TMEM→RMEM
TiledCopy tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
ThrCopy   thr_t2r   = tiled_t2r.get_slice(tid);

// 2. Read TMEM → RMEM
Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
Tensor tDrAcc = make_tensor<TypeC>(shape(thr_t2r.partition_D(gC_dummy)));
copy(tiled_t2r, tDtAcc, tDrAcc);  // tcgen05.ld → registers

// 3. Each thread iterates over its (m,n) elements in tDrAcc
//    and computes dist2 = sq_X[i0+m] + sq_X[j0+n] - 2*dot
//    needs (m,n) coordinates for each element in tDrAcc
```

---

## Key obstacle: thread → (m,n) coordinate mapping

From `copy_traits_sm100.hpp`:

```
SM100_TMEM_LOAD_32dp32b1x:
  ThrID     = Layout<_32>              ← only 32 threads (1 warp!)
  ValID     = Layout<(32,32),(1,DP_b)> ← 32 values per thread
  SrcLayout = Layout<(32,1024),(0,1)>  ← all threads same base, 1024 values
  DstLayout = Layout<(32,32),(32,1)>   ← 32 threads × 32 values = 1024 fp32
```

**Critical:** ThrID=32 means only one warp (32 threads) participates per TMEM load instruction. For 128-thread kernel, warp 0 does the TMEM load.

For BM=128, BN=256 accumulator = 32768 fp32 values:
- 32dp32b1x: 1024 values per call → needs 32 calls
- 32dp32b32x: 32768 values per call → 1 call, but 32 threads get 1024 values each
- Use `SM100_TMEM_LOAD_32dp32b32x` for single-call full tile readout

After `copy(tiled_t2r, tDtAcc, tDrAcc)`:
- Each of 32 threads in warp 0 holds 1024 fp32 dot products (tDrAcc)
- Need to map flat register index → (m, n) coordinate in BM×BN tile
- Then distribute to all 128 threads via SMEM

---

## Two implementation strategies for v7

### Strategy A: TMEM → SMEM scratch → all threads read

```
1. warp 0: copy(TMEM → sDot scratch[BM × BN])  // sDot: 128×256×4 = 128KB SMEM
2. __syncthreads()
3. All threads: read sDot[tid][j] for epilogue

SMEM budget (SM100, max 228KB per block):
  sA:   BM × BK_MMA × 2 =  128 × 64 × 2 = 16384 bytes = 16KB
  sB:   BN × BK_MMA × 2 =  256 × 64 × 2 = 32768 bytes = 32KB
  sDot: BM × BN  × 4    =  128 × 256 × 4 = 131072 bytes = 128KB
  mma_barrier + tmem_base_ptr: ~32 bytes
  TOTAL: 176KB ✓ (under 228KB limit)

Requires: cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 176*1024)
```

### Strategy B: warp 0 holds all dots, distributes via SMEM in BN=32 chunks

```
1. warp 0: load TMEM → 1024 registers (tDrAcc)
2. For each chunk of 32 j-values:
   a. warp 0 writes tDrAcc[chunk] → SMEM scratch[32 × 32] = 4KB
   b. __syncthreads()
   c. All threads read SMEM scratch for their i-row
   d. compute dist2 + exp for these 32 j-values
   e. Repeat for next chunk
Total chunks: 256/32 = 8 → 8 sync rounds per j-tile

Requires: only 4KB extra SMEM scratch (much cheaper)
Complexity: moderate (chunk loop)
```

**Strategy B is better** — only 4KB extra SMEM, avoids the 128KB scratch buffer limit risk.

---

## Exact API for Strategy B

```cpp
// After k-tile loop completes (TMEM has full BM×BN dot products):

// Setup TMEM load — use 32dp32b32x for full tile in one call
// 32 threads × 1024 values = 32768 fp32 = BM × BN ✓
TiledCopy tiled_t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, tCtAcc);
ThrCopy   thr_t2r   = tiled_t2r.get_slice(tid);
Tensor tDtAcc = thr_t2r.partition_S(tCtAcc);
Tensor tDrAcc = make_tensor<TypeC>(shape(thr_t2r.partition_D(gC_dummy)));

// Only warp 0 loads (ThrID=32 constraint)
if (elect_one_warp) {
    copy(tiled_t2r, tDtAcc, tDrAcc);  // TMEM → 1024 registers in warp 0
}
__syncthreads();

// Process in chunks of JB=32 j-values
// warp 0 writes chunk to SMEM scratch, all threads read
__shared__ float sDot_chunk[BM][32];  // 128×32×4 = 16KB scratch

for (int jb = 0; jb < bn; jb += 32) {
    int jb_size = min(32, bn - jb);
    
    // warp 0: write its registers for this j-chunk → SMEM
    // Need to map tDrAcc indices to (m, jb..jb+32) coords
    // tDrAcc layout from 32dp32b32x: (32threads × 1024vals) → (BM, BN)
    // m index = (val_idx / BN) in the flat layout
    // n index = (val_idx % BN) in the flat layout
    if (elect_one_warp) {
        for (int jj = tid; jj < jb_size; jj += 32) {
            for (int m = 0; m < BM; ++m) {
                // flat index in tDrAcc for element (m, jb+jj)
                int flat = m * BN + jb + jj;
                // which thread owns flat index flat?
                // thread owns flat/32 ... need exact mapping from copy_traits
                sDot_chunk[m][jj] = tDrAcc(flat);  // placeholder — needs correct indexing
            }
        }
    }
    __syncthreads();
    
    // all threads: epilogue for this j-chunk
    if (tid < bm) {
        for (int jj = 0; jj < jb_size; ++jj) {
            float dot   = sDot_chunk[tid][jj];
            float dist2 = fmaxf(sqi_local + sq_X[j0 + jb + jj] - 2.f * dot, 0.f);
            D_acc += fast_exp(-beta * dist2) * w[j0 + jb + jj];
        }
    }
    __syncthreads();
}
```

---

## Exact tDrAcc indexing — needs verification on B200

The key unknown: how does `SM100_TMEM_LOAD_32dp32b32x` map register index → (m, n)?

From `32dp32b1x` traits:
```
ValID = Layout<(32, 32), (1, DP_b)>
```
where `TMEM::DP_b` = 65536 (column stride in TMEM address space).

For `32dp32b32x`, the ValID should be `(32, 32×32) = (32, 1024)`.

The mapping is: register `r` of thread `t` corresponds to TMEM element at:
```
tmem_addr = base + t * 1 + r * TMEM::DP_b
```

To map to (m, n) in BM×BN, need to invert the TMEM layout from `mma_traits_sm100.hpp`.
From the TMEM accumulator layout: `tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)`

So: `tmem_addr = base + m * 65536 + n * 1`
→ `m = (tmem_addr - base) / 65536`
→ `n = (tmem_addr - base) % 65536`

And thread `t` at register `r` has `tmem_addr - base = t + r * 65536`:
→ `m = r`  (register index = m coordinate!)
→ `n = t`  (thread index = n coordinate, but only for n < 32)

For `32dp32b32x`: 32 repetitions → each thread covers n = t, t+32, t+64, ..., t+32×31
So thread `t`, register `r` → (m=r, n=t + rep*32) for rep in 0..31.

**This needs verification but the pattern suggests:**
```cpp
// thread t, register r in tDrAcc → (m=r, n=t + rep*32)
// For rep = register_index / BM, r_local = register_index % BM
// m = r_local, n = t + (register_index / BM) * 32
```

---

## Files to modify for v7

```
src/flashdiffusion/csrc/flash_diffusion_sm100.cu
  ← current: WMMA (flash_diffusion_sm100_.cu restored)
  ← v6:      flash_diffusion_sm100_umma_v6.cu (saved)
  ← v7:      implement Strategy B with correct tDrAcc indexing
```

## Verification plan

1. Unit test: N=256, d=64 → BM=128, BN=256, 1 j-tile
   - Compare TMEM readout vs numpy reference dot products
   - Should match to fp16 precision (~1e-3)

2. Swiss roll N=200k, d=3
   - precompute should be ~0.1s (vs 0.228s WMMA)
   - DC1 corr should match 0.978

3. Alanine dipeptide N=750k, d=20
   - precompute should be ~1s (vs 6.8s WMMA at N=750k)

---

## Expected performance after v7

```
Operation         WMMA v1    UMMA v7 (projected)
precompute N=200k  0.228s     ~0.05s  (5× speedup)
matvec N=200k      0.227s     ~0.05s  (5× speedup)
Lanczos N=200k    20.0s       ~5s     (4× speedup, GPU Lanczos)
total N=200k      20.5s       ~5.1s   (4× speedup over WMMA+GPU Lanczos)

precompute N=1M    6.1s       ~1.2s   (5× speedup)
Lanczos N=1M     534s        ~134s   (4× speedup)
```

---

## References

- CUTLASS main `examples/cute/tutorial/blackwell/01_mma_sm100.cu` — MMA + TMEM load epilogue
- `include/cute/atom/copy_traits_sm100.hpp` line 1455 — SM100_TMEM_LOAD_32dp32b1x traits
- `include/cute/arch/copy_sm100.hpp` line 3378 — raw PTX struct
- `include/cute/atom/mma_traits_sm100.hpp` line 481 — TMEM accumulator layout
- tCtAcc comment in example: `tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)`
  → m stride = 65536 = TMEM::DP_b, n stride = 1
  → thread t, register r → (m=r, n=t) for 32dp32b1x
  → thread t, register r → (m = r%BM, n = t + (r/BM)*32) for 32dp32b32x

## Current file locations

```
/tmp/FlashDiffusion/src/flashdiffusion/csrc/
  flash_diffusion_sm100_.cu    ← WMMA v1 (working, paper result)
  flash_diffusion_sm100.cu     ← currently WMMA (restored for paper)
  flash_diffusion_sm100_umma_v6.cu  ← save v6 here for reference
```

# Track 3 Technical Report — Implement a GPU Kernel for a Transformer Layer
### TikTok TechJam 2026

Historical rendered version (figures inline, print-to-PDF from the browser;
the maintained runtime audit below is newer):
[Rendered historical report](https://claude.ai/code/artifact/80227d3c-9682-42d0-a957-bf5188704088)

## 1. Environment

| | |
|---|---|
| **Development machine** | Intel Core i5-14500, 16 GB RAM, **Intel UHD 770 iGPU only (no NVIDIA GPU)**, Windows 11, Python 3.13. Used for authoring, the repo, this report, and the demo video — no GPU compute. |
| **Benchmark GPUs** | Two free Kaggle cards, both driven headlessly via the Kaggle API from the local machine. **Tesla P100-PCIE** (16 GB, sm_60) with torch 2.5.1+cu121 — the preinstalled 2.10+cu128 dropped sm_60, so the kernel detects the mismatch, installs a compatible build and re-execs. **Tesla T4** (16 GB, sm_75) with the preinstalled torch 2.10.0+cu128, no reinstall needed. The API allocates a P100 unless `machine_shape` / `--accelerator NvidiaTeslaT4` asks otherwise — the accepted names appear only in the SDK docstring for `ApiSaveKernelRequest`, and an unrecognised value is silently normalised back to a P100. Raw logs: `results/kaggle_p100_run.log`, `results/kaggle_t4_run.log`, `results/kaggle_t4_fp16_run.log`, `results/kaggle_t4_ablation.log`. |
| **Why cloud** | Track 3 is a GPU-kernel task; `CUDA`/`Triton`/tensor cores require an NVIDIA GPU. Free cloud GPUs (Colab is an explicitly allowed dev tool) were used and are reported honestly here. |

## 2. The problem and the grading contract

The harness compares a `UserOptimizedTransformer` against the reference
`BaselineTransformer` across 14 shapes. An entry passes a trial only if **every**
output element satisfies `abs_err ≤ 0.002` **OR** `rel_err ≤ 0.02`, and any
`NaN/Inf` is a hard fail. Only if accuracy passes is the median latency timed
(`speedup = baseline.median / optimized.median`). Weights are copied with
`strict=True`, so parameter names must match exactly.

## 3. Key insight: the memory wall at S=100000

The baseline computes attention explicitly: `scores = Q·Kᵀ` of shape
`[B, H, S, S]`. For shape 14 that is `32 · 16 · 10⁵ · 10⁵ = 5.12×10¹²` elements
≈ **20.5 TB** in fp32 — impossible on any GPU. The baseline therefore cannot run
shape 14 at all; only a memory-efficient (FlashAttention-style) attention with
`O(S)` memory can. This is the crux of the task.

## 4. Optimizations

1. **SDPA, memory-efficient backend.** `F.scaled_dot_product_attention(is_causal=True,
   attn_mask=None)` on the no-padding hot path. `O(S)` memory, fused softmax,
   tensor-core matmuls. Unlocks shape 14 and wins big on long-sequence shape 13.
2. **Internal fp16 autocast under fp32 grading.** Tensor cores on the T4 (which
   has fp16 MMA but no bf16/TF32). Reductions kept in fp32; `rtol=0.02` leaves
   ~40× margin over fp16 rounding. Verified per shape.
3. **Self-applied `torch.compile`.** Inductor fuses LayerNorm/bias/GELU
   epilogues into Triton kernels; `reduce-overhead` (CUDA graphs) for
   launch-bound small shapes, `default` otherwise. Independent of `--compile-user`.
   Because the ablation showed compilation *losing* on the launch-bound shape,
   the model now times eager against compiled once, on the real input, during
   the harness' warmup, and keeps the winner — with the eager kernels captured
   into a CUDA graph as a third candidate (`T3_COMPILE=auto`, `T3_CUDAGRAPH=1`).
4. **Batch chunking into a preallocated output, shape 14 only.** The chunk size
   is planned from the VRAM that is actually free (`cuda.mem_get_info` plus the
   allocator's reserved-but-unused blocks), minus the output buffer, divided by
   the ~8 activations a block keeps live at once. Each chunk is written straight
   into the preallocated `[B,S,D]` output; the earlier list-plus-`torch.cat`
   version held the pieces *and* the joined tensor simultaneously, which doubled
   peak VRAM precisely where it was tightest and was the actual cause of the
   seq_len=1e5 OOM. If the estimate is still too optimistic the chunk size halves
   and the pass restarts, so a mis-planned budget degrades instead of failing.

<!-- x3-opt:begin -->
5. **fp16x3 linear layers (shipped).** Turing has no fp32 tensor cores and no TF32,
   and SGEMM was 44–89% of the forward on the GEMM-heavy shapes. Each GEMM operand is
   represented as an fp16 hi + lo pair (\(x = x_{hi} + x_{lo}\), \(x_{lo} = \mathrm{fp16}(x - x_{hi})\),
   about 2^-22 relative), and one bare cuBLAS GEMM with K tripled,
   \([a_{lo}\,|\,a_{hi}\,|\,a_{hi}] \cdot [w_{hi}\,;\,w_{lo}\,;\,w_{hi}]^T\), accumulates the three
   cross terms in fp32 with an fp32 output — the tiny terms first, which cuts the
   tensor core's truncation error 1.6–3.7x. The weight is pre-scaled by a power of two
   so its lo part is a normal fp16 number; the scale and the bias are undone and added
   by the neighbouring kernels (SDPA's scale, the residual add, the GELU input), because
   cuBLAS's epilogue for them measured 40–90% slower than a bare `mm`. The split is
   fused into tiled Triton kernels that produce each activation (LayerNorm,
   add+LayerNorm, GELU), so it costs no extra pass; all four GEMM sites take it, after a
   three-run comparison against the operator table's narrower policy (`T3_X3_SITES`).
   Median 2.300x -> 2.834x, shape 8 1.09x -> 1.57x; worst error 9.5e-06 at K=1024
   (210x inside the gate), 2.9e-06 elsewhere. `T3_LINEAR=fp32` keeps the SGEMM path.

<!-- x3-opt:end -->

Per-shape dispatch table and the correctness checklist are in the project plan
and `user_optimized.py`.

## 5. Results

All 13 gradeable shapes PASS on both cards. The shipped T4 path lands 2.86e-06 from the fp32
reference on twelve shapes and 9.54e-06 on the K=1024 one (699× and 210× inside the `atol=0.002`
gate); the fp32-SGEMM path and the P100 sit at ~1.9e-6, ~1049× inside.

| regime | median | range | worst `max_abs` | margin vs `atol` |
|---|---|---|---|---|
| P100, SDPA only | 2.065× | 1.098-4.001× | 1.91e-6 | 1049× |
| **T4, SDPA + fp16x3 tensor-core GEMMs + first-forward autotune (shipped)** | **2.834×** | 1.566-5.222× | 9.54e-06 (K=1024); 2.86e-06 elsewhere | 210× / 699× |
| T4, `T3_LINEAR=fp32` (SGEMM) | 2.300× | 1.095-4.818× | 3.22e-06 | 621× |
| T4, `T3_LINEAR=auto` (two runs) | 2.299× / 2.214× | 1.429-4.967× | as shipped | |
| T4, + fp16 (`T3_AUTOCAST=fp16`) | 4.014× | 1.320-11.528× | 2.04e-3 | **0.98×** |
| T4, fp16 attention + fp32 FFN/LN (`T3_FP32_FFN=1`) | 2.953× | 1.467-9.609× | 1.72e-03 | 1.17× |

| # | shape [B,D,H,S] | P100 | T4 fp32 | **T4** | | # | shape [B,D,H,S] | P100 | T4 fp32 | **T4** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64,128,4,128 | 1.75× | 2.28× | 2.70× | | 8 | 64,1024,4,128 | 1.10× | 1.09× | 1.57× |
| 2 | 1,128,4,128 | 2.14× | 3.83× | 4.40× | | 9 | 64,128,1,128 | 1.24× | 1.26× | 1.86× |
| 3 | 4,128,4,128 | 2.17× | 3.38× | 3.65× | | 10 | 64,128,2,128 | 1.52× | 1.58× | 2.29× |
| 4 | 16,128,4,128 | 2.29× | 2.48× | 2.99× | | 11 | 64,128,16,128 | 2.56× | 3.08× | 3.86× |
| 5 | 128,128,4,128 | 1.78× | 2.30× | 2.59× | | 12 | 64,128,4,32 | 2.33× | 2.09× | 2.83× |
| 6 | 10000,128,4,128 | 1.85× | 2.16× | 2.49× | | 13 | 64,128,4,1024 | **4.00×** | 4.82× | **5.22×** |
| 7 | 64,32,4,128 | 2.06× | 2.76× | 4.50× | | 14 | 32,1024,16,100000 | infeasible→**runs** | | |

**Measurement protocol.** Every T4 cell above is the **median of three independent
runs** — the shipped path (run medians 2.828x / 2.912x / 2.834x) and the fp32-SGEMM
reference (2.261x / 2.300x / 2.332x), every run 13/13 PASS — and `max_abs` is the worst
of the three. The per-shape spread, (max − min) / median, reaches 25% on shape
4; shape 6 drifts by up to 20% between sessions on identical code. Differences smaller
than that between two configurations are noise, and the text says so wherever it
applies. All six runs are in `results/results_t4_runs.csv`. One correction: earlier
versions of this table were measured with `torch.compile` silently inactive on shapes
6, 8 and 13 (our sweep runs every shape in one process and tripped Dynamo's recompile
limit; the harness, one shape per process, does not). Fixed in the model, re-measured.

Two things in that table are worth stating rather than glossing:

**The T4's baselines are slower than the P100's** (shape 13: 324.7 ms vs
168.6 ms; shape 6: 1516.3 ms vs 772.3 ms). The P100 has higher fp32 throughput
and roughly twice the memory bandwidth. Our ratios improve on the T4 anyway,
because the optimized path gains `torch.compile` there while the baseline stays
bandwidth-bound — the attention kernel is the same memory-efficient one on both
cards (see the probe below). A speedup is a ratio and
it matters which side moved.

**fp16 is nearly twice as fast and we do not ship it.** `T3_AUTOCAST=fp16`
passes all 13 shapes at a median 4.014×. But its worst absolute error,
`max_abs = 0.0020388` on shape 6, has *already crossed* `atol=0.002`; it survives
only because the gate is `abs<=0.002` **OR** `rel<=0.02` and that element's
reference happened to be large enough (`|ref| >= 0.102`) for the relative branch.
Move the same error onto a near-zero reference and the element fails -- and one
failing element fails the shape and forfeits the speed score entirely. We took
the 2.286× that sits 1049× inside tolerance and left fp16 as a documented,
measured flag. This is the one place where our earlier reasoning was wrong: the
repo previously asserted fp16 "breaks the gate", which measurement disproved --
the conclusion survived, the justification did not.

**Which attention kernel ran.** Earlier drafts said FlashAttention. Forcing each
SDPA backend alone, for every dtype and head_dim in the sweep, on both cards
(`results/kaggle_t4_probe.log`, `results/kaggle_p100_probe.log`):

| dtype | head_dim | flash | efficient | math |
|---|---|---|---|---|
| fp32 | 8 / 32 / 64 / 128 / 256 | **no** | yes | yes |
| fp16 | 8 / 32 / 64 / 128 / 256 | **no** | yes | yes |

PyTorch's flash backend is fp16/bf16-only and needs sm_80+; the graded path is fp32
on sm_60 / sm_75. No run here used FlashAttention. Every call used the
memory-efficient backend, which is the `O(S)`-memory fused kernel the shape-14
result depends on — the mechanism was right and the name was not.

**Shape 14 is the result we care most about.** The baseline needs ~20.5 TB for
its scores and cannot run, so there is no ratio to report; the meaningful claim
is that the shape goes from impossible to possible. Measured on the same free
16 GB P100:

```
trunc S=2048 correctness: PASS max_abs=1.19e-06 max_rel=0.127
vram free=16.64/17.06 GB | baseline scores would be 20.5 TB -> infeasible
full S=100000: median=293376.9 ms | 10,907 tok/s | peak_vram=14.61 GB | chunk_bs=1
```

293 s per forward across 3.2 M tokens, peak 14.61 GB of the 17.06 GB card. On a
**T4** — same memory-efficient backend, but fp16 tensor cores for the natively-fp16
matmuls — the same forward takes
**184 s at 17,402 tok/s** with a 14.17 GB peak — on a card with only 15.64 GB
total, tighter than the P100, and it still fits
(`results/kaggle_t4_shape14.log`).
The separate truncated fp32 check passes (`max_abs 1.2e-6`); this does not prove
full-length fp16 numerical equivalence. Getting here required the memory fix
in §4.4 — before it, the run died in
the final `torch.cat`, not in the attention.

**Precision.** The truncated correctness check runs in **fp32**, matching the
graded shapes. The full-length timing runs in **fp16** by necessity: an fp32
input for this shape is 13.1 GB and its output another 13.1 GB, so 26.2 GB is
committed before any activation — more than any free 16 GB GPU has. fp16 halves
that to 13.1 GB and leaves room for the per-chunk working set. So the 293 s /
10,907 tok/s / 14.61 GB numbers are fp16 numbers, and are labelled as such
wherever they appear; the 13 graded shapes in the table above are all fp32.

Figures: `figures/memory_wall.png` (the 20.5 TB wall), `figures/speedups.png`
(per-shape speedup).

<!-- x3-report:begin -->
### 5.x fp16x3: the kernel that ships

The profile that motivated it, the arithmetic, and the operator-level tables are in the
README and `results/ablation.md`; the decision rests on three runs per configuration:
median **2.300x -> 2.834x**, minimum 1.095x -> 1.566x, wins on shapes 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 and 13,
losses of at most a few percent on none. The cost is
accuracy: 9.5e-06 at K=1024 against 3.2e-06 for SGEMM — the Turing tensor core's
truncating accumulator, growing linearly with K, halved for free by summing the small
terms first — still 210x inside the gate. Four variants lost and are published: a
PyTorch-side split, a two-GEMM form, a cuBLAS epilogue for bias and scale (40-90% slower
than a bare `mm`), and a first-forward tuner for the fp32/fp16x3 choice (retired at the
noise floor). A fifth, int8 tensor cores for the two cross terms, was a wash on speed
and ten times worse in error and is declined with its table. Every activation the path
splits is bounded from the weights, and the first forward is checked for finiteness, so
the fp16 range is guarded without a sync in steady state.

<!-- x3-report:end -->

## 6. Limitations & what we'd improve with more time

- fp16 is measured, not assumed: 13/13 PASS at 4.014x median, with `max_abs`
  2.04e-3 against a 2e-3 gate. Shipped off; see the ablation for the reasoning.
  So is the mixed assignment (fp16 attention, fp32 FFN and LayerNorm): 2.953x at
  `max_abs` 1.72e-03, margin 1.17x. The error floor sits in the fp16 attention
  matmuls, so there is no cheap middle ground on this axis.
- Fused QKV projection, measured and declined: median 2.387x vs 2.280x is one
  shape moving inside noise; the mean is flat (2.490x vs 2.494x) and shape 6,
  where it should pay most, did not move — the efficient backend most likely
  re-copies the strided q/k/v views and gives back the saved reads.
- `--dtype bfloat16` fails the accuracy gate. This is a property of the
  configuration rather than of our kernel: the reference compared against itself
  recomputed in fp32 fails identically (7603 vs our 6131 elements, same
  `max_abs 0.047`), because bf16's ulp near 1.0 is 0.0078 against `atol=0.002`.
  The graded configuration is fp32.
- The maintained padded path (2026-10-09) packs valid tokens by length and
  scatters outputs, avoiding the previous dense `[B,1,S,S]` bias. It preserves
  order, supports internal holes and empty rows, and is equivalent here because
  the model has no position-dependent operators. Many distinct valid lengths
  require many smaller forwards, so there is still launch-overhead work to do.
- The fused add+LayerNorm Triton kernel is written, registered as a
  `torch.library` op so it composes with `torch.compile`, and measured
  (`kernels/fused_layernorm.py`). Inside Inductor's graph it is within ±5% of
  Inductor's own fusion on the memory-bound shapes (1.01–1.05× ours) and loses
  on the small ones (0.82–0.86×), where a custom op is an opaque boundary the
  compiler cannot fuse across. End to end: 1.929× as a raw launch with compile
  off, 2.190× registered with compile on, 2.282× with compilation fixed on and
  none of it (2.286× under the autotune default). We matched
  the compiler and did not beat it; it stays **off by default** with every number
  published.
- The fused bias+GELU epilogue was scoped and not built; Inductor already fuses
  it.
- The tensor-core attention kernel (`kernels/attention.py`) is written and measured.
  With operand splitting it reaches fp32-class error (1.4e-06—4.2e-06 against an
  fp64 reference; fp16 SDPA is 1.8e-03—2.8e-03) from fp16 tensor cores, and end to end
  passes 13/13 at `max_abs` 1.9e-06. It runs at 0.04—0.72× the speed of fp32 SDPA on a T4,
  median 1.093× against 2.286× shipped, because a T4's fp16:fp32 MMA ratio (~8×) cannot
  absorb three matmuls per product and Triton's Turing codegen trails cutlass. It
  ships off; on an Ampere-class card the arithmetic favours it. One finding from
  it applies everywhere: Inductor folds `x - x.half().float()` to zero inside fused
  kernels, so precision tricks must sit behind an opaque op boundary.

<!-- x3-limits:begin -->
- The fp16x3 GEMM error grows with K (truncating tensor-core accumulation):
  1.658e-06 at K=128, 9.562e-06 at K=1024, 2.749e-05 at K=4096, measured.
  Split-K with fp32 reduction outside the tensor core halves it at twice the GEMM cost
  (`T3_X3_SPLITK`, off).
<!-- x3-limits:end -->

## 7. Reproducibility

`python run_all.py --shapes 1-13` → `results/results.csv`;
`python scripts/shape14_optimized_only.py` for shape 14. Fresh Colab/Kaggle
session, `git clone`, run — numbers reproduce within free-tier variance.

## 8. AI tooling

See `docs/AI_TOOLS.md`.

## 9. Maintained runtime audit (2026-10-09)

The post-submission update caches stable ordinary-mask reductions, drops an
unused CUDA-graph mask copy, and fixes cached-weight invalidation on Parameter
replacement and LayerNorm updates. It also packs padding without a dense
S-by-S bias and extends eager-vs-compiled autotuning to large unchunked shapes;
manual CUDA graphs remain small-shape-only. The September results above are
historical and remain intact.

The paired T4 repeat (`results/runtime_repeat.json`, raw log alongside it) uses
the unmodified official `benchmark_once` loop: 9 rotated rounds of 50 samples
per model, identical weights and inputs, reference revision `6a52565`, current
source SHA256 recorded in the result. B=1 latency falls from 0.4383 to 0.3809 ms;
B=4 falls from 0.4112 to 0.3592 ms. Shapes 8 and 12 are 1.0% and 2.4% slower,
so this is a small-batch improvement, not a claim that every shape gets faster.
All four numerical checks pass; separate wall-clock measurements are retained.

The final-code sweep (`results/runtime_full.json`) passes all 13 gradeable
shapes on three random input trials each. It uses 3 rotated rounds of 30
official timing samples per model. Extending eager-vs-compiled selection
recovers a PyTorch 2.11 compiler regression: shape 6 goes from 1479.81 to
671.80 ms (-54.6%), shape 13 from 109.52 to 68.18 ms (-37.7%), both choosing
eager. Wall-clock loop latency confirms the gain (1465.55 → 673.68 ms and
105.44 → 65.50 ms). These are paired improvements over revision `6a52565` in
the current environment, not extra multipliers over the September headline.
Worst absolute error is 9.96e-6. Shapes 8 and 9 are slightly slower (0.6% and
0.4%), and the small-shape repeat above is preserved with its earlier source hash.
The median of the 13 paired improvements is 1.012×, not the 2.20× of shape 6.

On the 32,768-token irregular-padding case (1,024 valid tokens per row), the
current model uses 93.5 MB peak additional tensor allocation, compared with an
8.59 GB theoretical dense bias in the old implementation. Valid outputs pass
against an independent compact-sequence reference (`max_abs=1.43e-6`) and
invalid positions are zero. The medium padded comparison in the initial logs
measured memory after warmup, excluding persistent graph allocations; those
memory columns must not be used as a total-footprint comparison.

The corrected audit (`results/runtime_memory.json`) measures from before the
first forward, excluding only the comparison's own temporary tensors. On the
4×512×128 padded case it records 22.27 MB vs 9.43 MB peak extra allocation and
2.603 vs 2.266 ms latency. This includes setup caches and persistent CUDA-graph
allocations. Both implementations pass the official comparison rule.
The final full-process sweep records a higher previous-model setup peak
(316.85 MB) while the current packed path remains 9.43 MB. The dedicated and
full-sweep contexts are retained separately; the old peak is not a constant
independent of the surrounding run.

The final-code full-length capacity audit (`results/runtime_shape14.json`)
records one cold fp16 forward at 157.57 s and 14.37 GB peak allocation. All
outputs are finite. An independent fp32 reference on the same original weights
checks the first batch's first 512 causal outputs: zero failing elements under
the official OR gate, `max_abs=0.0021081` (some pass via relative tolerance).
No numerical equivalence is claimed for the remaining full-length outputs,
and this single cold timing is not comparable to the historical steady median.

Local regression tests: 51 passed, 1 CUDA-only test skipped. CPU smoke checks
also pass the 12 CPU-feasible official shapes and two padded cases. The cloud
audit separately validates graph ownership and mutation/invalidation contracts.

## 10. Follow-up kernel audit (2026-10-09)

The next experiment series refreshes profiling of the actual compensated-GEMM
path. Shape 8 is 80.2% GEMM time; shape 13 is 69.8% attention time. The previous
SGEMM profile is not used to infer these current shares.

An opt-in, zero-workspace cuBLASLt algorithm search preserves FP32 accumulation
and output. Its first public-forward integration reduces shape-8 event latency
from 71.74 to 67.07 ms, with both models using owned-output CUDA Graphs; wall
latency confirms 72.07 to 67.26 ms. All 13 FP32 shapes pass three input trials,
worst absolute error 1.17e-5. This is a paired per-shape result, not an extra
all-shape multiplier. Default dispatch remains PyTorch; building a native
extension and tuning its algorithms are explicit opt-in setup costs.

An independent final-code T4 session confirms 77.71 → 72.10 ms CUDA-event
latency and 78.63 → 72.43 ms synchronized wall latency, with three rotated
paired rounds. All 13 FP32 shapes pass three inputs again, worst error 1.17e-5.
Every recorded core-source hash matches the implementation.

Inductor's C++ wrapper was built after repairing Kaggle's missing development
linker name in a private directory, without changing system libraries. It
beats the compiler's Python wrapper but not the established manual graph.
Bulk native parameter metadata reads also lose once traversal, binding and
key reconstruction are included. Neither becomes a default dependency.

A separate guarded adapter tests a pinned third-party Turing attention backend
only for native-FP16 causal inference. It checks dtype, shape, device,
head-width and stream assumptions, makes packed QKV views fully contiguous,
and falls back on unsupported cases. The FP32 grading path is not downcast.
Full native-FP16 backend agreement, feasible original-FP32 causal-prefix checks,
and full-length timings are separate evidence categories, not interchangeable
accuracy claims.

The isolated integration candidate's full shape-14 paired experiment records
three steady SDPA forwards at 138.1881/138.6050/137.9473 seconds and three
Turing forwards at 77.8718/77.1328/77.1079 seconds. Median latency falls
138.19 → 77.13 seconds (1.7916× throughput, 44.2% less time), including layout
copies. Setup/build and first forwards are separate. Both record 14.604 GB
peak GPU allocation in the shared two-model process, not the older
single-model memory protocol.

All 3,276,800,000 outputs are finite; full agreement with native-FP16 SDPA
has zero OR-gate failures (max absolute error 0.0078125). The independent
original-FP32 first-batch 512-token prefix also passes (max error 0.0055175).
There is no full original-FP32 equivalence claim. Candidate evidence is in
`results/next/flash-candidate/`; its JSON is recovered from the final printed
cloud-log payload, with an explicit provenance receipt.

A separate session verifies the actual production adapter on the final core
source. All 3.277 billion outputs pass native-FP16 SDPA equivalence, the original
FP32 causal prefix passes, and 73 actual adapter calls are recorded. Nine probes
cover head widths 64/96/128 and three seeds each. This is a cold verification
pair only (SDPA 188.12 s, adapter 120.26 s), not another steady-state speedup;
it is not pooled with the three-round candidate timing. Final local checks are
72 passed, one CUDA-only skip, plus all CPU-feasible official shapes and two
padded smoke cases. All recorded final core hashes match the implementation.

Exact methods, raw logs, repaired/invalid trials and primary-source links are in
[`docs/NEXT_OPTIMIZATION_AUDIT.md`](../docs/NEXT_OPTIMIZATION_AUDIT.md). Earlier
reported results and the frozen submission are retained unchanged.

## 11. Clean repository release verification (2026-10-09)

A fresh T4 session clones an exact GitHub commit and normally imports the real
benchmark/model/kernel modules. The current FP32 baseline/default sweep passes
all 13 shapes on three inputs each, maximum absolute error 9.8348e-6. Three
rotated steady rounds give an unweighted median shape speedup of **3.546×**.
First public-forward costs and cold/steady memory peaks are separate; memory
workers isolate one model per process rather than retaining comparison models.
These results are not pooled with the September headline.

The normal-import wide-QKV Lt gate confirms **91.77 → 86.12 ms** event latency,
**91.97 → 86.42 ms** synchronized wall, zero original-reference failures and a
36.80-second first call. Its dependency/setup cost and narrow geometry keep it
opt-in. Seven FP32 attention layout/dispatch candidates all pass accuracy but
lose end-to-end: current 66.53 ms, alternatives 67.48–70.94 ms in the paired
eager experiment. No new layout becomes a default.

Exact protocols and failed-instrumentation repairs are documented in
[`docs/RELEASE_VERIFICATION.md`](../docs/RELEASE_VERIFICATION.md). JSON, exact
launchers, raw normalized logs and independent file hashes are retained in
`results/next/release-fp32/` and `results/next/release-attention/`.

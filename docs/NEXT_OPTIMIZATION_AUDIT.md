# October 9: follow-up optimization audit

This is a new experiment series against maintained revision
`46154d19ceeba0c50af2588a0247ce43b56c0465`. It does not replace the September
three-session headline. The reference benchmark is unchanged. Candidate kernels
are not enabled in the submitted model merely because an isolated test wins.

## What the current profiler actually says

The profiler uses PyTorch 2.11.0+cu128 on a Tesla T4. Percentages sum device
kernel events, not nested CPU operator times. Three warmed public forwards were
recorded; these shares are diagnostic, not unprofiled speed measurements.

| Official shape | Attention share | GEMM share | Interpretation |
|---|---:|---:|---|
| 6: large batch | 38.6% | 35.7% | Mixed attention, GEMM and split/normalization traffic |
| 8: wide model | 10.4% | 80.2% | GEMM remains the main target |
| 13: long sequence | 69.8% | 16.2% | Attention is now the main target |

Evidence: [profile JSON](../results/next/initial-profile/next_profile.json).
Unlike the old SGEMM profile, this records the shipped fp16x3 tensor-core GEMMs.
Small shapes also measure public forward, owned-output graph replay, eager inner
compute, and replay alone. Replay alone is **not** a valid model interface: it
omits input copying, mutation checks, and ownership of returned output.

## Candidates and acceptance rules

1. **Algorithm-level cuBLASLt search.** Query up to 64 heuristics with a 32 MiB
   workspace allowance for the actual fp16x3 operands, including padded leading
   dimensions and the tail-free strided view. Keep FP32 accumulation and FP32
   output. Check against the original GEMM and a small FP64 oracle before timing.
   A microbenchmark win needs an end-to-end win, then repeat confirmation.
2. **C++ dispatch wrapper.** Compare Inductor Python/C++ wrappers against the
   existing manual CUDA Graph with weight/mask checks and output ownership kept.
   Distinct entry code objects prevent Dynamo reusing the first candidate's
   wrapper. A private linker path repairs Kaggle's missing `libcuda.so` name;
   system libraries and the installed PyTorch version are not changed.
3. **Bulk native metadata reads.** Test whether reading parameter identity,
   mutation version, storage pointer, device and dtype in one native call is
   worthwhile. Parameter lists are re-read, and same-version replacement, mask
   mutation and owned outputs must still pass. Lower CPU checking time alone
   does not establish lower end-to-end latency.
4. **Turing-specific attention.** Fetch an exact upstream revision for a private
   experiment, build against the actual cloud environment, and test supported
   head widths and strided packed-QKV input. Only native FP16 is eligible; FP32
   grading is not silently cast to half. Unsupported widths, non-default streams
   and CUDA graph capture fall back. Full shape-14 checks distinguish native-FP16
   backend equivalence from the original FP32 oracle's feasible causal prefix.

The shared timing protocol is three rotated paired rounds, with both the
official CUDA-event loop and synchronized host-inclusive timing. Full shape 14
uses complete forwards, with first-call cost separated from timed rounds.

## Evidence integrity and known discarded trials

Each artifact records model/kernel/driver hashes and the cloud environment.
`cloud_script.py` beside the JSON is the exact generated script used, so earlier
driver variants remain recoverable after repairs. An artifact manifest also
hashes the imported files. Logs are normalized from Kaggle's JSON event stream
without changing their text.

- `initial-profile/next_cpp.json` is **not valid C++ wrapper evidence**: the
  initial variant did not isolate Dynamo code-object caches. Use the isolated,
  repaired follow-up instead.
- `cpp-link-failure/` records the genuine isolated build failure before the
  private driver-link repair; it is not a performance verdict.
- `gemm-initial/` retains an oversized validation allocation on shape 6. That
  comparison itself ran out of memory; it is not a model-capacity failure or a
  reason to rule out those algorithms. The repaired driver compares bounded
  tiles and re-runs the search.

## Integrated GEMM and dispatch findings

The first public-forward cuBLASLt integration runs with the normal runtime
selection, not an eager-only substitute. Both candidates choose manual CUDA
Graphs. Shape 8 changes from 71.74 to 67.07 ms in the official event loop and
72.07 to 67.26 ms synchronized wall latency (three rotated paired rounds).
All 13 FP32 shapes pass three inputs each, worst absolute error 1.17e-5.
[Integrated result](../results/next/integrated/next_lt_integrated.json),
[regression gate](../results/next/integrated/next_regression.json).

A second independent T4 session on the final integrated core confirms the
gain: **77.71 → 72.10 ms** event latency (-7.2%) and **78.63 → 72.43 ms** wall
latency (-7.9%), again with three rotated rounds and both models choosing
owned-output graphs. Its 13-shape, three-input regression gate also passes with
worst absolute error 1.17e-5. All recorded core-source hashes match the working
implementation, and the official benchmark remains unchanged.
[Final paired result](../results/next/final/next_lt_integrated.json),
[final regression gate](../results/next/final/next_regression.json).

The optional `T3_X3_BLAS=lt` backend restricts its production search to this
measured wide-QKV geometry and zero-workspace algorithms. Unsupported shapes,
devices and failed builds retain the established GEMM. Thread-local choices
match native handle ownership, and no activation/output tensors are cached.
It is not enabled by default: a local C++/CUDA build and first-use search are
real setup costs, excluded from steady-state timing.

The broader eager-only search did not establish a useful end-to-end gain for
shapes 6 or 13. In the repaired search, shape 13 selects a faster isolated small
GEMM but whole-model wall latency is flat (64.75 vs 64.83 ms). Therefore that
geometry is not added to production selection.

The isolated, linker-repaired C++ wrapper works and preserves accuracy/owned
outputs, but on B=1 it takes 1.184 ms versus 0.326 ms for the existing graph.
It beats the 1.977 ms Python compiler wrapper, which is not the default winner.
[C++ result](../results/next/followup/next_cpp.json).

Native metadata batching was re-tested after explicitly recapturing graphs
following mutation tests. On B=1 its host checks take 283 us versus 119 us,
and whole-model wall latency is 0.393 versus 0.243 ms. Traversal, binding and
Python key reconstruction erase the proposed benefit. It is rejected, without
removing any safety checks from the real model.
[Corrected graph comparison](../results/next/integrated/next_native_keys.json).

## Long-sequence native-FP16 attention findings

The pinned Turing extension was built against the unchanged cloud torch/CUDA
installation. On full shape 14 (`32 × 100000 × 1024`, 16 heads, two layers),
the isolated integration candidate and native-FP16 SDPA both use eager execution
and batch chunks of one. Three rotated paired full forwards give:

| Backend | Three steady rounds (seconds) | Median (seconds) |
|---|---|---:|
| Native-FP16 SDPA | 138.1881, 138.6050, 137.9473 | 138.1881 |
| Guarded Turing candidate | 77.8718, 77.1328, 77.1079 | 77.1328 |

This is **1.7916× throughput**, or **44.2% less full-forward time**, including
QKV layout copies and the returned output. First calls are separate (138.52
and 77.46 seconds), as is the roughly ten-minute external build. Both backends
record 14,603,990,016 bytes peak allocated GPU memory in this two-model process;
this is not the previous single-model capacity audit's memory protocol.

All **3,276,800,000 outputs** are finite and pass the official elementwise
absolute-OR-relative rule against the native-FP16 SDPA model: zero failed
elements, max absolute error 0.0078125. Some elements pass by relative tolerance.
An independent original-FP32 reference checks only the first batch's first 512
causal tokens: zero failures, max absolute error 0.0055175. **This does not
establish full-length equivalence to the original FP32 reference.** FP32 graded
inference never selects this backend or casts down to half.

The 100000-token, head-width-64 attention-only comparison is 2033.19 → 1087.12
ms in the official event loop, including guards and contiguous copies. Short
128-token requests lose, supporting the production adapter's minimum sequence
length of 8192. Packed strided QKV, head widths 64/96/128, unsupported widths,
FP32 input and non-default-stream rejection are exercised by the candidate.

[Candidate result](../results/next/flash-candidate/next_flash.json) and its
exact cloud script/log retain this separate candidate implementation. Artifact
enumeration was rate-limited by the large private build tree; the JSON was
recovered from the **last complete printed payload** in Kaggle's persisted log,
not represented as the original artifact's bytes. A
[recovery receipt](../results/next/flash-candidate/next_recovery_receipt.json)
and artifact hashes record that derivation. No upstream source or binary is
copied into this repository. A later artifact download succeeded: its
[original JSON](../results/next/flash-artifact-download/next_flash.json) is
semantically identical to the recovered payload, and the original build log
is retained alongside it. Both variants have independent artifact hashes.

### Final production-adapter integration gate

A second T4 session builds the dependency from the same pinned source and calls
the actual maintained adapter through `UserOptimizedTransformer`, not the
isolated subclass. It passes nine interleaved-QKV attention checks (head widths
64/96/128, three seeds each; max error 0.0004883), rejects non-causal use, and
then checks full shape 14. All 3,276,800,000 outputs are finite with zero
native-FP16 SDPA OR-gate failures (max error 0.0078125). The independent original
FP32 causal prefix passes again (512 tokens, one batch, max error 0.0055175).
An execution counter records **73 actual production-adapter calls**, including
the nine probes and 64 layer calls across 32 batch chunks. All nine recorded
core-source hashes match the maintained implementation.

This final gate is explicitly a **cold pair only**, not another steady-state
speed measurement: SDPA 188.12 seconds, adapter 120.26 seconds. Its different
session/setup costs are not pooled with the earlier 138.19 → 77.13 second
three-round candidate experiment. The paired candidate establishes the
performance experiment; this independent final-core run establishes integration
correctness. [Final adapter evidence](../results/next/adapter-final/next_flash.json).

An earlier attempt to attach a prebuilt private artifact found no verifiable
binary and failed closed; the successful gate rebuilt from the pin instead.
No unverified artifact was loaded. Default SDPA and FP32 grading are unchanged.

## Reproduction

```bash
python scripts/build_kaggle_selfcontained.py --only next \
  --env T3_NEXT_PHASE=profile_cpp --accelerator NvidiaTeslaT4 \
  --id YOUR_ACCOUNT/track3-profile --out .kaggle_upload/profile
python scripts/build_kaggle_selfcontained.py --only next \
  --env T3_NEXT_PHASE=integrated --env T3_X3_BLAS=lt \
  --accelerator NvidiaTeslaT4 --id YOUR_ACCOUNT/track3-integrated \
  --out .kaggle_upload/integrated
python scripts/build_kaggle_selfcontained.py --only next \
  --env T3_NEXT_PHASE=flash --env T3_FLASH_USE_ADAPTER=1 \
  --accelerator NvidiaTeslaT4 \
  --id YOUR_ACCOUNT/track3-flash --out .kaggle_upload/flash
kaggle kernels push -p .kaggle_upload/flash --accelerator NvidiaTeslaT4
```

The flash phase builds the exact pinned optional dependency without modifying
torch, checks attention contracts, then runs a full-sized warmup pair and three
paired rounds. It requires two full native-FP16 outputs' worth of comparison
storage (one streamed to host), so host memory matters as well as GPU memory.
For an earlier exact experiment variant, run its committed `cloud_script.py`;
the current driver contains the repaired validation and integration checks.

Evidence integrity and the final tested core can be checked locally without
CUDA or Kaggle credentials:

```bash
python scripts/verify_next_evidence.py \
  --current-core results/next/final/next_lt_integrated.json
python scripts/verify_next_evidence.py \
  --current-core results/next/adapter-final/next_flash.json
python -m pytest -q tests/
```

The evidence test verifies saved file hashes on CI as well as locally. Historical
driver variants intentionally differ; the optional current-core check covers
the model, kernels and unchanged official benchmark rather than those drivers.
Final local checks: **72 passed, one CUDA-only skip**, plus all 12 CPU-feasible
official shapes and two padded smoke cases. CI uses `python -m pytest` so the
repository root is on Python's import path on Linux as well as Windows.

## Sources checked

- NVIDIA's [cuBLAS 12.8 API and heuristics cache](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#heuristics-cache),
  matching the measured CUDA generation. Cached algorithm selection is supported;
  that is not a promise of improvement over PyTorch's current choice.
- PyTorch's [C++ wrapper tutorial](https://docs.pytorch.org/tutorials/unstable/inductor_cpp_wrapper_tutorial.html).
  Availability and benefit were tested on the actual installed version.
- The [Turing attention repository](https://github.com/ssiu/flash-attention-turing),
  pinned to `9ef98fcb506bb1e2fe3cece50935e2935bf6b124`, advertises FP16 head widths
  64/96/128 and benchmarks T4. Its reported speedups are attention-only, not ours.
  Source inspection finds contiguous B,S,H,D indexing and default-stream launches;
  both conditions need guarding. There is no top-level LICENSE at that revision,
  so upstream source is not vendored in this repository.

GELU remains the reference's `approximate="none"` erf implementation. The
cuBLASLt built-in GELU epilogue is not substituted silently: its documented tanh
approximation is a different operation.

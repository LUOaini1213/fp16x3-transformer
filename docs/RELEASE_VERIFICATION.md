# Clean GitHub / T4 release verification

This audit closes the gap between a self-contained experiment script and the
maintained repository. Each fresh private Kaggle session clones the public
GitHub repository into `/tmp`, checks out an exact commit, asserts a clean Git
state, and runs `python -m scripts.benchmark_release` with normal imports.
The model, benchmark and kernel modules are not inlined. Source clones and
native builds stay outside the downloadable working directory.

## What is and is not measured

- **FP32 shapes 1–13:** same original weights copied strictly, three inputs
  per shape, original FP32 baseline, default compensated-GEMM model. Shape 8
  additionally tests the opt-in cuBLASLt provider. Outputs remain FP32.
- **Steady latency:** three rotated paired rounds using the untouched official
  CUDA-event loop, plus separately synchronized host-wall loops. Model planning,
  compilation and algorithm search happen before these rounds.
- **First-call cost:** first public forward, including initialization, planning,
  compilation and tuning. The fresh job shares disk caches across child
  processes, so these are not thirteen independent fresh-machine startups.
- **Memory:** a separate child process per shape and variant, with only that
  model on CUDA. Peak allocated memory includes its input/output and graph pools,
  but excludes reference/comparison tensors. Cold and steady peaks are retained
  separately. Reserved memory and other CUDA process overhead are not this metric.
- **Shape 14:** native-FP16 formal production adapter, one cold pair followed by
  three rotated full-forward pairs. Both sides use `T3_COMPILE=0`,
  `T3_CUDAGRAPH=0`, `T3_CHUNK_BS=1` to isolate the attention backend. These are
  synchronized whole-forward wall seconds, not a separate CUDA-event series
  or a comparison against an independently autotuned SDPA model.
  Full-output equivalence is against
  native-FP16 SDPA. The original FP32 oracle checks only the first batch's
  512-token causal prefix; it is not a full original-FP32 equivalence claim.
- **Shape 13 attention experiment:** FP32 throughout. Test contiguous BHSD,
  contiguous BSHD views, folded batch/head, and batch chunks of 8/16/32 against
  existing strided packed-QKV SDPA. All copies, launches and concat costs are
  timed. Both microbenchmarks and original-reference end-to-end checks are
  required; eager-only measurements do not by themselves change automatic dispatch.

Every correctness gate requires finite outputs and the official elementwise
rule `abs_error <= 0.002 OR abs_error <= 0.02 * abs(reference)`. This is not the
additive tolerance used by `torch.isclose`. Historical September measurements
and the frozen submission are left unchanged and are not pooled into this table.
The driver does not lock GPU clocks. Per-round arrays are retained so runtime
variance is visible; three paired rounds within one session are not three
independent machines or a fleet-wide confidence interval.

## Reproduction

```bash
python scripts/build_clean_kaggle.py --phase fp32 --ref 62bb64c05ce55c27cbe10d034087939c9f1d7fed \
  --id YOUR_ACCOUNT/track3-release-fp32 --out .kaggle_upload/release_fp32
python scripts/build_clean_kaggle.py --phase flash --ref f30c8aa55f5d3dd1eb74a2f0e61727bcda6e3190 \
  --id YOUR_ACCOUNT/track3-release-flash --out .kaggle_upload/release_flash
python scripts/build_clean_kaggle.py --phase attention --ref 23d070f64df8c293e27de87a83dea4b829c51e39 \
  --id YOUR_ACCOUNT/track3-release-attention --out .kaggle_upload/release_attention
kaggle kernels push -p .kaggle_upload/release_fp32
kaggle kernels push -p .kaggle_upload/release_flash
# Start attention after a GPU slot becomes available (Kaggle concurrency limit).
kaggle kernels push -p .kaggle_upload/release_attention
```

The launcher requests `NvidiaTeslaT4` explicitly. The driver rejects a different
GPU architecture. It does not install or upgrade Torch. The optional Turing
installer builds upstream revision
`9ef98fcb506bb1e2fe3cece50935e2935bf6b124` with `--no-deps` and
`--no-build-isolation`. Build time and build logs are separate artifacts.
Upstream has no top-level LICENSE at that pin; no upstream source or binary is
vendored into this public repository.

## Completed FP32 and attention gates

The FP32 session measures commit `62bb64c05ce55c27cbe10d034087939c9f1d7fed`:
all 13 shapes pass three original-reference comparisons, maximum absolute error
9.8347664e-6, unweighted median shape speedup **3.546×**. Each variant's memory
worker is a separate process. The opt-in Lt shape-8 event pair is
91.7719 → 86.1248 ms (6.15% reduction); synchronized wall latency is
91.9687 → 86.4221 ms. Its first public call costs 36.80 seconds.
[Source-stamped aggregate](../results/next/release-fp32/next_release_fp32.json).

The attention session measures commit `23d070f64df8c293e27de87a83dea4b829c51e39`.
All seven layouts pass three FP32 end-to-end comparisons. Existing packed-QKV
SDPA records 66.5319 ms event / 66.8282 ms wall. Every alternative loses:
67.4755–70.9448 ms event, 67.8371–71.2319 ms wall. Consequently no layout is
accepted and no normal-dispatch promotion test is needed. These eager paired
numbers are not substituted for the default-dispatch FP32 table.
[Full candidate evidence](../results/next/release-attention/next_release_attention.json).

Two harness mistakes are retained transparently as failed runs, not benchmark
evidence: a same-named package function shadowed the Turing submodule's counter,
and constructing candidate parameters inside an outer inference-mode context
removed their version counters. Explicit module lookup and normal-context model
construction repair the instrumentation; regression tests cover both. The model
and all nine core-source hashes remain unchanged. Failure logs are in
`results/next/release-flash-import-failure/` and
`results/next/release-attention-construction-failure/`.

## Completed formal-adapter steady gate

The production adapter is measured through normal imports at
`f30c8aa55f5d3dd1eb74a2f0e61727bcda6e3190`, with the same nine production-core
hashes as the FP32 and attention sessions. Dependency build takes **618.26 s**.
Cold calls are recorded separately: SDPA **156.18 s**, adapter **95.07 s**.
Three rotated steady pairs give:

| Round | SDPA seconds | Production adapter seconds |
|---:|---:|---:|
| 1 | 171.4410 | 92.3080 |
| 2 | 172.2202 | 94.3196 |
| 3 | 170.8376 | 95.0371 |
| Median | **171.4410** | **94.3196** |

The actual maintained adapter delivers **1.818×** in this paired session.
All **3,276,800,000** outputs pass full native-FP16 SDPA equivalence, with zero
OR-gate failures and maximum difference 0.0078125. Every full output is finite.
The independent original-FP32 first-batch 512-token causal prefix passes
(maximum difference 0.0055175). There are **256 confirmed adapter calls**,
exactly 32 chunks × two layers × four complete forwards. Shared two-model
steady peak allocation is **13.601 GiB** for both backends; this is not the
isolated FP32-memory protocol or an original-full-FP32 equivalence claim.
[Formal JSON and provenance](../results/next/release-flash/next_release_flash.json).

The complete [generated release table](../results/next/clean_release_summary.md)
is derived directly from the three successful JSON artifacts. It includes every
FP32 shape, setup/memory costs, full adapter rounds and all seven attention
candidates. Earlier 1.79× isolated-candidate measurements and the cold-only
production gate remain historical evidence and are not pooled into this result.
The measured driver's generic metadata protocol string describes the FP32
event/wall helper; the flash `round_seconds` fields use synchronized wall timing
only, as specified above and in the pinned source. The maintained metadata label
now names these phase-specific methods explicitly; immutable evidence is not edited.

Final local verification: **90 passed, one CUDA-only skip**; **92 artifact
hashes** and all **nine current core-source hashes** pass. The unchanged
official benchmark is checked against its original Git revision.

```bash
python scripts/summarize_release.py \
  --fp32 results/next/release-fp32/next_release_fp32.json \
  --flash results/next/release-flash/next_release_flash.json \
  --attention results/next/release-attention/next_release_attention.json \
  --output results/next/clean_release_summary.md
python scripts/verify_next_evidence.py \
  --current-core results/next/release-flash/next_release_flash.json
python -m pytest -q tests/
```

`scripts/import_next_results.py` retains exact launcher snapshots, result JSON,
normalized logs and SHA-256 manifests. `scripts/summarize_release.py` produces
the table only after all thirteen shapes and all three long-sequence rounds
exist, with matching core-source hashes. Incomplete evidence is rejected.

## Implementation and dependency decisions

The maintained default remains `T3_LINEAR=fp16x3`, `T3_X3_BLAS=torch`,
`T3_ATTN=sdpa`, with existing automatic compile/manual-graph selection.
The development-toolchain-dependent Lt provider and third-party native-FP16
Turing adapter remain explicit opt-ins. They are workload-specific improvements,
not universal replacements for the graded FP32 path.

Attention layout changes require at least 5% lower end-to-end event **and** wall
latency over three paired rounds, then confirmation under normal automatic
dispatch and the full correctness suite before becoming a default. An isolated
attention win or sub-threshold difference is not an acceptance result.

The relevant backend constraints are documented by PyTorch's
[SDPA API](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)
and [CUDA graph requirements](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).
SDPA is an interface that chooses a backend; calling it does not establish that
the T4 FP32 path uses FlashAttention. Memory-efficient attention avoids storing
the full score matrix but does not change the quadratic attention arithmetic.
The version-matched [CUDA implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/attention.cu)
passes explicit batch/token/head strides to its memory-efficient kernel; making
packed QKV contiguous is therefore a measured candidate, not a required fix.

## Independent repeats and explicit startup profiles

Two further private T4 jobs, `wenjiluo/track3-release-repeat-a` and
`wenjiluo/track3-release-repeat-b`, clone exact commit
`38e191d9ba1778b7f837b790f79abe941ee98c66`. They use the same normal-import
FP32 protocol and the same nine production-core hashes as the first audit.
All 13 shapes and all isolated memory workers finish in each session. The
original session and repeats have distinct fresh checkout paths/caches.

The respective session median shape speedups are **3.546× / 2.801× / 2.875×**.
The combined summary takes each shape's median paired ratio over those three
sessions, then the unweighted median across shapes: **2.875×**. All **117**
original-FP32 input checks pass (maximum error 9.8347664e-6). Three-session
ranges are retained. The original 3.546× remains a valid single-session result,
not a stability promise; no code regression or new optimization is inferred
from the differing aggregate alone. These are not pooled with September's
different Torch/protocol.

`python -m scripts.run` adds `quick` (eager FP32 + SDPA) and `steady` (unchanged
shipped fp16x3 dispatch) profiles, an environment probe, strict weight copy and
three original-input checks. CPU timings are labeled non-GPU evidence and an
explicit CUDA request fails when no CUDA GPU is available. No package installs
or source modifications occur. See [Quickstart](QUICKSTART.md).

The source-pinned usability job at
`b335fb64a0fc76ad4105abf58e23589ef8af80ae` compares profiles with a fresh child
per mode/shape, first public-forward cost and ten synchronized wall samples.
Import/model/input setup and environment CUDA probes are excluded from first
forward; compiler caches are shared only within that cloud job. Approximate
amortization crossovers are computed from these costs, not claimed as measured
deployment SLAs. Its generated results remain a separate protocol from the
three-session paired throughput audit.

All **78** profile/input checks pass (maximum error 9.8347664e-6). Quick first
public forwards span **0.054–0.826 s**, versus steady **6.75–30.23 s** in that
job. For shape 8, quick/steady first cost is **0.197 / 11.85 s**, steady wall
is **122.14 / 82.35 ms**; estimated setup crossover is about **294 calls**.
This is a startup/throughput trade, not faster steady compute. The exact samples
and all thirteen crossover estimates are in the generated report.

The same pinned revision's shape-8 workspace pilot keeps production files
unchanged. It derives a separate CPP module with an explicit 0/1/4/16/32 MiB
budget in each plan key and scratch per invocation/current stream. All 21
checks pass, with best event 80.77 ms versus existing Lt 82.90 ms (2.57%). It
allocated the whole budget even when the selected algorithm required zero
scratch; this unnecessarily penalizes larger budgets. That pilot and its
32.59-second build remain under `results/next/usability-workspace/`, not pooled
with the corrected confirmation or used as the final adoption decision.

A fresh session at `c00bc91d711f0c7502883d5693e39f5d0e9cdc17` allocates only
the chosen algorithm's actual `workspaceSize` per invocation, not the budget
ceiling. Each call owns scratch on its current stream, preventing a persistent
shared-scratch race; searches/builds occur before capture. Allocation/copy and
output ownership costs remain inside public-forward timing. All **21** checks
pass; all five budget winners select zero-workspace algorithms. Build takes
**28.17 s**. Existing opt-in Lt is **91.24 ms** event / **92.47 ms** wall;
the best research candidate is **93.22 / 94.24 ms**, and every experimental
budget loses. None clears 5% on both metrics, so default Torch and existing
narrow Lt remain unchanged. Host-extension counters include capture, not graph
replays; incremental shared-process memory is not isolated deployment memory.
Corrected evidence is in `results/next/usability-workspace-actual/`.

[Generated evidence and exact source links](../results/next/usability_summary.md)
retain all independent sessions and all losing workspace variants. Primary
workspace API reference: [cuBLASLt workspace preferences](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html).

```bash
python -m scripts.summarize_usability \
  --sessions results/next/release-fp32/next_release_fp32.json \
    results/next/release-repeat-a/next_release_fp32.json \
    results/next/release-repeat-b/next_release_fp32.json \
  --quick results/next/usability-quick/next_usability_quick.json \
  --workspace results/next/usability-workspace-actual/next_usability_workspace.json \
  --output results/next/usability_summary.md
python scripts/verify_next_evidence.py \
  --current-core results/next/usability-quick/next_usability_quick.json
python -m pytest -q tests/
```

Final local verification for this pass: **115 passed, one CUDA-only skip**,
**187 artifact hashes** and **nine unchanged current core hashes**. The official
benchmark still has no diff against its original revision. Evidence JSON/logs
and launcher bytes are preserved across Windows/Linux checkouts; generated
report tables use portable LF endings.

## Balanced, dispatch recovery/cache and FFN fusion (completed)

The next implementation is tested at exact commit
`c8fe1cf7807d5633a1b75773d2ab03461318cb22`, through clean clones and normal
imports on T4 / Torch 2.11.0+cu128. The new model and two research/cache kernels
have **11** current core-source hashes. Earlier results refer to their original
revision; the pre-change runner/model are preserved as source snapshots.
Neither the frozen submission nor the official reference is modified.

Private jobs `wenjiluo/track3-balanced-pilot-1009` and
`wenjiluo/track3-balanced-full-1009` compare quick/balanced/steady. The full
13-shape sweep has **234 passing profile/input checks**: three seeds per
profile in separate cold workers and in a paired process. Balanced skips
Inductor but retains compensated fp16x3 and optional manual graphs. Separate
empty per-mode compiler directories give first public-forward costs of
**3.707–6.404 s** versus steady **10.609–34.542 s**. These exclude process
startup, imports, model/input creation and CUDA context probes, not Triton JIT
or graph setup. Rotated paired event/wall rounds include public-forward costs.

The predeclared lower-first-forward and <=2%-regression-on-both-metrics gate
clears **12/13** shapes. Shape 2 has -0.41% event change but **+6.95% wall**.
A separate T4 job, `wenjiluo/track3-balanced-shape2-repeat-1009`, runs three
fresh interpreter workers sharing compiler caches, each with three paired
rounds of 200 calls and three correctness seeds. All **27** additional input
checks pass; **2/3** workers meet the steady gate, while one has **+3.63% event
time**. This is one GPU session, not three independent GPU repeats or another
cold-start audit. Original misses remain published, and no default is changed.
The full sweep's single-session 3.402x balanced median is not pooled with the
historical three-session 2.875x cohort.

Both pilot and full jobs verify real graph capture, changed inputs, weight
mutation followed by one re-selection, stable weights without a tuning loop,
mutable normal/inference masks, owned outputs and sequential alternate-stream
use. Concurrent calls on one mutable graph instance are not supported by this
test. Cache producer, cache hit and uncached control are fresh workers; hit and
control share warmed compiler directories. Observed first-forward savings are
23.5 / 71.4 ms, not a general SLA or steady-throughput win. Only environment/
source/layout-keyed JSON eager/graph choices persist; a hit recaptures the graph
and must match fresh eager output bitwise. Corrupt/foreign files cause a miss
without being overwritten.

`wenjiluo/track3-ffn-fusion-1009` profiles the current shape-6 implementation
and tests four compensated FFN GEMM + exact-erf GELU + split epilogues. All
**15** input/variant checks pass, but all candidates lose: baseline **647.95 ms**
event versus candidates **1117.71–4631.51 ms**. None clears 3% reduction in
both event and wall, so the experimental module is never imported by production.
Fresh profiling attributes device-event sums to attention 39.03%, GEMM 35.82%,
LayerNorm/residual split 16.14%, activation split 9.00%; these are not fractions
of end-to-end latency or promises of optimization headroom.

Exact launchers, logs, immutable JSON and artifact manifests are retained under
`results/next/balanced-pilot/`, `balanced-full/`, `balanced-shape2-repeat/` and
`ffn-fusion/`. The targeted repeat's launcher explicitly changes only the
repetition count; it imports the same pinned driver normally. Reproduce full
or fusion jobs with `scripts/build_clean_kaggle.py --ref c8fe1cf7807d5633a1b75773d2ab03461318cb22
--phase full` (or `fusion`), plus the account-specific `--id` and `--out`.
The [generated combined report](../results/next/improvements_summary.md)
preserves the failed gates and is checked against the source JSON.

```bash
python -m scripts.summarize_improvements \
  --profiles results/next/balanced-full/next_improvements_full.json \
  --cache results/next/balanced-full/next_improvements_cache.json \
  --contracts results/next/balanced-full/next_improvements_contracts.json \
  --fusion results/next/ffn-fusion/next_improvements_fusion.json \
  --shape2-repeat results/next/balanced-shape2-repeat/next_improvements_shape2_repeat.json \
  --output results/next/improvements_summary.md
python scripts/verify_next_evidence.py \
  --current-core results/next/balanced-full/next_improvements_full.json
python -m pytest -q tests/
```

Final local verification of this improvement pass: **153 passed, two CUDA-only
skips**, **219 artifact hashes** and **11 matching current core-source hashes**.
CPU tests check the complete generated reports, rejected performance gates,
cache-file safeguards and source snapshots; the cloud artifacts provide the
actual CUDA evidence. These counts do not replace earlier cohort counts.

## Separate portfolio-full session

`wenjiluo/track3-portfolio-full-1009` independently measures the same
`c8fe1cf7807d5633a1b75773d2ab03461318cb22` sources on a fresh T4 checkout.
All 234 profile/input checks and 13/13 startup/paired steady gates pass in that
session. Balanced/steady first public forwards are 3.998–8.016 / 11.220–37.740 s,
with the same setup exclusions. Its cache controls and real CUDA recovery
contracts are preserved in a [separate generated report](../results/next/portfolio_runtime_summary.md).
This successful session does not replace balanced-full's 12/13 result or the
shape-2 repeat's 2/3 worker comparisons. The immutable raw collections and
historical speedups remain separate; no default is promoted.

`python -m scripts.verify_next_evidence --maintained-runtime` performs an
offline, read-only audit of both full sessions, the retained pilots, fusion and
the targeted repeat. It requires their complete receipts/files/raw logs, source
hashes and all generated reports, and reports performance gates per session.
Missing evidence and optimized Python are refused with exit code 2.

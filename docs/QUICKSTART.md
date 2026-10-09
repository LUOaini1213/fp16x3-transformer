# Quick startup and repeatable inference

Run commands from the repository root. No NVIDIA GPU is needed for authoring
or the small CPU correctness check. GPU measurements need a compatible CUDA
runtime; the maintained evidence uses Linux, a Tesla T4, and Torch 2.11.0+cu128.

## Audit the committed evidence first

This offline command uses only Python's standard library. It works before
installing Torch, requires no GPU or model downloads, and does not modify the
measurements or regenerate their receipts/report:

```bash
python -m scripts.verify_next_evidence --maintained-runtime
```

Success reports 216 artifact hashes, 11 current core-source hashes, 234 recorded
profile/input checks and 13/13 balanced gates. It requires the full, pilot and
fusion receipts, every required JSON/launcher file and the raw execution logs;
all measured JSON/log files in those folders must be listed in their receipts.
It also verifies the current sources and the generated runtime report.

A partial checkout cannot pass by silently skipping its missing receipt. Missing
files/entries, altered bytes, different current core sources and stale reports
produce an actionable error and exit code 2. Restore the original committed
files or use the measured revision; recalculating a receipt is not a substitute
for the missing GPU run. This audits published evidence rather than executing
new inference or establishing performance on your machine.

Run strict mode without Python's `-O`/`-OO` options and with `PYTHONOPTIMIZE`
unset: its source/report validators use assertions, so optimized Python is
explicitly refused with exit code 2 instead of silently skipping those checks.

The original optional check remains available for historical subsets:
`python -m scripts.verify_next_evidence --current-core PATH_TO_RESULT_JSON`.
Use `--maintained-runtime` to require the complete maintained-profile evidence.

## Choose a runtime profile

```bash
python -m scripts.run --check-only
python -m scripts.run --mode quick --shape 2
python -m scripts.run --mode balanced --device cuda --shape 2
python -m scripts.run --mode steady --device cuda --shape 2
```

The command automatically chooses CUDA when available; otherwise it explicitly
labels CPU results as correctness-only. `--device cuda` refuses silent CPU
fallback. It probes whether the selected wheel runs CUDA and supports FP16-input,
FP32-output GEMM, reports Triton/toolchain availability, and never performs an
installation, downgrade, package upgrade, or login.

| Profile | Compute | Startup work | Intended use |
|---|---|---|---|
| `quick` | FP32 GEMMs + memory-efficient SDPA | No Inductor, manual graphs, Triton launches, or native builds | Trying the model, short-lived inference |
| `balanced` | Same compensated fp16x3 + SDPA as steady | No Inductor; Triton JIT and eager/manual-graph selection remain | Explicit opt-in for repeated inference; consult the per-shape T4 acceptance table |
| `steady` | Shipped compensated fp16x3 + SDPA | First-forward compile/eager/manual-graph selection | Many repeated forwards |

Quick mode is not FP16 autocast or a lower-accuracy shortcut. All profiles keep
FP32 input/output, exact-erf GELU, strict reference weight names and the original
per-element correctness rule: absolute error ≤ 0.002 **OR** relative error ≤ 0.02;
non-finite outputs fail. Every run checks three deterministic input seeds before
reporting ten synchronized steady-wall samples (override `--repeats` if desired).

`submission.py` and `UserOptimizedTransformer` retain their existing defaults.
The new profiles are explicit command-line choices, not a silent change to the
official benchmark or existing integrations. The CLI clears inherited `T3_*`
experiment settings to make its named profiles deterministic; use the original
entry points for custom ablation switches. Profile selection belongs at process
startup, not midway through an already imported, live model's execution.

### Balanced mode: avoid rejected Inductor candidates

`balanced` sets `T3_LINEAR=fp16x3`, `T3_COMPILE=0`, `T3_CUDAGRAPH=1`.
It keeps steady's compensated arithmetic, FP32 attention, and static/first-call
precision guards. Manual graphs remain an optional candidate on supported,
all-valid small/medium shapes; they are not forced. Large or padded shapes and
unsupported devices retain the existing eager/fallback policies. Triton kernels
can still JIT-compile, and graph warmup/capture/selection still take time: this
is not a zero-setup or no-compilation-at-all mode.

The three historical T4 sessions selected no compiled path in their 39 final
configurations. This motivates a separate profile, not a blanket claim that
Inductor is slower on every environment. The CLI still defaults to `quick` and
direct model imports retain the shipped automatic policy.

Balanced, dispatch-cache hits and weight/mask recovery now have source-pinned
T4 evidence: a three-shape pilot followed by a full 13-shape sweep. The
[runtime validation report](../results/next/improvements_summary.md) lists
the first-public-forward cost, three-round paired event/wall results and each
shape's acceptance gate. The profile remains an explicit opt-in; model and CLI
defaults are unchanged. The old quick/steady numbers and 2.875x aggregate remain
separate and must not be presented as balanced measurements. The earlier runner is preserved in
`results/next/usability-quick/run.py.snapshot` and checked against its recorded
source hash; it is historical evidence, not an executable entry point.

### Optional safe dispatch hints

```bash
python -m scripts.run --mode balanced --device cuda --shape 2 --tune-cache results/local_dispatch.json
```

The file contains only JSON eager/graph choices, keyed by model/kernel source,
GPU identity/capability, Torch/CUDA/Triton versions, dtype/strides/shape, model
configuration and math/precision settings. Compiled functions, graph pointers,
inputs and outputs are never serialized. A graph hit creates a new capture and
must match fresh eager output bitwise; otherwise normal tuning/fallback applies.
Current weight bounds, mask checks and output ownership remain active.

The cache is opt-in, primarily for graph-eligible small/medium shapes. CPU runs
do not write it; large shapes have no manual-graph candidate. Corrupt, foreign,
unreadable or oversized files cause a safe miss and are not overwritten. It is
a performance hint, not a guarantee of the best choice on a busy GPU. JSON
reports the actual hit/miss/storage status. Use separate cache files for
unrelated deployments; concurrent writers may lose a hint, never a tensor.

Weight/norm changes now invalidate dispatch and permit one new eager/graph
selection. Rejecting a slower graph in the normal tuner does not reset that
selection, so stable weights cannot trigger a tuning loop.

The older model source is preserved in
`results/next/usability-quick/user_optimized.py.snapshot`. The new cache/recovery
checks and cloud source hashes are in `results/next/improvements-full/`.
The historical 2.875x result is not a test of these changes. Cache-hit and
warm-uncached controls are retained together; their single-worker times do
not establish a statistically reliable cache speedup.

## Sweep and inspect results

```bash
python -m scripts.run --mode steady --device cuda --shapes 1-13 --output results/local_run.json
python -m scripts.run --mode quick --device cuda --shapes 2,8,13 --output results/local_run_quick.json
python -m scripts.run --mode balanced --device cuda --shapes 1-13 --output results/local_run_balanced.json
```

Each shape runs in a fresh child process. JSON reports the effective profile,
environment, all three correctness checks, selected dispatch, first public
forward, and steady-wall samples. First-forward timing excludes Python process
startup/imports, model/input creation and the environment's CUDA context probe;
it must not be called total application cold-start latency. Compiler caches can
already be warm when using the same runtime for several shapes.

Before promoting balanced, require all 13 shapes and all three correctness
seeds to pass, measure lower first-public-forward cost, and confirm no more than
2% steady event **and** wall regression against steady in rotated paired rounds
on the same T4/software stack. The CLI sweep alone is unpaired correctness and
timing evidence, not that adoption gate or a new speedup headline.

Shape 6 is intentionally refused on CPU to avoid the large reference allocation.
Shape 14 is not exposed here: the unchanged full FP32 reference needs about
20.5 TB of scores, so the independent extreme-shape runner has a different,
explicitly restricted correctness protocol.

## Dependencies and reproducibility

The fallback/API dependency floor is `torch>=2.1`; this is not a claim that every
version, OS, or GPU reaches the measured speed. Older Torch, missing Triton or
unsupported GPUs may use the correct FP32 path without compensated tensor-core
speed. All profile results record which dispatch actually ran. Current CPU CI checks
correctness; T4 performance evidence is recorded separately.

Use an existing compatible Colab/Kaggle T4 runtime without reinstalling Torch.
To repeat the source-pinned cloud protocol from an authenticated local machine:

```bash
python scripts/build_clean_kaggle.py --ref EXACT_GIT_COMMIT --phase fp32 --id ACCOUNT/UNIQUE_KERNEL --out .kaggle_upload/repeat
kaggle kernels push -p .kaggle_upload/repeat --accelerator NvidiaTeslaT4
kaggle kernels status ACCOUNT/UNIQUE_KERNEL
kaggle kernels output ACCOUNT/UNIQUE_KERNEL -p .kaggle_out_repeat
```

The builder resolves a ref to an exact 40-character commit, clones it in a fresh
directory and imports normal repository modules. Use `--phase quick` to compare
the two explicit profiles and `--phase workspace` for the isolated shape-8
research experiment. The latter builds a separate experimental extension; it
does not alter the production provider or defaults. Cloud kernels are private.
Kaggle's two-concurrent-GPU-session limit requires waiting for a slot; a rejected
push is not a queued task.

The new `--phase pilot` (shapes 2/8/13) and `--phase full` (all 13) run paired
quick/balanced/steady comparisons, separate cold workers, cache-hit controls and
real graph/weight/mask/output contracts. `--phase fusion` profiles shape 6 and
tests four experimental compensated GEMM + exact-GELU + split epilogues. The
fusion must pass all three original accuracy gates and win at least 3% in both
event and wall timing, then be independently confirmed before any promotion.

The [generated usability report](../results/next/usability_summary.md) includes
independent-session ranges, startup/steady trade-offs, and losing workspace
variants. Its generator rejects incomplete checks, mismatched core hashes,
duplicate checkout paths and unlike GPU/Torch environments. Three observed
sessions support a reproducibility statement, not a statistical SLA.

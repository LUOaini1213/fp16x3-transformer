# fp16x3-transformer

[![CPU correctness](https://github.com/LUOaini1213/fp16x3-transformer/actions/workflows/cpu-smoke.yml/badge.svg)](https://github.com/LUOaini1213/fp16x3-transformer/actions/workflows/cpu-smoke.yml)

A Transformer layer that uses **FP16 tensor cores with compensated arithmetic
to retain FP32-class accuracy**. Fused Triton kernels split operands into FP16
hi/lo parts; one cuBLAS GEMM with K tripled accumulates the three cross terms.
Memory-efficient attention and per-shape eager/compile/graph selection complete
the forward pass.

Built for TikTok TechJam 2026 Track 3. **The entry did not place.** The competition
submission is frozen on `submission` / `submitted-2026-09-01`; `main` is maintained.

[Quickstart](docs/QUICKSTART.md) · [Measured results](results/next/usability_summary.md) ·
[Runtime validation](results/next/improvements_summary.md) ·
[Independent runtime session](results/next/portfolio_runtime_summary.md) ·
[Technical report](report/report.md) · [Full experiment history](docs/EXPERIMENT_HISTORY.md)

## What the measurements support

| Result | Evidence and scope |
|---|---|
| **2.875× median speedup** | Three independent Tesla T4 sessions, 13 official FP32 shapes, same production-core hashes; paired original reference / optimized runs. Median paired ratio per shape over sessions, then unweighted median over shapes. Session medians are 3.546×, 2.801× and 2.875×. |
| **117/117 FP32 accuracy checks pass** | Three input seeds per shape per session; maximum absolute error 9.83e-6. Official per-element gate: absolute error ≤ 0.002 **OR** relative error ≤ 0.02; non-finite outputs fail. |
| **100,000-token shape runs within a 16 GB T4** | The original materialized-score reference would need about 20.5 TB. Extreme-shape timing uses native FP16, with separately stated equivalence and restricted FP32-prefix checks; it is not part of the FP32 speedup above. |
| **Balanced reduces setup, with session-dependent steady gates** | Two independent full T4 sessions at `c8fe1cf` each pass 234/234 accuracy checks. The balanced-full session clears **12/13** startup/steady gates (shape 2: **+6.95% wall**); portfolio-full clears **13/13**. A separate shape-2 repeat clears **2/3** worker comparisons (one: **+3.63% event**). All measurements and misses remain published; balanced stays opt-in. |

These tables measure Linux / T4 / Torch 2.11.0+cu128 and exact recorded revisions.
The three-session speedup belongs to the pre-cache/recovery production revision.
The [balanced-full report](results/next/improvements_summary.md) and separate
[portfolio-full report](results/next/portfolio_runtime_summary.md) validate the
maintained runtime; the latter does not erase the former's failed gate.
Balanced/steady first public forwards span **3.7–6.4 / 10.6–34.5 s** and
**4.0–8.0 / 11.2–37.7 s**, respectively. Imports, input/model setup and
CUDA-context probes are excluded. Old throughput is not attributed to new code.
The reference is the unchanged competition implementation, so the
headline does not claim an advantage over every modern Transformer library.

## The engineering contribution

- **Compensated tensor-core GEMMs.** Fused LayerNorm/GELU/split kernels emit
  hi/lo FP16 operands. The packed product computes `hi·hi + hi·lo + lo·hi` with
  FP32 accumulation. In the September matched ablation, this raises median
  speedup from 2.30× to 2.83×; the GEMM-bound shape improves from 1.09× to 1.57×.
- **Memory and execution policy.** SDPA removes the materialized S×S score
  matrix; batch chunking writes into a preallocated output for the extreme
  shape. Smaller shapes select eager, compiled or CUDA-graph execution using
  actual timings. Returned outputs own their storage.
- **Measurement as a selection rule.** Losing custom attention, fusion and
  workspace variants remain published. New backends require correctness and
  paired event/wall wins before promotion. Source hashes, raw rounds, cold
  costs and failure cases accompany the tables.

The parameter names match the original model exactly, allowing strict weight
copying. The official reference and grading harness are unchanged.

## Try it

Use Python 3.11+ and an existing Torch environment. For a small CPU correctness
check, install `requirements.txt` and run:

```bash
python -m pip install -r requirements.txt
python -m scripts.run --check-only
python -m scripts.run --mode quick --shape 2
```

On a compatible CUDA runtime:

```bash
python -m scripts.run --mode balanced --device cuda --shape 2
python -m scripts.run --mode steady --device cuda --shapes 1-13 --output results/local_run.json
python -m scripts.run --mode balanced --device cuda --shape 2 --tune-cache results/local_dispatch.json
```

| Profile | When to use it | Setup work |
|---|---|---|
| `quick` (CLI default) | First inspection or short-lived inference | Eager FP32 GEMMs + SDPA; no Inductor, manual graph or Triton launches |
| `balanced` (explicit choice) | Compare compensated throughput with less planning | FP16x3 + SDPA, no Inductor; Triton JIT and optional manual graph selection remain |
| `steady` | Repeated inference with the shipped policy | FP16x3 + eager/compile/manual-graph selection |

Direct model imports keep the shipped defaults. `--device cuda` fails clearly
when CUDA is unavailable; CPU output is labelled correctness-only. The command
does not install or replace Torch/CUDA. `torch>=2.1` is an API fallback floor,
not validation of every version's acceleration. Windows CPU checks do not
establish Linux/CUDA performance.

## Verify and reproduce

Check the committed GPU evidence before installing Torch. This audit uses only
Python's standard library, requires no GPU or model download, and leaves the
measurement files unchanged:

```bash
python -m scripts.verify_next_evidence --maintained-runtime
```

It checks both full-session collections, pilots, fusion and targeted-repeat
receipts, artifact hashes, current core sources and all generated reports.
It reports each session's gates separately. Missing receipts, unlisted or
changed measurements, and stale reports fail with exit code 2 and file names.
It verifies the recorded run; it does not rerun or certify a new GPU benchmark.

```bash
python -m pip install pytest
python -m pytest -q tests
python -m scripts.summarize_portfolio_runtime --check
```

CPU CI checks the reference math, runtime contracts, evidence integrity and
generated summaries. GPU evidence comes from private Kaggle T4 jobs that clone
an exact public Git commit and import normal repository modules. The
[release protocol](docs/RELEASE_VERIFICATION.md) and
[current runtime protocol](docs/QUICKSTART.md) provide the exact commands and
acceptance gates. Historical September Torch 2.10 results are kept separately.

## Limits worth reading

- The measured accelerated path is inference on a T4. Other GPUs, training,
  broader shape families and large real-model deployments need their own runs.
- FP16x3 assumes bounded activations. First-call/static guards are the default;
  `T3_X3_GUARD=every` checks later inputs at the cost of a synchronization.
  The measured error also grows with K.
- Shape 14 cannot have a full original-FP32 oracle on this hardware. Its native
  FP16 equivalence and FP32-prefix tests have a narrower claim than shapes 1–13.
- Setup can dominate short workloads. Dispatch-cache hits still recapture and
  validate graphs; a saved hint is neither a serialized executable nor a
  guarantee of a speedup. Concurrent use of one mutable graph instance is not
  validated.

## Where to look

| Path | Role |
|---|---|
| [user_optimized.py](user_optimized.py) | Production Transformer and runtime policy |
| [kernels/fp16x3.py](kernels/fp16x3.py) | Fused split kernels and compensated GEMMs |
| [kernels/dispatch_cache.py](kernels/dispatch_cache.py) | Environment/source/layout-keyed JSON dispatch hints |
| [scripts/run.py](scripts/run.py) | CLI profiles, correctness checks and timing JSON |
| [results/next/](results/next/) | Source-stamped measurements and generated reports |
| [docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md) | Detailed ablations, precision ladders, losing variants and history |

The narrated [demo video](build/track3_demo.mp4) and
[AI tooling log](docs/AI_TOOLS.md) document the original development and
measurement process.

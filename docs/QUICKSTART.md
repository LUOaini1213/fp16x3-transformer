# Quick startup and repeatable inference

Run commands from the repository root. No NVIDIA GPU is needed for authoring
or the small CPU correctness check. GPU measurements need a compatible CUDA
runtime; the maintained evidence uses Linux, a Tesla T4, and Torch 2.11.0+cu128.

## Choose a runtime profile

```bash
python -m scripts.run --check-only
python -m scripts.run --mode quick --shape 2
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
| `steady` | Shipped compensated fp16x3 + SDPA | First-forward compile/eager/manual-graph selection | Many repeated forwards |

Quick mode is not FP16 autocast or a lower-accuracy shortcut. Both profiles keep
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

## Sweep and inspect results

```bash
python -m scripts.run --mode steady --device cuda --shapes 1-13 --output results/local_run.json
python -m scripts.run --mode quick --device cuda --shapes 2,8,13 --output results/local_run_quick.json
```

Each shape runs in a fresh child process. JSON reports the effective profile,
environment, all three correctness checks, selected dispatch, first public
forward, and steady-wall samples. First-forward timing excludes Python process
startup/imports, model/input creation and the environment's CUDA context probe;
it must not be called total application cold-start latency. Compiler caches can
already be warm when using the same runtime for several shapes.

Shape 6 is intentionally refused on CPU to avoid the large reference allocation.
Shape 14 is not exposed here: the unchanged full FP32 reference needs about
20.5 TB of scores, so the independent extreme-shape runner has a different,
explicitly restricted correctness protocol.

## Dependencies and reproducibility

The fallback/API dependency floor is `torch>=2.1`; this is not a claim that every
version, OS, or GPU reaches the measured speed. Older Torch, missing Triton or
unsupported GPUs may use the correct FP32 path without compensated tensor-core
speed. `steady` results record which dispatch actually ran. Current CPU CI checks
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

The [generated usability report](../results/next/usability_summary.md) includes
independent-session ranges, startup/steady trade-offs, and losing workspace
variants. Its generator rejects incomplete checks, mismatched core hashes,
duplicate checkout paths and unlike GPU/Torch environments. Three observed
sessions support a reproducibility statement, not a statistical SLA.

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
  three rotated full-forward pairs. Full-output equivalence is against
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

## Reproduction

```bash
python scripts/build_clean_kaggle.py --phase fp32 --ref EXACT_MEASURED_COMMIT \
  --id YOUR_ACCOUNT/track3-release-fp32 --out .kaggle_upload/release_fp32
python scripts/build_clean_kaggle.py --phase flash --ref EXACT_MEASURED_COMMIT \
  --id YOUR_ACCOUNT/track3-release-flash --out .kaggle_upload/release_flash
python scripts/build_clean_kaggle.py --phase attention --ref EXACT_MEASURED_COMMIT \
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

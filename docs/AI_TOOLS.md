# AI Tools Used

This project was built with an AI-assisted workflow. Per the Track 3 rules,
using AI tools to analyze the workload and generate/optimize kernels is
explicitly in scope and earns bonus points. This log documents that usage
honestly.

## Tools

| Tool | Role |
|---|---|
| **Claude (Claude Code, Opus 4.8)** | Read and line-by-line analyzed the official `torch_transformer_benchmark.py`; designed the optimization strategy (multi-agent design pass exploring an SDPA/compile MVP, a Triton/dispatch track, and an infra/deliverables track); implemented `UserOptimizedTransformer`; wrote the run harness, notebooks, and this documentation. |
| **Codex** | Maintained the project after submission: audited the current code, removed repeated mask synchronization, repaired cache invalidation on parameter/LayerNorm changes, implemented packed padding, extended large-shape runtime selection, and ran paired T4 checks using the official timing loop (2026-10-09). |
| **Google Colab / Kaggle** | Free GPU runtimes (T4 / P100) used to benchmark and validate — an explicitly allowed development tool. |
| **PyTorch Inductor (`torch.compile`)** | AI/compiler-driven kernel generation: automatically fuses LayerNorm, bias, and GELU epilogues into Triton kernels. |

## How AI shaped the technical decisions

The October follow-up additionally checked current primary documentation and
upstream source, refreshed profiling of the actual fp16x3 path, built an
algorithm-level cuBLASLt search and guarded optional Turing attention adapter,
and tested C++ dispatch and native metadata batching. Candidate decisions are
based on GPU output comparisons, paired timings and ownership/mutation checks;
losing variants and repaired experiment mistakes remain in
`docs/NEXT_OPTIMIZATION_AUDIT.md` and `results/next/`. AI-generated proposals
are not treated as benchmark results.

The clean release audit additionally clones exact GitHub commits on fresh T4
sessions and imports the real modules, separates first-call cost from paired
steady timing, measures each variant's memory in an isolated process, and tests
seven FP32 attention layouts end-to-end. Two harness mistakes exposed by normal
imports/construction were repaired with regression tests; their failure logs
remain archived. The original benchmark and production core are unchanged.
See `docs/RELEASE_VERIFICATION.md` for this separate protocol.
The final production adapter is additionally timed over three complete paired
forwards through normal imports, with build/cold costs, full native-FP16
equivalence and the restricted original-FP32 oracle documented separately.
The unified Markdown table is generated from JSON and regression-checked
against those immutable artifacts; AI-written prose cannot silently change its
numbers. Default decisions retain the existing FP32 path and explicit opt-ins.

The subsequent usability pass adds an environment-checking, accuracy-gated
quick/steady command without changing production compute, repeats the exact
protocol in two independent T4 sessions, and tests shape-8 workspace budgets
in a separate extension. Three-session speedup is reported as 2.875×, including
the lower repeats rather than selecting the original 3.546×. Workspace variants
pass numerics but fail the predeclared end-to-end adoption threshold and remain
research-only. Generated tables validate source hashes, round counts and input
checks; raw evidence includes the losing variants and startup trade-offs.

1. **Workload analysis.** The AI extracted the exact grading contract from the
   harness code (tolerances `atol=0.002`/`rtol=0.02`, per-element all-pass rule,
   `strict=True` weight copy, fp32 softmax reference, `padding_ratio=0` hot path)
   rather than guessing — several of these directly constrain the implementation.
2. **The headline insight.** The AI computed that shape 14's baseline score
   matrix is `[32,16,1e5,1e5] = 5.12e12` elements ≈ **20.5 TB**, proving the
   baseline is infeasible and that memory-efficient attention is
   mandatory, not merely faster. This reframed the whole submission as
   "impossible → possible."
3. **Per-shape dispatch.** The AI classified the 14 shapes into launch-bound,
   compute/throughput-bound, and memory-bound buckets and picked the
   `torch.compile` mode + chunking policy per bucket.
4. **Correctness-first discipline.** The AI enumerated the harness' failure traps
   (NaN on fully-masked rows, tanh-GELU drift, mixing `is_causal` with a dense
   mask that can't be allocated at S=1e5) and encoded guards for each.

## Prompt → design → diff → verify loop

- **Prompt:** analyze the benchmark and produce a correctness-verified plan.
- **Design:** parallel design agents (MVP / Triton / infra) → synthesized plan
  (see `report/` and the project plan).
- **Diff:** implemented `user_optimized.py`, `submission.py`, `run_all.py`,
  `scripts/shape14_optimized_only.py`.
- **Verify:** numerical checks use the official comparison rule. Historical
  tables came from the custom sweep driver documented in the README; the
  maintained runtime audit uses the official timing loop. Raw logs and
  machine-readable results identify which protocol produced each table.

_This file is intentionally specific so judges can see exactly where AI was used
and where human review/verification gated it._

# Clean GitHub/T4 release evidence

Generated from source-stamped JSON, not historical headlines.

Measured Git commits: FP32 `62bb64c05ce55c27cbe10d034087939c9f1d7fed`; flash `f30c8aa55f5d3dd1eb74a2f0e61727bcda6e3190`; attention `23d070f64df8c293e27de87a83dea4b829c51e39`. Core-source hashes match across all three.

GPU: Tesla T4; PyTorch 2.11.0+cu128.

Source evidence: [FP32 JSON](release-fp32/next_release_fp32.json), [flash JSON](release-flash/next_release_flash.json), [attention JSON](release-attention/next_release_attention.json).

## Current FP32 sweep

Three rotated rounds; medians of per-round medians. Event and synchronized host-wall latency are separate. Every default output is FP32. Memory is isolated single-model peak allocated GiB, including input/output and graph pools, excluding the oracle. Cold seconds include first-forward planning/compile/tuning; job-local caches may already be warm.

| Shape | Baseline event ms | Default event ms | Speedup | Default wall ms | First call s | Steady baseline / default GiB | Default cold peak GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.6727 | 3.4726 | 2.785× | 3.5141 | 17.56 | 0.077 / 0.027 | 0.090 |
| 2 | 3.3886 | 0.5388 | 6.289× | 0.5524 | 9.77 | 0.011 / 0.012 | 0.030 |
| 3 | 3.4245 | 0.6222 | 5.504× | 0.6315 | 7.93 | 0.015 / 0.013 | 0.032 |
| 4 | 3.3864 | 0.9205 | 3.679× | 0.9378 | 8.11 | 0.027 / 0.016 | 0.044 |
| 5 | 18.8637 | 7.2034 | 2.619× | 7.2017 | 8.33 | 0.143 / 0.043 | 0.151 |
| 6 | 1509.7961 | 615.2455 | 2.454× | 619.4121 | 31.11 | 10.390 / 8.330 | 8.330 |
| 7 | 6.3999 | 1.3058 | 4.901× | 1.3228 | 9.59 | 0.049 / 0.012 | 0.041 |
| 8 | 148.2783 | 91.7719 | 1.616× | 91.9687 | 13.52 | 0.415 / 0.368 | 0.745 |
| 9 | 6.8792 | 3.4901 | 1.971× | 3.5389 | 8.13 | 0.053 / 0.027 | 0.090 |
| 10 | 8.3184 | 3.4493 | 2.412× | 3.4737 | 8.24 | 0.061 / 0.027 | 0.090 |
| 11 | 22.5980 | 5.7775 | 3.911× | 5.7407 | 8.35 | 0.171 / 0.027 | 0.090 |
| 12 | 3.3931 | 0.9568 | 3.546× | 0.9790 | 8.07 | 0.021 / 0.016 | 0.044 |
| 13 | 324.2063 | 63.2362 | 5.127× | 63.4705 | 9.94 | 2.293 / 0.439 | 0.439 |

Unweighted median shape speedup: **3.546×**. All 39 default comparisons pass the official OR gate; maximum absolute error **9.8347664e-06**.

Shape 8 opt-in Lt: **91.7719 → 86.1248 ms** event (6.15% reduction), **91.9687 → 86.4221 ms** wall; first Lt call **36.80 s**. Algorithm selection and three original-reference accuracy checks are recorded in the JSON.

## Formal production Turing adapter

Native-FP16 shape 14 only; not the official FP32 grading speedup. Both sides are eager (compile/graphs disabled), batch-chunked by one; these are synchronized whole-forward wall seconds, not an independently autotuned SDPA comparison. All 3,276,800,000 outputs pass against native-FP16 SDPA. The independent original-FP32 oracle is restricted to the first batch's 512-token causal prefix. Build and first calls are excluded from steady rounds.

| Backend | First call s | Round 1 s | Round 2 s | Round 3 s | Median s |
|---|---:|---:|---:|---:|---:|
| sdpa | 156.183 | 171.441 | 172.220 | 170.838 | 171.441 |
| turing | 95.068 | 92.308 | 94.320 | 95.037 | 94.320 |

Steady speedup: **1.818×**. Dependency build: **618.3 s**. Confirmed production-adapter calls: **256**. Full-output maximum absolute difference: **0.0078125**. Peak memory is from a shared two-model process, not isolated deployment memory.

Shared-process steady peak allocated memory: **sdpa: 13.601 GiB**, **turing: 13.601 GiB**.

## Shape 13 FP32 attention candidates

All conversion, copy, launch and concatenation costs are included. Each candidate is tested end-to-end against the original FP32 model on three inputs. Full-model paired results below keep both sides eager (compile and graphs disabled) to isolate attention.

| Layout / dispatch | Attention event ms | Full-model event ms | Full-model wall ms | vs current |
|---|---:|---:|---:|---:|
| current | 9.9502 | 66.5319 | 66.8282 | 1.000× |
| bhsd | 10.7089 | 67.4755 | 67.8371 | 0.986× |
| bshd | 10.6168 | 67.5566 | 68.0715 | 0.985× |
| fold | 10.7086 | 69.3432 | 69.6240 | 0.959× |
| chunk8 | 10.8186 | 70.9448 | 71.2319 | 0.938× |
| chunk16 | 10.5689 | 69.7037 | 70.1113 | 0.954× |
| chunk32 | 10.4223 | 69.3387 | 69.6374 | 0.960× |

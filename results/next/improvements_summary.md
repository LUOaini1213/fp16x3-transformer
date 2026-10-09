# Balanced profiles, dispatch hints/recovery, and FFN fusion

Generated from complete source-stamped cloud artifacts. This report does not replace older cohorts.

Exact tested Git commit: `c8fe1cf7807d5633a1b75773d2ab03461318cb22`.

- [next_improvements_full.json](balanced-full/next_improvements_full.json)
- [next_improvements_cache.json](balanced-full/next_improvements_cache.json)
- [next_improvements_contracts.json](balanced-full/next_improvements_contracts.json)
- [next_improvements_fusion.json](ffn-fusion/next_improvements_fusion.json)
- [next_improvements_shape2_repeat.json](balanced-shape2-repeat/next_improvements_shape2_repeat.json)

## Balanced versus steady

Separate first-forward workers use empty per-mode compiler directories. First public forward excludes process/import/model/input/context setup. Steady timings come from a different, rotated paired process with identical weights/input. These are not total application cold starts or confidence intervals.

| Shape | Balanced / steady first s | Balanced / steady event ms | Event regression | Wall regression | First gate |
|---:|---:|---:|---:|---:|---|
| 1 | 5.8545 / 15.1749 | 3.4417 / 3.4858 | -1.27% | -0.43% | pass |
| 2 | 3.8833 / 10.6812 | 0.3760 / 0.3775 | -0.41% | +6.95% | not cleared |
| 3 | 3.7992 / 10.7239 | 0.4868 / 0.4840 | +0.59% | -0.86% | pass |
| 4 | 3.8777 / 10.8181 | 0.8783 / 0.9163 | -4.15% | -2.83% | pass |
| 5 | 3.9867 / 11.2622 | 7.3172 / 7.3266 | -0.13% | +0.26% | pass |
| 6 | 4.0844 / 34.5421 | 602.3126 / 602.8912 | -0.10% | -0.22% | pass |
| 7 | 3.8599 / 10.6088 | 1.1450 / 1.2439 | -7.95% | -2.29% | pass |
| 8 | 6.4040 / 14.3853 | 89.1208 / 87.6653 | +1.66% | +0.86% | pass |
| 9 | 3.9051 / 11.0975 | 3.1891 / 3.3114 | -3.69% | +1.07% | pass |
| 10 | 3.8315 / 10.7872 | 3.2809 / 3.3729 | -2.73% | +0.27% | pass |
| 11 | 3.9294 / 10.7203 | 5.6147 / 5.6995 | -1.49% | -0.10% | pass |
| 12 | 3.8505 / 10.6872 | 0.9197 / 0.9628 | -4.48% | -4.43% | pass |
| 13 | 3.7070 / 12.4735 | 62.2191 / 62.0214 | +0.32% | +0.29% | pass |

Lower first-forward cost AND <=2% event/wall regression: **12/13 shapes**, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].
All **234** profile/input checks pass (three profiles, three seeds, both cold and paired workers).
No model or CLI default is promoted by this single session.
Single-session balanced median paired event speedup over original FP32: **3.402x**. Not pooled with the historical three-session 2.875x cohort.

## Shape 2: targeted repeat (original miss retained)

One new T4 session, three fresh interpreter workers with shared compiler caches. Each uses three rotated paired rounds of 200 calls per variant, the original timing helpers, three accuracy seeds and output-ownership checks. This is not three independent GPU sessions, an isolated cold-start repeat, or a replacement for the original +6.95% wall result.

| Worker | Balanced / steady event ms | Balanced / steady wall ms | Event regression | Wall regression | Within 2% on both |
|---:|---:|---:|---:|---:|---|
| 1 | 0.2316 / 0.2328 | 0.2200 / 0.2572 | -0.51% | -14.47% | yes |
| 2 | 0.2373 / 0.2373 | 0.2214 / 0.2217 | -0.02% | -0.12% | yes |
| 3 | 0.2374 / 0.2291 | 0.2217 / 0.2219 | +3.63% | -0.09% | no |

Fresh-worker steady comparisons within 2% on both metrics: **2/3**. All 27 additional profile/input checks pass. The original sweep's 12/13 gate and opt-in defaults remain unchanged.


## Dispatch cache and weight recovery

Three fresh workers share warmed compiler directories; compare the cache hit with the uncached warm control, not with the initial compiler-cold producer. Only JSON hints persist; graph pointers and outputs do not.

- Producer: stored-graph. Hit: hit-validated-graph.
- First public forward: hit **1.3805 s**, warm uncached **1.4519 s**.
- Ten-sample synchronized wall medians: hit **0.6973 ms**, control **0.6959 ms**.
- This is one cache trial, not a universal startup/throughput guarantee.

Real CUDA graph contracts: **PASS**, including one re-selection after weight change, mutable masks, owned outputs and sequential alternate-stream use. Concurrent use of one mutable graph instance is not claimed.

## Shape 6: GEMM + exact GELU + split fusion

The research candidate removes the materialized FP32 FFN-in result. Production is unchanged. All retained variants pass three original-FP32 seeds and owned-output checks. Three rotated paired event/wall rounds include normal public-forward costs.

| Variant | Event ms | Wall ms | Event change vs balanced | First gate |
|---|---:|---:|---:|---|
| balanced | 647.9456 | 648.9020 | +0.00% | retain baseline |
| m32k32 | 1254.6770 | 1257.1611 | +93.64% | reject |
| m64k32 | 1134.7774 | 1134.1378 | +75.13% | reject |
| m64k64 | 4631.5088 | 4632.8729 | +614.80% | reject |
| m128k32 | 1117.7051 | 1127.4760 | +72.50% | reject |

Candidates clearing >=3% event AND wall reduction: **none**.
Any winner still requires an independent confirmation before promotion.

Fresh source-stamped baseline profiling (sum of device events across three forwards, not end-to-end fractions):
- Attention: 39.03%.
- GEMM: 35.82%.
- LayerNorm/residual + split: 16.14%.
- Activation + split: 9.00%.


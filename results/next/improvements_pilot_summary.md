# Balanced profiles, dispatch hints/recovery, and FFN fusion

Generated from complete source-stamped cloud artifacts. This report does not replace older cohorts.

Exact tested Git commit: `c8fe1cf7807d5633a1b75773d2ab03461318cb22`.

- [next_improvements_pilot.json](balanced-pilot/next_improvements_pilot.json)
- [next_improvements_cache.json](balanced-pilot/next_improvements_cache.json)
- [next_improvements_contracts.json](balanced-pilot/next_improvements_contracts.json)
- [next_improvements_fusion.json](ffn-fusion/next_improvements_fusion.json)

## Balanced versus steady

Separate first-forward workers use empty per-mode compiler directories. First public forward excludes process/import/model/input/context setup. Steady timings come from a different, rotated paired process with identical weights/input. These are not total application cold starts or confidence intervals.

| Shape | Balanced / steady first s | Balanced / steady event ms | Event regression | Wall regression | First gate |
|---:|---:|---:|---:|---:|---|
| 2 | 3.8820 / 16.6981 | 0.5392 / 0.5407 | -0.28% | -0.39% | pass |
| 8 | 6.3831 / 14.0198 | 93.7687 / 95.9027 | -2.23% | -2.46% | pass |
| 13 | 3.7130 / 12.2235 | 64.8927 / 64.6169 | +0.43% | +0.13% | pass |

Lower first-forward cost AND <=2% event/wall regression: **3/3 shapes**, [2, 8, 13].
All **54** profile/input checks pass (three profiles, three seeds, both cold and paired workers).
No model or CLI default is promoted by this single session.

## Dispatch cache and weight recovery

Three fresh workers share warmed compiler directories; compare the cache hit with the uncached warm control, not with the initial compiler-cold producer. Only JSON hints persist; graph pointers and outputs do not.

- Producer: stored-graph. Hit: hit-validated-graph.
- First public forward: hit **1.3954 s**, warm uncached **1.4190 s**.
- Ten-sample synchronized wall medians: hit **0.6921 ms**, control **0.6953 ms**.
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


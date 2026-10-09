# Usability, independent reproduction and targeted optimization

Generated from complete, source-stamped JSON.

- [release-fp32/next_release_fp32.json](release-fp32/next_release_fp32.json) — Git `62bb64c05ce55c27cbe10d034087939c9f1d7fed`
- [release-repeat-a/next_release_fp32.json](release-repeat-a/next_release_fp32.json) — Git `38e191d9ba1778b7f837b790f79abe941ee98c66`
- [release-repeat-b/next_release_fp32.json](release-repeat-b/next_release_fp32.json) — Git `38e191d9ba1778b7f837b790f79abe941ee98c66`
- [usability-quick/next_usability_quick.json](usability-quick/next_usability_quick.json) — Git `b335fb64a0fc76ad4105abf58e23589ef8af80ae`
- [usability-workspace-actual/next_usability_workspace.json](usability-workspace-actual/next_usability_workspace.json) — Git `c00bc91d711f0c7502883d5693e39f5d0e9cdc17`

## Three independent T4 sessions

All nine production-core hashes match. Each session has a fresh clone/cache environment, 13 shapes, three rotated paired event/wall rounds and three original-FP32 input checks. Aggregate = median of the three paired speedup ratios per shape, then unweighted median over shapes. Ranges are observed three-session ranges, not confidence intervals or universal guarantees.

Session median shape speedups: **3.546×**, **2.801×**, **2.875×**.

| Shape | Median default event ms | Median event speedup | Session range | Median wall speedup | Median first call s |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.4726 | 2.801× | 2.785–2.814× | 2.743× | 17.41 |
| 2 | 0.5405 | 6.289× | 5.393–6.351× | 5.882× | 8.89 |
| 3 | 0.7936 | 3.916× | 3.773–5.504× | 3.796× | 7.36 |
| 4 | 0.9205 | 3.277× | 3.219–3.679× | 3.259× | 7.40 |
| 5 | 7.2034 | 2.619× | 2.585–2.667× | 2.616× | 7.79 |
| 6 | 592.2823 | 2.486× | 2.454–2.604× | 2.491× | 31.11 |
| 7 | 1.2314 | 5.063× | 4.901–5.113× | 4.989× | 8.96 |
| 8 | 86.1241 | 1.616× | 1.599–1.635× | 1.617× | 12.31 |
| 9 | 3.2409 | 2.002× | 1.971–2.004× | 1.909× | 7.41 |
| 10 | 3.2622 | 2.446× | 2.412–2.457× | 2.367× | 7.56 |
| 11 | 5.6324 | 3.970× | 3.911–4.003× | 3.963× | 7.63 |
| 12 | 1.0294 | 2.875× | 2.689–3.546× | 2.804× | 7.69 |
| 13 | 62.4877 | 5.142× | 5.127–5.205× | 5.115× | 9.27 |

Aggregate median speedup: **2.875×**. All **117** default FP32 checks pass; maximum absolute error **9.8347664e-06**.

## Fast first use versus steady throughput

One fresh child process per mode/shape, three original-FP32 input checks and ten synchronized wall samples. The first public forward excludes Python imports, model/input setup and CUDA context initialization/environment probes. Job-local compiler caches are shared across children. These are not three-session paired performance figures.

Quick = eager FP32 GEMMs + SDPA; steady = unchanged fp16x3 + automatic compile/manual graphs. Crossover is an estimate from first-forward plus repeated steady latency, not a deployment SLA.

| Shape | Quick / steady first s | Quick / steady wall ms | Approx. steady crossover calls |
|---:|---:|---:|---:|
| 1 | 0.3582 / 19.75 | 8.1076 / 3.3952 | 4116 |
| 2 | 0.0552 / 8.49 | 1.3290 / 0.7203 | 13864 |
| 3 | 0.0544 / 6.79 | 1.2337 / 1.0344 | 33779 |
| 4 | 0.0567 / 6.95 | 2.1562 / 1.8255 | 20853 |
| 5 | 0.0714 / 7.13 | 14.6935 / 7.2908 | 955 |
| 6 | 0.8262 / 30.23 | 786.2178 / 569.1984 | 137 |
| 7 | 0.0568 / 8.24 | 4.5089 / 2.5193 | 4116 |
| 8 | 0.1973 / 11.85 | 122.1438 / 82.3537 | 294 |
| 9 | 0.0574 / 6.90 | 6.9087 / 3.1399 | 1817 |
| 10 | 0.0594 / 6.93 | 7.3365 / 3.2292 | 1674 |
| 11 | 0.0639 / 6.92 | 7.5199 / 6.0629 | 4704 |
| 12 | 0.0560 / 6.75 | 2.6514 / 1.8324 | 8171 |
| 13 | 0.1470 / 8.60 | 72.2981 / 63.2947 | 941 |

All **78** profile/input checks pass. Maximum absolute error: **9.8347664e-06**.

## Shape 8 targeted workspace experiment

Original-FP32 gate on three inputs for each of seven variants; all 21 pass. Three rotated paired rounds include public-forward dispatch, input copies, output ownership and per-invocation/current-stream workspace allocations. No production core is changed. Memory records are incremental peaks above seven resident models, not isolated deployment peaks.

Scratch policy: only actual selected workspaceSize allocated per invocation/current stream; allocations included in timing/capture.

Acceptance requires at least 5% lower event AND wall latency than the existing opt-in zero-workspace production Lt provider, followed by independent confirmation before any promotion.

| Variant | Event ms | Wall ms | Event reduction vs existing Lt | Host extension calls |
|---|---:|---:|---:|---:|
| default | 95.2898 | 94.6373 | -4.43% | — |
| production_lt | 91.2437 | 92.4723 | 0.00% | — |
| ws0 | 93.2186 | 94.2391 | -2.16% | 76 |
| ws1 | 93.8322 | 93.2996 | -2.84% | 76 |
| ws4 | 97.0681 | 96.6930 | -6.38% | 76 |
| ws16 | 95.9604 | 95.2263 | -5.17% | 76 |
| ws32 | 94.3291 | 93.8846 | -3.38% | 76 |

Counters include capture-time extension invocations; CUDA graph replays do not increment Python counters.

Experimental extension build: **28.17 s**.

Candidates clearing the first gate: **none; retain the existing backend**.

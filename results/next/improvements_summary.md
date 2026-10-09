# Maintained runtime: balanced, cache and recovery validation

Generated with `python -m scripts.summarize_improvements` from committed Kaggle artifacts.

Measured Git: `c8fe1cf7807d5633a1b75773d2ab03461318cb22`. Linux / Tesla T4 / Torch 2.11.0+cu128 / CUDA 12.8.

A three-shape pilot and a separate fresh-checkout full sweep use the same source hashes. This validates the new runtime; the historical three-session 2.875× result remains separate.

**All 234 full-sweep profile/input checks pass** (13 shapes × 3 profiles × 3 seeds, both separate cold workers and paired workers); maximum absolute error 9.8347664e-06. Each profile's returned-output ownership is checked.

## Balanced acceptance

Gate per shape: lower first-public-forward cost than steady, and no more than 2% event AND wall regression. Paired timing uses three rotated rounds and reported medians. Every cold profile uses a separate empty compiler directory. Startup excludes process/import, model/input setup and CUDA-context probes. Shared-process paired models are not isolated memory measurements.

| Shape | Quick / balanced / steady first s | Balanced / steady event ms | Balanced / steady wall ms | Event / wall delta % | Gate |
|---:|---:|---:|---:|---:|---|
| 1 | 0.374 / 8.016 / 17.470 | 3.5240 / 3.5818 | 3.5951 / 3.6039 | -1.61 / -0.25 | PASS |
| 2 | 0.059 / 3.998 / 11.220 | 0.4024 / 0.4027 | 0.4112 / 0.4111 | -0.06 / +0.02 | PASS |
| 3 | 0.059 / 4.119 / 11.262 | 0.4319 / 0.4299 | 0.4414 / 0.4389 | +0.46 / +0.57 | PASS |
| 4 | 0.060 / 4.037 / 11.391 | 0.9266 / 0.9658 | 0.9476 / 0.9820 | -4.06 / -3.50 | PASS |
| 5 | 0.072 / 4.322 / 11.800 | 7.7286 / 7.7392 | 7.7599 / 7.8144 | -0.14 / -0.70 | PASS |
| 6 | 0.894 / 4.259 / 37.740 | 667.9045 / 668.6889 | 668.4501 / 668.6709 | -0.12 / -0.03 | PASS |
| 7 | 0.064 / 4.046 / 11.342 | 1.1812 / 1.3254 | 1.2594 / 1.3255 | -10.88 / -4.98 | PASS |
| 8 | 0.205 / 7.224 / 15.246 | 97.5813 / 96.4516 | 96.6608 / 96.0301 | +1.17 / +0.66 | PASS |
| 9 | 0.064 / 4.111 / 11.557 | 3.3786 / 3.6244 | 3.6364 / 3.6193 | -6.78 / +0.47 | PASS |
| 10 | 0.068 / 4.093 / 11.512 | 3.4351 / 3.7272 | 3.7155 / 3.7585 | -7.84 / -1.14 | PASS |
| 11 | 0.070 / 4.242 / 11.537 | 6.1752 / 6.0888 | 6.1250 / 6.1151 | +1.42 / +0.16 | PASS |
| 12 | 0.061 / 4.175 / 11.505 | 0.9072 / 1.0301 | 0.9614 / 1.0499 | -11.93 / -8.42 | PASS |
| 13 | 0.156 / 4.031 / 13.320 | 65.0121 / 64.9567 | 64.8560 / 64.8705 | +0.09 / -0.02 | PASS |

All shapes clear the measured gate. **Defaults are unchanged; balanced remains an explicit CLI choice.** One full session and the limited pilot are not a universal performance guarantee.

## Dispatch-cache control

Shape 2 runs in fresh processes. Cache-hit and uncached controls share warmed compiler directories; only JSON dispatch hints persist. Graph hits recapture and compare against fresh eager output bitwise.

| Run | First public forward s | Steady wall ms | Cache status |
|---|---:|---:|---|
| producer | 4.0890 | 0.7139 | stored-graph |
| hit | 1.4812 | 0.6998 | hit-validated-graph |
| warm_uncached | 1.5356 | 0.6941 | disabled |

These are single unpaired workers: the table proves hit validation and retains the warmed uncached control, but does not establish a statistically reliable cache speedup.

## GPU runtime contracts

All recorded checks pass: owned output; changed input; mutable normal mask; mutable inference mask; weight change re-selects once; stable weights do not retune; sequential alternate stream.
Sequential alternate-stream use is tested; concurrent calls on one mutable graph instance are not.

## FFN fusion experiment

Shape 6 compares four compensated GEMM + exact-GELU + split epilogues against balanced. Each timed variant passes three original-reference checks. Promotion requires at least 3% event AND wall reduction and independent confirmation; no research backend is installed by this report.

| Variant | Event ms | Wall ms |
|---|---:|---:|
| balanced | 647.9456 | 648.9020 |
| m32k32 | 1254.6770 | 1257.1611 |
| m64k32 | 1134.7774 | 1134.1378 |
| m64k64 | 4631.5088 | 4632.8729 |
| m128k32 | 1117.7051 | 1127.4760 |

Qualified candidates: **none**. Losing variants are retained as evidence; production remains unchanged.

## Evidence and reproduction

- [Full sweep](improvements-full/next_improvements_full.json), [cache](improvements-full/next_improvements_cache.json), [contracts](improvements-full/next_improvements_contracts.json).
- [Pilot](improvements-pilot/next_improvements_pilot.json) and [fusion](improvements-fusion/next_improvements_fusion.json).
- Each folder retains the source-pinned cloud launcher, raw log, round-level JSON and SHA-256 receipt.
- Build a fresh private T4 job with `scripts/build_clean_kaggle.py --ref c8fe1cf7807d5633a1b75773d2ab03461318cb22 --phase full --id ACCOUNT/UNIQUE_KERNEL --out .kaggle_upload/repeat-full`; push using `kaggle kernels push -p .kaggle_upload/repeat-full --accelerator NvidiaTeslaT4`.
- Use `--phase pilot` for shapes 2/8/13 or `--phase fusion` for the isolated shape-6 experiment.

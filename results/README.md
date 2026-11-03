# results/

Two generations of evidence live here. Start with `next/`.

| Path | What it is |
|---|---|
| [`next/`](next/) | **Current evidence** (October 2026, Torch 2.11 on a Kaggle T4): one folder per GPU session with its cloud script, raw log and JSON, plus the generated reports the README cites ([usability](next/usability_summary.md), [clean release](next/clean_release_summary.md), [balanced-full](next/improvements_summary.md), [portfolio-full](next/portfolio_runtime_summary.md)). `python -m scripts.verify_next_evidence --maintained-runtime` checks its receipts, artifact hashes and generated reports. |
| `results*.csv`, `kaggle_*.log` (this folder) | **September 2026 record** (Torch 2.10): the competition-era sweeps and ablations behind [docs/EXPERIMENT_HISTORY.md](../docs/EXPERIMENT_HISTORY.md) and [report/report.md](../report/report.md). `results_t4.csv` is the September T4 headline (2.834x, three runs in `results_t4_runs.csv`), `results.csv` the P100 run (SDPA only, no tensor cores); `tests/test_reported_numbers.py` recomputes the published figures from these CSVs. |
| `runtime_*.json` | Early-October runtime sweeps (Torch 2.11 on a T4; full sweep, memory audit, shape-14 runs) that predate `next/`. The history page and the report cite them; they are not part of the README headline. |
| [`ablation.md`](ablation.md), `ablation_t4.csv` | The September per-stage ablation. |

These older files stay at these paths because the report, the history page
and the tests cite them by path. They are not pooled with the `next/` results.

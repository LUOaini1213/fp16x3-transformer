# `build/` — the demo video, and which run it is showing

`track3_demo.mp4` and `track3_demo.srt` were rendered on **2026-09-01** by
`scripts/make_video.py` from that day's sweep — `results/kaggle_t4_run.log` and
`results/kaggle_t4_shape14.log`. Every figure spoken in the narration and shown
in the terminal replay is that run's:

| in the video | that run | current, `results/results_t4.csv` |
|---|---|---|
| median speedup | **2.282x** (fp32-SGEMM path) | **2.834x** (fp16x3 default) |
| shape 1 | 2.282x | 2.702x |
| shape 8 (K=1024) | 1.086x | 1.566x |
| shape 13 | 4.439x | 5.222x |
| shape 14, full `S=100000` | 204,132.3 ms · 15,676 tok/s · 14.58 GB | 183,884.9 ms · 17,402 tok/s · 14.17 GB |

Two changes landed after the render and moved all of it: the Dynamo
recompile-limit fix (the sweep had been running with `torch.compile` silently
inactive from the fifth shape on — see the README section "A measurement bug we
found late") and fp16x3 becoming the shipped default for the linear layers.

The cut has not been re-rendered against the new numbers. Re-rendering is not a
matter of re-running the script: `scripts/make_video.py` reads
`results/results_t4.csv` for the results slide and its chart, but the narration
(`SECTIONS`) and the terminal replay (`DEMO`) are fixed text transcribed from
the logs above, so those have to be rewritten first or the video would say one
number and show another.

The maintained figures are the results table in the top-level `README.md` and
the CSVs under `results/`; `tests/test_reported_numbers.py` recomputes them from
those CSVs on every CI run.

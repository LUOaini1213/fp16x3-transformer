"""Every speedup and error figure this repo publishes is recomputed here from
the committed CSVs and matched against the prose, so a stale sentence fails CI
instead of surviving in a README nobody re-reads.

The repo used to carry three different medians at once (2.50x in the README's
TL;DR table, 2.28x in the demo video, 2.834x in the results table) because the
only thing keeping them in step was remembering to. This file is the thing that
remembers.

Rounding conventions match the documents: speedups to 3 decimals in the results
tables and 2 in prose, errors to 2 significant digits, tolerance margins to the
nearest integer.

    pytest -q tests/test_reported_numbers.py
"""
import csv
import re
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATOL = 0.002  # the grader's absolute gate


def _rows(name):
    with open(ROOT / "results" / name, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["pass"] == "PASS" and r["speedup"]]


def _stats(name):
    rows = _rows(name)
    sp = sorted(float(r["speedup"]) for r in rows)
    return {
        "n": len(sp),
        "median": statistics.median(sp),
        "min": sp[0],
        "max": sp[-1],
        "worst_abs": max(float(r["max_abs"]) for r in rows),
    }


def _text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


T4 = _stats("results_t4.csv")
T4_FP32 = _stats("results_t4_fp32.csv")
P100 = _stats("results.csv")

README = _text("README.md")
REPORT = _text("report/report.md")
HANDOFF = _text("docs/DEVPOST_HANDOFF.md")


# --------------------------------------------------------------- the data
def test_t4_sweep_shape():
    """13 graded shapes pass; shape 14 has no baseline and no speedup."""
    assert T4["n"] == 13
    with open(ROOT / "results" / "results_t4.csv", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    assert len(all_rows) == 14
    assert all_rows[13]["pass"] == "PASS(trunc)" and not all_rows[13]["speedup"]


def test_t4_per_run_medians():
    """The three independent shipped runs, each a median over 13 shapes."""
    with open(ROOT / "results" / "results_t4_runs.csv", encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if r["config"] == "fp16x3 (shipped)"]
    by_run = {}
    for r in runs:
        by_run.setdefault(r["run"], []).append(float(r["speedup"]))
    assert sorted(by_run) == ["A", "B", "C"]
    assert all(len(v) == 13 for v in by_run.values())
    medians = [f"{statistics.median(by_run[k]):.3f}" for k in "ABC"]
    assert medians == ["2.828", "2.912", "2.834"]
    # the headline is the per-shape median of those runs, not one lucky run
    assert f"{T4['median']:.3f}" == "2.834"


# ------------------------------------------------------- the prose vs data
def _fmt(x, nd=3):
    return f"{x:.{nd}f}"


def test_readme_results_table_matches_csv():
    for s in (f"**{_fmt(T4['median'])}x**", f"{_fmt(T4['min'])}x - {_fmt(T4['max'])}x",
              f"{_fmt(P100['median'])}x", f"{_fmt(T4_FP32['median'])}x"):
        assert s in README, f"README lost {s!r}"


def test_readme_tldr_row_matches_csv():
    """The TL;DR row that describes the shipped default -- the one a reader
    hits first, and the one that carried 2.50x / 1.60x / 4.1e-05 for a week."""
    row = [ln for ln in README.splitlines()
           if "fp16x3 linear layers" in ln and "**default**" in ln]
    assert len(row) == 1
    row = row[0]
    shape8 = float(next(r["speedup"] for r in _rows("results_t4.csv") if r["shape"] == "8"))
    assert f"**{_fmt(T4['median'], 2)}x**" in row          # 2.83x
    assert f"{_fmt(shape8, 2)}x" in row                    # 1.57x
    assert f"{T4['worst_abs']:.1e}" in row                 # 9.5e-06
    # the values this row used to carry, scoped to this row so the correct
    # 1.60x in the qkv micro-benchmark tables below is not caught by mistake
    for stale in ("2.50x", "1.60x", "4.1e-05"):
        assert stale not in row


def test_worst_error_and_tolerance_margins():
    rows = _rows("results_t4.csv")
    worst = max(float(r["max_abs"]) for r in rows)
    elsewhere = max(float(r["max_abs"]) for r in rows if r["shape"] != "8")
    assert f"{worst:.2e}" == "9.54e-06" and f"{elsewhere:.2e}" == "2.86e-06"
    margins = (round(ATOL / worst), round(ATOL / elsewhere),
               round(ATOL / T4_FP32["worst_abs"]), round(ATOL / P100["worst_abs"]))
    assert margins == (210, 699, 621, 1049)
    # the shipped path's margin is 210x; 1049x is the P100's and may only be
    # claimed for it
    for doc, name in ((README, "README.md"), (REPORT, "report/report.md")):
        for m in re.finditer(r"1049", doc):
            window = doc[max(0, m.start() - 200):m.start()]
            assert "P100" in window or "1.91e-6" in window, \
                f"{name}: 1049x claimed outside a P100 context"


def test_no_superseded_measurements_in_the_prose():
    """Values from runs that later fixes invalidated. Each is anchored to the
    sentence that used to carry it, not grepped repo-wide, because some of
    these digits are correct elsewhere (1.60x is a real qkv micro-benchmark)."""
    assert "4.1e-05" not in README and "4.1e-5" not in README
    assert "4.1e-05" not in REPORT and "297" not in REPORT.split("## 5. Results")[1][:900]
    assert "2.499" not in HANDOFF
    assert "median 2.282x" not in _text("report/devpost.md")
    # the elevator pitch states a median, not a floor
    assert "on all 13 graded shapes" not in HANDOFF


def test_report_md_matches_csv():
    assert f"**{_fmt(T4['median'])}×**" in REPORT
    assert f"{_fmt(T4['min'])}-{_fmt(T4['max'])}×" in REPORT
    assert f"{T4['worst_abs']:.2e} (K=1024)" in REPORT


@pytest.mark.skipif(not (ROOT / "report" / "report.html").exists(),
                    reason="report/report.html is a build product")
def test_built_report_page_matches_csv():
    html = _text("report/report.html")
    assert "__MEDIAN__" not in html and "__RANGE__" not in html
    assert f"{_fmt(T4['median'])}&times;" in html
    assert f"{_fmt(T4['min'])}–{_fmt(T4['max'])}×" in html


def test_shape14_figures_come_from_the_final_rerun():
    """Shape 14 has no speedup, so its numbers live in a notes field; the video
    still quotes the earlier 204 s run, which README dates explicitly."""
    with open(ROOT / "results" / "results_t4.csv", encoding="utf-8") as f:
        note = list(csv.DictReader(f))[13]["notes"]
    for s in ("183884.9 ms", "17,402 tok/s", "14.17 GB"):
        assert s in note
    assert "17,402 tok/s" in README and "14.17 GB" in README
    # every place the superseded 204 s / 14.58 GB figures still appear must say
    # which run they are from
    for rel in ("README.md", "build/README.md"):
        body = _text(rel)
        hits = list(re.finditer(r"14\.58 GB", body))
        assert hits, f"{rel}: lost the pointer to the video's own figures"
        for m in hits:
            window = body[max(0, m.start() - 900):m.start()]
            assert "2026-09-01" in window, f"{rel}: undated 14.58 GB figure"

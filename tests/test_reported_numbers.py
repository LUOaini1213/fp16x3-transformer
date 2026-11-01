"""Every speedup and error figure this repo publishes is recomputed here from
the committed CSVs and matched against the prose, so a stale sentence fails CI
instead of surviving in a README nobody re-reads.

The repo used to carry three different medians at once (2.50x in the README's
TL;DR table, 2.28x in the demo video, 2.834x in the results table) because the
only thing keeping them in step was remembering to. This file is the thing that
remembers.

Since the October homepage rewrite, the September tables, the TL;DR row, the
fp16 paragraph and the demo-video notes live in docs/EXPERIMENT_HISTORY.md; the
README keeps one sentence of September figures and leads with the October
results/next/ sessions, whose generated reports scripts/verify_next_evidence.py
checks. Both documents are held to the CSVs here.

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
# the fp32 path the fp16 sweep was actually measured against, before fp16x3
# became the default -- it is not the shipped path and is not the P100
T4_GRAPH = _stats("results_t4_graph.csv")
FP16 = _stats("results_t4_fp16.csv")

README = _text("README.md")
# the September record the homepage used to carry, moved out of README.md
HISTORY = _text("docs/EXPERIMENT_HISTORY.md")
REPORT = _text("report/report.md")
ABLATION = _text("results/ablation.md")
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


def test_september_results_table_matches_csv():
    for s in (f"**{_fmt(T4['median'])}x**", f"{_fmt(T4['min'])}x - {_fmt(T4['max'])}x",
              f"{_fmt(P100['median'])}x", f"{_fmt(T4_FP32['median'])}x"):
        assert s in HISTORY, f"docs/EXPERIMENT_HISTORY.md lost {s!r}"


def _shape8(name):
    return float(next(r["speedup"] for r in _rows(name) if r["shape"] == "8"))


def test_readme_september_sentence_matches_csv():
    """The one place the streamlined README still quotes the September sweep."""
    assert (f"from {_fmt(T4_FP32['median'], 2)}× to {_fmt(T4['median'], 2)}×"
            in README)                                                      # 2.30x -> 2.83x
    assert (f"from {_fmt(_shape8('results_t4_fp32.csv'), 2)}× to "
            f"{_fmt(_shape8('results_t4.csv'), 2)}×" in README)            # 1.09x -> 1.57x


def test_tldr_row_matches_csv():
    """The TL;DR row that describes the shipped default -- the one a reader
    hits first, and the one that carried 2.50x / 1.60x / 4.1e-05 for a week."""
    row = [ln for ln in HISTORY.splitlines()
           if "fp16x3 linear layers" in ln and "**default**" in ln]
    assert len(row) == 1
    row = row[0]
    shape8 = _shape8("results_t4.csv")
    assert (f"**{_fmt(T4['median'])}x**" in row                             # 2.834x
            or f"**{_fmt(T4['median'], 2)}x**" in row)                      # 2.83x
    assert f"{_fmt(shape8, 2)}x" in row                    # 1.57x
    assert (f"{T4['worst_abs']:.2e}" in row                                 # 9.54e-06
            or f"{T4['worst_abs']:.1e}" in row)                             # 9.5e-06
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


# the four margins above are all real, but only one of them belongs to the
# shipped path. 1049x is what a worst-case 1.91e-06 buys, and 1.91e-06 was
# measured on the P100 (results.csv) and on the pre-fp16x3 T4 baseline
# (results_t4_graph.csv) -- never on the path we ship, whose worst case is
# 9.54e-06 and whose margin is 210x. An earlier pass fixed three copies of that
# mix-up and left four alive, in files the scan did not cover, so this scan
# walks the repository instead of a hand-written list of documents.
_M1049_ANCHORS = ("P100", "1.91e-6", "1.91e-06", "results_t4_graph.csv")
_M1049_SUFFIXES = {".md", ".py", ".html", ".yml", ".yaml", ".txt"}


def _scannable_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in _M1049_SUFFIXES:
            continue
        if any(part in (".git", "__pycache__", ".pytest_cache") for part in p.parts):
            continue
        if p == Path(__file__).resolve():
            continue  # the scanner itself quotes the digits it forbids
        yield p


def test_1049x_is_never_attached_to_the_shipped_path():
    """Every 1049 in the repo, in any prose, slide or template, has to sit next
    to the measurement it came from -- and must not sit next to the word
    'shipped'. Raw run logs and CSVs are excluded by suffix; the digits also
    occur inside timings there (5.428571701049805)."""
    seen = 0
    for p in _scannable_files():
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"1049", text):
            seen += 1
            window = text[max(0, m.start() - 120):m.start()]
            rel = p.relative_to(ROOT).as_posix()
            assert any(a in window for a in _M1049_ANCHORS), (
                f"{rel}: 1049x claimed with no P100 / 1.91e-06 anchor within "
                f"120 characters: ...{window[-90:]!r}")
            assert "shipped" not in window.lower(), (
                f"{rel}: 1049x claimed for the shipped path, whose margin is "
                f"{round(ATOL / T4['worst_abs'])}x: ...{window[-90:]!r}")
    assert seen >= 4, "the 1049x scan found nothing; did it stop reading files?"


def test_ablation_fp16_table_matches_csv():
    """The README's fp16 paragraph ends 'full numbers in results/ablation.md',
    so the table it points at has to carry the same three regimes and the same
    margins the README quotes, each recomputed from its own CSV."""
    table = ABLATION.split("Here is why we still do not ship it:")[1][:1200]
    for want in (f"{T4['median']:.3f}x",              # 2.834x, shipped
                 f"{T4['worst_abs']:.2e}",            # 9.54e-06
                 f"{round(ATOL / T4['worst_abs'])}x", # 210x
                 f"{T4_GRAPH['median']:.3f}x",        # 2.286x, the fp16 baseline
                 f"{FP16['median']:.3f}x"):           # 4.014x
        assert want in table, f"results/ablation.md fp16 table lost {want!r}"
    # the two ratios the prose quotes, each against a named path
    assert f"{FP16['median'] / T4_GRAPH['median']:.2f}x" in ABLATION   # 1.76x
    assert f"{FP16['median'] / T4['median']:.2f}x" in ABLATION         # 1.42x


def test_no_superseded_measurements_in_the_prose():
    """Values from runs that later fixes invalidated. Each is anchored to the
    sentence that used to carry it, not grepped repo-wide, because some of
    these digits are correct elsewhere (1.60x is a real qkv micro-benchmark)."""
    for doc in (README, HISTORY):
        assert "4.1e-05" not in doc and "4.1e-5" not in doc
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
    assert "17,402 tok/s" in HISTORY and "14.17 GB" in HISTORY
    # every place the superseded 204 s / 14.58 GB figures still appear must say
    # which run they are from; the video notes themselves must keep them
    for rel in ("README.md", "docs/EXPERIMENT_HISTORY.md", "build/README.md"):
        body = _text(rel)
        hits = list(re.finditer(r"14\.58 GB", body))
        assert hits or rel == "README.md", f"{rel}: lost the pointer to the video's own figures"
        for m in hits:
            window = body[max(0, m.start() - 900):m.start()]
            assert "2026-09-01" in window, f"{rel}: undated 14.58 GB figure"

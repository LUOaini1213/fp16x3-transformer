"""The README first screen and runtime-profile table are recomputed here from
the committed evidence, and the older September headline is only allowed where
it is labelled as such.

The repo once carried two different "historical three-session" headlines
(2.83x at "~1e-6 error" in docs/EXPERIMENT_HISTORY.md, 2.875x at 9.83e-6 in the
README). Both trace to committed artifacts, but an unlabelled 2.83x or an error
figure ten times smaller than the measured one reads as number-shopping. This
file fails if that comes back.

    pytest -q tests/test_readme_headline.py
"""
import csv
import json
import re
import statistics
from pathlib import Path

from scripts.summarize_usability import independent_metrics

ROOT = Path(__file__).resolve().parents[1]
NEXT = ROOT / "results" / "next"


def _text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


README = _text("README.md")
HISTORY = _text("docs/EXPERIMENT_HISTORY.md")
FIRST_SCREEN = README.split("\n## ", 1)[0]


def _paragraphs(text):
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def _sessions():
    return [json.loads((NEXT / folder / "next_release_fp32.json").read_text(encoding="utf-8"))
            for folder in ("release-fp32", "release-repeat-a", "release-repeat-b")]


def _short_exp(x):
    """9.834766e-06 -> '9.83e-6', the README's spelling."""
    mantissa, exp = f"{x:.2e}".split("e")
    return f"{mantissa}e{int(exp)}"


# ------------------------------------------------- headline vs the release JSONs
def test_readme_headline_matches_three_release_sessions():
    m = independent_metrics(_sessions())
    median = f"{m['median_speedup']:.3f}×"                                    # 2.875x
    sessions = ", ".join(f"{s:.3f}×" for s in m["session_medians"][:2])
    sessions += f" and {m['session_medians'][2]:.3f}×"                       # 3.546x, 2.801x and 2.875x
    checks = f"{m['comparisons']}/{m['comparisons']}"                         # 117/117
    err = _short_exp(m["max_abs"])                                            # 9.83e-6
    for s in (median, checks, err):
        assert s in FIRST_SCREEN, f"README first screen lost {s!r}"
    table = README.split("## What the measurements support", 1)[1].split("\n## ", 1)[0]
    for s in (f"**{median} median speedup**", sessions, f"**{checks} FP32 accuracy checks pass**",
              f"maximum absolute error {err}"):
        assert s in table, f"README results table lost {s!r}"


def test_readme_shape14_peak_matches_clean_release_summary():
    summary = _text("results/next/clean_release_summary.md")
    peaks = set(re.findall(r"\*\*(?:sdpa|turing): ([0-9.]+) GiB\*\*", summary))
    assert len(peaks) == 1, peaks
    peak = peaks.pop()
    assert f"{peak} GiB" in README
    # the 100k-token shape is a native FP16 run; the README must keep saying so
    row = next(ln for ln in README.splitlines() if "100,000-token" in ln and ln.startswith("|"))
    assert "native FP16" in row and "not part of the FP32 speedup" in row


# --------------------------------------------- runtime-profile table vs reports
def _table_rows(text, header_start):
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(header_start))
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return rows


def _span(values):
    return f"{min(values):.1f}–{max(values):.1f}"


def test_runtime_profile_table_matches_generated_reports():
    improvements = _text("results/next/improvements_summary.md")
    portfolio = _text("results/next/portfolio_runtime_summary.md")

    rows = _table_rows(improvements, "| Shape | Balanced / steady first s")
    assert len(rows) == 13
    first = [tuple(float(v) for v in r[1].split("/")) for r in rows]
    passed = sum(r[5] == "pass" for r in rows)
    shape2_wall = rows[1][4]
    balanced_full = (f"**{passed}/13** (shape 2: **{shape2_wall} wall**)",
                     f"**{_span([b for b, _ in first])} / {_span([s for _, s in first])} s**")

    rows = _table_rows(portfolio, "| Shape | Quick / balanced / steady first s")
    assert len(rows) == 13
    first = [tuple(float(v) for v in r[1].split("/")) for r in rows]
    passed = sum(r[5] == "PASS" for r in rows)
    portfolio_full = (f"**{passed}/13**",
                      f"**{_span([b for _, b, _ in first])} / {_span([s for _, _, s in first])} s**")

    section = README.split("### Runtime profile measurements", 1)[1].split("\n## ", 1)[0]
    bal_row = next(ln for ln in section.splitlines() if ln.startswith("| [balanced-full]"))
    port_row = next(ln for ln in section.splitlines() if ln.startswith("| [portfolio-full]"))
    for s in balanced_full:
        assert s in bal_row, f"balanced-full row lost {s!r}"
    for s in portfolio_full:
        assert s in port_row, f"portfolio-full row lost {s!r}"
    assert "All **234** profile/input checks pass" in improvements
    assert "All 234 full-sweep profile/input checks pass" in portfolio
    assert "234/234" in section
    repeat = re.search(r"\(one: \*\*(\+[0-9.]+%) event\*\*\)", README.replace("\n", " "))
    assert repeat and repeat.group(1) in improvements


def test_runtime_gate_threshold_matches_generated_reports():
    """The README states the gate's regression allowance; it has to be the one
    the generated reports were scored with."""
    pattern = r"(\d+)% event (?:and|AND) wall regression"
    readme = re.search(pattern, README.replace("\n", " "))
    report = re.search(pattern, _text("results/next/portfolio_runtime_summary.md"))
    assert readme and report and readme.group(1) == report.group(1)
    assert f"<={report.group(1)}% event/wall regression" in _text("results/next/improvements_summary.md")


# ------------------------------------------- history opening vs September CSV
def test_history_opening_matches_september_csv():
    """The history page opens with the September cohort. Its figures are
    recomputed from results/results_t4.csv, and no other speedup may be quoted
    there except the README's October headline it points to."""
    with open(ROOT / "results" / "results_t4.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    graded = [r for r in rows if r["pass"] == "PASS" and r["speedup"]]
    assert len(graded) == 13 and rows[13]["shape"] == "14"
    median = statistics.median(float(r["speedup"]) for r in graded)             # 2.834
    worst_row = max(graded, key=lambda r: float(r["max_abs"]))
    assert worst_row["D"] == "1024"                                             # the K=1024 shape
    worst = float(worst_row["max_abs"])                                         # 9.54e-06
    elsewhere = max(float(r["max_abs"]) for r in graded if r is not worst_row)  # 2.86e-06
    peak = float(re.search(r"peak_vram=([0-9.]+) GB", rows[13]["notes"]).group(1))  # 14.17
    october = f"{independent_metrics(_sessions())['median_speedup']:.3f}x"      # 2.875x

    history_head = "\n".join(HISTORY.splitlines()[:30])
    opening = next(p for p in _paragraphs(history_head) if "three-run cohort" in p).replace("\n", " ")
    for s in (f"{median:.2f}x median",
              f"worst error {worst:.1e} (the K=1024 shape; {elsewhere:.1e} elsewhere)",
              f"native fp16 at a {peak:.1f} GB peak",
              f"October {october} three-session result"):
        assert s in opening, f"docs/EXPERIMENT_HISTORY.md opening lost {s!r}"
    assert set(re.findall(r"\d+\.\d+[x×]", opening)) == {f"{median:.2f}x", october}


# ----------------------------------------------------- labels on old headlines
def test_september_headline_is_always_labelled():
    """2.83x is the September Torch 2.10 cohort; wherever the README or the
    history page's opening quotes it, the same paragraph must say so."""
    history_head = "\n".join(HISTORY.splitlines()[:30])
    for name, text in (("README.md", README), ("docs/EXPERIMENT_HISTORY.md (opening)", history_head)):
        for para in _paragraphs(text):
            if re.search(r"2\.83[0-9]*[x×]", para):
                assert "September" in para, f"{name}: unlabelled 2.83x in {para[:120]!r}"


def test_no_understated_error_in_headlines():
    """The measured worst error is ~1e-5 under either cohort, never '~1e-6'."""
    history_head = "\n".join(HISTORY.splitlines()[:30])
    for name, text in (("README.md", README), ("docs/EXPERIMENT_HISTORY.md (opening)", history_head)):
        assert "~1e-6" not in text, f"{name} understates the headline error"
        assert "1e-6 error" not in text, f"{name} understates the headline error"


def test_p100_figure_is_not_credited_to_tensor_cores():
    """The P100 has no tensor cores; its 2.07x is fused attention alone."""
    for name, text in (("README.md", README), ("docs/EXPERIMENT_HISTORY.md", HISTORY)):
        for para in _paragraphs(text):
            if re.search(r"2\.07x|2\.07×", para) and "P100" in para:
                assert "no tensor cores" in para, f"{name}: P100 2.07x without 'no tensor cores'"
            # sentence level: the clause quoting 2.07x must not also credit
            # tensor cores (a correct caveat elsewhere in the paragraph does
            # not excuse it)
            for sentence in re.split(r"(?<=\.)\s+", para.replace("\n", " ")):
                if re.search(r"2\.07x|2\.07×", sentence):
                    credited = re.sub(r"no tensor cores", "", sentence)
                    assert "tensor core" not in credited, \
                        f"{name}: 2.07x credited to tensor cores in {sentence[:120]!r}"

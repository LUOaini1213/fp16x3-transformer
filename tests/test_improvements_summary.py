"""Protect GPU claims against incomplete, mixed-source or altered evidence."""
import copy
import json
from pathlib import Path

import pytest

from scripts.summarize_improvements import generate, render, sweep_metrics


ROOT = Path("results/next")


def evidence():
    names = ("improvements-full/next_improvements_full.json",
             "improvements-pilot/next_improvements_pilot.json",
             "improvements-full/next_improvements_cache.json",
             "improvements-full/next_improvements_contracts.json",
             "improvements-fusion/next_improvements_fusion.json")
    return [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in names]


def test_committed_report_matches_current_source_and_complete_gpu_artifacts():
    report = ROOT / "improvements_summary.md"
    assert b"\r" not in report.read_bytes()
    assert report.read_text(encoding="utf-8") == generate()
    metrics = sweep_metrics(evidence()[0], range(1, 14))
    assert metrics["checks"] == 234
    assert len(metrics["gates"]) == 13


@pytest.mark.parametrize("change", ["missing_shape", "duplicate_shape", "missing_seed", "duplicate_seed", "failed_accuracy"])
def test_runtime_report_rejects_incomplete_or_failed_sweep(change):
    full = copy.deepcopy(evidence()[0])
    if change == "missing_shape":
        full["results"].pop()
    elif change == "duplicate_shape":
        full["results"][-1] = copy.deepcopy(full["results"][0])
    elif change == "missing_seed":
        full["results"][0]["paired"]["accuracy"].pop()
    elif change == "duplicate_seed":
        checks = full["results"][0]["cold"]["balanced"]["accuracy_trials"]
        checks[-1]["seed"] = checks[0]["seed"]
    else:
        full["results"][0]["paired"]["accuracy"][0]["passed"] = False
    with pytest.raises(AssertionError):
        sweep_metrics(full, range(1, 14))


@pytest.mark.parametrize("change", ["source", "dirty", "gpu", "timing"])
def test_runtime_report_rejects_unverified_source_or_timing(change):
    full = copy.deepcopy(evidence()[0])
    if change == "source":
        full["metadata"]["source_manifest"]["user_optimized.py"] = "0" * 64
    elif change == "dirty":
        full["metadata"]["clean_checkout"] = False
    elif change == "gpu":
        full["metadata"]["gpu"] = "CPU"
    else:
        full["results"][0]["paired"]["timing"]["balanced"]["event_ms"] *= .5
    with pytest.raises(AssertionError):
        sweep_metrics(full, range(1, 14))


@pytest.mark.parametrize("change", ["foreign_pilot", "cache_hit", "recovery"])
def test_runtime_report_rejects_mixed_runs_or_unvalidated_runtime_contracts(change):
    values = copy.deepcopy(evidence())
    if change == "foreign_pilot":
        values[1]["metadata"]["torch"] = "different"
    elif change == "cache_hit":
        values[2]["results"]["hit"]["tune_cache"]["status"] = "miss"
    else:
        values[3]["results"]["checks"].pop()
    with pytest.raises(AssertionError):
        render(*values)


def test_regressed_profile_is_not_reported_as_qualified():
    values = copy.deepcopy(evidence())
    timing = values[0]["results"][0]["paired"]["timing"]
    for kind in ("event", "wall"):
        steady = timing["steady"][kind + "_ms"]
        timing["balanced"][kind + "_round_ms"] = [steady * 1.03] * 3
        timing["balanced"][kind + "_ms"] = steady * 1.03
    assert sweep_metrics(values[0], range(1, 14))["gates"][0]["passed"] is False
    report = render(*values)
    assert "RETAIN OPT-IN" in report
    assert "Defaults are unchanged" in report


def test_losing_fusion_variants_and_cache_control_remain_visible():
    report = generate()
    assert "warm_uncached" in report and "single unpaired workers" in report
    assert "Qualified candidates: **none**" in report
    for name in evidence()[4]["results"]["timing"]:
        assert f"| {name} |" in report

import copy
import json
from pathlib import Path

import pytest

from scripts.summarize_usability import (core_manifest, crossover_calls, independent_metrics,
                                        independent_text, quick_text, workspace_text)


def payloads():
    original = json.loads(Path("results/next/release-fp32/next_release_fp32.json").read_text(encoding="utf-8"))
    values = [copy.deepcopy(original) for _ in range(3)]
    for i, value in enumerate(values):
        value["metadata"]["normal_imports"]["production"] = f"/tmp/independent-{i}/user_optimized.py"
    return values


def test_three_session_summary_checks_complete_matching_core():
    result = independent_metrics(payloads())
    assert result["comparisons"] == 117
    assert len(result["rows"]) == 13
    assert result["median_speedup"] == pytest.approx(3.546061413427432, rel=.001)


def test_duplicate_or_changed_core_sessions_rejected():
    values = payloads()
    values[1] = copy.deepcopy(values[0])
    with pytest.raises(ValueError, match="independent"):
        independent_metrics(values)
    values = payloads()
    values[1]["metadata"]["source_manifest"]["user_optimized.py"] = "changed"
    with pytest.raises(ValueError, match="core"):
        independent_metrics(values)


def test_incomplete_independent_session_rejected():
    values = payloads()
    values[1]["results"].pop()
    with pytest.raises(AssertionError, match="incomplete"):
        independent_metrics(values)


def test_crossover_counts_first_call_and_retains_no_win():
    assert crossover_calls({"first_forward_seconds": .01, "steady_wall_ms": 3},
                           {"first_forward_seconds": 10.01, "steady_wall_ms": 1}) == 5001
    assert crossover_calls({"first_forward_seconds": .01, "steady_wall_ms": 1},
                           {"first_forward_seconds": 10, "steady_wall_ms": 2}) is None


def test_incomplete_workspace_accuracy_rejected():
    with pytest.raises(AssertionError):
        workspace_text({"results": {"shape": 8, "accuracy": []}})


def test_real_independent_sessions_are_complete_and_core_matches_usability():
    root = Path("results/next")
    sessions = [json.loads((root / folder / "next_release_fp32.json").read_text(encoding="utf-8"))
                for folder in ("release-fp32", "release-repeat-a", "release-repeat-b")]
    result = independent_metrics(sessions)
    assert result["comparisons"] == 117
    assert result["median_speedup"] == pytest.approx(2.875157350767779)
    for phase in ("quick", "workspace"):
        payload = json.loads((root / f"usability-{phase}" / f"next_usability_{phase}.json").read_text(encoding="utf-8"))
        assert core_manifest(payload) == core_manifest(sessions[0])


def test_usability_report_is_generated_from_complete_artifacts():
    root = Path("results/next")
    path = root / "usability_summary.md"
    assert b"\r" not in path.read_bytes()
    report = path.read_text(encoding="utf-8")
    sessions = [json.loads((root / folder / "next_release_fp32.json").read_text(encoding="utf-8"))
                for folder in ("release-fp32", "release-repeat-a", "release-repeat-b")]
    assert independent_text(sessions) in report
    for phase, function in (("quick", quick_text), ("workspace", workspace_text)):
        payload = json.loads((root / f"usability-{phase}" / f"next_usability_{phase}.json").read_text(encoding="utf-8"))
        assert function(payload) in report
        if phase == "quick":
            partial = copy.deepcopy(payload)
            partial["results"][0]["profiles"]["quick"]["accuracy_trials"].pop()
            with pytest.raises(AssertionError):
                function(partial)


def test_measured_usability_driver_sources_match_current_checkout():
    import hashlib
    payload = json.loads(Path("results/next/usability-quick/next_usability_quick.json").read_text(encoding="utf-8"))
    for name in ("scripts/run.py", "scripts/benchmark_usability.py", "scripts/workspace_provider.py"):
        actual = hashlib.sha256(Path(name).read_text(encoding="utf-8").encode()).hexdigest()
        assert actual == payload["metadata"]["source_manifest"][name]

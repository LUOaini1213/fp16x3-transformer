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
        folder = "usability-workspace-actual" if phase == "workspace" else "usability-quick"
        payload = json.loads((root / folder / f"next_usability_{phase}.json").read_text(encoding="utf-8"))
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
        folder = "usability-workspace-actual" if phase == "workspace" else "usability-quick"
        payload = json.loads((root / folder / f"next_usability_{phase}.json").read_text(encoding="utf-8"))
        assert function(payload) in report
        if phase == "quick":
            partial = copy.deepcopy(payload)
            partial["results"][0]["profiles"]["quick"]["accuracy_trials"].pop()
            with pytest.raises(AssertionError):
                function(partial)


def test_historical_quick_driver_snapshot_matches_measured_source():
    import hashlib
    payload = json.loads(Path("results/next/usability-quick/next_usability_quick.json").read_text(encoding="utf-8"))
    # The old audit measured two profiles. Preserve that exact runner rather
    # than attributing its GPU numbers to the newly added balanced profile.
    snapshot = Path("results/next/usability-quick/run.py.snapshot")
    actual = hashlib.sha256(snapshot.read_text(encoding="utf-8").encode()).hexdigest()
    assert actual == payload["metadata"]["source_manifest"]["scripts/run.py"]
    from scripts.run import COMMON, PROFILES
    for row in payload["results"]:
        assert set(row["profiles"]) == {"quick", "steady"}
        for name, result in row["profiles"].items():
            assert result["profile"] == {**COMMON, **PROFILES[name]}


def test_historical_usability_core_is_preserved_by_source_or_snapshot():
    import hashlib
    payload = json.loads(Path("results/next/usability-quick/next_usability_quick.json").read_text(encoding="utf-8"))
    for name, measured in core_manifest(payload).items():
        source = Path("results/next/usability-quick/user_optimized.py.snapshot") if name == "user_optimized.py" else Path(name)
        actual = hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest()
        assert actual == measured


def test_actual_workspace_driver_is_the_measured_current_source():
    import hashlib
    payload = json.loads(Path("results/next/usability-workspace-actual/next_usability_workspace.json").read_text(encoding="utf-8"))
    for name in ("scripts/benchmark_usability.py", "scripts/workspace_provider.py"):
        actual = hashlib.sha256(Path(name).read_text(encoding="utf-8").encode()).hexdigest()
        assert actual == payload["metadata"]["source_manifest"][name]
    assert "actual selected workspaceSize" in payload["results"]["scratch_policy"]
    assert not payload["results"]["qualified_candidates"]

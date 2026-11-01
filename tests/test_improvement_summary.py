import ast
from pathlib import Path

import pytest

from scripts.summarize_improvements import (cache_text, checks, compatible, fusion_text,
                                           positive, profile_metrics, repeat_metrics,
                                           repeat_text, timing)


def test_improvement_summary_refuses_partial_sweeps_and_gates():
    with pytest.raises(ValueError, match="incomplete"):
        profile_metrics({"results": []}, range(1, 14))
    with pytest.raises(ValueError, match="incomplete"):
        checks([], ("balanced",))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_improvement_summary_refuses_invalid_timings(value):
    with pytest.raises(ValueError):
        positive(value)


def test_improvement_summary_requires_three_actual_paired_rounds():
    with pytest.raises(ValueError):
        timing({"event_round_ms": [1, 2], "wall_round_ms": [1, 2]})
    with pytest.raises(ValueError, match="median"):
        timing({"event_round_ms": [1, 2, 3], "event_ms": 1})
    timing({"event_round_ms": [1, 2, 3], "event_ms": 2,
            "wall_round_ms": [3, 4, 5], "wall_ms": 4})


def test_improvement_drivers_are_syntax_checked_without_importing_cuda():
    for name in ("scripts/benchmark_improvements.py", "scripts/summarize_improvements.py", "kernels/ffn_fusion.py"):
        ast.parse(Path(name).read_text(encoding="utf-8"))


def test_real_pilot_keeps_complete_gates_cache_and_losing_fusion():
    import copy
    import json
    root = Path("results/next")
    pilot = json.loads((root / "balanced-pilot/next_improvements_pilot.json").read_text(encoding="utf-8"))
    cache = json.loads((root / "balanced-pilot/next_improvements_cache.json").read_text(encoding="utf-8"))
    contract = json.loads((root / "balanced-pilot/next_improvements_contracts.json").read_text(encoding="utf-8"))
    fusion = json.loads((root / "ffn-fusion/next_improvements_fusion.json").read_text(encoding="utf-8"))
    compatible([pilot, cache, contract, fusion])
    assert all(r["gate"] for r in profile_metrics(pilot, (2, 8, 13)))
    assert contract["results"]["status"] == "PASS"
    assert "hit-validated-graph" in cache_text(cache)
    assert "**none**" in fusion_text(fusion)
    partial = copy.deepcopy(pilot)
    partial["results"][0]["paired"]["accuracy"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        profile_metrics(partial, (2, 8, 13))


def test_measured_core_and_frozen_runner_match_recorded_sources():
    import hashlib
    import json
    from scripts.verify_next_evidence import verify_current_core
    path = Path("results/next/balanced-pilot/next_improvements_pilot.json")
    assert verify_current_core(path) == 11
    payload = json.loads(path.read_text(encoding="utf-8"))
    from scripts.evidence_sources import measured_source
    for name in ("scripts/run.py", "scripts/benchmark_improvements.py", "scripts/build_clean_kaggle.py"):
        source = measured_source(Path.cwd(), payload["metadata"], name)
        actual = hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest()
        assert actual == payload["metadata"]["source_manifest"][name]
    current_cli = hashlib.sha256(Path("scripts/run.py").read_text(encoding="utf-8").encode()).hexdigest()
    assert current_cli != payload["metadata"]["source_manifest"]["scripts/run.py"]


def full_payloads():
    import json
    root = Path("results/next")
    return [json.loads((root / name).read_text(encoding="utf-8")) for name in (
        "balanced-full/next_improvements_full.json",
        "balanced-full/next_improvements_cache.json",
        "balanced-full/next_improvements_contracts.json",
        "ffn-fusion/next_improvements_fusion.json",
        "balanced-shape2-repeat/next_improvements_shape2_repeat.json")]


def test_full_t4_sweep_keeps_original_and_repeat_performance_misses():
    full, cache, contract, fusion, repeat = full_payloads()
    compatible([full, cache, contract, fusion, repeat])
    rows = profile_metrics(full, range(1, 14))
    assert len(rows) == 13 and sum(r["gate"] for r in rows) == 12
    shape2 = rows[1]
    assert shape2["shape"] == 2 and not shape2["gate"]
    assert shape2["wall_regression"] == pytest.approx(.0695, abs=.00005)
    repeats = repeat_metrics(repeat)
    assert sum(r["within_2_percent"] for r in repeats) == 2
    assert repeats[2]["event_regression"] == pytest.approx(.0363, abs=.00005)
    assert "**2/3**" in repeat_text(repeat)
    assert "original +6.95%" in repeat_text(repeat)
    assert "hit-validated-graph" in cache_text(cache)
    assert contract["results"]["status"] == "PASS"
    assert "**none**" in fusion_text(fusion)


def test_repeat_cannot_hide_missing_workers_checks_or_unlike_sources():
    import copy
    values = full_payloads()
    partial = copy.deepcopy(values[-1])
    partial["results"]["workers"].pop()
    with pytest.raises(ValueError, match="protocol"):
        repeat_metrics(partial)
    partial = copy.deepcopy(values[-1])
    partial["results"]["workers"][0]["accuracy"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        repeat_metrics(partial)
    changed = copy.deepcopy(values)
    changed[-1]["metadata"]["source_manifest"]["user_optimized.py"] = "unlike"
    with pytest.raises(ValueError, match="source/environment"):
        compatible(changed)


@pytest.mark.parametrize("pilot", [False, True])
def test_improvement_reports_regenerate_exactly_and_preserve_lf(tmp_path, monkeypatch, pilot):
    from scripts.summarize_improvements import main
    folder, phase = ("balanced-pilot", "pilot") if pilot else ("balanced-full", "full")
    root = Path("results/next")
    output = tmp_path / "regenerated.md"
    arguments = ["summarize_improvements", "--profiles", str(root / folder / f"next_improvements_{phase}.json"),
                 "--cache", str(root / folder / "next_improvements_cache.json"),
                 "--contracts", str(root / folder / "next_improvements_contracts.json"),
                 "--fusion", str(root / "ffn-fusion/next_improvements_fusion.json"), "--output", str(output)]
    if pilot:
        arguments.append("--pilot")
        committed = root / "improvements_pilot_summary.md"
    else:
        arguments += ["--shape2-repeat", str(root / "balanced-shape2-repeat/next_improvements_shape2_repeat.json")]
        committed = root / "improvements_summary.md"
    monkeypatch.setattr("sys.argv", arguments)
    main()
    assert b"\r" not in committed.read_bytes()
    assert output.read_bytes() == committed.read_bytes()

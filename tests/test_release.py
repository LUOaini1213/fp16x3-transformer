"""CPU structural checks before spending fresh cloud GPU sessions."""
import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.build_clean_kaggle import launcher
from scripts.benchmark_release import baseline, config, copy_candidate, layout_attention


def test_clean_launcher_pins_git_and_uses_normal_imports():
    source = launcher("a" * 40, "fp32")
    ast.parse(source)
    assert "--detach" in source and "scripts.benchmark_release" in source
    assert "pip" not in source and "class UserOptimizedTransformer" not in source


@pytest.mark.parametrize("revision,phase", [("HEAD", "fp32"), ("a" * 40, "oops")])
def test_clean_launcher_rejects_unpinned_inputs(revision, phase):
    with pytest.raises(ValueError):
        launcher(revision, phase)


def test_release_shapes_and_driver_syntax():
    for i in range(1, 14):
        config(i).validate()
    assert baseline(config(2)).config == config(2)
    ast.parse(Path("scripts/benchmark_release.py").read_text(encoding="utf-8"))


@pytest.mark.parametrize("layout", ["current", "bhsd", "bshd", "fold", "chunk8", "chunk16", "chunk32"])
def test_attention_layouts_preserve_fp32_and_causal_math(layout):
    torch.manual_seed(17)
    packed = torch.randn(17, 8, 3 * 16)
    q, k, v = [t.reshape(17, 8, 4, 4).transpose(1, 2) for t in packed.split(16, -1)]
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = layout_attention(layout, q, k, v, is_causal=True)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    torch.testing.assert_close(out, ref)


def test_table_rejects_incomplete_sweep():
    from scripts.summarize_release import fp32_table
    with pytest.raises(AssertionError, match="incomplete"):
        fp32_table({"results": []})


def test_production_adapter_counter_import_is_module_not_package_function():
    import importlib
    import types
    import kernels
    adapter = importlib.import_module("kernels.turing_attention")
    assert isinstance(adapter, types.ModuleType)
    assert callable(kernels.turing_attention)
    assert adapter.TURING_REVISION == "9ef98fcb506bb1e2fe3cece50935e2935bf6b124"
    assert isinstance(adapter._TURING_CALLS, int)
    source = Path("scripts/benchmark_release.py").read_text(encoding="utf-8")
    assert 'adapter = importlib.import_module("kernels.turing_attention")' in source


def test_experiment_construction_keeps_parameter_versions_under_inference():
    with torch.inference_mode():
        base = baseline(config(2))
        model = copy_candidate(config(2), base)
        for parameter in model.parameters():
            assert not parameter.is_inference()
            assert isinstance(parameter._version, int)
        x = torch.randn(1, 128, 128)
        torch.testing.assert_close(model(x), base(x), rtol=.02, atol=.002)


def test_unified_table_recomputes_current_fp32_metrics():
    import json
    from scripts.summarize_release import fp32_table
    payload = json.loads(Path("results/next/release-fp32/next_release_fp32.json").read_text(encoding="utf-8"))
    table = fp32_table(payload)
    assert "3.546×" in table and "All 39" in table
    assert "Default cold peak GiB" in table


def test_attention_report_retains_all_losing_candidates():
    import json
    from scripts.summarize_release import attention_text
    payload = json.loads(Path("results/next/release-attention/next_release_attention.json").read_text(encoding="utf-8"))
    table = attention_text(payload)
    assert "chunk32" in table and "bhsd" in table
    r = payload["results"]["full_eager_timing"]
    assert all(v["event_ms"] >= r["current"]["event_ms"] for v in r.values())


def test_round_validator_rejects_incomplete_or_nonfinite_data():
    from scripts.summarize_release import validate_rounds
    with pytest.raises(AssertionError):
        validate_rounds({"event_round_ms": [1, 2]})
    with pytest.raises(AssertionError):
        validate_rounds({"event_round_ms": [1, 2, float("nan")]})


def test_generated_summary_matches_complete_evidence_and_rejects_partial_flash():
    import copy
    import json
    from scripts.summarize_release import fp32_table, flash_text, attention_text
    report = Path("results/next/clean_release_summary.md")
    assert b"\r" not in report.read_bytes(), "generated Markdown must use portable LF endings"
    summary = report.read_text(encoding="utf-8")
    functions = {"fp32": fp32_table, "flash": flash_text, "attention": attention_text}
    for name, function in functions.items():
        payload = json.loads(Path(f"results/next/release-{name}/next_release_{name}.json").read_text(encoding="utf-8"))
        assert function(payload) in summary
        if name == "flash":
            incomplete = copy.deepcopy(payload)
            incomplete["results"]["full"]["sdpa"]["round_seconds"].pop()
            with pytest.raises(AssertionError):
                function(incomplete)

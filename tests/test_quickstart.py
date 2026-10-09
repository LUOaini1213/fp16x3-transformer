import json
import os
import subprocess
import sys

import pytest
import torch

from scripts.run import PROFILES, configure, environment_report, parse_shapes, run_case


@pytest.fixture(autouse=True)
def preserve_environment(monkeypatch):
    # configure intentionally changes the process; isolate that policy in tests.
    monkeypatch.setattr(os, "environ", os.environ.copy())


def test_profiles_ignore_inherited_experimental_flags():
    os.environ.update(T3_ATTN="turing", T3_X3_BLAS="lt", T3_COMPILE_MODE="max-autotune", T3_CHUNK_BS="1")
    profile = configure("quick")
    assert profile["T3_LINEAR"] == "fp32"
    assert profile["T3_COMPILE"] == profile["T3_CUDAGRAPH"] == "0"
    assert profile["T3_ATTN"] == "sdpa" and profile["T3_AUTOCAST"] == "off"
    assert "T3_COMPILE_MODE" not in os.environ and "T3_CHUNK_BS" not in os.environ
    assert configure("steady")["T3_LINEAR"] == "fp16x3"
    assert os.environ["T3_COMPILE"] == "auto" and os.environ["T3_CUDAGRAPH"] == "1"


def test_balanced_keeps_steady_arithmetic_and_guards_without_inductor():
    os.environ.update(T3_AUTOCAST="fp16", T3_X3_GUARD="off", T3_X3_BLAS="lt",
                      T3_COMPILE_MODE="max-autotune", T3_ATTN="turing")
    balanced = configure("balanced")
    steady = configure("steady")
    assert balanced["T3_COMPILE"] == "0"
    assert balanced["T3_CUDAGRAPH"] == "1" and balanced["T3_LINEAR"] == "fp16x3"
    assert balanced["T3_AUTOCAST"] == "off" and balanced["T3_X3_GUARD"] == "static+first"
    assert {k: v for k, v in balanced.items() if k != "T3_COMPILE"} == {
        k: v for k, v in steady.items() if k != "T3_COMPILE"}
    assert "T3_COMPILE_MODE" not in os.environ


def test_unknown_profile_does_not_clear_environment():
    before = dict(os.environ)
    with pytest.raises(ValueError, match="unknown mode"):
        configure("unknown")
    assert dict(os.environ) == before


@pytest.mark.parametrize("value", ["0", "14", "4-2", "1-2-3", "", "a"])
def test_bad_shape_ranges_are_rejected(value):
    with pytest.raises(ValueError):
        parse_shapes(value)


def test_shape_ranges_and_duplicates():
    assert parse_shapes("2,1-3,13") == [2, 1, 3, 13]


def test_cpu_doctor_makes_no_acceleration_claim():
    env = environment_report("cpu")
    assert env["device"] == "cpu"
    assert any("not GPU performance" in w for w in env["warnings"])


@pytest.mark.parametrize("mode", tuple(PROFILES))
def test_profiles_pass_official_gate_on_cpu(mode):
    row = run_case(2, mode, "cpu", repeats=1)
    assert row["accuracy"]["passed"] and row["accuracy"]["failed"] == 0
    assert len(row["accuracy_trials"]) == 3
    assert not row["dispatch"]["compiled"] and not row["dispatch"]["graph"]
    if mode == "quick":
        assert not row["dispatch"]["x3"]


def test_balanced_retains_mutation_checks_and_output_ownership(monkeypatch):
    import torch_transformer_benchmark as official
    from user_optimized import UserOptimizedTransformer

    configure("balanced")
    compile_calls = []

    def forbidden_compile(*args, **kwargs):
        compile_calls.append(True)
        raise RuntimeError("balanced must not call torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)
    torch.manual_seed(91)
    cfg = official.TransformerConfig(2, 16, 32, 4, 32, 2, True)
    base = official.BaselineTransformer(cfg).eval()
    model = UserOptimizedTransformer(cfg).eval()
    official.copy_model_weights(base, model, strict=True)
    assert set(base.state_dict()) == set(model.state_dict())
    assert not model._compile_ok and model._cudagraph
    x = torch.randn(2, 16, 32)
    mask = torch.ones(2, 16, dtype=torch.bool)

    def checked(xx):
        ref, out = base(xx, mask), model(xx, mask)
        assert out.dtype == torch.float32 and out.shape == xx.shape
        assert official.compare_outputs(ref, out, rtol=.02, atol=.002).passed
        return out

    with torch.inference_mode():
        old = checked(x)
        saved = old.clone()
        changed = checked(x * .5)
        assert torch.equal(old, saved) and not torch.equal(changed, old)
        mask[0, 7:] = False
        padded = checked(x)
        assert torch.count_nonzero(padded[0, 7:]) == 0
        mask.fill_(True)
        base.layers[0].norm1.bias.add_(.125)
        model.layers[0].norm1.bias.add_(.125)
        checked(x)
    assert not compile_calls and model._compiled is None


def test_balanced_cli_sweep_uses_fresh_workers_and_correctness_only_cpu(tmp_path):
    output = tmp_path / "balanced.json"
    process = subprocess.run([sys.executable, "-m", "scripts.run", "--mode", "balanced",
                              "--device", "cpu", "--shapes", "2,1", "--repeats", "1",
                              "--output", str(output)], capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["environment"]["device"] == "cpu"
    assert any("not GPU performance" in w for w in payload["environment"]["warnings"])
    assert [row["shape"] for row in payload["results"]] == [2, 1]
    for row in payload["results"]:
        assert row["mode"] == "balanced" and row["accuracy"]["passed"]
        assert len(row["accuracy_trials"]) == 3
        assert row["profile"]["T3_COMPILE"] == "0" and row["profile"]["T3_CUDAGRAPH"] == "1"
        assert not row["dispatch"]["compiled"] and not row["dispatch"]["graph"]
        assert "no paired speedup claim" in row["timing_scope"]
        assert output.with_name(f"balanced_shape{row['shape']}.json").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for balanced dispatch")
def test_balanced_cuda_forward_does_not_attempt_inductor():
    # A fresh interpreter isolates import-time kernel settings from other tests.
    code = '''
import torch
from scripts.run import run_case
attempts = []
def forbidden(*args, **kwargs):
    attempts.append(True)
    raise RuntimeError("unexpected Inductor call")
torch.compile = forbidden
row = run_case(2, "balanced", "cuda", repeats=1)
assert not attempts, "balanced attempted torch.compile even if it fell back"
assert row["accuracy"]["passed"] and len(row["accuracy_trials"]) == 3
assert not row["dispatch"]["compiled"]
assert row["profile"]["T3_CUDAGRAPH"] == "1"
'''
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr


def test_large_cpu_oracle_refused():
    with pytest.raises(ValueError, match="CPU"):
        run_case(6, "quick", "cpu")

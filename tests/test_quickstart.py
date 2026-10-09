import os

import pytest

from scripts.run import configure, environment_report, parse_shapes, run_case


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


@pytest.mark.parametrize("mode", ["quick", "steady"])
def test_profiles_pass_official_gate_on_cpu(mode):
    row = run_case(2, mode, "cpu", repeats=1)
    assert row["accuracy"]["passed"] and row["accuracy"]["failed"] == 0
    assert len(row["accuracy_trials"]) == 3
    assert not row["dispatch"]["compiled"] and not row["dispatch"]["graph"]
    if mode == "quick":
        assert not row["dispatch"]["x3"]


def test_large_cpu_oracle_refused():
    with pytest.raises(ValueError, match="CPU"):
        run_case(6, "quick", "cpu")

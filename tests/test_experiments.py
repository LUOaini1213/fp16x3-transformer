"""Cloud experiments must be structurally valid before spending a GPU session."""
import ast
import os

import pytest

from scripts.build_kaggle_selfcontained import build_selector
from scripts import benchmark_next
from torch_transformer_benchmark import TransformerConfig


def test_selector_quotes_and_windows_paths(monkeypatch):
    value = ' {"path": "C:\\tmp\\data", "text": "\'quoted\'"} '
    monkeypatch.delenv("T3_TEST_VALUE", raising=False)
    exec(build_selector("next", ["T3_TEST_VALUE=" + value]))
    assert os.environ["T3_TEST_VALUE"] == value


@pytest.mark.parametrize("value", ["no_equal", "=empty_name"])
def test_selector_rejects_invalid_env(value):
    with pytest.raises(ValueError):
        build_selector("next", [value])


def test_experiment_shapes_match_official_columns():
    expected = {2: (1, 128, 128, 4, 128, 4), 3: (4, 128, 128, 4, 128, 4),
                6: (10000, 128, 128, 4, 128, 4), 8: (64, 128, 1024, 4, 1024, 4),
                12: (64, 32, 128, 4, 128, 4), 13: (64, 1024, 128, 4, 128, 4)}
    assert benchmark_next.NEXT_SHAPES == expected
    for dims in expected.values():
        TransformerConfig(*dims, True).validate()


def test_driver_parses_without_running_cuda():
    from pathlib import Path
    ast.parse(Path(benchmark_next.__file__).read_text(encoding="utf-8"))


def test_evidence_log_normalization():
    from scripts.import_next_results import human_log
    assert human_log('[{"data":"hello\\n"},{"data":"world\\n"}]') == "hello\nworld\n"
    assert human_log("already plain\n") == "already plain\n"


def test_evidence_never_silently_overwrites(tmp_path):
    from scripts.import_next_results import store_generated
    target = tmp_path / "audit.json"
    store_generated(target, b"first")
    store_generated(target, b"first")
    with pytest.raises(ValueError):
        store_generated(target, b"other")
    assert target.read_bytes() == b"first"


def test_log_recovery_uses_last_source_stamped_payload():
    from scripts.recover_next_log import last_payload
    log = ('noise\nNEXT_FLASH {"metadata":{"source_manifest":{}},"results":{"round":1}}\n'
           'NEXT_FLASH {"metadata":{"source_manifest":{}},"results":{"round":3}}\n')
    assert last_payload(log, "next_flash")["results"]["round"] == 3
    with pytest.raises(ValueError):
        last_payload("NEXT_FLASH {\"results\":{}}\n", "next_flash")
    with pytest.raises(ValueError):
        last_payload("no result", "next_flash")


def test_lt_cpu_fallback_never_builds(monkeypatch):
    import torch
    from kernels import cublaslt_backend as backend

    def forbidden():
        raise AssertionError("CPU execution must not compile a CUDA extension")

    monkeypatch.setattr(backend, "_lt_load", forbidden)
    assert backend.lt_candidate_matmul(torch.randn(2, 3), torch.randn(4, 3)) is None


def test_lt_failed_build_is_cached_and_falls_back(monkeypatch):
    import torch.utils.cpp_extension
    from kernels import cublaslt_backend as backend
    monkeypatch.setattr(backend, "_LT_EXTENSION", None)
    monkeypatch.setattr(backend, "_LT_FAILED", False)
    calls = []

    def failed(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("no compiler")

    monkeypatch.setattr(torch.utils.cpp_extension, "load_inline", failed)
    assert backend._lt_load() is None
    assert backend._lt_load() is None
    assert len(calls) == 1


def test_turing_cpu_and_missing_extension_fallback(monkeypatch):
    import torch
    import importlib
    backend = importlib.import_module("kernels.turing_attention")
    q = torch.randn(1, 2, 16, 64)
    expected = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True, scale=.125)
    assert torch.equal(backend.turing_attention(q, q, q, .125), expected)
    monkeypatch.setattr(backend, "_TURING_LOOKED_UP", False)
    monkeypatch.setattr(backend, "_TURING_EXTENSION", None)
    monkeypatch.setattr(backend, "turing_eligible", lambda *args: True)

    def missing(name):
        raise ImportError("not installed")

    monkeypatch.setattr(backend.importlib, "import_module", missing)
    assert torch.equal(backend.turing_attention(q, q, q, .125), expected)


@pytest.mark.parametrize("change", ["fp32", "wrong_head", "short", "stream", "capture", "device", "grad"])
def test_turing_rejects_unsafe_contracts(monkeypatch, change):
    import importlib
    from types import SimpleNamespace
    import torch
    backend = importlib.import_module("kernels.turing_attention")
    q = SimpleNamespace(ndim=4, is_cuda=True, device=torch.device("cuda:0"),
                        dtype=torch.float16, shape=(1, 16, 8192, 64), requires_grad=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: 1)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device: 1)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert backend.turing_eligible(q, q, q)
    if change == "fp32":
        q.dtype = torch.float32
    elif change == "wrong_head":
        q.shape = (1, 16, 8192, 32)
    elif change == "short":
        q.shape = (1, 16, 128, 64)
    elif change == "stream":
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: 2)
    elif change == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif change == "device":
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    elif change == "grad":
        q.requires_grad = True
    assert not backend.turing_eligible(q, q, q)


def test_turing_noncausal_never_loads_extension(monkeypatch):
    import importlib
    import torch
    backend = importlib.import_module("kernels.turing_attention")

    def forbidden():
        raise AssertionError("non-causal path must not load the optional extension")

    monkeypatch.setattr(backend, "_turing_load", forbidden)
    q = torch.randn(1, 2, 16, 64)
    expected = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=False, scale=.125)
    assert torch.equal(backend.turing_attention(q, q, q, .125, False), expected)

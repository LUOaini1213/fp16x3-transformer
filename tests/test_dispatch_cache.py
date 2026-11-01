import copy
import json

import pytest
import torch

from kernels.dispatch_cache import key_for, lookup, remember
from scripts.run import configure
from torch_transformer_benchmark import BaselineTransformer, TransformerConfig, compare_outputs
from user_optimized import UserOptimizedTransformer


def test_cache_is_explicit_json_and_checks_complete_context(tmp_path):
    path = tmp_path / "dispatch.json"
    value = {"schema": 1, "gpu": "T4", "torch": "2.11", "shape": [1, 128, 128],
             "source": {"model": "source-hash"}, "dtype": "float32", "stride": [16384, 128, 1]}
    assert lookup(path, value) is None
    assert remember(path, value, "graph") and lookup(path, value) == "graph"
    payload = json.loads(path.read_text())
    assert payload["entries"][key_for(value)]["choice"] == "graph"
    for name in ("gpu", "torch", "shape", "source", "dtype", "stride"):
        changed = copy.deepcopy(value)
        changed[name] = "changed"
        assert lookup(path, changed) is None
    assert not remember(path, value, "compiled")
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("data", [b"not JSON", b'{}', b'{"schema":999,"entries":{}}',
                                 b'{"schema":1,"entries":[]}', b'\xff'])
def test_corrupt_or_foreign_cache_falls_back_without_overwriting(tmp_path, data):
    path = tmp_path / "cache.json"
    path.write_bytes(data)
    assert lookup(path, {}) is None and not remember(path, {}, "eager")
    assert path.read_bytes() == data


def test_no_executable_or_unknown_choices_are_loaded(tmp_path):
    path = tmp_path / "cache.json"
    value = {"shape": [2, 16, 32]}
    assert remember(path, value, "eager")
    payload = json.loads(path.read_text())
    payload["entries"][key_for(value)]["choice"] = "__import__('os').system('anything')"
    path.write_text(json.dumps(payload))
    assert lookup(path, value) is None


def test_cpu_cache_option_does_not_write_or_bypass_weight_guards(monkeypatch, tmp_path):
    monkeypatch.setattr("os.environ", __import__("os").environ.copy())
    path = tmp_path / "dispatch.json"
    configure("balanced", path)
    cfg = TransformerConfig(2, 16, 32, 4, 32, 2, True)
    model = UserOptimizedTransformer(cfg).eval()
    base = BaselineTransformer(cfg).eval()
    base.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(2, 16, 32)
    with torch.inference_mode():
        model(x)
        model._tuned = True
        model._tune_cache_attempted = True
        model.layers[0].norm1.bias.add_(.125)
        base.layers[0].norm1.bias.add_(.125)
        assert compare_outputs(base(x), model(x), rtol=.02, atol=.002).passed
    assert not model._tuned and not model._tune_cache_attempted
    assert not path.exists()
    model._tuned = True
    model._drop_graph()  # ordinary rejection must not restart tuning forever
    assert model._tuned


def test_cli_cache_setting_is_not_inherited_implicitly(monkeypatch, tmp_path):
    monkeypatch.setattr("os.environ", __import__("os").environ.copy())
    import os
    configure("balanced", tmp_path / "cache.json")
    assert os.environ["T3_TUNE_CACHE"] == str((tmp_path / "cache.json").resolve())
    configure("balanced")
    assert "T3_TUNE_CACHE" not in os.environ

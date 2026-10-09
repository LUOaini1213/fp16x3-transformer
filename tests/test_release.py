"""CPU structural checks before spending fresh cloud GPU sessions."""
import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.build_clean_kaggle import launcher
from scripts.benchmark_release import baseline, config, layout_attention


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

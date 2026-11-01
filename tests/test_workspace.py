import ast
from pathlib import Path

import pytest
import torch

from scripts.build_clean_kaggle import launcher
from scripts.workspace_provider import WorkspaceProvider, workspace_source


def test_workspace_source_changes_only_experimental_copy():
    path = Path("kernels/cublaslt_probe.cpp")
    before = path.read_bytes()
    generated = workspace_source(path.read_text(encoding="utf-8"))
    assert "scratch.data_ptr(), scratch.numel()" in generated
    assert "int64_t(p->candidates[index].workspaceSize)" in generated
    assert "key(x,w,bytes)" in generated
    assert "Plan>(x,w,limit,bytes)" in generated
    assert "at::cuda::getCurrentCUDAStream" in generated
    assert "workspace=at::empty" not in generated
    assert "size_t bytes = 0" not in generated
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="partial"):
        workspace_source("an unexpected source revision")


def test_workspace_cpu_falls_back_without_native_work():
    provider = WorkspaceProvider(None, 1 << 20)
    assert provider(torch.randn(2, 3), torch.randn(4, 3)) is None
    assert not provider.searches and provider.calls == 0
    with pytest.raises(ValueError):
        WorkspaceProvider(None, 7)


@pytest.mark.parametrize("phase", ["quick", "workspace"])
def test_usability_launcher_keeps_exact_commit_and_normal_imports(phase):
    source = launcher("a" * 40, phase)
    ast.parse(source)
    assert "scripts.benchmark_usability" in source and "pip" not in source
    ast.parse(Path("scripts/benchmark_usability.py").read_text(encoding="utf-8"))

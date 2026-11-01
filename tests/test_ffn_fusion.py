import pytest
import torch

from kernels import fp16x3 as k
from kernels.ffn_fusion import FusionTransformer, TILES, fused_ffn_split
from torch_transformer_benchmark import BaselineTransformer, TransformerConfig, compare_outputs


@pytest.mark.parametrize("tile", tuple(TILES))
def test_fused_epilogue_cpu_emulation_preserves_exact_gelu_and_split(tile):
    torch.manual_seed(47)
    a = torch.randn(67, 128)
    a3 = k._x3_split_ref(a)[:, :384]
    w3 = torch.randn(128, 384).half()
    bias = torch.randn(128)
    out = fused_ffn_split(a3, w3, bias, .03125, tile)
    ref = k._x3_act_split_ref(a3.float() @ w3.float().t(), True, .03125, bias)
    assert torch.equal(out, ref) and out.dtype == torch.float16


def test_fusion_class_passes_original_gate_and_strict_weight_copy_on_cpu(monkeypatch):
    monkeypatch.setenv("T3_COMPILE", "0")
    monkeypatch.setenv("T3_CUDAGRAPH", "0")
    monkeypatch.setenv("T3_LINEAR", "fp16x3")
    cfg = TransformerConfig(2, 16, 128, 4, 128, 2, True)
    base = BaselineTransformer(cfg).eval()
    candidate = FusionTransformer(cfg).eval()
    candidate.load_state_dict(base.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.randn(2, 16, 128)
        assert compare_outputs(base(x), candidate(x), rtol=.02, atol=.002).passed
    assert candidate.fused_calls == 2
    assert set(candidate.state_dict()) == set(base.state_dict())


@pytest.mark.parametrize("tile", ["unknown", ""])
def test_fusion_rejects_unknown_config(tile):
    with pytest.raises(ValueError):
        fused_ffn_split(torch.zeros(2, 384).half(), torch.zeros(128, 384).half(), None, 1., tile)

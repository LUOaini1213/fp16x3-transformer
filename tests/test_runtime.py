"""Mutation-sensitive caches must preserve the reference model's behavior."""

import pytest
import torch

from torch_transformer_benchmark import BaselineTransformer, TransformerConfig
from user_optimized import UserOptimizedTransformer


def models(monkeypatch, **env):
    for key, value in {"T3_COMPILE": "0", "T3_CUDAGRAPH": "0", **env}.items():
        monkeypatch.setenv(key, value)
    torch.manual_seed(47)
    cfg = TransformerConfig(2, 16, 32, 4, 32, 2, True)
    base = BaselineTransformer(cfg).eval()
    opt = UserOptimizedTransformer(cfg).eval()
    opt.load_state_dict(base.state_dict(), strict=True)
    return base, opt, torch.randn(2, 16, 32)


def assert_matches(base, opt, x, mask):
    with torch.inference_mode():
        ref, out = base(x, mask), opt(x, mask)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    return out


def test_repeated_mask_reuses_reduction(monkeypatch):
    _, opt, _ = models(monkeypatch)
    mask = torch.ones(2, 16, dtype=torch.bool)
    original = torch.Tensor.all
    reductions = []

    def counted(tensor, *args, **kwargs):
        if tensor is mask:
            reductions.append(1)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "all", counted)
    assert opt._all_valid(mask)
    assert opt._all_valid(mask)
    assert len(reductions) == 1


@pytest.mark.parametrize("write_context", [torch.no_grad, torch.inference_mode])
def test_mask_alias_mutation_invalidates_cache(monkeypatch, write_context):
    base, opt, x = models(monkeypatch)
    mask = torch.ones(2, 16, dtype=torch.bool)
    assert_matches(base, opt, x, mask)
    with write_context():
        mask.view(-1)[8:16].fill_(False)
    out = assert_matches(base, opt, x, mask)
    assert (out[0, 8:] == 0).all()
    mask.fill_(True)
    assert_matches(base, opt, x, mask)
    assert opt._mask_valid


def test_new_mask_and_changed_input_are_not_cached_outputs(monkeypatch):
    base, opt, x = models(monkeypatch)
    mask = torch.ones(2, 16, dtype=torch.bool)
    before = assert_matches(base, opt, x, mask).clone()
    different_mask = mask.clone()
    different_mask[1, 5:] = False
    assert_matches(base, opt, x, different_mask)
    after = assert_matches(base, opt, x * 2.0, mask)
    assert not torch.allclose(before, after)


def test_inference_masks_are_rechecked(monkeypatch):
    base, opt, x = models(monkeypatch)
    with torch.inference_mode():
        mask = torch.ones(2, 16, dtype=torch.bool)
        assert mask.is_inference()
        assert_matches(base, opt, x, mask)
        mask[0, 6:] = False
        out = assert_matches(base, opt, x, mask)
        assert (out[0, 6:] == 0).all()
    assert opt._mask_tensor is None


def test_external_mask_write_has_explicit_uncached_path(monkeypatch):
    base, opt, x = models(monkeypatch, T3_MASK_CACHE="0")
    mask = torch.ones(2, 16, dtype=torch.bool)
    assert_matches(base, opt, x, mask)
    mask.data[0, 6:] = False  # deliberately bypasses the mutation counter
    assert_matches(base, opt, x, mask)


def test_norm_mutation_rechecks_fp16_range_bound(monkeypatch):
    base, opt, x = models(monkeypatch)
    assert_matches(base, opt, x, None)
    assert opt._x3_on
    with torch.no_grad():
        base.layers[0].norm1.weight.fill_(1e5)
        opt.layers[0].norm1.weight.fill_(1e5)
    assert_matches(base, opt, x, None)
    assert not opt._x3_on


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("mode", ["fp16x3", "fp32"])
def test_packing_handles_holes_left_padding_and_empty_rows(monkeypatch, causal, mode):
    base, opt, x = models(monkeypatch, T3_LINEAR=mode)
    # Rebuild with more rows so repeated lengths and all-invalid rows coexist.
    cfg = TransformerConfig(5, 16, 32, 4, 32, 2, causal)
    base = BaselineTransformer(cfg).eval()
    opt = UserOptimizedTransformer(cfg).eval()
    opt.load_state_dict(base.state_dict(), strict=True)
    x = torch.randn(5, 16, 32)
    mask = torch.zeros(5, 16, dtype=torch.bool)
    mask[0, 4:] = True
    mask[1, ::2] = True
    mask[2, 1::2] = True
    mask[4, :1] = True
    out = assert_matches(base, opt, x, mask)
    assert (out[~mask] == 0).all()
    assert [len(rows) for rows, _ in opt._padding_groups] == [1, 2, 1]
    # A mask mutation invalidates both the all-valid cache and the pack plan.
    mask[0].fill_(False)
    assert_matches(base, opt, x, mask)


def test_padded_path_never_calls_dense_bias_fallback(monkeypatch):
    base, opt, x = models(monkeypatch, T3_LINEAR="fp32")
    attention = opt._attention

    def assert_compact(attn, xx, mask, causal, all_valid):
        assert all_valid and mask is None
        return attention(attn, xx, mask, causal, all_valid)

    monkeypatch.setattr(opt, "_attention", assert_compact)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[0, 3:] = False
    mask[1, ::2] = False
    assert_matches(base, opt, x, mask)


def test_packing_accepts_strided_inputs_masks_and_small_chunks(monkeypatch):
    base, opt, x = models(monkeypatch, T3_CHUNK_BS="1")
    x = x.transpose(1, 2).contiguous().transpose(1, 2)
    storage = torch.ones(2, 32, dtype=torch.bool)
    mask = storage[:, ::2]
    mask[:, 9:] = False
    assert not x.is_contiguous() and not mask.is_contiguous()
    assert_matches(base, opt, x, mask)


def test_eager_failure_is_not_retried_as_a_compile_failure(monkeypatch):
    _, opt, x = models(monkeypatch, T3_LINEAR="fp32")
    calls = []

    def fail(*args):
        calls.append(1)
        raise ValueError("invalid eager input")

    monkeypatch.setattr(opt, "_run_full", fail)
    with pytest.raises(ValueError, match="invalid eager input"):
        opt(x)
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["fp16x3", "fp32"])
@pytest.mark.parametrize("site", ["q_proj", "out_proj"])
def test_parameter_replacement_with_same_version_rebuilds_cache(monkeypatch, mode, site):
    base, opt, x = models(monkeypatch, T3_LINEAR=mode, T3_FUSED_QKV="1")
    assert_matches(base, opt, x, None)
    lin = getattr(opt.layers[0].attention, site)
    old_version = lin.weight._version
    replacement = torch.nn.Parameter(torch.randn_like(lin.weight) * 0.1)
    while replacement._version < old_version:
        with torch.no_grad():
            replacement.add_(0)
    assert replacement._version == old_version
    lin.weight = replacement
    getattr(base.layers[0].attention, site).weight = torch.nn.Parameter(replacement.detach().clone())
    assert_matches(base, opt, x, None)

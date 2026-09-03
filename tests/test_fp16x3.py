"""Tests for the fp16x3 linear path (kernels/fp16x3.py) and its integration in
user_optimized.py. Everything runs on CPU through the exact emulation; the
CUDA cases run the Triton kernels and cuBLAS path when a GPU is present.

    pytest -q tests/
"""
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kernels import fp16x3 as K  # noqa: E402

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _rows_with_edges(m, n, device):
    x = torch.randn(m, n, device=device)
    x[0] = 0.0                                   # zero row: LayerNorm -> beta
    x[1] = 3.0                                   # constant row: zero variance
    x[2] = torch.randn(n, device=device) * 1e-6  # lo parts in the subnormal range
    return x


def _distinct_halves(a3, n):
    """(hi, lo) column blocks of a split, whichever operand order is active."""
    a3 = a3.float()
    if K._X3_LO_FIRST:
        return a3[:, n:2 * n], a3[:, :n]
    return a3[:, :n], a3[:, 2 * n:]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [32, 96, 100, 1000, 1024])
def test_split_kernels_match_reference(device, n):
    x = _rows_with_edges(67, n, device)
    r = torch.randn_like(x)
    w = torch.rand(n, device=device) + 0.5
    b = torch.randn(n, device=device)
    tol = 2e-3  # hi halves may differ by one fp16 ulp between implementations
    got = K.x3_ln_split(x, w, b, 1e-5).float()
    ref = K._x3_ln_split_ref(x, w, b, 1e-5).view(-1, 3 * n + K._X3_TAIL).float()
    assert got.shape == ref.shape and (got - ref).abs().max().item() < tol
    s, a3 = K.x3_add_ln_split(x, r, w, b, 1e-5)
    s_ref, a3_ref = K._x3_add_ln_split_ref(x, r, w, b, 1e-5)
    assert (s - s_ref).abs().max().item() < 1e-5
    assert (a3.float() - a3_ref.view(-1, 3 * n + K._X3_TAIL).float()).abs().max().item() < tol
    for gelu in (True, False):
        got = K.x3_act_split(x, gelu).float()
        ref = K._x3_act_split_ref(x, gelu).view(-1, 3 * n + K._X3_TAIL).float()
        assert (got - ref).abs().max().item() < tol


@pytest.mark.parametrize("device", DEVICES)
def test_rows_wider_than_the_kernel_limit_use_the_reference(device):
    x = torch.randn(4, K.MAX_X3_WIDTH + 1, device=device)
    assert not K.x3_can_use(x)
    assert K.x3_act_split(x, False).shape == (4, 3 * (K.MAX_X3_WIDTH + 1) + K._X3_TAIL)


def test_split_reconstructs_fp32_to_2pow_minus_22():
    x = torch.randn(1000, 128) * 5
    hi, lo = _distinct_halves(K._x3_split_ref(x), 128)
    err = (hi + lo - x).abs()
    assert (err <= 2.0 ** -21 * x.abs() + 1e-7).all()


def test_fp16_edge_is_where_the_guard_lives():
    x = torch.tensor([[65504.0, 60000.0, -65504.0, 1e-8, 0.0, 1.0, -1.0, 2.5]])
    hi, lo = _distinct_halves(K._x3_split_ref(x), 8)
    assert torch.isfinite(hi).all() and torch.isfinite(lo).all()
    x = torch.tensor([[65520.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    hi, _ = _distinct_halves(K._x3_split_ref(x), 8)
    assert torch.isinf(hi[0, 0])  # beyond fp16 range: the model's guard must refuse this


@pytest.mark.parametrize("scale", [0.0, 1.0, 1e30, 1e-30])
def test_prepare_handles_any_finite_weight_scale(scale):
    """The contract is on the product, not the parts: whatever power-of-two
    scale x3_prepare picks (it must also keep the folded bias inside fp16), the
    emulated GEMM reproduces a @ w^T + b to fp32-class relative accuracy of the
    output. At 1e30 the bias is below fp32 resolution of the output and may be
    dropped; at 1e-30 the weights are."""
    torch.manual_seed(0)
    w = torch.randn(8, 16) * scale
    bias = torch.randn(8)
    a = torch.randn(64, 16)
    w3, b, inv = K.x3_prepare(w, bias)
    assert b is None
    assert w3.shape == (8, 48 + K._X3_TAIL) and w3.dtype == torch.float16 and torch.isfinite(w3).all()
    ref = a.double() @ w.double().t() + bias.double()
    out = K.x3_linear(K._x3_split_ref(a).view(-1, 48 + K._X3_TAIL), w3, None, inv).double()
    assert (out - ref).abs().max().item() <= 2.0 ** -19 * ref.abs().max().item() + 1e-30
    if scale == 1.0:
        hi, lo = w3[:, :16].float(), w3[:, 16:32].float()
        assert ((hi + lo) * inv - w).abs().max().item() <= 2.0 ** -21 * w.abs().max().item()
        tail = w3[:, 48:].float()
        assert ((tail[:, 0] + tail[:, 1]) * inv - bias).abs().max().item() <= 2.0 ** -21 * bias.abs().max().item()
        assert (tail[:, 2:] == 0).all()
        i = w.abs().argmax()   # the lo part of the largest entry is a normal fp16 number
        assert lo.flatten()[i].abs().item() == 0.0 or lo.flatten()[i].abs().item() >= 6.1e-5


def test_prepare_does_not_raise_on_nan_or_inf():
    w = torch.full((4, 8), float("nan"))
    w3, b, inv = K.x3_prepare(w, None)
    assert inv == 1.0 and torch.isnan(w3[:, :24]).all()
    w = torch.full((4, 8), float("inf"))
    w3, b, inv = K.x3_prepare(w, None)
    assert inv == 1.0


def test_prepare_keeps_a_large_bias_inside_fp16():
    w = torch.randn(8, 16) * 0.03
    bias = torch.full((8,), 5000.0)
    w3, b, inv = K.x3_prepare(w, bias)
    tail = w3[:, 48:].float()
    assert torch.isfinite(tail).all()
    assert ((tail[:, 0] + tail[:, 1]) * inv - bias).abs().max().item() < 1e-2


def test_consumer_folded_scale_matches_reference():
    """The model's calling convention: GEMM returns s * value, the consumer
    multiplies by 1/s."""
    torch.manual_seed(3)
    a = torch.nn.functional.layer_norm(torch.randn(256, 64), (64,))
    w = torch.randn(32, 64) * 0.03
    bias = torch.randn(32) * 0.1
    ref = a @ w.t() + bias
    w3, _, inv = K.x3_prepare(w, bias)
    scaled = K.x3_linear(K._x3_split_ref(a).view(-1, 3 * 64 + K._X3_TAIL), w3, None, inv, False)
    assert (scaled * inv - ref).abs().max().item() < 1e-5
    # the split kernels undo the scale on their input
    a3 = K.x3_act_split(scaled, False, inv)
    hi, lo = _distinct_halves(a3, 32)
    assert (hi + lo - ref).abs().max().item() < 1e-5
    x = torch.randn(256, 32)
    s_, _ = K.x3_add_ln_split(x, scaled, torch.ones(32), torch.zeros(32), 1e-5, inv)
    assert (s_ - (x + ref)).abs().max().item() < 1e-5


@pytest.mark.parametrize("k", [128, 1024])
def test_emulated_linear_is_fp32_class(k):
    torch.manual_seed(k)
    a = torch.nn.functional.layer_norm(torch.randn(2048, k), (k,))
    w = (torch.rand(1024, k) * 2 - 1) / math.sqrt(k)
    bias = (torch.rand(1024) * 2 - 1) / math.sqrt(k)
    ref = a.double() @ w.double().t() + bias.double()
    a3 = K._x3_split_ref(a).view(-1, 3 * k + K._X3_TAIL)
    out = K.x3_linear(a3, *K.x3_prepare(w, bias))
    assert (out.double() - ref).abs().max().item() < 3e-6   # the committed "1.2e-6 on CPU" claim, with margin


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
def test_cuda_linear_error_regression_fence():
    dev = "cuda"
    assert K.x3_available(dev)
    torch.manual_seed(1024)
    a = torch.nn.functional.layer_norm(torch.randn(2048, 1024, device=dev), (1024,))
    w = (torch.rand(1024, 1024, device=dev) * 2 - 1) / 32
    ref = a.double() @ w.double().t()
    out = K.x3_linear(K.x3_act_split(a, False), *K.x3_prepare(w, None))
    assert (out.double() - ref).abs().max().item() < 1e-4


# --------------------------------------------------------------------------
# Integration: the model's guard, weight tracking, chunked path.
# --------------------------------------------------------------------------
def _models(monkeypatch, gamma=None, env=None):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("T3_COMPILE", "0")
    from torch_transformer_benchmark import (BaselineTransformer, TransformerConfig,
                                             copy_model_weights, generate_random_case)
    from user_optimized import UserOptimizedTransformer
    cfg = TransformerConfig(batch_size=2, seq_len=16, d_model=32, num_heads=4,
                            ffn_dim=32, num_layers=2, causal=True)
    cfg.validate()
    torch.manual_seed(0)
    base = BaselineTransformer(cfg).eval()
    if gamma is not None:
        with torch.no_grad():
            base.layers[0].norm1.weight.fill_(gamma)
    opt = UserOptimizedTransformer(cfg)
    copy_model_weights(base, opt, strict=True)
    opt.eval()
    x, m = generate_random_case(cfg, torch.device("cpu"), torch.float32, 0, 0.0, 1.0)
    return base, opt, x, m


def test_static_bound_refuses_overflowing_weights(monkeypatch):
    base, opt, x, m = _models(monkeypatch, gamma=1e5, env={"T3_LINEAR": "fp16x3"})
    with torch.inference_mode():
        out = opt(x, m)
        ref = base(x, m)
    assert not opt._x3_on and opt._linear_policy == "fp32"
    assert torch.isfinite(out).all() and torch.allclose(out, ref, atol=1e-4)


def test_first_forward_check_catches_what_the_bound_missed(monkeypatch):
    base, opt, x, m = _models(monkeypatch, gamma=1e5, env={"T3_LINEAR": "fp16x3"})
    monkeypatch.setattr(opt, "_x3_static_bound", lambda: 0.0)  # pretend the bound passed
    with torch.inference_mode():
        out = opt(x, m)
        ref = base(x, m)
    assert not opt._x3_on
    assert torch.isfinite(out).all() and torch.allclose(out, ref, atol=1e-4)


def test_guard_off_really_is_off(monkeypatch):
    base, opt, x, m = _models(monkeypatch, gamma=1e5,
                              env={"T3_LINEAR": "fp16x3", "T3_X3_GUARD": "off"})
    with torch.inference_mode():
        out = opt(x, m)
    assert opt._x3_on and not torch.isfinite(out).all()  # proves the guard is load-bearing


def test_moderate_gamma_keeps_the_path_on(monkeypatch):
    base, opt, x, m = _models(monkeypatch, gamma=1e2, env={"T3_LINEAR": "fp16x3"})
    with torch.inference_mode():
        out = opt(x, m)
        ref = base(x, m)
    assert opt._x3_on
    assert (out - ref).abs().max().item() < 2e-3


def test_weight_update_after_first_forward_is_tracked(monkeypatch):
    base, opt, x, m = _models(monkeypatch, env={"T3_LINEAR": "fp16x3"})
    with torch.inference_mode():
        opt(x, m)
        new_w = torch.randn_like(base.layers[0].attention.out_proj.weight) * 0.1
        base.layers[0].attention.out_proj.weight.copy_(new_w)
        opt.layers[0].attention.out_proj.weight.copy_(new_w)
        out = opt(x, m)
        ref = base(x, m)
    assert (out - ref).abs().max().item() < 2e-3


@pytest.mark.parametrize("sites", ["qkv,ffn_in", "qkv,ffn_in,ffn_out", "out,ffn_out", "ffn_in"])
def test_site_policy_variants_match_reference(monkeypatch, sites):
    base, opt, x, m = _models(monkeypatch, env={"T3_LINEAR": "fp16x3", "T3_X3_SITES": sites})
    with torch.inference_mode():
        out = opt(x, m)
        ref = base(x, m)
    assert opt._x3_on and opt._x3_sites == frozenset(sites.split(","))
    assert (out - ref).abs().max().item() < 2e-3
    assert hasattr(opt.layers[0].ffn_in, "_x3") == ("ffn_in" in sites)


def test_chunked_path_allocates_no_split_weights(monkeypatch):
    base, opt, x, m = _models(monkeypatch, env={"T3_LINEAR": "fp16x3", "T3_CHUNK_BS": "1"})
    with torch.inference_mode():
        out = opt(x, m)
        ref = base(x, m)
    assert not opt._x3_on
    assert not hasattr(opt.layers[0].ffn_in, "_x3")
    assert torch.allclose(out, ref, atol=1e-4)

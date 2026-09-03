#!/usr/bin/env python3
"""
fp16x3 linear layers: fp32-class GEMMs on fp16 tensor cores.

**Why.** Turing (T4) has no fp32 tensor cores and no TF32. Its fp16 tensor cores
run about 8x the fp32 SIMT rate, and cuBLAS drives them near peak, while the
fp32 ``volta_sgemm`` kernels are 44-89% of the eager forward on the GEMM-heavy
graded shapes. fp16 *inputs* are out of the question under the harness' gate
(a plain fp16 GEMM lands 1e-3 off). fp16 *arithmetic with compensation* is not:

    x = x_hi + x_lo            x_hi = fp16(x), x_lo = fp16(x - x_hi)   (~2^-22 rel.)
    a.w ~= a_hi.w_hi + a_lo.w_hi + a_hi.w_lo                          (drop a_lo.w_lo)

Every product is exact in fp32 (11-bit x 11-bit mantissas), the tensor cores
accumulate in fp32, and cuBLAS writes an fp32 result (``out_dtype``), so the
whole thing is an fp32-class GEMM at 3x the fp16 FLOPs, i.e. ~2.7x the fp32
rate at the limit. One cuBLAS call does the three terms, with K tripled::

    out = (1/s) * [a_hi | a_hi | a_lo] @ [w_hi ; w_lo ; w_hi]^T + bias

where the weight was scaled by a power of two ``s`` before its split, so that
its lo part (2^-11 of a ~0.03 weight, below fp16's normal range) is a normal
fp16 number, and ``alpha = 1/s`` in the cuBLAS epilogue undoes the scale
exactly, together with the bias. (A two-GEMM variant with K doubled and a
second accumulating call was measured first and lost: the extra pass over the
fp32 output cost more than the third of the operand it saved.) Activations are
not scaled: their lo parts that fall into the subnormal range cost 6e-8
absolute per element, times a weight, invisible next to accumulation noise.

**The split is free.** A naive split (``x.half()``, ``x - hi.float()``, ``cat``)
is five kernels and ~10 bytes of traffic per element; on the GEMM-heavy shape it
cost exactly what the faster GEMM saved. Here the split is fused into the kernel
that produces the activation -- LayerNorm (with the residual add), GELU, or a
plain pass for the attention output -- so the operand is written once, in
``[M, 3K]`` fp16 (hi, hi, lo column blocks), and nothing else touches it.

Hand-written Triton kernels are registered through ``torch.library.triton_op``
so ``torch.compile`` schedules them inside its graph (their casts are explicit
in the kernel, so Inductor cannot fold them away -- the trap that hit the
attention op). The GEMM is an opaque ``custom_op`` around cuBLAS.

Falls back to plain PyTorch (exact emulation on CPU, where fp16 products are
formed in fp32) whenever Triton or ``mm(out_dtype=)`` is unavailable, so the
CPU correctness test exercises the same arithmetic.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton is absent on CPU-only installs
    HAVE_TRITON = False


MAX_X3_WIDTH = 1024          # row-per-program kernels keep one row in registers
_X3_MAX_SCALED_W = 1024.0    # put max|s*w| near 2^10 so s*w_lo is a normal fp16

# Operand order along the tripled K. The weight is always [w_hi | w_lo | w_hi];
# the activation is [a_hi | a_hi | a_lo] ("hhl", shipped) or [a_lo | a_hi | a_hi]
# ("lhh"). Same three products, different accumulation order: with a
# truncating accumulator the tiny cross terms lose less when they are summed
# first, into a small running total. Measured by T3_ONLY=x3k.
_X3_ORDER = os.environ.get("T3_X3_ORDER", "hhl").strip().lower()
if _X3_ORDER not in ("hhl", "lhh"):
    _X3_ORDER = "hhl"
_X3_LO_FIRST = _X3_ORDER == "lhh"
# Split the tripled K into this many cuBLAS calls whose partials are combined
# by the fp32 epilogue (round-to-nearest) instead of inside the tensor core.
try:
    _X3_SPLITK = int(os.environ.get("T3_X3_SPLITK", "1"))
except ValueError:
    _X3_SPLITK = 1
if _X3_SPLITK not in (1, 2, 4):
    _X3_SPLITK = 1


if HAVE_TRITON:

    @triton.jit
    def _x3_ln_split_fwd(
        X, R, SUM, OUT, W, B,
        stride_x, stride_out, N, eps,
        HAS_RESIDUAL: tl.constexpr,
        LO_FIRST: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """LayerNorm(x [+ r]) -> [hi | hi | lo] (or [lo | hi | hi]) fp16 row;
        optionally also x + r."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        s = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            s += tl.load(R + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(SUM + row * stride_x + cols, s, mask=mask)
        mean = tl.sum(s, axis=0) / N
        d = tl.where(mask, s - mean, 0.0)
        var = tl.sum(d * d, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = d * rstd * w + b
        hi = y.to(tl.float16)
        lo = (y - hi.to(tl.float32)).to(tl.float16)
        base = OUT + row * stride_out
        if LO_FIRST:
            tl.store(base + cols, lo, mask=mask)
            tl.store(base + N + cols, hi, mask=mask)
            tl.store(base + 2 * N + cols, hi, mask=mask)
        else:
            tl.store(base + cols, hi, mask=mask)
            tl.store(base + N + cols, hi, mask=mask)
            tl.store(base + 2 * N + cols, lo, mask=mask)

    @triton.jit
    def _x3_act_split_fwd(
        X, OUT, stride_x, stride_out, N,
        GELU: tl.constexpr,
        LO_FIRST: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """[gelu](x) -> [hi | hi | lo] (or [lo | hi | hi]) fp16 row."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        y = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        if GELU:
            # exact (erf) GELU, the reference's approximate="none"
            y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
        hi = y.to(tl.float16)
        lo = (y - hi.to(tl.float32)).to(tl.float16)
        base = OUT + row * stride_out
        if LO_FIRST:
            tl.store(base + cols, lo, mask=mask)
            tl.store(base + N + cols, hi, mask=mask)
            tl.store(base + 2 * N + cols, hi, mask=mask)
        else:
            tl.store(base + cols, hi, mask=mask)
            tl.store(base + N + cols, hi, mask=mask)
            tl.store(base + 2 * N + cols, lo, mask=mask)


def _x3_next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _x3_geometry(n: int):
    block = _x3_next_pow2(n)
    return block, (4 if block <= 512 else 8)


def _x3_launch_ln(flat, r, total, out, weight, bias, n, eps, has_res, kernel):
    block, warps = _x3_geometry(n)
    kernel[(flat.shape[0],)](
        flat, r, total, out, weight, bias,
        flat.stride(0), out.stride(0), n, eps,
        HAS_RESIDUAL=has_res, LO_FIRST=_X3_LO_FIRST, BLOCK=block, num_warps=warps,
    )


def _x3_launch_act(flat, out, n, gelu, kernel):
    block, warps = _x3_geometry(n)
    kernel[(flat.shape[0],)](
        flat, out, flat.stride(0), out.stride(0), n,
        GELU=gelu, LO_FIRST=_X3_LO_FIRST, BLOCK=block, num_warps=warps,
    )


# ---------------------------------------------------------------------------
# Plain-PyTorch reference implementations (CPU, no Triton, or self-test).
# ---------------------------------------------------------------------------
def _x3_split_ref(y: torch.Tensor) -> torch.Tensor:
    hi = y.to(torch.float16)
    lo = (y - hi.to(torch.float32)).to(torch.float16)
    if _X3_LO_FIRST:
        return torch.cat([lo, hi, hi], dim=-1)
    return torch.cat([hi, hi, lo], dim=-1)


def _x3_ln_split_ref(x, weight, bias, eps):
    return _x3_split_ref(F.layer_norm(x, (x.shape[-1],), weight, bias, eps))


def _x3_add_ln_split_ref(x, r, weight, bias, eps):
    s = x + r
    return s, _x3_ln_split_ref(s, weight, bias, eps)


def _x3_act_split_ref(x, gelu):
    return _x3_split_ref(F.gelu(x, approximate="none") if gelu else x)


def _x3_linear_ref(a3, w3, bias, inv):
    """Exact emulation: fp16 x fp16 products are exact in fp32."""
    out = inv * (a3.float() @ w3.float().t())
    return out if bias is None else out + bias


# ---------------------------------------------------------------------------
# cuBLAS path for the GEMM. ``mm``/``addmm`` grew an ``out_dtype`` keyword
# (fp16 inputs, fp32 accumulation and output) in recent PyTorch; without it the
# emulation above runs, which is correct but no faster than fp32.
# ---------------------------------------------------------------------------
_X3_OUT_DTYPE = None        # None = not probed yet; True/False afterwards
_X3_ADDMM_BIAS = False      # whether addmm(out_dtype=) broadcasts a 1-D bias


def _x3_probe_cublas(device) -> bool:
    global _X3_OUT_DTYPE, _X3_ADDMM_BIAS
    if _X3_OUT_DTYPE is not None:
        return _X3_OUT_DTYPE
    try:
        a = torch.randn(16, 32, device=device).half()
        w = torch.randn(8, 32, device=device).half()
        ref = a.float() @ w.float().t()
        out = torch.mm(a, w.t(), out_dtype=torch.float32)
        assert out.dtype == torch.float32 and (out - ref).abs().max().item() < 1e-3
        c = torch.randn(16, 8, device=device)
        out2 = torch.addmm(c, a, w.t(), alpha=0.5, out_dtype=torch.float32)
        assert (out2 - (c + 0.5 * ref)).abs().max().item() < 1e-3
        _X3_OUT_DTYPE = True
        try:
            b = torch.randn(8, device=device)
            out3 = torch.addmm(b, a, w.t(), alpha=0.25, out_dtype=torch.float32)
            _X3_ADDMM_BIAS = bool(
                out3.shape == (16, 8) and (out3 - (0.25 * ref + b)).abs().max().item() < 1e-3)
        except Exception:
            _X3_ADDMM_BIAS = False
    except Exception:
        _X3_OUT_DTYPE = False
    return _X3_OUT_DTYPE


def _x3_linear_cuda(a3, w3, bias, inv):
    if _X3_SPLITK == 1:
        if bias is not None and _X3_ADDMM_BIAS:
            # alpha and the bias both ride in the cuBLAS epilogue: one call.
            return torch.addmm(bias, a3, w3.t(), alpha=inv, out_dtype=torch.float32)
        out = torch.mm(a3, w3.t(), out_dtype=torch.float32)
        out *= inv
        if bias is not None:
            out += bias
        return out
    # Split-K over the tripled axis: column slices of a3/w3 are strided views
    # (lda = 3K) that cuBLAS takes as-is; the partials meet in the fp32
    # epilogue (beta=1), i.e. round-to-nearest instead of the tensor core's
    # truncation. Costs one extra read+write of the fp32 output per chunk.
    k3 = a3.shape[1]
    ch = k3 // _X3_SPLITK
    sl = slice(0, ch)
    if bias is not None and _X3_ADDMM_BIAS:
        out = torch.addmm(bias, a3[:, sl], w3[:, sl].t(), alpha=inv, out_dtype=torch.float32)
    else:
        out = torch.mm(a3[:, sl], w3[:, sl].t(), out_dtype=torch.float32)
        out *= inv
        if bias is not None:
            out += bias
    for c in range(1, _X3_SPLITK):
        sl = slice(c * ch, (c + 1) * ch if c < _X3_SPLITK - 1 else k3)
        out = torch.addmm(out, a3[:, sl], w3[:, sl].t(), alpha=inv, beta=1.0,
                          out_dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# Registration.
# ---------------------------------------------------------------------------
HAVE_X3_TRITON_OP = False
HAVE_X3_LINEAR_OP = False
_X3_TRITON_OK = HAVE_TRITON     # cleared if the self-test fails on this machine

if HAVE_TRITON:
    try:
        from torch.library import triton_op, wrap_triton

        @triton_op("exactswap::x3_ln_split", mutates_args={})
        def _x3_ln_split_op(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                            eps: float) -> torch.Tensor:
            n = x.shape[-1]
            flat = x.contiguous().view(-1, n)
            out = torch.empty((flat.shape[0], 3 * n), dtype=torch.float16, device=x.device)
            _x3_launch_ln(flat, flat, flat, out, weight.contiguous(), bias.contiguous(),
                          n, eps, False, wrap_triton(_x3_ln_split_fwd))
            return out

        @triton_op("exactswap::x3_add_ln_split", mutates_args={})
        def _x3_add_ln_split_op(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor,
                                bias: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
            n = x.shape[-1]
            flat = x.contiguous().view(-1, n)
            rc = r.contiguous().view(-1, n)
            total = torch.empty_like(flat)
            out = torch.empty((flat.shape[0], 3 * n), dtype=torch.float16, device=x.device)
            _x3_launch_ln(flat, rc, total, out, weight.contiguous(), bias.contiguous(),
                          n, eps, True, wrap_triton(_x3_ln_split_fwd))
            return total.view(x.shape), out

        @triton_op("exactswap::x3_act_split", mutates_args={})
        def _x3_act_split_op(x: torch.Tensor, gelu: bool) -> torch.Tensor:
            n = x.shape[-1]
            flat = x.contiguous().view(-1, n)
            out = torch.empty((flat.shape[0], 3 * n), dtype=torch.float16, device=x.device)
            _x3_launch_act(flat, out, n, gelu, wrap_triton(_x3_act_split_fwd))
            return out

        HAVE_X3_TRITON_OP = True
    except Exception:  # pragma: no cover - older torch, or a registration clash
        HAVE_X3_TRITON_OP = False

try:
    from torch.library import custom_op

    @custom_op("exactswap::x3_linear", mutates_args=())
    def _x3_linear_op(a3: torch.Tensor, w3: torch.Tensor,
                      bias: Optional[torch.Tensor], inv: float) -> torch.Tensor:
        if a3.is_cuda and _x3_probe_cublas(a3.device):
            return _x3_linear_cuda(a3, w3, bias, inv)
        return _x3_linear_ref(a3, w3, bias, inv)

    @_x3_linear_op.register_fake
    def _(a3, w3, bias, inv):
        return a3.new_empty((a3.shape[0], w3.shape[0]), dtype=torch.float32)

    HAVE_X3_LINEAR_OP = True
except Exception:  # pragma: no cover
    HAVE_X3_LINEAR_OP = False


# ---------------------------------------------------------------------------
# Public entry points. Each falls back to the reference implementation when the
# fast path does not apply, so callers never check.
# ---------------------------------------------------------------------------
def x3_can_use(x: torch.Tensor) -> bool:
    """Whether the fused Triton split kernels apply to rows of this tensor."""
    return (_X3_TRITON_OK and HAVE_X3_TRITON_OP and x.is_cuda
            and x.dtype == torch.float32 and x.shape[-1] <= MAX_X3_WIDTH)


def x3_ln_split(x, weight, bias, eps=1e-5):
    """``split(LayerNorm(x))`` as ``[M, 3N]`` fp16 (hi, hi, lo column blocks)."""
    if x3_can_use(x):
        return _x3_ln_split_op(x, weight, bias, float(eps))
    return _x3_ln_split_ref(x, weight, bias, eps).view(-1, 3 * x.shape[-1])


def x3_add_ln_split(x, r, weight, bias, eps=1e-5):
    """``(x + r, split(LayerNorm(x + r)))`` in one pass over the activation."""
    if x3_can_use(x):
        return _x3_add_ln_split_op(x, r, weight, bias, float(eps))
    s, a3 = _x3_add_ln_split_ref(x, r, weight, bias, eps)
    return s, a3.view(-1, 3 * x.shape[-1])


def x3_act_split(x, gelu: bool):
    """``split(gelu(x))`` (or ``split(x)``) as ``[M, 3N]`` fp16."""
    if x3_can_use(x):
        return _x3_act_split_op(x, bool(gelu))
    return _x3_act_split_ref(x, gelu).view(-1, 3 * x.shape[-1])


def x3_linear(a3, w3, bias, inv: float):
    """``a @ w^T + bias`` from the split operand and a prepared weight."""
    if HAVE_X3_LINEAR_OP:
        return _x3_linear_op(a3, w3, bias, float(inv))
    if a3.is_cuda and _x3_probe_cublas(a3.device):
        return _x3_linear_cuda(a3, w3, bias, inv)
    return _x3_linear_ref(a3, w3, bias, inv)


@torch.no_grad()
def x3_prepare(weight: torch.Tensor, bias: Optional[torch.Tensor]):
    """Split an ``nn.Linear`` weight once: ``(w3, bias, 1/s)``.

    ``w3 = [w_hi | w_lo | w_hi]`` is ``[N, 3K]`` fp16, from the weight scaled
    by the power of two ``s`` that puts its largest entry near 2^10, so the lo
    parts are normal fp16 numbers. Paired with the activation's
    ``[a_hi | a_hi | a_lo]`` (or ``[a_lo | a_hi | a_hi]`` under
    ``T3_X3_ORDER=lhh``) it yields ``s * (a . w)`` in one GEMM; ``1/s`` is
    exact.
    """
    w = weight.detach().to(torch.float32)
    amax = float(w.abs().max().item())
    # A zero, inf or NaN weight gets no scaling: the split then reproduces
    # whatever the reference would compute (zeros, or NaN) instead of raising.
    if amax > 0 and math.isfinite(amax):
        s = 2.0 ** math.floor(math.log2(_X3_MAX_SCALED_W / amax))
    else:
        s = 1.0
    ws = w * s
    hi = ws.to(torch.float16)
    lo = (ws - hi.to(torch.float32)).to(torch.float16)
    w3 = torch.cat([hi, lo, hi], dim=1).contiguous()
    b = None if bias is None else bias.detach().to(torch.float32).contiguous()
    return w3, b, 1.0 / s


def x3_selfcheck(device, thorough: bool = False) -> bool:
    """The registered split ops against the PyTorch reference.

    The quick form is what x3_available runs once per device; the thorough
    form (tests, the T3_ONLY=x3k driver run) adds ragged tiles, the widest row,
    and the edge rows the reference must reproduce: an all-zero row, a constant
    row and a row whose lo parts fall in fp16's subnormal range. Returns False
    instead of raising, so a broken Triton build degrades to the PyTorch split.
    """
    dev = torch.device(device)
    cases = [(64, 96)] if not thorough else [(64, 96), (67, 128), (67, 100), (5, 1024), (3, 32)]
    tol = 2e-3   # hi halves may differ by one fp16 ulp between implementations
    try:
        for (m, n) in cases:
            x = torch.randn(m, n, device=dev)
            if thorough and m >= 3:
                x[0] = 0.0
                x[1] = 3.0
                x[2] = torch.randn(n, device=dev) * 1e-6
            r = torch.randn_like(x)
            w = torch.rand(n, device=dev) + 0.5
            b = torch.randn(n, device=dev)
            a = _x3_ln_split_op(x, w, b, 1e-5)
            a_ref = _x3_ln_split_ref(x, w, b, 1e-5)
            s, a2 = _x3_add_ln_split_op(x, r, w, b, 1e-5)
            s_ref, a2_ref = _x3_add_ln_split_ref(x, r, w, b, 1e-5)
            g = _x3_act_split_op(x, True)
            g_ref = _x3_act_split_ref(x, True)
            p = _x3_act_split_op(x, False)
            p_ref = _x3_act_split_ref(x, False)
            for got, ref in ((a, a_ref), (a2, a2_ref), (g, g_ref), (p, p_ref)):
                if got.shape != ref.shape:
                    return False
                if not (got.float() - ref.float()).abs().max().item() < tol:
                    return False
            if not (s - s_ref).abs().max().item() < 1e-5:
                return False
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        return True
    except Exception:
        return False


_X3_AVAILABLE = {}


def x3_available(device) -> bool:
    """One-time check that the fast path works on this device (cuBLAS
    ``out_dtype`` support, tensor cores, and the Triton kernels agreeing with
    the reference on a small case). Cached per device."""
    global _X3_TRITON_OK
    key = str(device)
    if key in _X3_AVAILABLE:
        return _X3_AVAILABLE[key]
    ok = False
    try:
        dev = torch.device(device)
        if dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 7:
            ok = _x3_probe_cublas(dev)
            if ok:
                # fp16 GEMMs may otherwise reduce split-K partials in fp16.
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            if ok and _X3_TRITON_OK and HAVE_X3_TRITON_OP:
                if not x3_selfcheck(dev):
                    _X3_TRITON_OK = False   # cuBLAS path still usable, split via PyTorch
    except Exception:
        ok = False
    _X3_AVAILABLE[key] = ok
    return ok

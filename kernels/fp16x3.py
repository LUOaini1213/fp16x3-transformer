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
rate at the limit. One plain cuBLAS ``mm`` does the three terms and the bias,
with K tripled and padded by eight columns::

    s * (a . w) + s * b  =  [a_lo | a_hi | a_hi | 1 1 0 0 0 0 0 0] @ [w_hi ; w_lo ; w_hi ; (s b)_hi (s b)_lo 0 ...]^T

The weight is scaled by a power of two ``s`` before its split so that its lo
part (2^-11 of a ~0.03 weight, below fp16's normal range) is a normal fp16
number. The scale is *not* undone by the GEMM: cuBLAS' ``addmm`` epilogue
(alpha and a broadcast bias) measured 40-90% slower than a bare ``mm`` on a T4
-- an extra pass over the fp32 output -- so the GEMM returns ``s * (a.w + b)``
and whichever kernel consumes it multiplies by ``1/s`` for free: SDPA through
its ``scale`` argument (q and k carry s^2, v carries s into the attention
output), the add+LayerNorm-split kernel on its residual input, the GELU-split
kernel on its input, ``torch.add(alpha=1/s)`` on the last layer. The lo-first
operand order is the default: the tiny cross terms are accumulated first,
into a small running total, which measured 1.6-3.7x less truncation error on
Turing's tensor cores at identical cost (``results/x3_error_vs_k_t4.csv``).
(A two-GEMM variant with K doubled and an accumulating second call was
measured first and lost for the same reason: the extra pass over the fp32
output.) Activations are not scaled: their lo parts that fall into the
subnormal range cost 6e-8 absolute per element, times a weight, invisible
next to accumulation noise.

**The split is free.** A naive split (``x.half()``, ``x - hi.float()``, ``cat``)
is five kernels and ~10 bytes of traffic per element; on the GEMM-heavy shape it
cost exactly what the faster GEMM saved. Here the split is fused into the kernel
that produces the activation -- LayerNorm (with the residual add), GELU, or a
plain pass for the attention output -- so the operand is written once, in
``[M, 3K + 8]`` fp16 (lo, hi, hi column blocks and the constant tail), and
nothing else touches it.

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

_X3_BLAS = os.environ.get("T3_X3_BLAS", "torch").strip().lower()
if _X3_BLAS == "lt":
    try:
        from .cublaslt_backend import lt_candidate_matmul
    except ImportError:
        # Single-file cloud builds inline the provider before this module.
        if "lt_candidate_matmul" not in globals():
            _X3_BLAS = "torch"

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
_X3_ORDER = os.environ.get("T3_X3_ORDER", "lhh").strip().lower()
if _X3_ORDER not in ("hhl", "lhh"):
    _X3_ORDER = "lhh"
_X3_LO_FIRST = _X3_ORDER == "lhh"
_X3_MAX_SCALED_B = 30000.0   # |s * b| must stay inside fp16 range


def _x3_tail(k: int) -> int:
    """Padding columns after the tripled K: rounds 3K up to a multiple of 32
    (cuBLAS's fast Turing kernels tile K by 32) with at least two columns for
    the bias pair. 32 for every graded width."""
    t = 32 - (3 * k) % 32
    return t if t >= 2 else t + 32


def x3_tail(k: int) -> int:
    return _x3_tail(k)

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
        X, R, RB, SUM, OUT, W, B,
        M, stride_x, stride_out, eps, r_scale,
        N: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        HAS_RBIAS: tl.constexpr,
        WRITE_SUM: tl.constexpr,
        LO_FIRST: tl.constexpr,
        WRITE_SPLIT: tl.constexpr,
        TAIL: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """LayerNorm(x [+ r_scale * r [+ rb]]) for a [ROWS, N] tile.

        WRITE_SPLIT: rows go out as [lo | hi | hi | 1 1 0..] (or hi-first) fp16
        -- the fp16x3 GEMM operand, TAIL constant columns wide; otherwise as
        plain fp32. With HAS_RESIDUAL the residual is a GEMM output carrying
        the weight scale (undone by r_scale) and, with HAS_RBIAS, its bias is
        added here; WRITE_SUM stores the new residual stream.
        """
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        rmask = rows < M
        cmask = cols < N
        mask = rmask[:, None] & cmask[None, :]
        offs = rows[:, None] * stride_x + cols[None, :]
        s = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            s += tl.load(R + offs, mask=mask, other=0.0).to(tl.float32) * r_scale
            if HAS_RBIAS:
                s += tl.load(RB + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
            if WRITE_SUM:
                tl.store(SUM + offs, s, mask=mask)
        mean = tl.sum(s, axis=1) / N
        d = tl.where(mask, s - mean[:, None], 0.0)
        var = tl.sum(d * d, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=cmask, other=0.0).to(tl.float32)
        y = d * rstd[:, None] * w[None, :] + b[None, :]
        obase = OUT + rows[:, None] * stride_out
        if WRITE_SPLIT:
            hi = y.to(tl.float16)
            lo = (y - hi.to(tl.float32)).to(tl.float16)
            if LO_FIRST:
                tl.store(obase + cols[None, :], lo, mask=mask)
                tl.store(obase + N + cols[None, :], hi, mask=mask)
                tl.store(obase + 2 * N + cols[None, :], hi, mask=mask)
            else:
                tl.store(obase + cols[None, :], hi, mask=mask)
                tl.store(obase + N + cols[None, :], hi, mask=mask)
                tl.store(obase + 2 * N + cols[None, :], lo, mask=mask)
            # the constant tail that pairs with the weight's bias columns
            t32 = tl.arange(0, 32)
            tail = tl.broadcast_to((t32 < 2).to(tl.float16)[None, :], [ROWS, 32])
            tl.store(obase + 3 * N + t32[None, :], tail, mask=rmask[:, None] & (t32 < TAIL)[None, :])
        else:
            tl.store(obase + cols[None, :], y, mask=mask)

    @triton.jit
    def _x3_act_split_fwd(
        X, XB, OUT, M, stride_x, stride_out, scale,
        N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        GELU: tl.constexpr,
        LO_FIRST: tl.constexpr,
        TAIL: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """[gelu](scale * x [+ xb]) -> [lo | hi | hi | 1 1 0..] (or hi-first) fp16 tile."""
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        rmask = rows < M
        cmask = cols < N
        mask = rmask[:, None] & cmask[None, :]
        y = tl.load(X + rows[:, None] * stride_x + cols[None, :], mask=mask, other=0.0).to(tl.float32) * scale
        if HAS_BIAS:
            y += tl.load(XB + cols, mask=cmask, other=0.0).to(tl.float32)[None, :]
        if GELU:
            # exact (erf) GELU, the reference's approximate="none"
            y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
        hi = y.to(tl.float16)
        lo = (y - hi.to(tl.float32)).to(tl.float16)
        obase = OUT + rows[:, None] * stride_out
        if LO_FIRST:
            tl.store(obase + cols[None, :], lo, mask=mask)
            tl.store(obase + N + cols[None, :], hi, mask=mask)
            tl.store(obase + 2 * N + cols[None, :], hi, mask=mask)
        else:
            tl.store(obase + cols[None, :], hi, mask=mask)
            tl.store(obase + N + cols[None, :], hi, mask=mask)
            tl.store(obase + 2 * N + cols[None, :], lo, mask=mask)
        t32 = tl.arange(0, 32)
        tail = tl.broadcast_to((t32 < 2).to(tl.float16)[None, :], [ROWS, 32])
        tl.store(obase + 3 * N + t32[None, :], tail, mask=rmask[:, None] & (t32 < TAIL)[None, :])


def _x3_next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# Rows per program and warps by row width, from a sweep on the T4
# (results/kaggle_t4_b2geo2_run.log): a few rows per program with eight warps
# is what pays -- the LayerNorm-split at N=128 went from 156 GB/s one row per
# program to 206 GB/s, and the plain fp32 LayerNorm to 245 GB/s, 2.3x
# PyTorch's own -- while tall tiles (16-32 rows) bought nothing. Static rather
# than triton.autotune: autotuning synchronises on its first call, which is
# fatal inside a CUDA-graph capture and makes verdicts irreproducible. The
# bench overrides let T3_ONLY=x3geo sweep the table.
_X3_ROWS_OVERRIDE = None
_X3_WARPS_OVERRIDE = None


def _x3_geometry(n: int, m: int):
    block = _x3_next_pow2(n)
    if block <= 64:
        rows, warps = 8, 4
    elif block <= 256:
        rows, warps = 4, 8
    else:
        rows, warps = 2, 8
    if _X3_ROWS_OVERRIDE is not None:
        rows = _X3_ROWS_OVERRIDE
    if _X3_WARPS_OVERRIDE is not None:
        warps = _X3_WARPS_OVERRIDE
    # tiny M: keep ~80 programs alive rather than a handful of fat ones
    rows = max(1, min(rows, _x3_next_pow2(max(1, -(-m // 80)))))
    return block, rows, warps


def _x3_launch_ln(flat, r, rb, total, out, weight, bias, n, eps, has_res, r_scale,
                  write_sum, write_split, kernel):
    m = flat.shape[0]
    block, rows, warps = _x3_geometry(n, m)
    grid = (-(-m // rows),)
    kernel[grid](
        flat, r, rb if rb is not None else weight, total, out, weight, bias,
        m, flat.stride(0), out.stride(0), eps, r_scale,
        N=n, HAS_RESIDUAL=has_res, HAS_RBIAS=rb is not None, WRITE_SUM=write_sum,
        LO_FIRST=_X3_LO_FIRST, WRITE_SPLIT=write_split, TAIL=_x3_tail(n),
        ROWS=rows, BLOCK=block, num_warps=warps,
    )


def _x3_launch_act(flat, xb, out, n, gelu, scale, kernel):
    m = flat.shape[0]
    block, rows, warps = _x3_geometry(n, m)
    grid = (-(-m // rows),)
    kernel[grid](
        flat, xb if xb is not None else flat, out, m, flat.stride(0), out.stride(0), scale,
        N=n, HAS_BIAS=xb is not None, GELU=gelu, LO_FIRST=_X3_LO_FIRST, TAIL=_x3_tail(n),
        ROWS=rows, BLOCK=block, num_warps=warps,
    )


# ---------------------------------------------------------------------------
# Plain-PyTorch reference implementations (CPU, no Triton, or self-test).
# ---------------------------------------------------------------------------
def _x3_split_ref(y: torch.Tensor) -> torch.Tensor:
    hi = y.to(torch.float16)
    lo = (y - hi.to(torch.float32)).to(torch.float16)
    tail = torch.zeros(y.shape[:-1] + (_x3_tail(y.shape[-1]),), dtype=torch.float16, device=y.device)
    tail[..., :2] = 1.0
    if _X3_LO_FIRST:
        return torch.cat([lo, hi, hi, tail], dim=-1)
    return torch.cat([hi, hi, lo, tail], dim=-1)


def _x3_ln_split_ref(x, weight, bias, eps):
    return _x3_split_ref(F.layer_norm(x, (x.shape[-1],), weight, bias, eps))


def _x3_ln_ref(x, weight, bias, eps):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _x3_residual_ref(x, r, r_scale=1.0, rbias=None):
    s = x + r * r_scale if r_scale != 1.0 else x + r
    return s if rbias is None else s + rbias


def _x3_add_ln_split_ref(x, r, weight, bias, eps, r_scale=1.0, rbias=None):
    s = _x3_residual_ref(x, r, r_scale, rbias)
    return s, _x3_ln_split_ref(s, weight, bias, eps)


def _x3_add_ln_ref(x, r, weight, bias, eps, r_scale=1.0, rbias=None):
    return _x3_ln_ref(_x3_residual_ref(x, r, r_scale, rbias), weight, bias, eps)


def _x3_act_split_ref(x, gelu, scale=1.0, xbias=None):
    y = x * scale if scale != 1.0 else x
    if xbias is not None:
        y = y + xbias
    return _x3_split_ref(F.gelu(y, approximate="none") if gelu else y)


def _x3_linear_ref(a3, w3, bias, inv, apply_scale=True):
    """Exact emulation: fp16 x fp16 products are exact in fp32."""
    out = a3.float() @ w3.float().t()
    if apply_scale and inv != 1.0:
        out = out * inv
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


def _x3_linear_cuda(a3, w3, bias, inv, apply_scale=True):
    chosen = lt_candidate_matmul(a3, w3) if _X3_BLAS == "lt" and _X3_SPLITK == 1 else None
    if chosen is not None:
        out = chosen
    elif _X3_SPLITK == 1:
        # One bare mm: the bias is inside K and the scale is the consumer's.
        out = torch.mm(a3, w3.t(), out_dtype=torch.float32)
    else:
        # Split-K over the tripled axis: column slices of a3/w3 are strided
        # views (lda = 3K+8) that cuBLAS takes as-is; the partials meet in the
        # fp32 epilogue (beta=1), round-to-nearest instead of the tensor
        # core's truncation. Costs one extra read+write of the output per
        # chunk -- an accuracy option, off by default.
        k3 = a3.shape[1]
        ch = (k3 // _X3_SPLITK) // 8 * 8
        out = torch.mm(a3[:, :ch], w3[:, :ch].t(), out_dtype=torch.float32)
        for c in range(1, _X3_SPLITK):
            sl = slice(c * ch, (c + 1) * ch if c < _X3_SPLITK - 1 else k3)
            out = torch.addmm(out, a3[:, sl], w3[:, sl].t(), beta=1.0, out_dtype=torch.float32)
    if apply_scale and inv != 1.0:
        out *= inv
    if bias is not None:
        out += bias
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
            out = torch.empty((flat.shape[0], 3 * n + _x3_tail(n)), dtype=torch.float16, device=x.device)
            _x3_launch_ln(flat, flat, None, flat, out, weight.contiguous(), bias.contiguous(),
                          n, eps, False, 1.0, False, True, wrap_triton(_x3_ln_split_fwd))
            return out

        @triton_op("exactswap::x3_ln", mutates_args={})
        def _x3_ln_op(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                      eps: float) -> torch.Tensor:
            n = x.shape[-1]
            flat = x.contiguous().view(-1, n)
            out = torch.empty_like(flat)
            _x3_launch_ln(flat, flat, None, flat, out, weight.contiguous(), bias.contiguous(),
                          n, eps, False, 1.0, False, False, wrap_triton(_x3_ln_split_fwd))
            return out.view(x.shape)

        @triton_op("exactswap::x3_add_ln", mutates_args={})
        def _x3_add_ln_op(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor,
                          bias: torch.Tensor, eps: float, r_scale: float,
                          rbias: Optional[torch.Tensor]) -> torch.Tensor:
            n = x.shape[-1]
            flat = x.contiguous().view(-1, n)
            rc = r.contiguous().view(-1, n)
            out = torch.empty_like(flat)
            _x3_launch_ln(flat, rc, None if rbias is None else rbias.contiguous(), flat, out,
                          weight.contiguous(), bias.contiguous(),
                          n, eps, True, r_scale, False, False, wrap_triton(_x3_ln_split_fwd))
            return out.view(x.shape)

        @triton_op("exactswap::x3_add_ln_split", mutates_args={})
        def _x3_add_ln_split_op(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor,
                                bias: torch.Tensor, eps: float, r_scale: float,
                                rbias: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            n = x.shape[-1]
            flat = x.contiguous().view(-1, n)
            rc = r.contiguous().view(-1, n)
            total = torch.empty_like(flat)
            out = torch.empty((flat.shape[0], 3 * n + _x3_tail(n)), dtype=torch.float16, device=x.device)
            _x3_launch_ln(flat, rc, None if rbias is None else rbias.contiguous(), total, out,
                          weight.contiguous(), bias.contiguous(),
                          n, eps, True, r_scale, True, True, wrap_triton(_x3_ln_split_fwd))
            return total.view(x.shape), out

        @triton_op("exactswap::x3_act_split", mutates_args={})
        def _x3_act_split_op(x: torch.Tensor, gelu: bool, scale: float,
                             xbias: Optional[torch.Tensor]) -> torch.Tensor:
            n = x.shape[-1]
            flat = x.contiguous().view(-1, n)
            out = torch.empty((flat.shape[0], 3 * n + _x3_tail(n)), dtype=torch.float16, device=x.device)
            _x3_launch_act(flat, None if xbias is None else xbias.contiguous(), out, n, gelu, scale,
                           wrap_triton(_x3_act_split_fwd))
            return out

        HAVE_X3_TRITON_OP = True
    except Exception:  # pragma: no cover - older torch, or a registration clash
        HAVE_X3_TRITON_OP = False

try:
    from torch.library import custom_op

    @custom_op("exactswap::x3_linear", mutates_args=())
    def _x3_linear_op(a3: torch.Tensor, w3: torch.Tensor, bias: Optional[torch.Tensor],
                      inv: float, apply_scale: bool) -> torch.Tensor:
        if a3.is_cuda and _x3_probe_cublas(a3.device):
            return _x3_linear_cuda(a3, w3, bias, inv, apply_scale)
        return _x3_linear_ref(a3, w3, bias, inv, apply_scale)

    @_x3_linear_op.register_fake
    def _(a3, w3, bias, inv, apply_scale):
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
    """``split(LayerNorm(x))`` as ``[M, 3N + tail]`` fp16 (lo, hi, hi blocks + tail)."""
    if x3_can_use(x):
        return _x3_ln_split_op(x, weight, bias, float(eps))
    return _x3_ln_split_ref(x, weight, bias, eps).view(-1, 3 * x.shape[-1] + _x3_tail(x.shape[-1]))


def x3_ln(x, weight, bias, eps=1e-5):
    """Plain fp32 ``LayerNorm(x)`` through the tiled kernel (the final norm)."""
    if x3_can_use(x):
        return _x3_ln_op(x, weight, bias, float(eps))
    return _x3_ln_ref(x, weight, bias, eps)


def x3_add_ln(x, r, weight, bias, eps=1e-5, r_scale=1.0, rbias=None):
    """``LayerNorm(x + r_scale * r + rbias)`` -- the final norm with the last
    residual add and the last GEMM's bias folded in."""
    if x3_can_use(x):
        return _x3_add_ln_op(x, r, weight, bias, float(eps), float(r_scale), rbias)
    return _x3_add_ln_ref(x, r, weight, bias, eps, r_scale, rbias)


def x3_add_ln_split(x, r, weight, bias, eps=1e-5, r_scale=1.0, rbias=None):
    """``(x + r_scale * r + rbias, split(LayerNorm(...)))`` in one pass.

    ``r_scale`` undoes the power-of-two weight scale of the GEMM that produced
    ``r`` and ``rbias`` is that GEMM's bias (see ``x3_prepare``); both cost
    nothing here.
    """
    if x3_can_use(x):
        return _x3_add_ln_split_op(x, r, weight, bias, float(eps), float(r_scale), rbias)
    s, a3 = _x3_add_ln_split_ref(x, r, weight, bias, eps, r_scale, rbias)
    return s, a3.view(-1, 3 * x.shape[-1] + _x3_tail(x.shape[-1]))


def x3_act_split(x, gelu: bool, scale=1.0, xbias=None):
    """``split(gelu(scale * x + xbias))`` (or without gelu) as ``[M, 3N + tail]`` fp16."""
    if x3_can_use(x):
        return _x3_act_split_op(x, bool(gelu), float(scale), xbias)
    return _x3_act_split_ref(x, gelu, scale, xbias).view(-1, 3 * x.shape[-1] + _x3_tail(x.shape[-1]))


def x3_linear(a3, w3, bias=None, inv: float = 1.0, apply_scale: bool = True):
    """``a @ w^T + bias`` from the split operand and a prepared weight.

    With ``apply_scale=False`` the result is ``s * (a @ w^T + b)`` -- the
    caller's consumer undoes ``s`` (``inv = 1/s``) for free; with the default
    the scale is applied here, at the cost of a pass over the output.
    """
    if HAVE_X3_LINEAR_OP:
        return _x3_linear_op(a3, w3, bias, float(inv), bool(apply_scale))
    if a3.is_cuda and _x3_probe_cublas(a3.device):
        return _x3_linear_cuda(a3, w3, bias, inv, apply_scale)
    return _x3_linear_ref(a3, w3, bias, inv, apply_scale)


@torch.no_grad()
def x3_prepare(weight: torch.Tensor, bias: Optional[torch.Tensor], fold_bias: bool = True):
    """Split an ``nn.Linear`` weight once: ``(w3, bias_or_None, 1/s)``.

    The weight is scaled by the power of two ``s`` that puts its largest entry
    near 2^10, so the lo parts are normal fp16 numbers, then split:
    ``w3 = [w_hi | w_lo | w_hi | tail]`` fp16. With ``fold_bias`` the tail is
    ``x3_tail(K)`` columns (3K rounded up to a multiple of 32) carrying the
    scaled bias as an fp16 pair, which pairs with the constant ``[1, 1, 0..]``
    tail the split kernels write, and the GEMM yields ``s * (a . w + b)``;
    without it ``w3`` is ``[N, 3K]``, the (unscaled) bias comes back for the
    consumer kernel to add, and the caller feeds the GEMM the first 3K
    columns of the operand. ``1/s`` is exact and is the consumer's to apply;
    ``s`` is lowered if the scaled bias would leave fp16 range.
    """
    w = weight.detach().to(torch.float32)
    amax = float(w.abs().max().item())
    b = None if bias is None else bias.detach().to(torch.float32).contiguous()
    bmax = 0.0 if (b is None or not fold_bias) else float(b.abs().max().item())
    # A zero, inf or NaN weight gets no scaling: the split then reproduces
    # whatever the reference would compute (zeros, or NaN) instead of raising.
    if amax > 0 and math.isfinite(amax) and math.isfinite(bmax):
        s = 2.0 ** math.floor(math.log2(_X3_MAX_SCALED_W / amax))
        while s > 1.0 and bmax * s > _X3_MAX_SCALED_B:
            s /= 2.0
        while s < 1.0 and amax * s > _X3_MAX_SCALED_W * 32:   # a huge weight: scale down
            s /= 2.0
    else:
        s = 1.0
    ws = w * s
    hi = ws.to(torch.float16)
    lo = (ws - hi.to(torch.float32)).to(torch.float16)
    if not fold_bias:
        return torch.cat([hi, lo, hi], dim=1).contiguous(), b, 1.0 / s
    n, k = w.shape
    tail = torch.zeros((n, _x3_tail(k)), dtype=torch.float32, device=w.device)
    if b is not None:
        bs = b * s
        bh = bs.to(torch.float16).to(torch.float32)
        tail[:, 0] = bh
        tail[:, 1] = bs - bh
    w3 = torch.cat([hi, lo, hi, tail.to(torch.float16)], dim=1).contiguous()
    return w3, None, 1.0 / s


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
            rb = torch.randn(n, device=dev) * 0.1
            s, a2 = _x3_add_ln_split_op(x, r, w, b, 1e-5, 0.25, rb)
            s_ref, a2_ref = _x3_add_ln_split_ref(x, r, w, b, 1e-5, 0.25, rb)
            g = _x3_act_split_op(x, True, 1.0, rb)
            g_ref = _x3_act_split_ref(x, True, 1.0, rb)
            p = _x3_act_split_op(x, False, 0.5, None)
            p_ref = _x3_act_split_ref(x, False, 0.5, None)
            ln = _x3_ln_op(x, w, b, 1e-5)
            ln_ref = _x3_ln_ref(x, w, b, 1e-5)
            if not (ln - ln_ref).abs().max().item() < 1e-4:
                return False
            aln = _x3_add_ln_op(x, r, w, b, 1e-5, 0.25, rb)
            aln_ref = _x3_add_ln_ref(x, r, w, b, 1e-5, 0.25, rb)
            if not (aln - aln_ref).abs().max().item() < 1e-4:
                return False
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

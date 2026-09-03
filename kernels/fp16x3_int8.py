#!/usr/bin/env python3
"""
Experimental: the two cross terms of fp16x3 on the int8 IMMA tensor cores.

fp16x3 spends three fp16 GEMM units per fp32 product: a_hi.w_hi + a_lo.w_hi +
a_hi.w_lo. The two cross terms are 2^-11 of the result, so they need far less
than fp16's precision -- 8 bits leave the total at ~2^-19 relative -- and a
T4's int8 tensor cores run at twice the fp16 rate (``torch._int_mm``, probed at
~67 TFLOPS on sm_75). So::

    main   = fp16 GEMM        a_hi . w_hi                        (1 unit, fp32 out)
    cross  = int8 GEMM        [a_lo_q | a_hi_q] . [w_hi_q ; w_lo_q]   (K doubled = 1 unit, int32 out)
    out    = main + (s_r * t_n * 2^-11) * cross

with per-row power-of-two activation scales ``s_r`` (from the row max of |a|,
so a_hi/s_r fits int8, and a_lo/(s_r 2^-11) fits it too since |a_lo| <= 2^-11
|a|) and per-output-column scales ``t_n`` for the weight. Both cross terms
carry the same scale product, which is what lets them share one accumulator.

This module is the operator-level experiment behind T3_ONLY=x3i8: the split
kernel that emits ``a_hi`` (fp16, with the constant tail for a bias in K), the
int8 pair and the row scale in one pass, the weight preparation, and a
reference implementation. It is not wired into the model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

from .fp16x3 import _x3_geometry, _x3_tail


if HAVE_TRITON:

    @triton.jit
    def _x3i8_split_fwd(
        X, HI, Q, S, M, stride_x, stride_hi, stride_q, scale,
        N: tl.constexpr,
        GELU: tl.constexpr,
        TAIL: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """[gelu](scale * x) -> a_hi fp16 (+ tail), [a_lo_q | a_hi_q] int8, row scale."""
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK)
        rmask = rows < M
        cmask = cols < N
        mask = rmask[:, None] & cmask[None, :]
        y = tl.load(X + rows[:, None] * stride_x + cols[None, :], mask=mask, other=0.0).to(tl.float32) * scale
        if GELU:
            y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
        hi = y.to(tl.float16)
        hif = hi.to(tl.float32)
        lo = y - hif
        amax = tl.max(tl.where(mask, tl.abs(hif), 0.0), axis=1)              # [ROWS]
        amax = tl.where(amax > 0.0, amax, 1.0)
        # power-of-two scale: 2^ceil(log2(amax / 127)) puts the row max at <= 127
        e = tl.ceil(tl.log2(amax / 127.0))
        sr = tl.exp2(e)                                                      # [ROWS]
        inv_hi = 1.0 / sr
        inv_lo = 2048.0 / sr                                                 # 2^11 / s_r
        hq = tl.floor(hif * inv_hi[:, None] + 0.5)
        lq = tl.floor(lo * inv_lo[:, None] + 0.5)
        hq = tl.minimum(tl.maximum(hq, -127.0), 127.0).to(tl.int8)
        lq = tl.minimum(tl.maximum(lq, -127.0), 127.0).to(tl.int8)
        hbase = HI + rows[:, None] * stride_hi
        tl.store(hbase + cols[None, :], hi, mask=mask)
        t32 = tl.arange(0, 32)
        tail = tl.broadcast_to((t32 < 2).to(tl.float16)[None, :], [ROWS, 32])
        tl.store(hbase + N + t32[None, :], tail, mask=rmask[:, None] & (t32 < TAIL)[None, :])
        qbase = Q + rows[:, None] * stride_q
        tl.store(qbase + cols[None, :], lq, mask=mask)
        tl.store(qbase + N + cols[None, :], hq, mask=mask)
        tl.store(S + rows, sr, mask=rmask)


def x3i8_split(x: torch.Tensor, gelu: bool = False, scale: float = 1.0):
    """Returns ``(a_hi [M, N+tail] fp16, a_q [M, 2N] int8, s_r [M] fp32)``."""
    n = x.shape[-1]
    flat = x.contiguous().view(-1, n)
    m = flat.shape[0]
    tail = _x3_tail(n)
    if not (HAVE_TRITON and x.is_cuda):
        return _x3i8_split_ref(flat, gelu, scale)
    hi = torch.empty((m, n + tail), dtype=torch.float16, device=x.device)
    q = torch.empty((m, 2 * n), dtype=torch.int8, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    block, rows, warps = _x3_geometry(n, m)
    grid = (-(-m // rows),)
    _x3i8_split_fwd[grid](flat, hi, q, s, m, flat.stride(0), hi.stride(0), q.stride(0), float(scale),
                          N=n, GELU=gelu, TAIL=tail, ROWS=rows, BLOCK=block, num_warps=warps)
    return hi, q, s


def _x3i8_split_ref(flat, gelu=False, scale=1.0):
    import torch.nn.functional as F
    y = flat.float() * scale
    if gelu:
        y = F.gelu(y, approximate="none")
    m, n = y.shape
    hi = y.to(torch.float16)
    hif = hi.float()
    lo = y - hif
    amax = hif.abs().amax(dim=1)
    amax = torch.where(amax > 0, amax, torch.ones_like(amax))
    sr = torch.exp2(torch.ceil(torch.log2(amax / 127.0)))
    hq = torch.clamp(torch.floor(hif / sr[:, None] + 0.5), -127, 127).to(torch.int8)
    lq = torch.clamp(torch.floor(lo * (2048.0 / sr)[:, None] + 0.5), -127, 127).to(torch.int8)
    tail = torch.zeros((m, _x3_tail(n)), dtype=torch.float16, device=y.device)
    tail[:, :2] = 1.0
    return torch.cat([hi, tail], 1), torch.cat([lq, hq], 1), sr


@torch.no_grad()
def x3i8_prepare(weight: torch.Tensor, bias: Optional[torch.Tensor], fold_bias: bool = True):
    """``(w_hi [N, K(+tail)] fp16, w_q [N, 2K] int8, t_n [N] fp32, bias_or_None)``.

    ``w_hi = fp16(w)``; the int8 pair is ``w_hi / t_n`` and ``(w - w_hi) / (t_n
    2^-11)`` with ``t_n`` the power of two that puts the row max of |w_hi| at
    <= 127. With ``fold_bias`` the bias rides in the fp16 tail as usual.
    """
    w = weight.detach().float()
    n, k = w.shape
    hi = w.to(torch.float16)
    hif = hi.float()
    lo = w - hif
    tmax = hif.abs().amax(dim=1)
    tmax = torch.where(tmax > 0, tmax, torch.ones_like(tmax))
    t = torch.exp2(torch.ceil(torch.log2(tmax / 127.0)))
    hq = torch.clamp(torch.floor(hif / t[:, None] + 0.5), -127, 127).to(torch.int8)
    lq = torch.clamp(torch.floor(lo * (2048.0 / t)[:, None] + 0.5), -127, 127).to(torch.int8)
    wq = torch.cat([hq, lq], 1).contiguous()
    b = None if bias is None else bias.detach().float().contiguous()
    if fold_bias:
        tail = torch.zeros((n, _x3_tail(k)), dtype=torch.float32, device=w.device)
        if b is not None:
            bh = b.to(torch.float16).float()
            tail[:, 0] = bh
            tail[:, 1] = b - bh
        whi = torch.cat([hi, tail.to(torch.float16)], 1).contiguous()
        return whi, wq, t, None
    return hi.contiguous(), wq, t, b


def x3i8_linear(a_hi, a_q, s_r, w_hi, w_q, t_n, bias=None, k=None):
    """``main + dequant(cross)``: one fp16 GEMM and one int8 GEMM."""
    if k is None:
        k = w_q.shape[1] // 2
    if a_hi.shape[1] != w_hi.shape[1]:
        a_hi = a_hi[:, :w_hi.shape[1]]
    if a_hi.is_cuda:
        main = torch.mm(a_hi, w_hi.t(), out_dtype=torch.float32)
        cross = torch._int_mm(a_q, w_q.t())
    else:
        main = a_hi.float() @ w_hi.float().t()
        cross = (a_q.int() @ w_q.int().t())
    out = main + (s_r[:, None] * t_n[None, :] * (1.0 / 2048.0)) * cross.float()
    if bias is not None:
        out += bias
    return out

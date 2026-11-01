"""Research-only compensated GEMM + exact GELU + split epilogue.

The candidate avoids materializing the FP32 FFN-in result. It is not imported
by production and must win a complete-model T4 comparison before promotion.
Static tile parameters: no autotuning/synchronization inside graph capture.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import fp16x3 as k

if k.HAVE_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _ffn_gemm_gelu_split(A, W, B, OUT, M,
                             STRIDE_A: tl.constexpr, STRIDE_W: tl.constexpr,
                             STRIDE_O: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                             INV: tl.constexpr, HAS_BIAS: tl.constexpr,
                             LO_FIRST: tl.constexpr, TAIL: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        offsets = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0., tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + offsets
            a = tl.load(A + rows[:, None] * STRIDE_A + kk[None, :],
                        mask=(rows[:, None] < M) & (kk[None, :] < K), other=0.)
            w = tl.load(W + cols[None, :] * STRIDE_W + kk[:, None],
                        mask=(cols[None, :] < N) & (kk[:, None] < K), other=0.)
            acc = tl.dot(a, w, acc)
        y = acc * INV
        if HAS_BIAS:
            y += tl.load(B + cols, mask=cols < N, other=0.)[None, :]
        y = .5 * y * (1. + tl.erf(y * .7071067811865476))
        hi = y.to(tl.float16)
        lo = (y - hi.to(tl.float32)).to(tl.float16)
        base = OUT + rows[:, None] * STRIDE_O + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        if LO_FIRST:
            tl.store(base, lo, mask=mask)
            tl.store(base + N, hi, mask=mask)
            tl.store(base + 2 * N, hi, mask=mask)
        else:
            tl.store(base, hi, mask=mask)
            tl.store(base + N, hi, mask=mask)
            tl.store(base + 2 * N, lo, mask=mask)
        if tl.program_id(1) == 0:
            tail = tl.arange(0, 32)
            values = tl.broadcast_to((tail < 2).to(tl.float16)[None, :], (BM, 32))
            tl.store(OUT + rows[:, None] * STRIDE_O + 3 * N + tail[None, :], values,
                     mask=(rows[:, None] < M) & (tail[None, :] < TAIL))


TILES = {"m32k32": (32, 64, 32, 4), "m64k32": (64, 64, 32, 4),
         "m64k64": (64, 64, 64, 4), "m128k32": (128, 64, 32, 8)}


def fused_ffn_split(a3, w3, bias, inv, tile="m64k32"):
    if tile not in TILES:
        raise ValueError("unknown FFN fusion tile")
    if a3.ndim != 2 or w3.ndim != 2 or a3.shape[1] != w3.shape[1]:
        raise ValueError("matching 2-D compensated operands required")
    if a3.dtype != torch.float16 or w3.dtype != torch.float16:
        raise ValueError("compensated operands must be FP16")
    if a3.device != w3.device or (bias is not None and bias.device != a3.device):
        raise ValueError("operands and bias must share a device")
    m, kk = a3.shape
    n = w3.shape[0]
    if not (k.HAVE_TRITON and a3.is_cuda and torch.cuda.get_device_capability(a3.device) == (7, 5)):
        hidden = a3.float() @ w3.float().t()
        return k._x3_act_split_ref(hidden, True, inv, bias)
    if a3.stride(1) != 1 or w3.stride(1) != 1 or n != 128 or kk != 384:
        raise ValueError("T4 candidate supports only N=128, compensated K=384, unit inner strides")
    bm, bn, bk, warps = TILES[tile]
    tail = k.x3_tail(n)
    out = torch.empty((m, 3 * n + tail), device=a3.device, dtype=torch.float16)
    _ffn_gemm_gelu_split[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        a3, w3, bias if bias is not None else a3, out, m,
        STRIDE_A=a3.stride(0), STRIDE_W=w3.stride(0), STRIDE_O=out.stride(0),
        K=kk, N=n, INV=inv, HAS_BIAS=bias is not None, LO_FIRST=k._X3_LO_FIRST,
        TAIL=tail, BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=2,
        enable_fp_fusion=False)
    return out


from user_optimized import UserOptimizedTransformer


class FusionTransformer(UserOptimizedTransformer):
    """Shape-6 experiment only; other sites/layouts retain production compute."""
    def __init__(self, config, tile="m64k32"):
        super().__init__(config)
        if tile not in TILES:
            raise ValueError("unknown FFN fusion tile")
        self.fusion_tile = tile
        self.fused_calls = 0

    def _run_full_x3(self, x, causal):
        if (x.shape[2] != 128 or self.config.ffn_dim != 128
                or self._x3_sites != {"qkv", "out", "ffn_in", "ffn_out"}):
            return super()._run_full_x3(x, causal)
        b, s, d = x.shape
        h, hd = self.config.num_heads, d // self.config.num_heads
        x = x.contiguous().view(b * s, d)
        norm = self.layers[0].norm1
        a3 = k.x3_ln_split(x, norm.weight, norm.bias, norm.eps)
        for i, layer in enumerate(self.layers):
            attn = layer.attention
            wq, _, iq = attn._x3_qkv
            packed = k.x3_linear(a3, wq, None, iq, False)
            q, key, v = [t.reshape(b, s, h, hd).transpose(1, 2) for t in packed.split(d, -1)]
            attention = F.scaled_dot_product_attention(q, key, v, is_causal=causal,
                                                       scale=attn.scale * iq * iq)
            attention = attention.transpose(1, 2).reshape(b * s, d)
            wo, bo, io = attn.out_proj._x3
            projected = k.x3_linear(k.x3_act_split(attention, False, iq)[:, :3*d], wo, None, io, False)
            norm = layer.norm2
            x, a3 = k.x3_add_ln_split(x, projected, norm.weight, norm.bias, norm.eps, io, bo)
            wi, bi, ii = layer.ffn_in._x3
            activation = fused_ffn_split(a3[:, :3*d], wi, bi, ii, self.fusion_tile)
            self.fused_calls += 1
            wf, bf, iff = layer.ffn_out._x3
            f = k.x3_linear(activation[:, :3*d], wf, None, iff, False)
            if i + 1 < len(self.layers):
                norm = self.layers[i + 1].norm1
                x, a3 = k.x3_add_ln_split(x, f, norm.weight, norm.bias, norm.eps, iff, bf)
            else:
                norm = self.final_norm
                return k.x3_add_ln(x, f, norm.weight, norm.bias, norm.eps, iff, bf).view(b, s, d)
        return super()._run_full_x3(x.view(b, s, d), causal)

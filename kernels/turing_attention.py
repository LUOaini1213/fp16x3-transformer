"""Guarded adapter for an optional third-party Turing FP16 attention extension.

No upstream source is bundled. Install the pinned dependency explicitly before
using T3_ATTN=turing. Unsupported inputs and missing extensions use SDPA.
The FP32 graded path is never cast down. This adapter is inference-only.
"""
from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F


TURING_REVISION = "9ef98fcb506bb1e2fe3cece50935e2935bf6b124"
_TURING_EXTENSION = None
_TURING_LOOKED_UP = False
_TURING_CALLS = 0


def turing_eligible(q, k, v):
    """Check every assumption missing from upstream's raw C++ entry point."""
    if not (q.ndim == k.ndim == v.ndim == 4 and q.is_cuda and
            q.device == k.device == v.device and
            q.dtype == k.dtype == v.dtype == torch.float16 and
            q.shape == k.shape == v.shape and q.shape[-1] in (64, 96, 128) and
            q.shape[-2] >= 8192):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return False
    return (q.device.index == torch.cuda.current_device() and
            torch.cuda.get_device_capability(q.device) == (7, 5) and
            torch.cuda.current_stream(q.device) == torch.cuda.default_stream(q.device) and
            not torch.cuda.is_current_stream_capturing())


def _turing_load():
    global _TURING_EXTENSION, _TURING_LOOKED_UP
    if not _TURING_LOOKED_UP:
        _TURING_LOOKED_UP = True
        try:
            _TURING_EXTENSION = importlib.import_module("flash_attn_turing")
            assert callable(_TURING_EXTENSION.fwd)
        except (ImportError, AttributeError, AssertionError) as exc:
            _TURING_EXTENSION = None
            print(f"[Turing attention] optional extension unavailable ({exc}); using SDPA", flush=True)
    return _TURING_EXTENSION


def turing_attention(q, k, v, scale, causal=True):
    global _TURING_CALLS
    extension = _turing_load() if causal and turing_eligible(q, k, v) else None
    if extension is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                              is_causal=causal, scale=scale)
    # Upstream hardcodes contiguous B,S,H,D strides. Last-dimension stride==1
    # alone is not sufficient for a view of a packed QKV projection.
    out, _lse = extension.fwd(q.transpose(1, 2).contiguous(),
                              k.transpose(1, 2).contiguous(),
                              v.transpose(1, 2).contiguous(), float(scale), bool(causal))
    _TURING_CALLS += 1
    return out.transpose(1, 2)

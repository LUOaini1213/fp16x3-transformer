"""Opt-in, zero-workspace cuBLASLt search for the measured wide QKV GEMM.

T3_X3_BLAS=lt requests this backend. Default remains PyTorch. Compilation and
algorithm search happen before capture, never inside a timed graph replay.
Only algorithms/configurations are cached; activations, weights and outputs are
not. Unsupported devices/geometries and failed builds use the existing GEMM.
"""
from __future__ import annotations

import os
from pathlib import Path
import statistics
import threading

import torch


_LT_EXTENSION = None
_LT_FAILED = False
_LT_CHOICES = threading.local()
_LT_RESULTS = []


def _lt_supported(a, w):
    return (a.is_cuda and w.device == a.device and
            a.dtype == w.dtype == torch.float16 and
            a.ndim == w.ndim == 2 and a.stride(1) == w.stride(1) == 1 and
            tuple(a.shape) == (8192, 3104) and tuple(w.shape) == (3072, 3104) and
            a.stride(0) == w.stride(0) == 3104 and
            a.data_ptr() % 256 == w.data_ptr() % 256 == 0 and
            a.device.index == torch.cuda.current_device() and
            torch.cuda.get_device_capability(a.device) == (7, 5))


def _lt_load():
    global _LT_EXTENSION, _LT_FAILED
    if _LT_EXTENSION is None and not _LT_FAILED:
        try:
            from torch.utils.cpp_extension import load_inline
            source = globals().get("NEXT_LT_SOURCE")
            if source is None:
                source = Path(__file__).with_name("cublaslt_probe.cpp").read_text(encoding="utf-8")
            os.environ.setdefault("MAX_JOBS", "1")
            _LT_EXTENSION = load_inline("track3_lt_backend", cpp_sources=source,
                                        extra_cflags=["-O2"], extra_ldflags=["-lcublasLt"],
                                        with_cuda=True, verbose=False)
        except Exception as exc:
            _LT_FAILED = True
            print(f"[cuBLASLt] unavailable ({type(exc).__name__}: {str(exc)[:200]}); using torch", flush=True)
    return _LT_EXTENSION


def lt_candidate_matmul(a, w):
    """Return a fresh FP32 product, or None to request the established fallback."""
    if not _lt_supported(a, w):
        return None
    if not hasattr(_LT_CHOICES, "values"):
        _LT_CHOICES.values = {}
    choices = _LT_CHOICES.values
    key = (a.device.index, tuple(a.shape), tuple(w.shape), a.stride(0), w.stride(0))
    if key not in choices:
        # A graph cannot host a JIT build, heuristic query, event timing or sync.
        if torch.cuda.is_current_stream_capturing():
            return None
        extension = _lt_load()
        choices[key] = None
        if extension is None:
            return None
        try:
            descriptors = extension.algorithms(a, w, 64)
            reference = torch.mm(a, w.t(), out_dtype=torch.float32)
            calls = {"torch": lambda: torch.mm(a, w.t(), out_dtype=torch.float32)}
            for index, desc in enumerate(descriptors):
                if desc[0] != 0 or desc[1] != 0:
                    continue
                candidate = extension.matmul(a, w, index)
                # Preserve numerical behaviour before considering speed.
                good = bool(torch.isfinite(candidate).all()) and torch.allclose(
                    reference, candidate, rtol=1e-5, atol=1e-3)
                del candidate
                if good:
                    calls[str(index)] = lambda ii=index: extension.matmul(a, w, ii)
            del reference
            for call in calls.values():
                for _ in range(3):
                    call()
            times = {name: [] for name in calls}
            names = list(calls)
            for turn in range(3):
                for name in names[turn:] + names[:turn]:
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(12):
                        calls[name]()
                    end.record()
                    end.synchronize()
                    times[name].append(start.elapsed_time(end) / 12)
            medians = {name: statistics.median(values) for name, values in times.items()}
            winner = min(medians, key=medians.get)
            # Do not spend an extra dependency on a noise-level isolated win.
            if winner != "torch" and medians[winner] < medians["torch"] * .98:
                choices[key] = int(winner)
            _LT_RESULTS.append({"key": str(key), "round_ms": times, "median_ms": medians,
                                "selected": choices[key], "descriptors": descriptors})
            print("[cuBLASLt] " + str(_LT_RESULTS[-1]), flush=True)
        except Exception as exc:
            print(f"[cuBLASLt] search failed ({type(exc).__name__}: {str(exc)[:200]}); using torch", flush=True)
    choice = choices[key]
    if choice is None:
        return None
    return _LT_EXTENSION.matmul(a, w, choice)

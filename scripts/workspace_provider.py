"""Research-only wide-QKV Lt workspace search; never imported by production.

Plans retain descriptors/algorithms, not inputs or outputs. Scratch storage is
allocated per call on the current stream, including capture, so concurrent
streams cannot race on a single persistent scratch buffer.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import statistics

import torch

from kernels.cublaslt_backend import _lt_supported


def workspace_source(source):
    replacements = {
        "Plan(const at::Tensor& x, const at::Tensor& w, int limit)":
            "Plan(const at::Tensor& x, const at::Tensor& w, int limit, size_t bytes)",
        "size_t bytes = 0;": "// Workspace limit is passed explicitly by the experiment.",
        "workspace=at::empty({int64_t(bytes)}, x.options().dtype(at::kByte));":
            "// Scratch is per invocation, never shared across streams.",
        "key(const at::Tensor& x, const at::Tensor& w)":
            "key(const at::Tensor& x, const at::Tensor& w, size_t bytes)",
        "s << x.get_device()": "s << bytes << ':' << x.get_device()",
        "algorithms(const at::Tensor& x, const at::Tensor& w, int limit)":
            "algorithms(const at::Tensor& x, const at::Tensor& w, int limit, size_t bytes)",
        "key(x,w)": "key(x,w,bytes)",
        "std::make_unique<Plan>(x,w,limit)": "std::make_unique<Plan>(x,w,limit,bytes)",
        "matmul(const at::Tensor& x, const at::Tensor& w, int index)":
            "matmul(const at::Tensor& x, const at::Tensor& w, int index, size_t bytes)",
        "float alpha=1.0f, beta=0.0f;":
            "auto scratch=at::empty({int64_t(bytes)}, x.options().dtype(at::kByte));\n    float alpha=1.0f, beta=0.0f;",
        "p->workspace.data_ptr(), p->workspace.numel()": "scratch.data_ptr(), scratch.numel()",
        "// Handles are thread-local. Only zero-workspace algorithms are queried: there\n// is no scratch buffer to race when graph capture uses a different CUDA stream.":
            "// Handles are thread-local; workspace budget is part of the plan key.\n// Each invocation has stream-local scratch, including graph capture.",
    }
    for old, new in replacements.items():
        expected = 2 if old == "key(x,w)" else 1
        if source.count(old) != expected:
            raise ValueError(f"CPP source changed; refusing a partial workspace patch: {old}")
        source = source.replace(old, new)
    return source


def load_workspace_extension():
    from torch.utils.cpp_extension import load_inline
    source = workspace_source(Path("kernels/cublaslt_probe.cpp").read_text(encoding="utf-8"))
    suffix = hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline("track3_workspace_" + suffix, cpp_sources=source,
                       extra_cflags=["-O2"], extra_ldflags=["-lcublasLt"],
                       with_cuda=True, verbose=True)


class WorkspaceProvider:
    def __init__(self, extension, workspace_bytes):
        if workspace_bytes not in (0, 1 << 20, 4 << 20, 16 << 20, 32 << 20):
            raise ValueError("unsupported workspace budget")
        self.extension = extension
        self.workspace_bytes = workspace_bytes
        self.choices = {}
        self.searches = []
        self.calls = 0

    def __call__(self, a, w):
        if not _lt_supported(a, w):
            return None
        key = (a.device.index, tuple(a.shape), tuple(w.shape), a.stride(0), w.stride(0))
        if key not in self.choices:
            if torch.cuda.is_current_stream_capturing():
                return None
            self.choices[key] = self.search(a, w)
        index = self.choices[key]
        if index is None:
            return None
        self.calls += 1
        return self.extension.matmul(a, w, index, self.workspace_bytes)

    def search(self, a, w):
        descriptions = self.extension.algorithms(a, w, 64, self.workspace_bytes)
        ref = torch.mm(a, w.t(), out_dtype=torch.float32)
        calls = {"torch": lambda: torch.mm(a, w.t(), out_dtype=torch.float32)}
        failures = {}
        for index, desc in enumerate(descriptions):
            if desc[0] != 0 or desc[1] > self.workspace_bytes:
                continue
            try:
                out = self.extension.matmul(a, w, index, self.workspace_bytes)
                if not (bool(torch.isfinite(out).all()) and torch.allclose(ref, out, rtol=1e-5, atol=1e-3)):
                    raise RuntimeError("raw GEMM numerical gate failed")
                calls[str(index)] = lambda index=index: self.extension.matmul(a, w, index, self.workspace_bytes)
                del out
            except RuntimeError as exc:
                failures[str(index)] = str(exc)[:500]
        del ref
        for call in calls.values():
            for _ in range(3):
                call()
        rounds = {name: [] for name in calls}
        names = list(calls)
        for turn in range(3):
            shift = turn % len(names)
            for name in names[shift:] + names[:shift]:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(12):
                    calls[name]()
                end.record()
                end.synchronize()
                rounds[name].append(start.elapsed_time(end) / 12)
        medians = {name: statistics.median(values) for name, values in rounds.items()}
        winner = min(medians, key=medians.get)
        choice = int(winner) if winner != "torch" and medians[winner] < medians["torch"] * .98 else None
        self.searches.append({"workspace_limit_bytes": self.workspace_bytes, "descriptors": descriptions,
                              "failures": failures, "round_ms": rounds, "median_ms": medians,
                              "selected": choice,
                              "selected_workspace_bytes": descriptions[choice][1] if choice is not None else None})
        print("WORKSPACE_SEARCH " + str(self.searches[-1]), flush=True)
        return choice

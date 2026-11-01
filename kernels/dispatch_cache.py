"""Opt-in JSON dispatch hints, never code, tensors, outputs or CUDA graphs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from functools import lru_cache

SCHEMA = 1
MAX_BYTES = 1024 * 1024
MAX_ENTRIES = 128
ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def source_stamp():
    paths = [ROOT / "user_optimized.py", ROOT / "torch_transformer_benchmark.py",
             *sorted((ROOT / "kernels").glob("*.py")),
             *sorted((ROOT / "kernels").glob("*.cpp"))]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in paths}


def context(model, x, all_valid):
    import torch
    props = torch.cuda.get_device_properties(x.device)
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    cfg = model.config
    return {"schema": SCHEMA, "source": source_stamp(), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "triton": triton_version,
            "gpu": props.name, "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
            "device_uuid": str(getattr(props, "uuid", "unavailable")),
            "shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype),
            "device_index": x.device.index, "all_valid": bool(all_valid),
            "model": type(model).__module__ + "." + type(model).__qualname__,
            "config": {n: getattr(cfg, n) for n in
                       ("d_model", "num_heads", "ffn_dim", "num_layers", "causal")},
            "x3": model._x3_on, "autocast": str(model._autocast_dtype),
            "grad_enabled": torch.is_grad_enabled(), "inference": torch.is_inference_mode_enabled(),
            "parameters": [[list(p.shape), list(p.stride()), str(p.dtype)] for p in model.parameters()],
            "math": {"tf32": torch.backends.cuda.matmul.allow_tf32,
                     "fp16_reduced": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                     "deterministic": torch.are_deterministic_algorithms_enabled()},
            "settings": {k: v for k, v in sorted(os.environ.items())
                         if k.startswith("T3_") and k != "T3_TUNE_CACHE"}}


def key_for(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            return None
        if not isinstance(payload.get("entries"), dict):
            return None
        return payload
    except (OSError, ValueError, UnicodeError):
        return None


def lookup(path, value):
    payload = _read(Path(path))
    if payload is None:
        return None
    record = payload["entries"].get(key_for(value))
    if not isinstance(record, dict) or record.get("context") != value:
        return None
    choice = record.get("choice")
    return choice if choice in ("eager", "graph") else None


def remember(path, value, choice):
    """Atomic replacement; corrupt/foreign files are left untouched.

    Concurrent writers may lose a hint, which is only a future cache miss.
    Every read validates the complete context and a small strategy allowlist.
    """
    if choice not in ("eager", "graph"):
        return False
    path = Path(path)
    temporary = None
    try:
        payload = _read(path) if path.exists() else {"schema": SCHEMA, "entries": {}}
        if payload is None:
            return False
        entries = payload["entries"]
        cache_key = key_for(value)
        entries.pop(cache_key, None)
        entries[cache_key] = {"context": value, "choice": choice}
        payload["entries"] = dict(list(entries.items())[-MAX_ENTRIES:])
        data = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        if len(data) > MAX_BYTES:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
        temporary = None
        return True
    except (OSError, ValueError, TypeError):
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass

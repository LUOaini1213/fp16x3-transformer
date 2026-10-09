"""One-command, accuracy-checked quick/steady inference (no automatic installs).

python -m scripts.run --mode quick --shape 2
python -m scripts.run --mode steady --device cuda --shapes 1-13
python -m scripts.run --check-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time


PROFILES = {
    # No Inductor, CUDA graph capture, native extension build, or Triton kernels.
    # FP32 GEMMs deliberately trade steady throughput for cheap first use.
    "quick": {"T3_COMPILE": "0", "T3_CUDAGRAPH": "0", "T3_LINEAR": "fp32"},
    "steady": {"T3_COMPILE": "auto", "T3_CUDAGRAPH": "1", "T3_LINEAR": "fp16x3"},
}
COMMON = {"T3_AUTOCAST": "off", "T3_ATTN": "sdpa", "T3_X3_BLAS": "torch",
          "T3_FUSED_QKV": "0", "T3_TRITON": "0", "T3_X3_GUARD": "static+first",
          "T3_X3_ORDER": "lhh", "T3_X3_SPLITK": "1", "T3_X3_SITES": "auto",
          "T3_MASK_CACHE": "1", "T3_PACK_PADDING": "1"}


def configure(mode):
    """Explicit CLI profiles ignore inherited experimental T3_* toggles."""
    if mode not in PROFILES:
        raise ValueError(f"unknown mode: {mode}")
    for name in list(os.environ):
        if name.startswith("T3_"):
            del os.environ[name]
    os.environ.update(COMMON, **PROFILES[mode])
    return {**COMMON, **PROFILES[mode]}


def parse_shapes(value):
    shapes = []
    for token in value.split(","):
        endpoints = token.strip().split("-")
        if len(endpoints) == 1:
            part = [int(endpoints[0])]
        elif len(endpoints) == 2:
            lo, hi = map(int, endpoints)
            if hi < lo:
                raise ValueError("shape ranges must be ascending")
            part = list(range(lo, hi + 1))
        else:
            raise ValueError("use shape numbers or ascending ranges, e.g. 1-13")
        shapes.extend(part)
    if not shapes or any(i not in range(1, 14) for i in shapes):
        raise ValueError("only gradeable shapes 1-13 are supported; shape 14 has no full FP32 oracle")
    return list(dict.fromkeys(shapes))


def environment_report(requested="auto"):
    import torch
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("No usable CUDA GPU. Use --device cpu for correctness only, or a Linux Colab/Kaggle T4.")
    device = "cuda" if available and requested != "cpu" else "cpu"
    report = {"python": sys.version.split()[0], "torch": str(torch.__version__),
              "cuda_build": torch.version.cuda, "device": device,
              "triton_installed": importlib.util.find_spec("triton") is not None,
              "native_toolchain": {n: shutil.which(n) is not None for n in ("nvcc", "ninja", "c++")},
              "tested_accelerated_environment": "Linux Tesla T4 / PyTorch 2.11.0+cu128",
              "warnings": []}
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
        report["capability"] = list(torch.cuda.get_device_capability())
        # Verify that the installed wheel actually runs on the selected GPU.
        a = torch.ones(2, 2, device="cuda", dtype=torch.float16)
        torch.mm(a.float(), a.float())
        try:
            out = torch.mm(a, a, out_dtype=torch.float32)
            torch.cuda.synchronize()
            report["fp16_to_fp32_mm"] = out.dtype == torch.float32
        except (TypeError, RuntimeError) as exc:
            report["fp16_to_fp32_mm"] = False
            report["warnings"].append(f"Compensated GEMM unavailable; model can fall back: {type(exc).__name__}")
        if not report["triton_installed"]:
            report["warnings"].append("Triton is absent; accelerated steady kernels may fall back.")
        if not str(torch.__version__).startswith("2.11."):
            report["warnings"].append("Not the pinned T4 validation version; this run validates its own outputs only.")
    else:
        report["warnings"].append("CPU is for correctness/authoring only; these timings are not GPU performance evidence.")
    return report


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_case(shape, mode, device_name, repeats=10):
    profile = configure(mode)  # before importing import-time experimental flags
    import torch
    import torch_transformer_benchmark as official
    from scripts.benchmark_release import config
    from user_optimized import UserOptimizedTransformer

    if device_name == "cpu" and shape == 6:
        raise ValueError("Shape 6 is deliberately refused on CPU (large oracle); use a T4 instead.")
    cfg = config(shape)
    device = torch.device(device_name)
    torch.manual_seed(141009 + cfg.d_model + cfg.seq_len)
    base = official.BaselineTransformer(cfg).eval()
    model = UserOptimizedTransformer(cfg).eval()
    official.copy_model_weights(base, model, strict=True)
    model.to(device)
    x, mask = official.generate_random_case(cfg, device, torch.float32, 20261009, 0., 1.)
    with torch.inference_mode():
        sync(torch, device)
        start = time.perf_counter()
        out = model(x, mask)
        sync(torch, device)
        first = time.perf_counter() - start
        # Reference work is excluded from the first public forward and steady timing.
        base.to(device)
        ref = base(x, mask)
        check = official.compare_outputs(ref, out, rtol=.02, atol=.002)
        if not check.passed:
            raise RuntimeError(f"official correctness gate failed: {check}")
        checks = [{"seed": 20261009, "passed": check.passed, "failed": check.failed_elements,
                   "max_abs": check.max_abs_error}]
        del ref, out
        for seed in (20261010, 20261011):
            xx, mm = official.generate_random_case(cfg, device, torch.float32, seed, 0., 1.)
            ref, candidate = base(xx, mm), model(xx, mm)
            trial = official.compare_outputs(ref, candidate, rtol=.02, atol=.002)
            if not trial.passed:
                raise RuntimeError(f"official correctness gate failed at seed {seed}: {trial}")
            checks.append({"seed": seed, "passed": trial.passed, "failed": trial.failed_elements,
                           "max_abs": trial.max_abs_error})
            del xx, mm, ref, candidate
        del base
        times = []
        for _ in range(3):
            model(x, mask)
        for _ in range(repeats):
            sync(torch, device)
            start = time.perf_counter()
            out = model(x, mask)
            sync(torch, device)
            times.append((time.perf_counter() - start) * 1000)
            del out
    return {"shape": shape, "mode": mode, "profile": profile, "device": device_name,
            "first_forward_seconds": first, "steady_wall_ms": statistics.median(times),
            "steady_wall_samples_ms": times, "accuracy": {"passed": True,
            "failed": 0, "max_abs": max(c["max_abs"] for c in checks)}, "accuracy_trials": checks,
            "dispatch": {"x3": model._x3_on, "compiled": model._compiled is not None,
                         "graph": model._graph is not None},
            "timing_scope": "first public forward excludes process/import/model/input setup; synchronized wall; no paired speedup claim"}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--mode", choices=tuple(PROFILES), default="quick")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--shape", type=int, choices=range(1, 14), default=2)
    group.add_argument("--shapes", help="fresh worker per shape, e.g. 1-13 or 2,8")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--output", type=Path, default=Path("results/local_run.json"))
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    try:
        if args.repeats < 1:
            raise ValueError("--repeats must be positive")
        configure(args.mode)
        env = environment_report(args.device)
        print(json.dumps(env, indent=2), flush=True)
        if args.check_only:
            write_json(args.output, {"environment": env})
            return 0
        if args.shapes:
            rows = []
            for shape in parse_shapes(args.shapes):
                child = args.output.with_name(args.output.stem + f"_shape{shape}.json")
                subprocess.run([sys.executable, "-m", "scripts.run", "--worker", "--mode", args.mode,
                                "--device", env["device"], "--shape", str(shape), "--repeats", str(args.repeats),
                                "--output", str(child)], check=True)
                rows += json.loads(child.read_text(encoding="utf-8"))["results"]
                write_json(args.output, {"environment": env, "results": rows})
        else:
            row = run_case(args.shape, args.mode, env["device"], args.repeats)
            write_json(args.output, {"environment": env, "results": [row]})
            print(json.dumps(row, indent=2), flush=True)
        return 0
    except (ImportError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}\nInstall a suitable PyTorch build yourself; this command never changes the environment.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

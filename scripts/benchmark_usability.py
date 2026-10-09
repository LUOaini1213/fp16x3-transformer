"""Exact-commit cloud validation of quick startup and targeted shape-8 workspace.

Production core files remain unchanged. All experimental scratch allocation,
dispatch, graph ownership and input-copy costs are inside public-forward timing.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from scripts import benchmark_release as release
from scripts.run import configure


ROOT = Path(__file__).resolve().parents[1]
EXTRA = ["scripts/run.py", "scripts/benchmark_usability.py", "scripts/workspace_provider.py"]


def save(output, name, rows):
    meta = release.metadata()
    meta["source_manifest"].update({n: hashlib.sha256((ROOT / n).read_text(encoding="utf-8").encode()).hexdigest()
                                    for n in EXTRA})
    meta["protocol"] = "quick: one fresh process per mode/shape, first forward and 10 synchronized wall samples; workspace: three rotated paired event/wall rounds, normal public-forward dispatch"
    payload = {"metadata": meta, "results": rows}
    (output / f"next_usability_{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("USABILITY_" + name.upper() + " " + json.dumps(payload), flush=True)


def quick(output):
    rows = []
    for index in range(1, 14):
        row = {"shape": index, "profiles": {}}
        order = ("quick", "steady") if index % 2 else ("steady", "quick")
        for mode in order:
            target = output / f"usability_case_{index}_{mode}.json"
            subprocess.run([sys.executable, "-m", "scripts.run", "--mode", mode, "--device", "cuda",
                            "--shape", str(index), "--output", str(target)], cwd=ROOT, check=True)
            payload = json.loads(target.read_text(encoding="utf-8"))
            row["profiles"][mode] = payload["results"][0]
        rows.append(row)
        save(output, "quick", rows)


def workspace(output):
    from scripts.workspace_provider import load_workspace_extension, WorkspaceProvider
    configure("steady")
    torch.manual_seed(1783)
    start = time.perf_counter()
    ext = load_workspace_extension()
    build_seconds = time.perf_counter() - start
    cfg = release.config(8)
    base = release.baseline(cfg).cuda()
    x3 = importlib.import_module("kernels.fp16x3")
    models, providers, cold, accuracy = {}, {}, {}, []
    variants = ("default", "production_lt", "ws0", "ws1", "ws4", "ws16", "ws32")
    for name in variants:
        models[name] = release.make(cfg, base, "lt" if name == "production_lt" else "default")
        if name.startswith("ws"):
            providers[name] = WorkspaceProvider(ext, int(name[2:]) << 20)
    production_backend = importlib.import_module("kernels.cublaslt_backend")
    production_provider = production_backend.lt_candidate_matmul

    def forward(name, x, mask):
        x3._X3_BLAS = "torch" if name == "default" else "lt"
        x3.lt_candidate_matmul = providers.get(name, production_provider)
        return models[name](x, mask)

    calls = {n: lambda xx, mm, n=n: forward(n, xx, mm) for n in variants}
    x, mask = release.inputs(cfg)
    with torch.inference_mode():
        for name, call in calls.items():
            out, duration = release.synchronized(call, x, mask)
            cold[name] = duration
            del out
        for seed in range(3):
            xx, mm = release.inputs(cfg, 20261009 + seed)
            ref = base(xx, mm)
            for name, call in calls.items():
                out = call(xx, mm)
                accuracy.append({"variant": name, "seed": seed, **release.checked(ref, out)})
                prior = out.clone()
                call(xx * .5, mm)
                assert torch.equal(out, prior), (name, "output alias")
                del out, prior
            del ref, xx, mm
        times = release.paired(calls, x, mask, 20)
        assert any(s["selected"] is not None for s in production_backend._LT_RESULTS), "production Lt did not execute"
        extra_peaks = {}
        for name, call in calls.items():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            out = call(x, mask)
            torch.cuda.synchronize()
            extra_peaks[name] = torch.cuda.max_memory_allocated() - resident
            del out
    results = {"shape": 8, "build_seconds": build_seconds, "cold_seconds": cold,
               "accuracy": accuracy, "timing": times,
               "dispatch": {n: release.modes(m) for n, m in models.items()},
               "search": {n: p.searches for n, p in providers.items()},
               "production_lt_search": production_backend._LT_RESULTS,
               "workspace_calls": {n: p.calls for n, p in providers.items()},
               "shared_process_extra_peak_bytes": extra_peaks,
               "memory_scope": "incremental allocated peak above seven resident models; not isolated deployment memory",
               "scratch_policy": "allocation per invocation/current stream; allocations are included in timing/capture",
               "counter_scope": "host extension invocations, including capture; graph replays do not increment Python counters",
               "acceptance": "at least 5% event AND wall reduction vs production_lt plus independent confirmation; no default promotion from this experiment"}
    current = times["production_lt"]
    results["qualified_candidates"] = [n for n in providers if providers[n].calls and
        times[n]["event_ms"] < current["event_ms"] * .95 and times[n]["wall_ms"] < current["wall_ms"] * .95]
    save(output, "workspace", results)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--phase", choices=("quick", "workspace"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5), "T4 required"
    os.environ.setdefault("MAX_JOBS", "1")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/track3-usability-inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/track3-usability-triton")
    {"quick": quick, "workspace": workspace}[args.phase](args.output.resolve())


if __name__ == "__main__":
    main()

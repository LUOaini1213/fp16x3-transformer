"""Source-pinned T4 validation: profiles, cache/recovery contracts, FFN fusion.

Cold profiles have separate empty compiler directories. Cache-hit and uncached
controls share warmed compiler directories but run in fresh processes. Neither
timing includes Python imports/context probes. Production defaults are unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from scripts.run import configure
configure("balanced")  # before importing any import-time kernel settings

import torch
from scripts import benchmark_release as release

ROOT = Path(__file__).resolve().parents[1]
EXTRA = ["scripts/run.py", "scripts/benchmark_improvements.py", "kernels/dispatch_cache.py",
         "kernels/ffn_fusion.py"]


def save(output, name, results):
    metadata = release.metadata()
    metadata["source_manifest"].update({n: hashlib.sha256((ROOT / n).read_text(encoding="utf-8").encode()).hexdigest()
                                        for n in EXTRA})
    metadata["protocol"] = __doc__
    metadata["clean_checkout"] = not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    payload = {"metadata": metadata, "results": results}
    (output / f"next_improvements_{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("IMPROVEMENTS_" + name.upper() + " " + json.dumps(payload), flush=True)


def pair_worker(index, output):
    cfg = release.config(index)
    base = release.baseline(cfg)
    calls = {"baseline": base.cuda()}
    for name in ("quick", "balanced", "steady"):
        configure(name)
        calls[name] = release.copy_candidate(cfg, base).cuda()
    x, mask = release.inputs(cfg)
    accuracy, setup = [], {}
    with torch.inference_mode():
        for name, model in calls.items():
            out, elapsed = release.synchronized(model, x, mask)
            setup[name] = elapsed
            del out
        for seed in range(3):
            xx, mm = release.inputs(cfg, 20261009 + seed)
            ref = base(xx, mm)
            for name in ("quick", "balanced", "steady"):
                out = calls[name](xx, mm)
                accuracy.append({"variant": name, "seed": seed, **release.checked(ref, out)})
                previous = out.clone()
                calls[name](xx * .5, mm)
                assert torch.equal(out, previous), (name, "output alias")
                del out, previous
            del ref, xx, mm
        timing = release.paired(calls, x, mask, 6 if index == 6 else 20)
    save(output, f"pair_{index}", {"shape": index, "accuracy": accuracy, "timing": timing,
         "shared_process_setup_seconds": setup, "dispatch": {n: release.modes(m) for n, m in calls.items()},
         "output_ownership": "PASS", "cache": "disabled",
         "memory_scope": "paired models share a process; not isolated deployment memory"})


def cli_case(output, shape, mode, label, compiler_dir, cache=None):
    path = output / f"profile_{shape}_{label}.json"
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = str(compiler_dir / "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_dir / "inductor")
    command = [sys.executable, "-m", "scripts.run", "--mode", mode, "--device", "cuda",
               "--shape", str(shape), "--repeats", "10", "--output", str(path)]
    if cache is not None:
        command += ["--tune-cache", str(cache)]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return json.loads(path.read_text(encoding="utf-8"))["results"][0]


def contracts(output):
    configure("balanced")
    cfg = release.config(2)
    base = release.baseline(cfg).cuda()
    model = release.copy_candidate(cfg, base).cuda()
    x, mask = release.inputs(cfg)
    with torch.inference_mode():
        ref = base(x, mask)
        release.checked(ref, model(x, mask))
        # Guarantee the mutation test starts from a real captured graph, even
        # if this short shape's noisy initial selection happened to pick eager.
        assert model._capture_graph(x, mask, lambda fn, xx, mm, av: fn(xx, mm, True, av))
        model._tuned = True
        owned = model(x, mask)
        previous = owned.clone()
        release.checked(base(x * .5, mask), model(x * .5, mask))
        assert torch.equal(owned, previous)
        model.layers[0].norm1.bias.add_(.125)
        base.layers[0].norm1.bias.add_(.125)
        release.checked(base(x, mask), model(x, mask))
        assert model._tuned and model._tune_result is not None, "weight refresh did not trigger one re-selection"
        result_after_change = model._tune_result
        release.checked(base(x, mask), model(x, mask))
        assert model._tune_result is result_after_change, "stable weights retuned again"
        mask[0, 17:] = False
        release.checked(base(x, mask), model(x, mask))
        mask.fill_(True)
        inference_mask = mask.clone()
        inference_mask[0, 23:] = False
        release.checked(base(x, inference_mask), model(x, inference_mask))
        # Sequential use on another stream, not concurrent shared-model use.
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            side_out = model(x, mask)
        side.synchronize()
        release.checked(base(x, mask), side_out)
    save(output, "contracts", {"status": "PASS", "dispatch": release.modes(model),
         "checks": ["owned output", "changed input", "mutable normal mask", "mutable inference mask",
                    "weight change re-selects once", "stable weights do not retune", "sequential alternate stream"],
         "concurrency_scope": "no concurrent calls on a single mutable graph instance"})


def cache_trials(output):
    directory = Path(tempfile.mkdtemp(prefix="track3-dispatch-compiler-"))
    cache = output / "dispatch_hints.json"
    producer = cli_case(output, 2, "balanced", "cache_producer", directory, cache)
    consumer = cli_case(output, 2, "balanced", "cache_hit", directory, cache)
    control = cli_case(output, 2, "balanced", "warm_uncached", directory)
    assert producer["tune_cache"]["status"].startswith("stored-")
    assert consumer["tune_cache"]["status"].startswith("hit-validated-")
    assert not consumer["dispatch"]["compiled"]
    save(output, "cache", {"producer": producer, "hit": consumer, "warm_uncached": control,
         "scope": "fresh workers, same warmed compiler directories for hit/control; only hints persist, graph recaptured"})


def sweep(output, indices, name):
    rows = []
    for index in indices:
        cold = {}
        order = ("quick", "balanced", "steady") if index % 2 else ("steady", "balanced", "quick")
        for mode in order:
            # Each cold worker gets an empty compiler directory, rather than
            # benefiting from another mode's compiled kernels in the sweep.
            directory = Path(tempfile.mkdtemp(prefix=f"track3-cold-{index}-{mode}-"))
            cold[mode] = cli_case(output, index, mode, mode, directory)
        subprocess.run([sys.executable, "-m", "scripts.benchmark_improvements", "--phase", "pair",
                        "--shape", str(index), "--output", str(output)], cwd=ROOT, check=True)
        pair = json.loads((output / f"next_improvements_pair_{index}.json").read_text())["results"]
        rows.append({"shape": index, "cold": cold, "paired": pair})
        save(output, name, rows)
    cache_trials(output)
    subprocess.run([sys.executable, "-m", "scripts.benchmark_improvements", "--phase", "contracts",
                    "--output", str(output)], cwd=ROOT, check=True)


def profile(model, x, mask):
    with torch.inference_mode():
        for _ in range(3):
            model(x, mask)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                model(x, mask)
            torch.cuda.synchronize()
    device = defaultdict(float)
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            device[event.name] += event.time_range.elapsed_us()
    total = sum(device.values())
    return {"device_total_us": total, "kernels": [{"name": n, "total_us": v,
             "device_share": v / total} for n, v in sorted(device.items(), key=lambda item: -item[1])],
            "scope": "three profiled forwards, summed device events; not end-to-end latency shares"}


def fusion(output):
    from kernels.ffn_fusion import FusionTransformer, TILES
    configure("balanced")
    cfg = release.config(6)
    base = release.baseline(cfg).cuda()
    models = {"balanced": release.copy_candidate(cfg, base).cuda()}
    for tile in TILES:
        model = FusionTransformer(cfg, tile).eval()
        model.load_state_dict(base.state_dict(), strict=True)
        models[tile] = model.cuda()
    x, mask = release.inputs(cfg)
    accuracy, failures, setup = [], {}, {}
    accepted = {}
    with torch.inference_mode():
        for name, model in models.items():
            try:
                out, elapsed = release.synchronized(model, x, mask)
                assert torch.isfinite(out).all()
                del out
                setup[name] = elapsed
                accepted[name] = model
            except Exception as exc:
                if name == "balanced":
                    raise
                failures[name] = f"{type(exc).__name__}: {exc}"
        for seed in range(3):
            xx, mm = release.inputs(cfg, 20261009 + seed)
            ref = base(xx, mm)
            for name, model in list(accepted.items()):
                try:
                    out = model(xx, mm)
                    accuracy.append({"variant": name, "seed": seed, **release.checked(ref, out)})
                    saved = out.clone()
                    model(xx * .5, mm)
                    assert torch.equal(saved, out)
                    del out, saved
                except Exception as exc:
                    if name == "balanced":
                        raise
                    failures[name] = f"{type(exc).__name__}: {exc}"
                    del accepted[name]
            del xx, mm, ref
        # Oracle no longer contributes resident memory during profiler/timing.
        del base
        gc.collect()
        torch.cuda.empty_cache()
        timing = release.paired(accepted, x, mask, 6)
        profiles = {"balanced": profile(models["balanced"], x, mask)}
        winners = [name for name in accepted if name != "balanced"
                   and timing[name]["event_ms"] < timing["balanced"]["event_ms"] * .97
                   and timing[name]["wall_ms"] < timing["balanced"]["wall_ms"] * .97]
        for name in winners:
            profiles[name] = profile(models[name], x, mask)
    save(output, "fusion", {"shape": 6, "accuracy": accuracy, "failures": failures, "timing": timing,
         "setup_seconds": setup, "profiles": profiles, "qualified_candidates": winners,
         "fused_calls": {n: getattr(m, "fused_calls", 0) for n, m in accepted.items()},
         "dispatch": {n: release.modes(m) for n, m in accepted.items()},
         "acceptance": "all three original gates; at least 3% event AND wall win plus independent confirmation before promotion",
         "default_changed": False})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("pilot", "full", "pair", "contracts", "fusion"), required=True)
    parser.add_argument("--shape", type=int, choices=range(1, 14), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5), "Tesla T4 required"
    os.environ.setdefault("MAX_JOBS", "1")
    if args.phase == "pair":
        pair_worker(args.shape, output)
    elif args.phase == "contracts":
        contracts(output)
    elif args.phase == "fusion":
        fusion(output)
    else:
        sweep(output, (2, 8, 13) if args.phase == "pilot" else range(1, 14), args.phase)


if __name__ == "__main__":
    main()

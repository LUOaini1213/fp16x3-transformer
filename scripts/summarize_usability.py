"""Build a checked report from three independent sessions and usability trials."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from scripts.summarize_release import validate_rounds


def core_manifest(payload):
    return {n: h for n, h in payload["metadata"]["source_manifest"].items()
            if n.startswith("kernels/") or n in ("user_optimized.py", "torch_transformer_benchmark.py")}


def independent_metrics(payloads):
    if len(payloads) != 3:
        raise ValueError("exactly three independent sessions required")
    if len({p["metadata"]["normal_imports"]["production"] for p in payloads}) != 3:
        raise ValueError("independent normal-import checkout paths required")
    if len({hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() for p in payloads}) != 3:
        raise ValueError("duplicated session evidence")
    core = core_manifest(payloads[0])
    if len(core) != 9 or any(core_manifest(p) != core for p in payloads[1:]):
        raise ValueError("production core hashes differ")
    env = [(p["metadata"]["torch"], p["metadata"]["cuda"], p["metadata"]["gpu"]) for p in payloads]
    if len(set(env)) != 1:
        raise ValueError("different GPU/Torch environments must not be pooled")
    errors = []
    for p in payloads:
        assert [r["shape"] for r in p["results"]] == list(range(1, 14)), "incomplete session"
        for row in p["results"]:
            for name in ("baseline", "default"):
                validate_rounds(row["timing"][name])
            checks = [c for c in row["accuracy"] if c["variant"] == "default"]
            assert len(checks) == 3 and {c["seed"] for c in checks} == {0, 1, 2}
            assert all(c["passed"] and c["failed"] == 0 and math.isfinite(c["max_abs"]) for c in checks)
            errors.extend(c["max_abs"] for c in checks)
    rows = []
    session_ratios = [[] for _ in payloads]
    for index in range(13):
        shapes = [p["results"][index] for p in payloads]
        assert len({tuple(r["dims"]) for r in shapes}) == 1
        ratios, wall_ratios = [], []
        for session, row in enumerate(shapes):
            timing = row["timing"]
            ratios.append(timing["baseline"]["event_ms"] / timing["default"]["event_ms"])
            wall_ratios.append(timing["baseline"]["wall_ms"] / timing["default"]["wall_ms"])
            session_ratios[session].append(ratios[-1])
        rows.append({"shape": index + 1, "speedup": statistics.median(ratios),
                     "range": (min(ratios), max(ratios)), "wall_speedup": statistics.median(wall_ratios),
                     "default_event_ms": statistics.median(r["timing"]["default"]["event_ms"] for r in shapes),
                     "first_seconds": statistics.median(r["cold_seconds"]["default"] for r in shapes)})
    return {"rows": rows, "median_speedup": statistics.median(r["speedup"] for r in rows),
            "session_medians": [statistics.median(values) for values in session_ratios],
            "max_abs": max(errors), "comparisons": len(errors)}


def independent_text(payloads):
    result = independent_metrics(payloads)
    text = ["## Three independent T4 sessions", "",
            "All nine production-core hashes match. Each session has a fresh clone/cache environment, "
            "13 shapes, three rotated paired event/wall rounds and three original-FP32 input checks. "
            "Aggregate = median of the three paired speedup ratios per shape, then unweighted median over shapes. "
            "Ranges are observed three-session ranges, not confidence intervals or universal guarantees.", "",
            "Session median shape speedups: " + ", ".join(f"**{v:.3f}×**" for v in result["session_medians"]) + ".", "",
            "| Shape | Median default event ms | Median event speedup | Session range | Median wall speedup | Median first call s |",
            "|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        text.append(f'| {r["shape"]} | {r["default_event_ms"]:.4f} | {r["speedup"]:.3f}× | '
                    f'{r["range"][0]:.3f}–{r["range"][1]:.3f}× | {r["wall_speedup"]:.3f}× | {r["first_seconds"]:.2f} |')
    text.extend(["", f'Aggregate median speedup: **{result["median_speedup"]:.3f}×**. '
                 f'All **{result["comparisons"]}** default FP32 checks pass; maximum absolute error **{result["max_abs"]:.8g}**.'])
    return "\n".join(text)


def crossover_calls(quick, steady):
    startup = steady["first_forward_seconds"] - quick["first_forward_seconds"]
    saving = quick["steady_wall_ms"] - steady["steady_wall_ms"]
    if saving <= 0:
        return None
    return 1 + max(0, math.ceil(startup * 1000 / saving))


def quick_text(payload):
    rows = payload["results"]
    assert [r["shape"] for r in rows] == list(range(1, 14)), "incomplete quick/steady sweep"
    text = ["## Fast first use versus steady throughput", "",
            "One fresh child process per mode/shape, three original-FP32 input checks and ten synchronized "
            "wall samples. The first public forward excludes Python imports, model/input setup and CUDA "
            "context initialization/environment probes. Job-local compiler caches are shared across children. "
            "These are not three-session paired performance figures.", "",
            "Quick = eager FP32 GEMMs + SDPA; steady = unchanged fp16x3 + automatic compile/manual graphs. "
            "Crossover is an estimate from first-forward plus repeated steady latency, not a deployment SLA.", "",
            "| Shape | Quick / steady first s | Quick / steady wall ms | Approx. steady crossover calls |",
            "|---:|---:|---:|---:|"]
    errors = []
    for row in rows:
        q, s = (row["profiles"][name] for name in ("quick", "steady"))
        for v in (q, s):
            trials = v["accuracy_trials"]
            assert len(trials) == 3 and all(c["passed"] and c["failed"] == 0 for c in trials)
            assert len(v["steady_wall_samples_ms"]) == 10
            assert v["steady_wall_ms"] == statistics.median(v["steady_wall_samples_ms"])
            assert all(math.isfinite(t) and t > 0 for t in v["steady_wall_samples_ms"])
            assert math.isfinite(v["first_forward_seconds"]) and v["first_forward_seconds"] > 0
            errors += [c["max_abs"] for c in trials]
        assert not any(q["dispatch"].values()), "quick must not compile, graph-capture or use x3"
        cross = crossover_calls(q, s)
        text.append(f'| {row["shape"]} | {q["first_forward_seconds"]:.4f} / {s["first_forward_seconds"]:.2f} | '
                    f'{q["steady_wall_ms"]:.4f} / {s["steady_wall_ms"]:.4f} | {cross if cross is not None else "no steady advantage measured"} |')
    text.extend(["", f"All **78** profile/input checks pass. Maximum absolute error: **{max(errors):.8g}**."])
    return "\n".join(text)


def workspace_text(payload):
    r = payload["results"]
    assert r["shape"] == 8 and len(r["accuracy"]) == 21
    assert all(c["passed"] and c["failed"] == 0 for c in r["accuracy"])
    base = r["timing"]["production_lt"]
    for v in r["timing"].values():
        validate_rounds(v)
    qualified = [n for n, calls in r["workspace_calls"].items() if calls and
                 r["timing"][n]["event_ms"] < base["event_ms"] * .95 and
                 r["timing"][n]["wall_ms"] < base["wall_ms"] * .95]
    assert qualified == r["qualified_candidates"], "workspace decision does not match evidence"
    text = ["## Shape 8 targeted workspace experiment", "",
            "Original-FP32 gate on three inputs for each of seven variants; all 21 pass. "
            "Three rotated paired rounds include public-forward dispatch, input copies, output ownership "
            "and per-invocation/current-stream workspace allocations. No production core is changed. "
            "Memory records are incremental peaks above seven resident models, not isolated deployment peaks.", "",
            "Scratch policy: " + r["scratch_policy"] + ".", "",
            "Acceptance requires at least 5% lower event AND wall latency than the existing opt-in zero-workspace "
            "production Lt provider, followed by independent confirmation before any promotion.", "",
            "| Variant | Event ms | Wall ms | Event reduction vs existing Lt | Host extension calls |",
            "|---|---:|---:|---:|---:|"]
    for name, v in r["timing"].items():
        reduction = 100 * (1 - v["event_ms"] / base["event_ms"])
        count = r["workspace_calls"].get(name, "—")
        text.append(f'| {name} | {v["event_ms"]:.4f} | {v["wall_ms"]:.4f} | {reduction:.2f}% | {count} |')
    text.extend(["", "Counters include capture-time extension invocations; CUDA graph replays do not increment Python counters.", "",
                 f'Experimental extension build: **{r["build_seconds"]:.2f} s**.', "",
                 "Candidates clearing the first gate: " + (", ".join(qualified) if qualified else "**none; retain the existing backend**") + "."])
    return "\n".join(text)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--sessions", nargs=3, type=Path, required=True)
    ap.add_argument("--quick", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    paths = [*args.sessions, args.quick, args.workspace]
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    assert all(core_manifest(p) == core_manifest(payloads[0]) for p in payloads)
    body = "# Usability, independent reproduction and targeted optimization\n\nGenerated from complete, source-stamped JSON.\n\n"
    for p, payload in zip(paths, payloads):
        relative = Path(os.path.relpath(p.resolve(), args.output.parent.resolve())).as_posix()
        body += f'- [{p.parent.name}/{p.name}]({relative}) — Git `{payload["metadata"]["git_commit"]}`\n'
    body += "\n" + independent_text(payloads[:3]) + "\n\n" + quick_text(payloads[3]) + "\n\n" + workspace_text(payloads[4]) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()

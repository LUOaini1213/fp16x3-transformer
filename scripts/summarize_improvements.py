"""Recompute the maintained-runtime acceptance report from source-pinned T4 evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from scripts.summarize_release import validate_rounds


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("quick", "balanced", "steady")
CONTRACTS = {"owned output", "changed input", "mutable normal mask", "mutable inference mask",
             "weight change re-selects once", "stable weights do not retune", "sequential alternate stream"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_metadata(payload):
    metadata = payload["metadata"]
    assert metadata["gpu"] == "Tesla T4" and metadata["capability"] == [7, 5], "T4 evidence required"
    assert metadata["clean_checkout"] is True, "clean source-pinned checkout required"
    assert re.fullmatch(r"[a-f0-9]{40}", metadata["git_commit"]), "exact Git revision required"
    assert set(metadata["normal_imports"]) == {"official", "production"}, "normal imports required"
    manifest = metadata["source_manifest"]
    required = {"user_optimized.py", "torch_transformer_benchmark.py", "kernels/fp16x3.py",
                "kernels/dispatch_cache.py", "scripts/run.py", "scripts/benchmark_improvements.py"}
    assert required <= manifest.keys(), "runtime source hashes required"
    for name, measured_hash in manifest.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), "source path leaves repository"
        actual = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
        assert actual == measured_hash, f"measured source differs: {name}"
    return metadata


def check_accuracy(checks, seeds):
    assert len(checks) == len(seeds) and {c["seed"] for c in checks} == set(seeds), "incomplete accuracy seeds"
    assert all(c["passed"] is True and c["failed"] == 0 and math.isfinite(c["max_abs"])
               and c["max_abs"] >= 0 for c in checks), "failed accuracy check"


def check_profile(profile, shape, mode):
    assert profile["shape"] == shape and profile["mode"] == mode and profile["device"] == "cuda"
    check_accuracy(profile["accuracy_trials"], range(20261009, 20261012))
    assert math.isfinite(profile["first_forward_seconds"]) and profile["first_forward_seconds"] > 0
    samples = profile["steady_wall_samples_ms"]
    assert len(samples) == 10 and all(math.isfinite(v) and v > 0 for v in samples)
    assert profile["steady_wall_ms"] == statistics.median(samples), "profile median differs from samples"
    if mode == "balanced":
        assert profile["profile"]["T3_COMPILE"] == "0" and not profile["dispatch"]["compiled"]


def sweep_metrics(payload, shapes):
    check_metadata(payload)
    rows = payload["results"]
    assert [r["shape"] for r in rows] == list(shapes), "incomplete or duplicate shape sweep"
    gates, errors = [], []
    for row in rows:
        shape, pair, cold = row["shape"], row["paired"], row["cold"]
        assert pair["shape"] == shape and pair["output_ownership"] == "PASS"
        assert set(cold) == set(PROFILES)
        assert set(pair["timing"]) == {"baseline", *PROFILES}
        assert len(pair["accuracy"]) == 9, "incomplete paired checks"
        for mode in PROFILES:
            check_profile(cold[mode], shape, mode)
            checks = [c for c in pair["accuracy"] if c["variant"] == mode]
            check_accuracy(checks, range(3))
            errors.extend(c["max_abs"] for c in checks)
            errors.extend(c["max_abs"] for c in cold[mode]["accuracy_trials"])
        for timing in pair["timing"].values():
            validate_rounds(timing)
        assert pair["cache"] == "disabled", "paired profile timing must exclude dispatch cache"
        assert not pair["dispatch"]["balanced"]["compiled"], "balanced compiled unexpectedly"
        t = pair["timing"]
        event_ratio = t["balanced"]["event_ms"] / t["steady"]["event_ms"]
        wall_ratio = t["balanced"]["wall_ms"] / t["steady"]["wall_ms"]
        startup_lower = cold["balanced"]["first_forward_seconds"] < cold["steady"]["first_forward_seconds"]
        gates.append({"shape": shape, "event_ratio": event_ratio, "wall_ratio": wall_ratio,
                      "startup_lower": startup_lower,
                      "passed": startup_lower and event_ratio <= 1.02 and wall_ratio <= 1.02})
    return {"gates": gates, "checks": len(errors), "max_abs": max(errors)}


def render(full, pilot, cache, contracts, fusion):
    payloads = (full, pilot, cache, contracts, fusion)
    metadatas = [check_metadata(p) for p in payloads]
    identity = lambda m: (m["git_commit"], m["torch"], m["cuda"], m["gpu"], m["source_manifest"])
    assert all(identity(m) == identity(metadatas[0]) for m in metadatas), "source/environment mismatch"
    assert metadatas[0]["normal_imports"] != metadatas[1]["normal_imports"], "independent pilot/full checkouts required"
    metrics = sweep_metrics(full, range(1, 14))
    sweep_metrics(pilot, (2, 8, 13))
    c = cache["results"]
    for name in ("producer", "hit", "warm_uncached"):
        check_profile(c[name], 2, "balanced")
    assert c["producer"]["tune_cache"]["status"].startswith("stored-")
    assert c["hit"]["tune_cache"]["status"].startswith("hit-validated-"), "validated graph/eager cache hit required"
    assert c["warm_uncached"]["tune_cache"]["status"] == "disabled"
    recovery = contracts["results"]
    assert recovery["status"] == "PASS" and set(recovery["checks"]) == CONTRACTS, "incomplete recovery contracts"
    f = fusion["results"]
    for timing in f["timing"].values():
        validate_rounds(timing)
    for name in f["timing"]:
        check_accuracy([x for x in f["accuracy"] if x["variant"] == name], range(3))
    winners = [name for name, timing in f["timing"].items() if name != "balanced"
               and timing["event_ms"] < f["timing"]["balanced"]["event_ms"] * .97
               and timing["wall_ms"] < f["timing"]["balanced"]["wall_ms"] * .97]
    assert set(winners) == set(f["qualified_candidates"]) and f["default_changed"] is False
    failed = [g["shape"] for g in metrics["gates"] if not g["passed"]]
    m = metadatas[0]
    text = ["# Maintained runtime: balanced, cache and recovery validation", "",
            "Generated with `python -m scripts.summarize_improvements` from committed Kaggle artifacts.", "",
            f"Measured Git: `{m['git_commit']}`. Linux / Tesla T4 / Torch {m['torch']} / CUDA {m['cuda']}.", "",
            "A three-shape pilot and a separate fresh-checkout full sweep use the same source hashes. "
            "This validates the new runtime; the historical three-session 2.875× result remains separate.", "",
            f"**All {metrics['checks']} full-sweep profile/input checks pass** "
            "(13 shapes × 3 profiles × 3 seeds, both separate cold workers and paired workers); "
            f"maximum absolute error {metrics['max_abs']:.8g}. Each profile's returned-output ownership is checked.", "",
            "## Balanced acceptance", "",
            "Gate per shape: lower first-public-forward cost than steady, and no more than 2% event "
            "AND wall regression. Paired timing uses three rotated rounds and reported medians. "
            "Every cold profile uses a separate empty compiler directory. Startup excludes process/import, "
            "model/input setup and CUDA-context probes. Shared-process paired models are not isolated memory measurements.", "",
            "| Shape | Quick / balanced / steady first s | Balanced / steady event ms | Balanced / steady wall ms | Event / wall delta % | Gate |",
            "|---:|---:|---:|---:|---:|---|"]
    for row, gate in zip(full["results"], metrics["gates"]):
        cold, timing = row["cold"], row["paired"]["timing"]
        startup = " / ".join(f"{cold[n]['first_forward_seconds']:.3f}" for n in PROFILES)
        text.append(f"| {row['shape']} | {startup} | {timing['balanced']['event_ms']:.4f} / "
                    f"{timing['steady']['event_ms']:.4f} | {timing['balanced']['wall_ms']:.4f} / "
                    f"{timing['steady']['wall_ms']:.4f} | {(gate['event_ratio']-1)*100:+.2f} / "
                    f"{(gate['wall_ratio']-1)*100:+.2f} | {'PASS' if gate['passed'] else 'RETAIN OPT-IN'} |")
    text += ["", ("All shapes clear the measured gate. " if not failed else
                  "Shapes " + ", ".join(map(str, failed)) + " do not clear the complete gate. ") +
             "**Defaults are unchanged; balanced remains an explicit CLI choice.** "
             "One full session and the limited pilot are not a universal performance guarantee.", "",
             "## Dispatch-cache control", "",
             "Shape 2 runs in fresh processes. Cache-hit and uncached controls share warmed compiler directories; "
             "only JSON dispatch hints persist. Graph hits recapture and compare against fresh eager output bitwise.", "",
             "| Run | First public forward s | Steady wall ms | Cache status |", "|---|---:|---:|---|"]
    for name in ("producer", "hit", "warm_uncached"):
        row = c[name]
        text.append(f"| {name} | {row['first_forward_seconds']:.4f} | {row['steady_wall_ms']:.4f} | "
                    f"{row['tune_cache']['status']} |")
    text += ["", "These are single unpaired workers: the table proves hit validation and retains the warmed "
             "uncached control, but does not establish a statistically reliable cache speedup.", "",
             "## GPU runtime contracts", "", "All recorded checks pass: " + "; ".join(recovery["checks"]) + ".",
             "Sequential alternate-stream use is tested; concurrent calls on one mutable graph instance are not.", "",
             "## FFN fusion experiment", "",
             "Shape 6 compares four compensated GEMM + exact-GELU + split epilogues against balanced. "
             "Each timed variant passes three original-reference checks. Promotion requires at least 3% event "
             "AND wall reduction and independent confirmation; no research backend is installed by this report.", "",
             "| Variant | Event ms | Wall ms |", "|---|---:|---:|"]
    for name, timing in f["timing"].items():
        text.append(f"| {name} | {timing['event_ms']:.4f} | {timing['wall_ms']:.4f} |")
    text += ["", "Qualified candidates: " + (", ".join(winners) if winners else "**none**") + ". "
             "Losing variants are retained as evidence; production remains unchanged.", "",
             "## Evidence and reproduction", "",
             "- [Full sweep](improvements-full/next_improvements_full.json), "
             "[cache](improvements-full/next_improvements_cache.json), "
             "[contracts](improvements-full/next_improvements_contracts.json).",
             "- [Pilot](improvements-pilot/next_improvements_pilot.json) and "
             "[fusion](improvements-fusion/next_improvements_fusion.json).",
             "- Each folder retains the source-pinned cloud launcher, raw log, round-level JSON and SHA-256 receipt.",
             "- Build a fresh private T4 job with `scripts/build_clean_kaggle.py --ref " + m["git_commit"] +
             " --phase full --id ACCOUNT/UNIQUE_KERNEL --out .kaggle_upload/repeat-full`; "
             "push using `kaggle kernels push -p .kaggle_upload/repeat-full --accelerator NvidiaTeslaT4`.",
             "- Use `--phase pilot` for shapes 2/8/13 or `--phase fusion` for the isolated shape-6 experiment.", ""]
    return "\n".join(text)


def generate():
    root = ROOT / "results" / "next"
    return render(load(root / "improvements-full/next_improvements_full.json"),
                  load(root / "improvements-pilot/next_improvements_pilot.json"),
                  load(root / "improvements-full/next_improvements_cache.json"),
                  load(root / "improvements-full/next_improvements_contracts.json"),
                  load(root / "improvements-fusion/next_improvements_fusion.json"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when the committed report differs from evidence")
    args = parser.parse_args()
    path = ROOT / "results" / "next" / "improvements_summary.md"
    text = generate()
    if args.check:
        assert path.read_text(encoding="utf-8") == text, "report differs from measured evidence"
        print("Runtime report matches source-pinned evidence")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()

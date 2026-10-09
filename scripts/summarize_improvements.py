"""Generate a checked report; no adoption/speedup claim from partial trials."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def positive(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("non-positive or nonfinite timing")
    return value


def checks(rows, names):
    if len(rows) != len(names) * 3:
        raise ValueError("incomplete accuracy checks")
    for name in names:
        selected = [r for r in rows if r["variant"] == name]
        if sorted(r["seed"] for r in selected) != [0, 1, 2]:
            raise ValueError("missing or duplicate input seed")
        if any(not r["passed"] or r["failed"] != 0 or not math.isfinite(r["max_abs"]) for r in selected):
            raise ValueError("accuracy failure")


def timing(value):
    for metric in ("event", "wall"):
        samples = value[metric + "_round_ms"]
        if len(samples) != 3:
            raise ValueError("three paired rounds required")
        for sample in samples:
            positive(sample)
        if value[metric + "_ms"] != statistics.median(samples):
            raise ValueError("round median does not match")


def cold_case(value, name):
    trials = value["accuracy_trials"]
    if len(trials) != 3 or sorted(r["seed"] for r in trials) != [20261009, 20261010, 20261011]:
        raise ValueError("incomplete cold input checks")
    if any(not r["passed"] or r["failed"] for r in trials):
        raise ValueError("cold accuracy failure")
    if any(not math.isfinite(r["max_abs"]) for r in trials):
        raise ValueError("nonfinite cold error")
    if (not value["accuracy"]["passed"] or value["accuracy"]["failed"]
            or value["accuracy"]["max_abs"] != max(r["max_abs"] for r in trials)):
        raise ValueError("cold accuracy summary mismatch")
    positive(value["first_forward_seconds"])
    if len(value["steady_wall_samples_ms"]) != 10:
        raise ValueError("ten cold-worker steady samples required")
    for sample in value["steady_wall_samples_ms"]:
        positive(sample)
    if value["steady_wall_ms"] != statistics.median(value["steady_wall_samples_ms"]):
        raise ValueError("cold-worker median does not match")
    if value["mode"] != name or value["device"] != "cuda":
        raise ValueError("incorrect cold profile/device")
    if name == "balanced" and (value["profile"]["T3_COMPILE"] != "0" or value["dispatch"]["compiled"]):
        raise ValueError("balanced attempted to retain Inductor")


def profile_metrics(payload, indices):
    rows = payload["results"]
    if [r["shape"] for r in rows] != list(indices):
        raise ValueError("incomplete shape sweep")
    result = []
    for row in rows:
        for name in ("quick", "balanced", "steady"):
            cold_case(row["cold"][name], name)
            if row["cold"][name]["shape"] != row["shape"]:
                raise ValueError("cold shape mismatch")
        paired = row["paired"]
        if paired["shape"] != row["shape"]:
            raise ValueError("paired shape mismatch")
        checks(paired["accuracy"], ("quick", "balanced", "steady"))
        if paired["output_ownership"] != "PASS" or paired["cache"] != "disabled":
            raise ValueError("unverified ownership or contaminated profile timing")
        for value in paired["timing"].values():
            timing(value)
        if paired["dispatch"]["balanced"]["compiled"]:
            raise ValueError("balanced used Inductor in paired timing")
        b, s = (paired["timing"][n] for n in ("balanced", "steady"))
        b_cold = row["cold"]["balanced"]["first_forward_seconds"]
        s_cold = row["cold"]["steady"]["first_forward_seconds"]
        event_regression = b["event_ms"] / s["event_ms"] - 1.
        wall_regression = b["wall_ms"] / s["wall_ms"] - 1.
        result.append({"shape": row["shape"], "balanced_first": b_cold, "steady_first": s_cold,
                       "balanced_event": b["event_ms"], "steady_event": s["event_ms"],
                       "event_regression": event_regression, "wall_regression": wall_regression,
                       "speedup": paired["timing"]["baseline"]["event_ms"] / b["event_ms"],
                       "gate": b_cold < s_cold and event_regression <= .02 and wall_regression <= .02})
    return result


def compatible(payloads):
    first = payloads[0]["metadata"]
    for payload in payloads:
        meta = payload["metadata"]
        if not meta["clean_checkout"]:
            raise ValueError("dirty cloud checkout")
        for name in ("source_manifest", "torch", "cuda", "gpu", "capability", "git_commit"):
            if meta[name] != first[name]:
                raise ValueError("source/environment mismatch: " + name)


def cache_text(payload):
    row = payload["results"]
    for name in ("producer", "hit", "warm_uncached"):
        cold_case(row[name], "balanced")
    if not row["producer"]["tune_cache"]["status"].startswith("stored-"):
        raise ValueError("cache was not generated")
    if not row["hit"]["tune_cache"]["status"].startswith("hit-validated-"):
        raise ValueError("cache was not validated/reused")
    hit, control = row["hit"], row["warm_uncached"]
    return ("## Dispatch cache and weight recovery\n\n"
            "Three fresh workers share warmed compiler directories; compare the cache hit with the uncached warm control, "
            "not with the initial compiler-cold producer. Only JSON hints persist; graph pointers and outputs do not.\n\n"
            f"- Producer: {row['producer']['tune_cache']['status']}. Hit: {hit['tune_cache']['status']}.\n"
            f"- First public forward: hit **{hit['first_forward_seconds']:.4f} s**, warm uncached **{control['first_forward_seconds']:.4f} s**.\n"
            f"- Ten-sample synchronized wall medians: hit **{hit['steady_wall_ms']:.4f} ms**, control **{control['steady_wall_ms']:.4f} ms**.\n"
            "- This is one cache trial, not a universal startup/throughput guarantee.\n")


def fusion_text(payload):
    row = payload["results"]
    names = list(row["timing"])
    checks([r for r in row["accuracy"] if r["variant"] in names], names)
    baseline = row["timing"]["balanced"]
    for value in row["timing"].values():
        timing(value)
    winners = [n for n in names if n != "balanced" and
               row["timing"][n]["event_ms"] < baseline["event_ms"] * .97 and
               row["timing"][n]["wall_ms"] < baseline["wall_ms"] * .97]
    if winners != row["qualified_candidates"] or row["default_changed"]:
        raise ValueError("fusion gate/default mismatch")
    lines = ["## Shape 6: GEMM + exact GELU + split fusion", "",
             "The research candidate removes the materialized FP32 FFN-in result. Production is unchanged. "
             "All retained variants pass three original-FP32 seeds and owned-output checks. "
             "Three rotated paired event/wall rounds include normal public-forward costs.", "",
             "| Variant | Event ms | Wall ms | Event change vs balanced | First gate |", "|---|---:|---:|---:|---|"]
    for name in names:
        value = row["timing"][name]
        if name != "balanced" and row["fused_calls"][name] <= 0:
            raise ValueError("fusion did not execute")
        lines.append(f"| {name} | {value['event_ms']:.4f} | {value['wall_ms']:.4f} | "
                     f"{(value['event_ms']/baseline['event_ms']-1)*100:+.2f}% | {'candidate' if name in winners else 'retain baseline' if name == 'balanced' else 'reject'} |")
    lines += ["", "Candidates clearing >=3% event AND wall reduction: **" + (", ".join(winners) or "none") + "**.",
              "Any winner still requires an independent confirmation before promotion."]
    if row["failures"]:
        lines += ["", "Compilation/accuracy failures (not included as speed wins):"]
        lines += [f"- {name}: {reason}" for name, reason in row["failures"].items()]
    device = row["profiles"]["balanced"]
    groups = defaultdict(float)
    for event in device["kernels"]:
        name = event["name"].lower()
        category = ("LayerNorm/residual + split" if "_x3_ln_split" in name else
                    "Activation + split" if "_x3_act_split" in name else
                    "Attention" if "fmha" in name or "attention" in name else
                    "GEMM" if "gemm" in name else "Other")
        groups[category] += event["total_us"]
    lines += ["", "Fresh source-stamped baseline profiling (sum of device events across three forwards, not end-to-end fractions):"]
    lines += [f"- {name}: {value/device['device_total_us']*100:.2f}%." for name, value in groups.items()]
    return "\n".join(lines) + "\n"


def repeat_metrics(payload):
    row = payload["results"]
    if (row["shape"] != 2 or row["repetitions_per_round"] != 200
            or row["fresh_processes"] != 3 or len(row["workers"]) != 3):
        raise ValueError("incomplete shape-2 repeat protocol")
    result = []
    for index, worker in enumerate(row["workers"], 1):
        if worker["shape"] != 2 or worker["output_ownership"] != "PASS" or worker["cache"] != "disabled":
            raise ValueError("unverified repeat shape/ownership/cache")
        checks(worker["accuracy"], ("quick", "balanced", "steady"))
        if worker["dispatch"]["balanced"]["compiled"]:
            raise ValueError("repeat balanced used Inductor")
        for value in worker["timing"].values():
            timing(value)
        b, s = (worker["timing"][n] for n in ("balanced", "steady"))
        event = b["event_ms"] / s["event_ms"] - 1.
        wall = b["wall_ms"] / s["wall_ms"] - 1.
        result.append({"worker": index, "balanced_event": b["event_ms"], "steady_event": s["event_ms"],
                       "balanced_wall": b["wall_ms"], "steady_wall": s["wall_ms"],
                       "event_regression": event, "wall_regression": wall,
                       "within_2_percent": event <= .02 and wall <= .02})
    return result


def repeat_text(payload):
    metrics = repeat_metrics(payload)
    lines = ["## Shape 2: targeted repeat (original miss retained)", "",
             "One new T4 session, three fresh interpreter workers with shared compiler caches. "
             "Each uses three rotated paired rounds of 200 calls per variant, the original timing helpers, "
             "three accuracy seeds and output-ownership checks. This is not three independent GPU sessions, "
             "an isolated cold-start repeat, or a replacement for the original +6.95% wall result.", "",
             "| Worker | Balanced / steady event ms | Balanced / steady wall ms | Event regression | Wall regression | Within 2% on both |",
             "|---:|---:|---:|---:|---:|---|"]
    for row in metrics:
        lines.append(f"| {row['worker']} | {row['balanced_event']:.4f} / {row['steady_event']:.4f} | "
                     f"{row['balanced_wall']:.4f} / {row['steady_wall']:.4f} | {row['event_regression']*100:+.2f}% | "
                     f"{row['wall_regression']*100:+.2f}% | {'yes' if row['within_2_percent'] else 'no'} |")
    count = sum(r["within_2_percent"] for r in metrics)
    lines += ["", f"Fresh-worker steady comparisons within 2% on both metrics: **{count}/3**. "
              "All 27 additional profile/input checks pass. The original sweep's 12/13 gate and opt-in defaults remain unchanged."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--shape2-repeat", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.profiles, args.cache, args.contracts, args.fusion]
    if args.shape2_repeat:
        paths.append(args.shape2_repeat)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    compatible(payloads)
    indices = (2, 8, 13) if args.pilot else range(1, 14)
    metrics = profile_metrics(payloads[0], indices)
    if payloads[2]["results"]["status"] != "PASS":
        raise ValueError("GPU contracts did not pass")
    lines = ["# Balanced profiles, dispatch hints/recovery, and FFN fusion", "",
             "Generated from complete source-stamped cloud artifacts. This report does not replace older cohorts.", "",
             f"Exact tested Git commit: `{payloads[0]['metadata']['git_commit']}`.", ""]
    lines += [f"- [{p.name}]({p.parent.name}/{p.name})" for p in paths]
    lines += ["", "## Balanced versus steady", "",
              "Separate first-forward workers use empty per-mode compiler directories. First public forward excludes "
              "process/import/model/input/context setup. Steady timings come from a different, rotated paired process "
              "with identical weights/input. These are not total application cold starts or confidence intervals.", "",
              "| Shape | Balanced / steady first s | Balanced / steady event ms | Event regression | Wall regression | First gate |",
              "|---:|---:|---:|---:|---:|---|"]
    for row in metrics:
        lines.append(f"| {row['shape']} | {row['balanced_first']:.4f} / {row['steady_first']:.4f} | "
                     f"{row['balanced_event']:.4f} / {row['steady_event']:.4f} | {row['event_regression']*100:+.2f}% | "
                     f"{row['wall_regression']*100:+.2f}% | {'pass' if row['gate'] else 'not cleared'} |")
    passed = [r["shape"] for r in metrics if r["gate"]]
    lines += ["", f"Lower first-forward cost AND <=2% event/wall regression: **{len(passed)}/{len(metrics)} shapes**, {passed}.",
              f"All **{len(metrics)*18}** profile/input checks pass (three profiles, three seeds, both cold and paired workers).",
              "No model or CLI default is promoted by this single session."]
    if not args.pilot:
        lines += [f"Single-session balanced median paired event speedup over original FP32: **{statistics.median(r['speedup'] for r in metrics):.3f}x**. "
                  "Not pooled with the historical three-session 2.875x cohort."]
    if args.shape2_repeat:
        lines += ["", repeat_text(payloads[4])]
    lines += ["", cache_text(payloads[1]), "Real CUDA graph contracts: **PASS**, including one re-selection after weight change, "
              "mutable masks, owned outputs and sequential alternate-stream use. Concurrent use of one mutable graph instance is not claimed.",
              "", fusion_text(payloads[3])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

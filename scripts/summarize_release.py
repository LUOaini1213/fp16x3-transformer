"""Generate the current-version table from immutable release JSON evidence."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["normal_imports"], "normal-import evidence required"
    return data


def validate_rounds(timing):
    for kind in ("event", "wall"):
        values = timing[kind + "_round_ms"]
        assert len(values) == 3 and all(math.isfinite(v) and v > 0 for v in values)
        assert timing[kind + "_ms"] == statistics.median(values)


def fp32_table(payload):
    rows = payload["results"]
    assert [r["shape"] for r in rows] == list(range(1, 14)), "incomplete shape sweep"
    text = ["## Current FP32 sweep", "",
            "Three rotated rounds; medians of per-round medians. Event and synchronized host-wall "
            "latency are separate. Every default output is FP32. Memory is isolated single-model "
            "peak allocated GiB, including input/output and graph pools, excluding the oracle. "
            "Cold seconds include first-forward planning/compile/tuning; job-local caches may already be warm.", "",
            "| Shape | Baseline event ms | Default event ms | Speedup | Default wall ms | First call s | Steady baseline / default GiB | Default cold peak GiB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ratios = []
    errors = []
    for r in rows:
        t, m = r["timing"], r["memory"]
        validate_rounds(t["default"])
        validate_rounds(t["baseline"])
        checks = [c for c in r["accuracy"] if c["variant"] == "default"]
        assert len(checks) == 3 and all(c["passed"] and c["failed"] == 0 for c in checks)
        errors += [c["max_abs"] for c in checks]
        ratio = t["baseline"]["event_ms"] / t["default"]["event_ms"]
        ratios.append(ratio)
        base_gb = m["baseline"]["steady_peak_allocated_bytes"] / 2**30
        opt_gb = m["default"]["steady_peak_allocated_bytes"] / 2**30
        cold_gb = m["default"]["cold_peak_allocated_bytes"] / 2**30
        text.append(f'| {r["shape"]} | {t["baseline"]["event_ms"]:.4f} | {t["default"]["event_ms"]:.4f} | '
                    f'{ratio:.3f}× | {t["default"]["wall_ms"]:.4f} | {r["cold_seconds"]["default"]:.2f} | {base_gb:.3f} / {opt_gb:.3f} | {cold_gb:.3f} |')
    text += ["", f"Unweighted median shape speedup: **{statistics.median(ratios):.3f}×**. "
             f"All 39 default comparisons pass the official OR gate; maximum absolute error **{max(errors):.8g}**."]
    wide = rows[7]
    if "lt" in wide["timing"]:
        t = wide["timing"]
        validate_rounds(t["lt"])
        assert any(r["selected"] is not None for r in wide["lt_search"])
        reduction = 100 * (1 - t["lt"]["event_ms"] / t["default"]["event_ms"])
        text += ["", f'Shape 8 opt-in Lt: **{t["default"]["event_ms"]:.4f} → {t["lt"]["event_ms"]:.4f} ms** '
                 f'event ({reduction:.2f}% reduction), **{t["default"]["wall_ms"]:.4f} → {t["lt"]["wall_ms"]:.4f} ms** '
                 f'wall; first Lt call **{wide["cold_seconds"]["lt"]:.2f} s**. '
                 'Algorithm selection and three original-reference accuracy checks are recorded in the JSON.']
    return "\n".join(text)


def flash_text(payload):
    r = payload["results"]
    assert r["full_equivalence"]["failed"] == 0 and r["adapter_calls"] == 256
    assert r["full_equivalence"]["elements"] == 3276800000
    assert r["fp32_causal_prefix"]["passed"] and r["fp32_causal_prefix"]["failed"] == 0
    text = ["## Formal production Turing adapter", "",
            "Native-FP16 shape 14 only; not the official FP32 grading speedup. Both sides are eager "
            "(compile/graphs disabled), batch-chunked by one; these are synchronized whole-forward "
            "wall seconds, not an independently autotuned SDPA comparison. All 3,276,800,000 "
            "outputs pass against native-FP16 SDPA. The independent original-FP32 oracle is restricted "
            "to the first batch's 512-token causal prefix. Build and first calls are excluded from steady rounds.", "",
            "| Backend | First call s | Round 1 s | Round 2 s | Round 3 s | Median s |",
            "|---|---:|---:|---:|---:|---:|"]
    for name in ("sdpa", "turing"):
        v = r["full"][name]
        assert len(v["round_seconds"]) == 3
        assert len(v["peak_bytes"]) == 3
        assert v["median_seconds"] == statistics.median(v["round_seconds"])
        text.append(f'| {name} | {v["cold_seconds"]:.3f} | ' + " | ".join(f"{n:.3f}" for n in v["round_seconds"])
                    + f' | {v["median_seconds"]:.3f} |')
    text += ["", f'Steady speedup: **{r["speedup"]:.3f}×**. Dependency build: **{r["build_seconds"]:.1f} s**. '
             f'Confirmed production-adapter calls: **{r["adapter_calls"]}**. '
             f'Full-output maximum absolute difference: **{r["full_equivalence"]["max_abs"]:.8g}**. '
             'Peak memory is from a shared two-model process, not isolated deployment memory.']
    text += ["", "Shared-process steady peak allocated memory: " + ", ".join(
             f'**{name}: {max(r["full"][name]["peak_bytes"]) / 2**30:.3f} GiB**'
             for name in ("sdpa", "turing")) + "."]
    return "\n".join(text)


def attention_text(payload):
    r = payload["results"]
    assert len(r["full_accuracy"]) == 21 and len(r["micro_accuracy"]) == 21
    assert all(c["passed"] and c["failed"] == 0 for c in r["full_accuracy"])
    text = ["## Shape 13 FP32 attention candidates", "",
            "All conversion, copy, launch and concatenation costs are included. Each candidate is "
            "tested end-to-end against the original FP32 model on three inputs. Full-model paired "
            "results below keep both sides eager (compile and graphs disabled) to isolate attention.", "",
            "| Layout / dispatch | Attention event ms | Full-model event ms | Full-model wall ms | vs current |",
            "|---|---:|---:|---:|---:|"]
    base = r["full_eager_timing"]["current"]["event_ms"]
    for name, v in r["full_eager_timing"].items():
        validate_rounds(v)
        validate_rounds(r["micro_timing"][name])
        text.append(f'| {name} | {r["micro_timing"][name]["event_ms"]:.4f} | '
                    f'{v["event_ms"]:.4f} | {v["wall_ms"]:.4f} | {base / v["event_ms"]:.3f}× |')
    return "\n".join(text)


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--fp32", type=Path, required=True)
    ap.add_argument("--flash", type=Path, required=True)
    ap.add_argument("--attention", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    payloads = [load(p) for p in (args.fp32, args.flash, args.attention)]
    core = {k: v for k, v in payloads[0]["metadata"]["source_manifest"].items()
            if k.startswith("kernels/") or k in ("user_optimized.py", "torch_transformer_benchmark.py")}
    for payload in payloads[1:]:
        assert all(payload["metadata"]["source_manifest"][k] == v for k, v in core.items())
    body = "# Clean GitHub/T4 release evidence\n\nGenerated from source-stamped JSON, not historical headlines.\n\n"
    body += "Measured Git commits: " + "; ".join(f"{name} `{p['metadata']['git_commit']}`" for name, p in
            zip(("FP32", "flash", "attention"), payloads)) + ". Core-source hashes match across all three.\n\n"
    body += f"GPU: Tesla T4; PyTorch {payloads[0]['metadata']['torch']}.\n\n"
    body += "Source evidence: " + ", ".join(f"[{name} JSON]({Path(os.path.relpath(path.resolve(), args.output.resolve().parent)).as_posix()})"
            for name, path in zip(("FP32", "flash", "attention"), (args.fp32, args.flash, args.attention))) + ".\n\n"
    body += "\n\n".join(fn(p) for fn, p in zip((fp32_table, flash_text, attention_text), payloads)) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()

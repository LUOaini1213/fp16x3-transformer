"""Audit committed GPU evidence without Torch, a GPU, downloads or modifying artifacts.

python -m scripts.verify_next_evidence --maintained-runtime
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

# Preserve the documented legacy `python scripts/verify_next_evidence.py` entry.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evidence_sources import RUNNER_ARCHIVE_FOLDER, measured_source


ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_ARTIFACTS = {
    "improvements-full": {"cloud_script.py", "next_improvements_full.json",
                          "next_improvements_cache.json", "next_improvements_contracts.json",
                          *(f"next_improvements_pair_{i}.json" for i in range(1, 14))},
    "improvements-pilot": {"cloud_script.py", "next_improvements_pilot.json",
                           "next_improvements_cache.json", "next_improvements_contracts.json",
                           *(f"next_improvements_pair_{i}.json" for i in (2, 8, 13))},
    "improvements-fusion": {"cloud_script.py", "next_improvements_fusion.json"},
    "balanced-full": {"cloud_script.py", "next_improvements_full.json",
                      "next_improvements_cache.json", "next_improvements_contracts.json",
                      *(f"next_improvements_pair_{i}.json" for i in range(1, 14))},
    "balanced-pilot": {"cloud_script.py", "next_improvements_pilot.json",
                       "next_improvements_cache.json", "next_improvements_contracts.json",
                       *(f"next_improvements_pair_{i}.json" for i in (2, 8, 13))},
    "balanced-shape2-repeat": {"cloud_script.py", "next_improvements_shape2_repeat.json"},
    "ffn-fusion": {"cloud_script.py", "next_improvements_fusion.json"},
}
FROZEN_RUNNER_ARTIFACTS = {RUNNER_ARCHIVE_FOLDER: {"run.py.snapshot", "source.json"}}


def _unique_entries(pairs):
    receipt = {}
    for name, value in pairs:
        if name in receipt:
            raise ValueError(f"duplicate artifact receipt entry: {name}")
        receipt[name] = value
    return receipt


def check_required_artifacts(root, required_artifacts, require_raw_logs=True):
    """Require the full inventory before verifying hashes; missing receipts never certify a run."""
    missing = [f"{folder}/artifact_sha256.json" for folder in sorted(required_artifacts)
               if not (root / folder / "artifact_sha256.json").is_file()]
    if missing:
        raise ValueError("missing required evidence receipts: " + ", ".join(missing))
    for folder, required in required_artifacts.items():
        directory = root / folder
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"evidence directory leaves root: {folder}")
        path = directory / "artifact_sha256.json"
        receipt = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_entries)
        if not isinstance(receipt, dict) or not receipt:
            raise ValueError(f"empty or invalid artifact receipt: {folder}/artifact_sha256.json")
        for name, digest in receipt.items():
            if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"artifact receipt must use filenames inside {folder}: {name}")
            if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError(f"invalid SHA-256 receipt entry: {folder}/{name}")
        missing = sorted(required - receipt.keys())
        if missing:
            raise ValueError(f"missing required artifact entries in {folder}: " + ", ".join(missing))
        inventory = {p.name for p in directory.iterdir() if p.is_file()
                     and (p.suffix in (".json", ".log", ".snapshot") or p.name == "cloud_script.py")
                     and p.name != "artifact_sha256.json"}
        unreceipted = sorted(inventory - receipt.keys())
        if unreceipted:
            raise ValueError(f"unreceipted evidence files in {folder}: " + ", ".join(unreceipted))
        missing = sorted(name for name in receipt if not (directory / name).is_file())
        if missing:
            raise ValueError(f"missing evidence files in {folder}: " + ", ".join(missing))
        if require_raw_logs and not any(name.endswith(".log") for name in receipt):
            raise ValueError(f"missing raw execution log in {folder}")


def verify_artifacts(root, required_artifacts=None):
    if required_artifacts is not None:
        check_required_artifacts(root, required_artifacts)
    checked = 0
    manifests = sorted(root.glob("*/artifact_sha256.json"))
    if not manifests:
        raise ValueError("no evidence manifests found")
    for manifest in manifests:
        for name, expected in json.loads(manifest.read_text(encoding="utf-8")).items():
            path = manifest.parent / name
            if not path.resolve().is_relative_to(manifest.parent.resolve()):
                raise ValueError(f"artifact outside evidence directory: {name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
            checked += 1
    return checked


def verify_current_core(result, repository=ROOT):
    metadata = json.loads(result.read_text(encoding="utf-8"))["metadata"]
    checked = 0
    for name, expected in metadata["source_manifest"].items():
        # Earlier driver snapshots are intentionally retained. Only the model,
        # kernels and untouched official benchmark must match this final gate.
        if not (name.startswith("kernels/") or name in
                ("user_optimized.py", "torch_transformer_benchmark.py")):
            continue
        text = (repository / name).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
            raise ValueError(f"final tested core does not match {name}")
        checked += 1
    if not checked:
        raise ValueError("no core-source hashes found")
    return checked


def verify_maintained_runtime(root, repository=ROOT):
    """Check receipts, current sources and the report; never regenerate missing evidence."""
    if not __debug__:
        raise ValueError("strict maintained-runtime audit requires assertions; rerun without -O/-OO "
                         "and unset PYTHONOPTIMIZE")
    check_required_artifacts(root, FROZEN_RUNNER_ARTIFACTS, require_raw_logs=False)
    hashes = verify_artifacts(root, required_artifacts=MAINTAINED_ARTIFACTS)
    full_path = root / "improvements-full/next_improvements_full.json"
    core = verify_current_core(full_path, repository)
    from scripts.summarize_portfolio_runtime import (CONTRACTS, check_metadata, load,
                                                     render, sweep_metrics)
    from scripts.summarize_improvements import render as render_main, repeat_metrics
    full = load(full_path)
    runner = measured_source(repository, full["metadata"], "scripts/run.py", evidence_root=root)
    runner_hash = hashlib.sha256(runner.read_text(encoding="utf-8").encode()).hexdigest()
    if runner_hash != full["metadata"]["source_manifest"]["scripts/run.py"]:
        raise ValueError("frozen measured runner differs from the recorded source hash")
    expected = render(full, load(root / "improvements-pilot/next_improvements_pilot.json"),
                      load(root / "improvements-full/next_improvements_cache.json"),
                      load(root / "improvements-full/next_improvements_contracts.json"),
                      load(root / "improvements-fusion/next_improvements_fusion.json"))
    def check_report(name, expected_text):
        report = root / name
        if not report.is_file():
            raise ValueError(f"missing maintained-runtime report: {name}")
        if report.read_text(encoding="utf-8") != expected_text:
            raise ValueError(f"maintained-runtime report differs from verified measurements: {name}")

    check_report("portfolio_runtime_summary.md", expected)
    metrics = sweep_metrics(full, range(1, 14))
    main_paths = [root / name for name in (
        "balanced-full/next_improvements_full.json", "balanced-full/next_improvements_cache.json",
        "balanced-full/next_improvements_contracts.json", "ffn-fusion/next_improvements_fusion.json",
        "balanced-shape2-repeat/next_improvements_shape2_repeat.json")]
    main_payloads = [load(path) for path in main_paths]
    pilot_paths = [root / name for name in (
        "balanced-pilot/next_improvements_pilot.json", "balanced-pilot/next_improvements_cache.json",
        "balanced-pilot/next_improvements_contracts.json", "ffn-fusion/next_improvements_fusion.json")]
    pilot_payloads = [load(path) for path in pilot_paths]
    for payload in (*main_payloads, *pilot_payloads):
        check_metadata(payload)
    # Apply the same complete source/seed/cache/contract checks to the other full
    # session; a failed performance gate is reported rather than an integrity error.
    render(main_payloads[0], pilot_payloads[0], *main_payloads[1:4])
    for payload in (main_payloads[2], pilot_payloads[2]):
        if payload["results"]["status"] != "PASS" or set(payload["results"]["checks"]) != CONTRACTS:
            raise ValueError("incomplete recovery contracts in balanced-full/pilot")
    check_report("improvements_summary.md", render_main(main_payloads, main_paths))
    check_report("improvements_pilot_summary.md", render_main(pilot_payloads, pilot_paths, pilot=True))
    main_metrics = sweep_metrics(main_payloads[0], range(1, 14))
    repeat = repeat_metrics(main_payloads[4])
    return {"artifact_hashes": hashes, "core_source_hashes": core,
            "cohorts": [{"name": name, "profile_input_checks": value["checks"],
                         "shapes": len(value["gates"]),
                         "balanced_gates_passed": sum(g["passed"] for g in value["gates"])}
                        for name, value in (("balanced-full", main_metrics), ("portfolio-full", metrics))],
            "shape2_repeat_passed": sum(row["within_2_percent"] for row in repeat),
            "shape2_repeat_workers": len(repeat),
            "measured_git": full["metadata"]["git_commit"]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--current-core", type=Path,
                        help="optional source-stamped final result to check against this checkout")
    parser.add_argument("--maintained-runtime", action="store_true",
                        help="strict inventories, hashes, current sources and both session reports")
    args = parser.parse_args()
    try:
        if args.maintained_runtime:
            result = verify_maintained_runtime(ROOT / "results/next")
            print(f"PASS: {result['artifact_hashes']} artifact hashes; complete maintained-runtime receipts")
            print(f"PASS: {result['core_source_hashes']} current core-source hashes and all generated reports")
            print("Historical c8 runner is verified from its frozen snapshot; current CLI parsing is outside GPU timing evidence")
            print(f"Measured Git: {result['measured_git']}")
            for cohort in result["cohorts"]:
                print(f"{cohort['name']}: {cohort['profile_input_checks']} profile/input checks; "
                      f"{cohort['balanced_gates_passed']}/{cohort['shapes']} balanced gates")
            print(f"shape-2 repeat: {result['shape2_repeat_passed']}/{result['shape2_repeat_workers']} "
                  "steady comparisons within 2%; original miss retained")
        else:
            print(f"PASS: {verify_artifacts(ROOT / 'results/next')} artifact hashes")
        if args.current_core:
            print(f"PASS: {verify_current_core(args.current_core)} current core-source hashes")
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        parser.exit(2, f"ERROR: {exc}\nEvidence was not certified; restore the original artifacts/receipts or use the measured revision.\n")


if __name__ == "__main__":
    main()

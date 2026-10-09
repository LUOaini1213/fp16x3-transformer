"""Verify the follow-up evidence hashes and optional final core-source stamp."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_artifacts(root):
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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--current-core", type=Path,
                        help="optional source-stamped final result to check against this checkout")
    args = parser.parse_args()
    print(f"PASS: {verify_artifacts(ROOT / 'results/next')} artifact hashes")
    if args.current_core:
        print(f"PASS: {verify_current_core(args.current_core)} current core-source hashes")


if __name__ == "__main__":
    main()

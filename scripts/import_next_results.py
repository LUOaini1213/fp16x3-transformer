"""Import generated Kaggle evidence without changing historical results.

Example: python scripts/import_next_results.py --input .kaggle_out_next_followup
  --tag followup --snapshot .kaggle_upload/next_followup/track3_sc.py
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def human_log(raw):
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(entries, list) and all(isinstance(row, dict) and "data" in row for row in entries):
        return "".join(row["data"] for row in entries)
    return raw


def store_generated(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"refusing to overwrite different evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if not args.tag.replace("_", "").replace("-", "").isalnum():
        parser.error("tag must be letters, digits, underscores or hyphens")
    output = ROOT / "results" / "next" / args.tag
    imported = {}
    for path in sorted(args.input.glob("next_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "metadata" in payload and "source_manifest" in payload["metadata"], path
        data = path.read_bytes()
        store_generated(output / path.name, data)
        imported[path.name] = hashlib.sha256(data).hexdigest()
    for path in sorted(args.input.glob("*.log")):
        data = human_log(path.read_text(encoding="utf-8")).encode("utf-8")
        store_generated(output / path.name, data)
        imported[path.name] = hashlib.sha256(data).hexdigest()
    data = args.snapshot.read_bytes()
    store_generated(output / "cloud_script.py", data)
    imported["cloud_script.py"] = hashlib.sha256(data).hexdigest()
    store_generated(output / "artifact_sha256.json",
                    (json.dumps(imported, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"Imported {len(imported)} artifacts to {output}")


if __name__ == "__main__":
    main()

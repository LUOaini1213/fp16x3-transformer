"""Recover a completed NEXT result from the cloud's persisted log stream.

This avoids enumerating thousands of unrelated output files. The recovered
JSON is explicitly derived from the last matching printed payload, not claimed
to be a byte-for-byte download of the original JSON artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path


def last_payload(log, result_name):
    prefix = result_name.upper() + " "
    matches = [json.loads(line[len(prefix):]) for line in log.splitlines()
               if line.startswith(prefix)]
    if not matches:
        raise ValueError(f"no {prefix.strip()} payload found in persisted log")
    payload = matches[-1]
    if "source_manifest" not in payload.get("metadata", {}) or "results" not in payload:
        raise ValueError("result is missing source provenance or results")
    return payload


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--result", default="next_flash")
    args = parser.parse_args()
    if not args.result.replace("_", "").isalnum():
        parser.error("result name must contain only letters, digits or underscores")
    # Import lazily so parsing and tests do not require Kaggle credentials.
    import kaggle
    events = list(kaggle.api.kernels_logs_stream(args.kernel))
    log = "".join(row.get("data", "") for row in events)
    payload = last_payload(log, args.result)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / (args.kernel.split("/")[-1] + ".log")
    log_path.write_text(log, encoding="utf-8", newline="")
    result = args.out / (args.result + ".json")
    result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="")
    receipt = {"kernel": args.kernel, "method": "last matching JSON payload in persisted cloud log",
               "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
               "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
               "not_original_json_bytes": True}
    (args.out / "next_recovery_receipt.json").write_text(
        json.dumps({"metadata": payload["metadata"], "results": receipt}, indent=2) + "\n",
        encoding="utf-8", newline="")
    print(f"Recovered {args.result} from {args.kernel}; provenance receipt saved")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn the `X3K,...` lines of a T3_ONLY=x3k Kaggle log into a CSV.

    python scripts/parse_x3k_log.py results/kaggle_t4_s0x3k_run.log results/x3_error_vs_k_t4.csv
"""
import csv
import io
import sys


def main(log, out):
    rows = []
    for line in io.open(log, encoding="utf-8"):
        if not line.startswith("X3K,K="):
            continue
        parts = line.strip().split(",")
        k = int(parts[1][2:])
        variant = parts[2]
        kv = dict(p.split("=", 1) for p in parts[3:] if "=" in p)
        rows.append(dict(K=k, variant=variant, **kv))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["K", "variant", "max_abs", "mean_abs", "bias", "frac_below", "ms"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results/x3_error_vs_k_t4.csv")

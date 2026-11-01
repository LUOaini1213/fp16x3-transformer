#!/usr/bin/env python3
"""
Shape 14 (batch=32, d_model=1024, heads=16, seq_len=100000, layers=2, ffn=1024).

The baseline cannot run this: its explicit attention needs a [B,H,S,S] score
matrix = 32*16*1e5*1e5 = 5.12e12 elements = ~20.5 TB in fp32. No GPU can hold
it, so the standard harness (which runs baseline first) dies before timing.

This script demonstrates the value of the optimization:
  A) TIMING: run ONLY the optimized model at the full seq_len=100000 (fp16,
     batch-chunked, memory-efficient SDPA) and report latency + tokens/s.
  B) TRUNCATED CORRECTNESS: at a truncated seq_len the baseline CAN run,
     compare optimized vs baseline element-wise with the official tolerances.
     This does not prove full-length numerical equivalence, and the truncated
     fp32 check is separate from the full fp16 capacity/timing demonstration.

Run on a 16 GB GPU (Kaggle T4/P100 recommended). Example:
    python scripts/shape14_optimized_only.py --trunc-seq 2048
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)                    # flat layout (e.g. Kaggle working dir)
sys.path.insert(0, os.path.dirname(_here))   # repo layout (scripts/ under repo root)

import torch_transformer_benchmark as bench
from user_optimized import UserOptimizedTransformer

FULL = dict(batch_size=32, seq_len=100000, d_model=1024,
            num_heads=16, ffn_dim=1024, num_layers=2, causal=True)


def build_pair(cfg, device, dtype):
    baseline = bench.BaselineTransformer(cfg)
    optimized = UserOptimizedTransformer(cfg)
    bench.copy_model_weights(baseline, optimized, strict=True)
    return (baseline.to(device=device, dtype=dtype).eval(),
            optimized.to(device=device, dtype=dtype).eval())


def timing_full(args, device):
    print("\n=== A) full seq_len=100000, optimized only (fp16) ===")
    cfg = bench.TransformerConfig(**{**FULL, "num_layers": args.layers})
    dtype = torch.float16
    scores_tb = cfg.batch_size * cfg.num_heads * cfg.seq_len ** 2 * 4 / 1e12
    print(f"  baseline would need {scores_tb:.1f} TB of attention scores -> "
          f"cannot run; only the optimized path is timed here")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free0, total0 = torch.cuda.mem_get_info(device)
    print(f"  vram free={free0/1e9:.2f} GB / total={total0/1e9:.2f} GB")
    try:
        baseline = bench.BaselineTransformer(cfg)  # keep unused reference on CPU
        optimized = UserOptimizedTransformer(cfg)
        bench.copy_model_weights(baseline, optimized, strict=True)
        del baseline
        optimized = optimized.to(device=device, dtype=dtype).eval()
        x, mask = bench.generate_random_case(cfg, device, dtype, seed=1234,
                                             padding_ratio=0.0, input_scale=1.0)
        with torch.inference_mode():
            for _ in range(args.warmup):
                optimized(x, mask)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            samples = []
            for _ in range(args.iters):
                start.record()
                optimized(x, mask)
                end.record()
                torch.cuda.synchronize(device)
                samples.append(start.elapsed_time(end))
        samples.sort()
        med = samples[len(samples) // 2]
        tokens = cfg.batch_size * cfg.seq_len
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  median={med:.2f} ms | tokens/call={tokens} | "
              f"throughput={tokens*1000.0/med:,.0f} tok/s | peak_vram={peak:.2f} GB")
        print(f"  chunk_bs={optimized._chunk_bs} autocast={optimized._autocast_dtype}")
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  [OOM] full seq_len did not fit (peak {peak:.2f} GB): {e}")
            print("  -> set T3_CHUNK_BS=1, or use a GPU with more VRAM.")
            return False
        else:
            raise


def correctness_truncated(args, device):
    print(f"\n=== B) truncated seq_len={args.trunc_seq}, correctness vs baseline ===")
    # Small batch so the baseline's [B,H,S,S] scores fit (baseline is the reference).
    cfg = bench.TransformerConfig(**{**FULL, "seq_len": args.trunc_seq,
                                     "num_layers": args.layers, "batch_size": 2})
    dtype = torch.float32  # match grading semantics for the reference
    baseline, optimized = build_pair(cfg, device, dtype)
    x, mask = bench.generate_random_case(cfg, device, dtype, seed=1234,
                                         padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        ref = baseline(x, mask)
        opt = optimized(x, mask)
    res = bench.compare_outputs(ref, opt, rtol=0.02, atol=0.002)
    print(f"  {'PASS' if res.passed else 'FAIL'} | max_abs={res.max_abs_error:.6g} | "
          f"max_rel={res.max_relative_error:.6g} | failed={res.failed_elements}/"
          f"{res.total_elements}")
    return res.passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunc-seq", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed full forwards (default 1; each takes minutes on a T4)")
    ap.add_argument("--iters", type=int, default=3,
                    help="timed full forwards (default 3)")
    ap.add_argument("--skip-full", action="store_true")
    args = ap.parse_args()
    if args.warmup < 0 or args.iters < 1:
        ap.error("warmup must be nonnegative and iters must be positive")

    if not torch.cuda.is_available():
        print("CUDA required. Run on Colab/Kaggle GPU.")
        return 1
    device = torch.device("cuda")
    print(f"gpu={torch.cuda.get_device_name(device)} torch={torch.__version__}")

    ok = correctness_truncated(args, device)
    if not ok:
        print("\nSummary: correctness(truncated)=FAIL; full timing skipped.")
        return 2
    full_ok = None if args.skip_full else timing_full(args, device)
    full_status = "SKIPPED" if full_ok is None else "COMPLETE" if full_ok else "FAILED"
    print(f"\nSummary: correctness(truncated)=PASS; full-seq={full_status} "
          "(fp16 capacity/timing only, no full-length reference).")
    return 1 if full_ok is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

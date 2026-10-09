#!/usr/bin/env python3
"""
Assemble a SINGLE self-contained Kaggle kernel script from:
  torch_transformer_benchmark.py  (reference harness, minus its __main__ guard)
+ user_optimized.py               (minus the from-benchmark import and __future__)
+ _kaggle_driver.py               (the sweep driver)

No dataset attach needed -> robust headless run.

The Kaggle kernels API picks the GPU through the ``machine_shape`` metadata
field (equivalently ``kaggle kernels push --accelerator``). The accepted names
are documented only in the SDK docstring for ``ApiSaveKernelRequest``:
``NvidiaTeslaT4``, ``NvidiaTeslaP100``, ``Tpu1VmV38``. Anything else is silently
normalised back to a generic ``"Gpu"``, which lands on a P100.

Examples:
    python scripts/build_kaggle_selfcontained.py
    python scripts/build_kaggle_selfcontained.py --only 1-13 --accelerator NvidiaTeslaT4 \
        --id wenjiluo/track3-t4 --out .kaggle_upload/kernel_t4
"""

import argparse
import hashlib
import json
import os
import py_compile
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTDIR = os.path.join(HERE, ".kaggle_upload", "kernel_sc")


BOOTSTRAP = '''# --- bootstrap: ensure a torch build compatible with the allocated GPU ---
# Kaggle's API-allocated GPU is a Tesla P100 (sm_60) unless machine_shape asks for
# something else, and the preinstalled torch 2.10+cu128 does NOT support sm_60
# (sm_70+ only). Detect an incompatible build, reinstall a P100+T4-compatible
# torch, then re-exec so the new build is loaded. On a T4 (sm_75) this is a no-op.
import os as _os, sys as _sys, subprocess as _sp
# Expandable segments keep the allocator from fragmenting when the seq_len=1e5
# shape holds two ~6.5 GB tensors (input + output) plus per-chunk activations.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if _os.environ.get("_T3_BOOT") != "1":
    _need = False
    try:
        import torch as _t
        if _t.cuda.is_available():
            _cc = "sm_%d%d" % _t.cuda.get_device_capability()
            _need = _cc not in _t.cuda.get_arch_list()
        else:
            _need = True
    except Exception:
        _need = True
    if _need:
        print("[bootstrap] GPU/torch mismatch -> installing torch 2.5.1+cu121 ...", flush=True)
        _sp.run([_sys.executable, "-m", "pip", "install", "-q",
                 "--index-url", "https://download.pytorch.org/whl/cu121",
                 "torch==2.5.1"], check=False)
        _os.environ["_T3_BOOT"] = "1"
        _os.execv(_sys.executable, [_sys.executable] + _sys.argv)
    _os.environ["_T3_BOOT"] = "1"
'''


def read(p):
    with open(os.path.join(HERE, p), encoding="utf-8") as f:
        return f.read()


def build_selector(only, extra_env):
    """Kaggle kernels take no environment, so bake the knobs into the script."""
    lines = ["import os as _os2", '_os2.environ["T3_ONLY"] = "%s"' % only]
    for kv in extra_env:
        key, _, val = kv.partition("=")
        lines.append('_os2.environ["%s"] = "%s"' % (key.strip(), val.strip()))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "1-13", "14", "ablation", "triton", "probe", "attn", "profile", "x3", "gemm", "x3k", "x3geo", "x3i8", "runtime", "runtime14"],
                    help="which section of the sweep the kernel runs (default: all)")
    ap.add_argument("--id", default="wenjiluo/track3-bench",
                    help="Kaggle kernel id to push to")
    ap.add_argument("--accelerator", default=None,
                    choices=["NvidiaTeslaT4", "NvidiaTeslaP100", "Tpu1VmV38"],
                    help="machine_shape baked into kernel-metadata.json")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="extra environment variable baked into the kernel "
                         "(repeatable), e.g. --env T3_COMPILE=0")
    ap.add_argument("--reference-ref", default="HEAD",
                    help="git revision of the previous model for the paired runtime audit")
    args = ap.parse_args()

    bench = read("torch_transformer_benchmark.py")
    marker = 'if __name__ == "__main__":'
    assert marker in bench
    bench = bench[: bench.index(marker)].rstrip() + "\n"
    # Inject the GPU-compat bootstrap and the run selector right after the
    # mandatory __future__ import.
    fut = "from __future__ import annotations\n"
    assert fut in bench
    source_sha = hashlib.sha256(read("user_optimized.py").encode("utf-8")).hexdigest()
    source_env = [*args.env, "T3_RUNTIME_SOURCE_SHA256=" + source_sha]
    prologue = fut + "\n" + BOOTSTRAP + "\n" + build_selector(args.only, source_env)
    bench = bench.replace(fut, prologue, 1)

    # The Kaggle kernel is a single file, so the hand-written Triton kernels
    # have to be inlined too, and the package import in user_optimized.py
    # rewritten to point at the names that inlining leaves in scope.
    tk = read(os.path.join("kernels", "fused_layernorm.py"))
    tk = tk.replace("from __future__ import annotations", "")
    ta = read(os.path.join("kernels", "attention.py"))
    ta = ta.replace("from __future__ import annotations", "")
    tx = read(os.path.join("kernels", "fp16x3.py"))
    tx = tx.replace("from __future__ import annotations", "")
    ti = read(os.path.join("kernels", "fp16x3_int8.py"))
    ti = ti.replace("from __future__ import annotations", "")
    ti = ti.replace("from .fp16x3 import _x3_geometry, _x3_tail", "")

    uo = read("user_optimized.py")
    uo = uo.replace("from __future__ import annotations\n", "")
    uo = uo.replace("from torch_transformer_benchmark import BaselineTransformer\n", "")
    # Everything between the markers is the package import; in the single-file
    # build the kernels are inlined above it, under their module-level names.
    b0 = uo.index("# --- kernels import (begin) ---")
    b1 = uo.index("# --- kernels import (end) ---")
    b1 = uo.index("\n", b1)
    uo = uo[:b0] + ("HAVE_KERNELS = True\ntriton_attention = attention\n"
         "can_use_attention = can_use\n" + uo[b1 + 1:])

    driver = read(os.path.join("scripts", "_kaggle_driver.py"))
    if args.only in ("runtime", "runtime14"):
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", args.reference_ref], cwd=HERE, text=True).strip()
        previous = subprocess.check_output(
            ["git", "show", f"{revision}:user_optimized.py"], cwd=HERE, encoding="utf-8")
        previous = previous.replace("from __future__ import annotations\n", "")
        previous = previous.replace("from torch_transformer_benchmark import BaselineTransformer\n", "")
        p0 = previous.index("# --- kernels import (begin) ---")
        p1 = previous.index("# --- kernels import (end) ---")
        p1 = previous.index("\n", p1)
        previous = previous[:p0] + "HAVE_KERNELS = True\n" + previous[p1 + 1:]
        previous = previous.replace("class UserOptimizedTransformer(", "class PreviousTransformer(", 1)
        driver = (previous + "\n" + read(os.path.join("scripts", "benchmark_runtime.py"))
                  + ("\nruntime_benchmark(PreviousTransformer, " + repr(revision) + ")\n"
                     if args.only == "runtime" else "\nruntime_shape14(" + repr(revision) + ")\n"))

    combined = (
        bench
        + "\n\n# ====== kernels/fused_layernorm.py (inlined) ======\n\n"
        + tk
        + "\n\n# ============ kernels/attention.py (inlined) ============\n\n"
        + ta
        + "\n\n# ============ kernels/fp16x3.py (inlined) ============\n\n"
        + tx
        + "\n\n# ============ kernels/fp16x3_int8.py (inlined) ============\n\n"
        + ti
        + "\n\n# ================= user_optimized.py (inlined) =================\n\n"
        + uo
        + "\n\n# ================= sweep driver =================\n\n"
        + driver
    )

    outdir = args.out or DEFAULT_OUTDIR
    if not os.path.isabs(outdir):
        outdir = os.path.join(HERE, outdir)
    os.makedirs(outdir, exist_ok=True)
    out_py = os.path.join(outdir, "track3_sc.py")
    with open(out_py, "w", encoding="utf-8") as f:
        f.write(combined)

    meta = {
        "id": args.id,
        "title": args.id.split("/")[-1],
        "code_file": "track3_sc.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if args.accelerator:
        meta["machine_shape"] = args.accelerator
    with open(os.path.join(outdir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    py_compile.compile(out_py, doraise=True)
    print(f"wrote {out_py} ({len(combined)} chars, {combined.count(chr(10))+1} lines)")
    print(f"  only={args.only} id={args.id} accelerator={args.accelerator} env={args.env}")
    print("py_compile: OK")


if __name__ == "__main__":
    main()

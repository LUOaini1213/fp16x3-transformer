"""Normal-import, exact-commit CUDA release verification; never CPU timing evidence.

FP32 timing is paired. Memory is measured in a separate process for each single
model, excluding the reference/comparison buffers. Compile caches are private to
this job and shared across workers: first-call cost is not fresh-machine cost.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
import torch_transformer_benchmark as official
import user_optimized as production
from scripts.benchmark_runtime import RUNTIME_SHAPES
from scripts import benchmark_next

ROOT = Path(__file__).resolve().parents[1]
CORE = ["user_optimized.py", "torch_transformer_benchmark.py", *[
    "kernels/" + n for n in ("fp16x3.py", "fp16x3_int8.py", "attention.py",
    "fused_layernorm.py", "cublaslt_backend.py", "cublaslt_probe.cpp", "turing_attention.py")]]
SOURCES = CORE + ["scripts/benchmark_release.py", "scripts/build_clean_kaggle.py",
                  "scripts/benchmark_next.py", "scripts/benchmark_runtime.py"]
OUT = None
REAL_SDPA = F.scaled_dot_product_attention


def metadata():
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "python": sys.version, "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_manifest": {n: hashlib.sha256((ROOT / n).read_text(encoding="utf-8").encode()).hexdigest()
                                for n in SOURCES},
            "normal_imports": {"official": official.__file__, "production": production.__file__},
            "protocol": "three rotated paired steady rounds; separate event/host wall timing; same weights/input",
            "cache_scope": "fresh job caches, shared across subprocesses; cold costs are first public forward, not machine cold"}


def save(name, results):
    payload = {"metadata": metadata(), "results": results}
    (OUT / ("next_release_" + name + ".json")).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("RELEASE_" + name.upper() + " " + json.dumps(payload), flush=True)


def config(index):
    _, b, d, h, s, l, f = RUNTIME_SHAPES[index - 1]
    return official.TransformerConfig(b, s, d, h, f, l, True)


def inputs(cfg, seed=20261009, dtype=torch.float32):
    return official.generate_random_case(cfg, torch.device("cuda"), dtype, seed, 0., 1.)


def baseline(cfg):
    torch.manual_seed(141009 + cfg.d_model + cfg.seq_len)
    return official.BaselineTransformer(cfg).eval()


def make(cfg, base, variant):
    import kernels.fp16x3 as x3
    from kernels.cublaslt_backend import lt_candidate_matmul
    # The production module imports this provider only when its import-time
    # environment requests lt. Paired experiments switch variants explicitly.
    x3.lt_candidate_matmul = lt_candidate_matmul
    x3._X3_BLAS = "lt" if variant == "lt" else "torch"
    if variant == "baseline":
        return base.cuda()
    model = production.UserOptimizedTransformer(cfg).eval()
    official.copy_model_weights(base, model, strict=True)
    return model.cuda()


def modes(model):
    return {"graph": getattr(model, "_graph", None) is not None,
            "compiled": getattr(model, "_compiled", None) is not None,
            "x3": getattr(model, "_x3_on", False)}


def checked(ref, out):
    result = official.compare_outputs(ref, out, rtol=.02, atol=.002)
    assert result.passed, result
    return {"max_abs": result.max_abs_error, "failed": result.failed_elements,
            "elements": result.total_elements, "passed": result.passed}


def synchronized(call, x, mask):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = call(x, mask)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def memory_worker(index, variant):
    cfg = config(index)
    base = baseline(cfg)
    model = make(cfg, base, variant)
    if variant != "baseline":
        del base
    x, mask = inputs(cfg)
    gc.collect()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()
    with torch.inference_mode():
        torch.cuda.reset_peak_memory_stats()
        out, cold = synchronized(model, x, mask)
        cold_peak = torch.cuda.max_memory_allocated()
        del out
        for _ in range(3):
            model(x, mask)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = model(x, mask)
        torch.cuda.synchronize()
        steady_peak = torch.cuda.max_memory_allocated()
        assert out.dtype == torch.float32 and torch.isfinite(out).all()
    save(f"memory_{index}_{variant}", {"shape": index, "variant": variant,
         "resident_input_weights_bytes": resident, "cold_seconds": cold,
         "cold_peak_allocated_bytes": cold_peak, "steady_peak_allocated_bytes": steady_peak,
         "mode": modes(model), "memory_scope": "isolated single model; input, output and graph pools included; oracle excluded"})


def paired(calls, x, mask, repeats):
    benchmark_next.benchmark_once = official.benchmark_once
    return benchmark_next.next_paired(calls, x, mask, repeats)


def timing_worker(index):
    cfg = config(index)
    base = baseline(cfg)
    current = make(cfg, base, "default")
    tuned = make(cfg, base, "lt") if index == 8 else None
    base.cuda()
    import kernels.fp16x3 as x3
    import kernels.cublaslt_backend as lt

    def original(xx, mm):
        x3._X3_BLAS = "torch"
        return current(xx, mm)

    def optional(xx, mm):
        x3._X3_BLAS = "lt"
        return tuned(xx, mm)

    calls = {"baseline": base, "default": original}
    if tuned is not None:
        calls["lt"] = optional
    x, mask = inputs(cfg)
    row = {"shape": index, "dims": list(RUNTIME_SHAPES[index - 1]), "cold_seconds": {}, "accuracy": []}
    with torch.inference_mode():
        for name, call in calls.items():
            out, duration = synchronized(call, x, mask)
            row["cold_seconds"][name] = duration
            del out
        for seed in range(3):
            xx, mm = inputs(cfg, 20261009 + seed)
            ref = base(xx, mm)
            for name, call in list(calls.items())[1:]:
                out = call(xx, mm)
                row["accuracy"].append({"seed": seed, "variant": name, **checked(ref, out)})
                del out
            del ref, xx, mm
        old = original(x, mask)
        snap = old.clone()
        original(x * .5, mask)
        assert torch.equal(old, snap), "output alias"
        del old, snap
        row["output_ownership"] = "PASS"
        row["timing"] = paired(calls, x, mask, 6 if index == 6 else 20)
        row["mode"] = {"default": modes(current), **({"lt": modes(tuned)} if tuned is not None else {})}
        row["lt_search"] = lt._LT_RESULTS
        if tuned is not None:
            assert any(r["selected"] is not None for r in lt._LT_RESULTS), "Lt did not execute"
    save(f"timing_{index}", row)


def subprocess_phase(phase, index=None, variant=None):
    cmd = [sys.executable, "-m", "scripts.benchmark_release", "--phase", phase, "--output", str(OUT)]
    if index is not None:
        cmd += ["--shape", str(index)]
    if variant is not None:
        cmd += ["--variant", variant]
    subprocess.run(cmd, cwd=ROOT, check=True)


def fp32():
    rows = []
    for index in range(1, 14):
        subprocess_phase("timing", index)
        memory = {}
        for variant in ("baseline", "default", "lt") if index == 8 else ("baseline", "default"):
            subprocess_phase("memory", index, variant)
            memory[variant] = json.loads((OUT / f"next_release_memory_{index}_{variant}.json").read_text())["results"]
        row = json.loads((OUT / f"next_release_timing_{index}.json").read_text())["results"]
        row["memory"] = memory
        rows.append(row)
        save("fp32", rows)


def flash():
    # kernels.__init__ exports a same-named function. Dotted import-as can
    # resolve that attribute instead of the submodule; request the module.
    adapter = importlib.import_module("kernels.turing_attention")
    before = str(torch.__version__)
    start = time.perf_counter()
    with (OUT / "release_turing_build.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-m", "scripts.install_turing_attention"], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    assert str(torch.__version__) == before
    build_seconds = time.perf_counter() - start
    cfg = official.TransformerConfig(32, 100000, 1024, 16, 1024, 2, True)
    torch.manual_seed(141009)
    base = official.BaselineTransformer(cfg).eval()
    os.environ.update(T3_COMPILE="0", T3_CUDAGRAPH="0", T3_CHUNK_BS="1", T3_ATTN="sdpa")
    current = make(cfg, base, "default").half()
    os.environ["T3_ATTN"] = "turing"
    candidate = make(cfg, base, "default").half()
    x, mask = inputs(cfg, 141009, torch.float16)
    rows = {"implementation": "normal-import production adapter", "dtype": "float16",
            "external_revision": adapter.TURING_REVISION, "build_seconds": build_seconds,
            "scope": "full native-fp16 equivalence against SDPA; original fp32 oracle only first batch 512-token causal prefix",
            "full": {n: {"cold_seconds": None, "round_seconds": [], "peak_bytes": []} for n in ("sdpa", "turing")}}
    saved = None
    with torch.inference_mode():
        for turn in range(4):
            order = [("sdpa", current), ("turing", candidate)]
            if turn % 2:
                order.reverse()
            for name, model in order:
                torch.cuda.reset_peak_memory_stats()
                out, elapsed = synchronized(model, x, mask)
                peak = torch.cuda.max_memory_allocated()
                assert out.dtype == x.dtype and out.shape == x.shape
                assert all(bool(torch.isfinite(out[b:b + 1]).all()) for b in range(32))
                record = rows["full"][name]
                if turn == 0:
                    record["cold_seconds"] = elapsed
                else:
                    record["round_seconds"].append(elapsed)
                    record["peak_bytes"].append(peak)
                print(f"RELEASE_FLASH_PROGRESS round={turn} model={name} seconds={elapsed}", flush=True)
                if turn == 0 and name == "sdpa":
                    saved = out.cpu()
                if turn == 0 and name == "turing":
                    failed, max_abs = 0, 0.
                    for batch in range(32):
                        for token in range(0, 100000, 8192):
                            check = checked(saved[batch, token:token + 8192], out[batch, token:token + 8192].cpu())
                            failed += check["failed"]
                            max_abs = max(max_abs, check["max_abs"])
                    saved = None
                    rows["full_equivalence"] = {"elements": x.numel(), "failed": failed, "max_abs": max_abs}
                    prefix = x[:1, :512].float().contiguous()
                    base.cuda()
                    ref = base(prefix, None)
                    rows["fp32_causal_prefix"] = checked(ref, out[:1, :512])
                    base.cpu()
                    del prefix, ref
                del out
                rows["adapter_calls"] = adapter._TURING_CALLS
                save("flash", rows)
                gc.collect()
                torch.cuda.empty_cache()
    for value in rows["full"].values():
        value["median_seconds"] = statistics.median(value["round_seconds"])
    rows["speedup"] = rows["full"]["sdpa"]["median_seconds"] / rows["full"]["turing"]["median_seconds"]
    rows["memory_scope"] = "shared two-model process, not isolated deployment VRAM"
    assert adapter._TURING_CALLS == 256, adapter._TURING_CALLS
    save("flash", rows)


def layout_attention(layout, q, k, v, **kwargs):
    if layout == "bhsd":
        q, k, v = [t.contiguous() for t in (q, k, v)]
    elif layout == "bshd":
        q, k, v = [t.transpose(1, 2).contiguous().transpose(1, 2) for t in (q, k, v)]
    elif layout == "fold":
        b, h, s, d = q.shape
        q, k, v = [t.reshape(b * h, 1, s, d) for t in (q, k, v)]
        return REAL_SDPA(q, k, v, **kwargs).reshape(b, h, s, d)
    elif layout.startswith("chunk"):
        n = int(layout[5:])
        # Fresh output; include launch and concat costs in the measurement.
        return torch.cat([REAL_SDPA(q[i:i+n], k[i:i+n], v[i:i+n], **kwargs)
                          for i in range(0, q.shape[0], n)], dim=0)
    elif layout != "current":
        raise ValueError(layout)
    return REAL_SDPA(q, k, v, **kwargs)


@contextmanager
def attention_layout(layout):
    F.scaled_dot_product_attention = lambda q, k, v, **kw: layout_attention(layout, q, k, v, **kw)
    try:
        yield
    finally:
        F.scaled_dot_product_attention = REAL_SDPA


def attention():
    layouts = ("current", "bhsd", "bshd", "fold", "chunk8", "chunk16", "chunk32")
    rows = {"dtype": "float32", "copies_included": True, "micro_accuracy": [], "full_accuracy": []}
    with torch.inference_mode():
        for seed in range(3):
            torch.manual_seed(20261009 + seed)
            packed = torch.randn(64, 1024, 3 * 128, device="cuda")
            q, k, v = [t.reshape(64, 1024, 4, 32).transpose(1, 2) for t in packed.split(128, -1)]
            ref = REAL_SDPA(q, k, v, is_causal=True, scale=32 ** -.5)
            for layout in layouts:
                out = layout_attention(layout, q, k, v, is_causal=True, scale=32 ** -.5)
                rows["micro_accuracy"].append({"layout": layout, "seed": seed, **checked(ref, out)})
            if seed == 0:
                rows["micro_timing"] = paired({n: lambda xx, mm, n=n: layout_attention(
                    n, q, k, v, is_causal=True, scale=32 ** -.5) for n in layouts}, q, None, 20)
            del packed, q, k, v, ref, out
        # Test every candidate end-to-end, even if its microbenchmark loses.
        os.environ.update(T3_COMPILE="0", T3_CUDAGRAPH="0")
        cfg = config(13)
        base = baseline(cfg)
        models = {n: make(cfg, base, "default") for n in layouts}
        base.cuda()

        def full(name, xx, mm):
            with attention_layout(name):
                return models[name](xx, mm)

        calls = {n: lambda xx, mm, n=n: full(n, xx, mm) for n in layouts}
        for seed in range(3):
            x, mask = inputs(cfg, 20261009 + seed)
            ref = base(x, mask)
            for name, call in calls.items():
                out = call(x, mask)
                rows["full_accuracy"].append({"layout": name, "seed": seed, **checked(ref, out)})
            del ref, out
        rows["full_eager_timing"] = paired(calls, x, mask, 20)
        rows["production_default_unchanged"] = True
    save("attention", rows)


def main():
    global OUT
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--phase", choices=("fp32", "flash", "attention", "timing", "memory"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shape", type=int, choices=range(1, 14))
    ap.add_argument("--variant", choices=("baseline", "default", "lt"))
    args = ap.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5), "T4 required"
    assert Path(production.__file__).parent == ROOT
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/track3-release-inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/track3-release-triton")
    print("RELEASE_START " + json.dumps(metadata()), flush=True)
    if args.phase == "timing":
        timing_worker(args.shape)
    elif args.phase == "memory":
        memory_worker(args.shape, args.variant)
    else:
        {"fp32": fp32, "flash": flash, "attention": attention}[args.phase]()


if __name__ == "__main__":
    main()

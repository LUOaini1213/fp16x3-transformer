"""Paired runtime audit, using the official timing loop, on identical weights.

The kernel builder inlines this with a named git revision of the previous model.
Results are separate from the existing measurements; historical CSVs stay intact.
"""

import gc
import json
import os
import statistics
import time

import torch


RUNTIME_SHAPES = [
    (1, 64, 128, 4, 128, 4, 128), (2, 1, 128, 4, 128, 4, 128),
    (3, 4, 128, 4, 128, 4, 128), (4, 16, 128, 4, 128, 4, 128),
    (5, 128, 128, 4, 128, 4, 128), (6, 10000, 128, 4, 128, 4, 128),
    (7, 64, 32, 4, 128, 4, 32), (8, 64, 1024, 4, 128, 4, 1024),
    (9, 64, 128, 1, 128, 4, 128), (10, 64, 128, 2, 128, 4, 128),
    (11, 64, 128, 16, 128, 4, 128), (12, 64, 128, 4, 32, 4, 128),
    (13, 64, 128, 4, 1024, 4, 128),
]


def runtime_shape14(reference_revision):
    """One full-length fp16 forward, plus an oracle for the causal prefix.

    This is deliberately a cold-call measurement, not a steady-state median.
    The reference never allocates a full S-by-S matrix; for a causal model,
    the first 512 outputs are independent of the later 99,488 input tokens.
    """
    cfg = TransformerConfig(32, 100000, 1024, 16, 1024, 2, True)
    torch.manual_seed(141009)
    base = BaselineTransformer(cfg).eval()
    model = UserOptimizedTransformer(cfg).eval()
    copy_model_weights(base, model, strict=True)
    model = model.cuda().half()
    x, mask = generate_random_case(cfg, torch.device("cuda"), torch.float16, 141009, 0.0, 1.0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(x, mask)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    # Never allocate a bool tensor as large as the complete 6.55 GB output.
    with torch.inference_mode():
        finite = all(bool(torch.isfinite(out[i:i + 1]).all()) for i in range(len(out)))
        assert finite and out.shape == x.shape and out.dtype == x.dtype
        prefix = x[:1, :512].float().contiguous()
        base = base.cuda().float()
        ref = base(prefix, None)
        accuracy = compare_outputs(ref, out[:1, :512], rtol=0.02, atol=0.002)
        assert accuracy.passed, accuracy
    row = {"shape": 14, "status": "FULL_FINITE_PREFIX_PASS", "dtype": "float16",
           "cold_seconds": elapsed, "peak_allocated_bytes": peak, "chunk_bs": model._chunk_bs,
           "prefix_tokens_checked": 512, "prefix_batches_checked": 1,
           "prefix_max_abs": accuracy.max_abs_error, "prefix_failed_elements": accuracy.failed_elements,
           "source_sha256": os.environ.get("T3_RUNTIME_SOURCE_SHA256"),
           "reference_revision": reference_revision, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(),
           "correctness_scope": "all output elements finite; first batch's causal prefix compared against fp32 reference; not full-length numerical equivalence"}
    print("RUNTIME_SHAPE14 " + json.dumps(row), flush=True)
    with open("runtime_shape14.json", "w", encoding="utf-8") as output:
        json.dump(row, output, indent=2)


def runtime_contracts():
    """Exercise cache invalidation and output ownership on a real CUDA graph."""
    saved = {k: os.environ.get(k) for k in ("T3_COMPILE", "T3_CUDAGRAPH")}
    os.environ.update(T3_COMPILE="0", T3_CUDAGRAPH="0")
    try:
        cfg = TransformerConfig(2, 16, 32, 4, 32, 2, True)
        torch.manual_seed(47)
        base = BaselineTransformer(cfg).cuda().eval()
        opt = UserOptimizedTransformer(cfg).cuda().eval()
        copy_model_weights(base, opt, strict=True)
        x = torch.randn(2, 16, 32, device="cuda")
        mask = torch.ones(2, 16, dtype=torch.bool, device="cuda")

        def check(xx, mm):
            ref, out = base(xx, mm), opt(xx, mm)
            result = compare_outputs(ref, out, rtol=0.02, atol=0.002)
            assert result.passed, result
            return out

        with torch.inference_mode():
            saved_output = check(x, mask)
            snapshot = saved_output.clone()
            assert opt._capture_graph(x, mask, lambda fn, xx, mm, av: fn(xx, mm, True, av))
            check(x * 1.5, mask)
            assert opt._g_m is None
            assert torch.equal(saved_output, snapshot), "a later call overwrote an owned output"
            mask.view(-1)[8:16].fill_(False)
            check(x, mask)
            mask.fill_(True)
            check(x, mask)
            inference_mask = mask.clone()
            assert inference_mask.is_inference()
            check(x, inference_mask)
            inference_mask[1, 7:] = False
            check(x, inference_mask)
            check(x, mask.clone())
            # Irregular padding, shared lengths and an entirely empty row.
            mask[0].fill_(False)
            mask[1, ::2] = False
            check(x, mask)
            mask.fill_(True)

        # A new Parameter can have the same mutation counter as the old one.
        lin = opt.layers[0].attention.out_proj
        replacement = torch.nn.Parameter(torch.randn_like(lin.weight) * 0.1)
        with torch.no_grad():
            while replacement._version < lin.weight._version:
                replacement.add_(0)
        assert replacement._version == lin.weight._version
        lin.weight = replacement
        base.layers[0].attention.out_proj.weight = torch.nn.Parameter(replacement.detach().clone())
        with torch.inference_mode():
            check(x, mask)
            opt.layers[0].norm1.weight.fill_(1e5)
            base.layers[0].norm1.weight.fill_(1e5)
            check(x, mask)
            assert not opt._x3_on
        print("RUNTIME_CONTRACTS: PASS (CUDA graph, masks, weights, range, owned outputs)", flush=True)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    del base, opt, x, mask, saved_output, snapshot
    gc.collect()
    torch.cuda.empty_cache()


def runtime_padding(previous_class):
    """Measured padded performance and memory, plus a sparse long-sequence case."""
    cfg = TransformerConfig(4, 512, 128, 4, 128, 2, True)
    base = BaselineTransformer(cfg).cuda().eval()
    previous = previous_class(cfg).cuda().eval()
    current = UserOptimizedTransformer(cfg).cuda().eval()
    for model in (previous, current):
        copy_model_weights(base, model, strict=True)
    x = torch.randn(4, 512, 128, device="cuda")
    mask = torch.ones(4, 512, dtype=torch.bool, device="cuda")
    mask[:2, 256:] = False
    mask[2:, ::2] = False
    timing, memory, retained = {}, {}, {}
    with torch.inference_mode():
        ref = base(x, mask)
        for name, model in (("previous", previous), ("current", current)):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = model(x, mask)
            torch.cuda.synchronize()
            first_peak = torch.cuda.max_memory_allocated() - before
            result = compare_outputs(ref, out, rtol=0.02, atol=0.002)
            assert result.passed, (name, result)
            del out
            # Exclude the comparison's temporary tensors, but count prepared
            # weights, pack indices and the previous model's persistent CUDA
            # graph pool. Measuring only after warmup hides that pool and made
            # the previous path misleadingly look like a 1 MB allocation.
            torch.cuda.reset_peak_memory_stats()
            warmup_model(model, x, mask, 5, torch.device("cuda"))
            timing[name] = statistics.median(benchmark_once(model, x, mask, 30, torch.device("cuda")))
            torch.cuda.synchronize()
            memory[name] = max(first_peak, torch.cuda.max_memory_allocated() - before)
            retained[name] = torch.cuda.memory_allocated() - before
    row = {"case": "padded_4x512x128", "status": "PASS", "previous_ms": timing["previous"],
           "current_ms": timing["current"], "improvement": timing["previous"] / timing["current"],
           "previous_peak_extra_bytes": memory["previous"], "current_peak_extra_bytes": memory["current"],
           "memory_protocol": "peak includes first-forward caches and persistent graph allocations; comparison tensors excluded",
           "previous_retained_bytes": retained["previous"], "current_retained_bytes": retained["current"]}
    print("PADDING_RESULT " + json.dumps(row), flush=True)
    del base, previous, current, model, x, mask, ref
    gc.collect()
    torch.cuda.empty_cache()

    # The baseline cannot expand attention at this original length. Its
    # compact counterpart is an independent reference on exactly the valid
    # tokens (same ordering, no position-dependent layers in this model).
    large = TransformerConfig(2, 32768, 128, 4, 128, 2, True)
    short = TransformerConfig(2, 1024, 128, 4, 128, 2, True)
    base = BaselineTransformer(short).cuda().eval()
    current = UserOptimizedTransformer(large).cuda().eval()
    copy_model_weights(base, current, strict=True)
    x = torch.randn(2, 32768, 128, device="cuda")
    mask = torch.zeros(2, 32768, dtype=torch.bool, device="cuda")
    mask[:, ::32] = True
    with torch.inference_mode():
        ref = base(x[:, ::32], None)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = current(x, mask)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        result = compare_outputs(ref, out[:, ::32], rtol=0.02, atol=0.002)
        assert result.passed and bool((out[~mask] == 0).all())
    long_row = {"case": "sparse_2x32768x128", "status": "PASS", "valid_per_row": 1024,
                "max_abs": result.max_abs_error, "peak_extra_bytes": peak,
                "old_dense_bias_bytes": 2 * 32768 * 32768 * 4,
                "reference": "same weights, compact valid tokens, original order"}
    print("PADDING_RESULT " + json.dumps(long_row), flush=True)
    return [row, long_row]


def runtime_benchmark(previous_class, reference_revision):
    runtime_contracts()
    selected = {int(s) for s in os.environ.get("T3_RUNTIME_SHAPES", "1,2,3,4,7,8,11,12").split(",")}
    repeats = int(os.environ.get("T3_RUNTIME_REPEATS", "50"))
    rounds = int(os.environ.get("T3_RUNTIME_ROUNDS", "3"))
    trials = int(os.environ.get("T3_RUNTIME_TRIALS", "3"))
    metadata = {"reference_revision": reference_revision,
                "current_source_sha256": os.environ.get("T3_RUNTIME_SOURCE_SHA256"), "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                "timing": "official benchmark_once, interleaved three-model order",
                "repeats": repeats, "rounds": rounds, "accuracy_trials": trials}
    print("RUNTIME_METADATA " + json.dumps(metadata), flush=True)
    rows = []
    for idx, b, d, h, s, l, f in RUNTIME_SHAPES:
        if idx not in selected:
            continue
        torch.manual_seed(1234 + idx)
        cfg = TransformerConfig(b, s, d, h, f, l, True)
        base = BaselineTransformer(cfg).cuda().eval()
        previous = previous_class(cfg).cuda().eval()
        current = UserOptimizedTransformer(cfg).cuda().eval()
        copy_model_weights(base, previous, strict=True)
        copy_model_weights(base, current, strict=True)
        models = {"reference": base, "previous": previous, "current": current}
        maxima = {name: 0.0 for name in ("previous", "current")}
        for trial in range(trials):
            x, m = generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                        20261009 + trial, 0.0, 1.0)
            with torch.inference_mode():
                ref = base(x, m)
                for name in ("previous", "current"):
                    result = compare_outputs(ref, models[name](x, m), rtol=0.02, atol=0.002)
                    assert result.passed, (idx, name, result)
                    maxima[name] = max(maxima[name], result.max_abs_error)
            del ref
        # A different, ordinary tensor mask has a version counter and can be
        # cached. This is the same case the official benchmark generates.
        x, m = generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                    30361009, 0.0, 1.0)
        samples = {name: [] for name in models}
        for model in models.values():
            warmup_model(model, x, m, 20, torch.device("cuda"))
        order = list(models)
        for r in range(rounds):
            for name in order[r % 3:] + order[:r % 3]:
                samples[name].extend(benchmark_once(models[name], x, m, repeats, torch.device("cuda")))
        med = {name: statistics.median(vals) for name, vals in samples.items()}
        # Also measure end-to-end host latency; do not mistake a queued GPU
        # launch improvement for the application's wall-clock latency.
        wall = {}
        for name in ("previous", "current"):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(10):
                    models[name](x, m)
            torch.cuda.synchronize()
            wall[name] = (time.perf_counter() - t0) * 100.0
        row = {"shape": idx, "status": "PASS", **{k + "_ms": v for k, v in med.items()},
               "previous_speedup": med["reference"] / med["previous"],
               "current_speedup": med["reference"] / med["current"],
               "improvement": med["previous"] / med["current"],
               "previous_wall_ms": wall["previous"], "current_wall_ms": wall["current"],
               "max_abs": maxima["current"]}
        rows.append(row)
        print("RUNTIME_RESULT " + json.dumps(row), flush=True)
        # Autotuning / graph ownership diagnostics, kept separate from metrics.
        print("RUNTIME_MODES " + json.dumps({name: {
            "graph": model._graph is not None, "compiled": model._compiled is not None,
            "tune": model._tune_result} for name, model in models.items() if name != "reference"}), flush=True)
        del models, base, previous, current, x, m, model
        gc.collect()
        torch.cuda.empty_cache()
    summary = {"shapes": len(rows), "all_passed": True,
               "median_previous_speedup": statistics.median(r["previous_speedup"] for r in rows),
               "median_current_speedup": statistics.median(r["current_speedup"] for r in rows),
               "median_improvement": statistics.median(r["improvement"] for r in rows)}
    padding = runtime_padding(previous_class)
    print("RUNTIME_SUMMARY " + json.dumps(summary), flush=True)
    with open("runtime_results.json", "w", encoding="utf-8") as out:
        json.dump({"metadata": metadata, "results": rows, "summary": summary, "padding": padding}, out, indent=2)

"""New, source-stamped experiments; inlined by build_kaggle_selfcontained.py.

No result in this file changes production dispatch automatically. Timing uses
three rotated paired rounds and includes public-forward checks and ownership.
"""
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import types
from collections import defaultdict

import torch


NEXT_SHAPES = {
    # B, S, D, H, F, layers (unlike the legacy sweep's column order).
    2: (1, 128, 128, 4, 128, 4), 3: (4, 128, 128, 4, 128, 4),
    6: (10000, 128, 128, 4, 128, 4), 8: (64, 128, 1024, 4, 1024, 4),
    12: (64, 32, 128, 4, 128, 4), 13: (64, 1024, 128, 4, 128, 4),
}


def next_metadata():
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "source_manifest": json.loads(os.environ["T3_SOURCE_MANIFEST"]),
            "reference_revision": "46154d19ceeba0c50af2588a0247ce43b56c0465"}


def next_regression():
    cases = [(1, 64, 128, 4, 128, 4, 128), (2, 1, 128, 4, 128, 4, 128),
             (3, 4, 128, 4, 128, 4, 128), (4, 16, 128, 4, 128, 4, 128),
             (5, 128, 128, 4, 128, 4, 128), (6, 10000, 128, 4, 128, 4, 128),
             (7, 64, 32, 4, 128, 4, 32), (8, 64, 1024, 4, 128, 4, 1024),
             (9, 64, 128, 1, 128, 4, 128), (10, 64, 128, 2, 128, 4, 128),
             (11, 64, 128, 16, 128, 4, 128), (12, 64, 128, 4, 32, 4, 128),
             (13, 64, 128, 4, 1024, 4, 128)]
    os.environ.pop("T3_COMPILE", None)
    os.environ.pop("T3_CUDAGRAPH", None)
    rows = []
    for index, b, d, h, s, l, f in cases:
        cfg = TransformerConfig(b, s, d, h, f, l, True)
        torch.manual_seed(141009 + index)
        base = BaselineTransformer(cfg).cuda().eval()
        model = UserOptimizedTransformer(cfg).cuda().eval()
        copy_model_weights(base, model, strict=True)
        row = {"shape": index, "passed": True, "max_abs": 0., "failed": 0, "trials": 3}
        with torch.inference_mode():
            for trial in range(3):
                x, mask = generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                              20261009 + trial, 0., 1.)
                out = model(x, mask)
                ref = base(x, mask)
                check = compare_outputs(ref, out, rtol=.02, atol=.002)
                assert check.passed, (index, trial, check)
                row["max_abs"] = max(row["max_abs"], check.max_abs_error)
                del x, mask, out, ref
        row["graph"] = model._graph is not None
        row["compiled"] = model._compiled is not None
        rows.append(row)
        next_save("next_regression", rows)
        del base, model
        gc.collect()
        torch.cuda.empty_cache()


def next_save(name, results):
    payload = {"metadata": next_metadata(), "results": results}
    with open(name + ".json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(name.upper() + " " + json.dumps(payload), flush=True)


def next_paired(calls, x, mask, repeats=30):
    """Official event timing + separate synchronized host-inclusive timing."""
    result = {name: {"event_round_ms": [], "wall_round_ms": []} for name in calls}
    names = list(calls)
    with torch.inference_mode():
        for call in calls.values():
            for _ in range(3):
                call(x, mask)
        torch.cuda.synchronize()
        for turn in range(3):
            order = names[turn % len(names):] + names[:turn % len(names)]
            for name in order:
                samples = benchmark_once(calls[name], x, mask, repeats, torch.device("cuda"))
                result[name]["event_round_ms"].append(statistics.median(samples))
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(repeats):
                    calls[name](x, mask)
                torch.cuda.synchronize()
                result[name]["wall_round_ms"].append((time.perf_counter() - start) * 1000 / repeats)
    for value in result.values():
        value["event_ms"] = statistics.median(value["event_round_ms"])
        value["wall_ms"] = statistics.median(value["wall_round_ms"])
    return result


def next_profile():
    rows = []
    for index, dims in NEXT_SHAPES.items():
        cfg = TransformerConfig(*dims, True)
        torch.manual_seed(141009 + index)
        model = UserOptimizedTransformer(cfg).cuda().eval()
        x, mask = generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       141009 + index, 0.0, 1.0)
        with torch.inference_mode():
            model(x, mask)
            for _ in range(3):
                model(x, mask)
            torch.cuda.synchronize()
            # Aggregate device kernel events, not overlapping/nested CPU ops.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    model(x, mask)
                torch.cuda.synchronize()
            device = defaultdict(float)
            for event in prof.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    device[event.name] += event.time_range.elapsed_us()
            total = sum(device.values())
            kernels = [{"name": name, "total_us": duration,
                        "device_share": duration / total if total else None}
                       for name, duration in sorted(device.items(), key=lambda pair: -pair[1])[:20]]
            row = {"shape": index, "graph": model._graph is not None,
                   "compiled": model._compiled is not None, "x3": model._x3_on,
                   "kernel_events": kernels, "device_total_us": total}
            if index in (2, 3, 12):
                # Host checks only: no tensor values changed or cache checks disabled.
                start = time.perf_counter()
                for _ in range(3000):
                    model._refresh_x3()
                row["weight_checks_host_us"] = (time.perf_counter() - start) * 1e6 / 3000
                start = time.perf_counter()
                for _ in range(3000):
                    model._all_valid(mask)
                row["mask_checks_host_us"] = (time.perf_counter() - start) * 1e6 / 3000
                calls = {"public": model, "inner_eager": lambda xx, mm: model._run_full(xx, mm, True, True)}
                if model._graph is not None:
                    calls["graph_owned_output"] = model._replay
                    calls["graph_replay_only_NOT_PUBLIC_CONTRACT"] = lambda xx, mm: model._graph.replay()
                row["decomposition"] = next_paired(calls, x, mask, 50)
        rows.append(row)
        print("PROFILE_ROW " + json.dumps(row), flush=True)
        next_save("next_profile", rows)
        del model, x, mask, prof
        gc.collect()
        torch.cuda.empty_cache()


def next_cpp():
    from torch._inductor import config as inductor_config
    from pathlib import Path
    # Kaggle exposes the real driver as libcuda.so.1 but omits the development
    # libcuda.so linker name. Repair only this experiment's private search path.
    candidates = []
    for folder in ("/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64", "/usr/lib64-nvidia"):
        candidates.extend(Path(folder).glob("libcuda.so.1"))
    for candidate in candidates:
        if candidate.exists():
            directory = Path("/kaggle/working/cuda_link")
            directory.mkdir(exist_ok=True)
            link = directory / "libcuda.so"
            if not link.exists():
                link.symlink_to(candidate.resolve())
            os.environ["LIBRARY_PATH"] = str(directory) + ":" + os.environ.get("LIBRARY_PATH", "")
            print("CPP_DRIVER_LINK " + str(link) + " -> " + str(candidate.resolve()), flush=True)
            break
    rows = []
    # C++ wrapper is an alternative to dispatch only, not permission to remove
    # mutable-weight / mutable-mask checks or return an aliased graph buffer.
    for index in (2, 3, 12):
        cfg = TransformerConfig(*NEXT_SHAPES[index], True)
        os.environ.update(T3_COMPILE="0", T3_CUDAGRAPH="1")
        torch.manual_seed(141009 + index)
        model = UserOptimizedTransformer(cfg).cuda().eval()
        base = BaselineTransformer(cfg).cuda().eval()
        copy_model_weights(model, base, strict=True)
        x, mask = generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       141009 + index, 0.0, 1.0)
        row = {"shape": index, "candidates": {}, "failures": {}}
        with torch.inference_mode():
            model(x, mask)
            calls = {"current_public": model}
            for name, cpp in (("python_wrapper", False), ("cpp_wrapper", True)):
                try:
                    with inductor_config.patch({"cpp_wrapper": cpp, "triton.cudagraphs": False}):
                        # Dynamo caches by code object. Separate entry code objects
                        # prevent the second candidate reusing the first wrapper.
                        method = model._run_full.__func__
                        unique = types.FunctionType(method.__code__.replace(), method.__globals__,
                                                    name=method.__name__, argdefs=method.__defaults__)
                        bound = types.MethodType(unique, model)
                        inner = torch.compile(bound, dynamic=False, fullgraph=True)
                        inner(x, mask, True, True)  # compile inside the config context

                    def checked(xx, mm, core=inner):
                        model._plan(xx)
                        if model._fused_qkv:
                            model._refresh_fused_qkv()
                        if model._x3_on:
                            model._refresh_x3()
                        valid = model._all_valid(mm)
                        return core(xx, mm, True, valid).to(xx.dtype)

                    accuracies = []
                    for seed in range(3):
                        xx = torch.randn_like(x) * (1 + seed * .25)
                        result = compare_outputs(base(xx, mask), checked(xx, mask),
                                                 rtol=.02, atol=.002)
                        assert result.passed, result
                        accuracies.append(result.max_abs_error)
                    old = checked(x, mask)
                    snapshot = old.clone()
                    checked(x * .5, mask)
                    assert torch.equal(old, snapshot), "output ownership failed"
                    calls[name] = checked
                    row["candidates"][name] = {"max_abs": max(accuracies), "ownership": "PASS"}
                except Exception as exc:
                    row["failures"][name] = {"type": type(exc).__name__, "message": str(exc)[:6000]}
                    traceback.print_exc()
            row["timing"] = next_paired(calls, x, mask, 50)
        rows.append(row)
        next_save("next_cpp", rows)
        del model, base, x, mask, calls
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()


def next_gemm():
    from torch.utils.cpp_extension import load_inline
    os.environ["MAX_JOBS"] = "1"
    print("LT_BUILD_START", flush=True)
    extension = load_inline("track3_lt_probe", cpp_sources=NEXT_LT_SOURCE,
                            extra_cflags=["-O2"], extra_ldflags=["-lcublasLt"],
                            with_cuda=True, verbose=True)
    print("LT_BUILD_COMPLETE", flush=True)
    rows = []
    original = globals()["_x3_linear_cuda"]

    def geometry(a, w):
        return (a.shape[0], w.shape[0], a.shape[1], a.stride(0), w.stride(0))

    for index in (8, 6, 13):
        cfg = TransformerConfig(*NEXT_SHAPES[index], True)
        os.environ.update(T3_COMPILE="0", T3_CUDAGRAPH="0")
        torch.manual_seed(141009 + index)
        model = UserOptimizedTransformer(cfg).cuda().eval()
        base = BaselineTransformer(cfg).cuda().eval()
        copy_model_weights(model, base, strict=True)
        x, mask = generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       141009 + index, 0.0, 1.0)
        operands = {}

        def record(a, w, bias, inv, apply_scale=True):
            # Retain actual split operands, including padded leading dimensions;
            # never replace a strided tail-free view by a contiguous synthetic one.
            operands.setdefault(geometry(a, w), (a, w, inv))
            return original(a, w, bias, inv, apply_scale)

        with torch.inference_mode():
            globals()["_x3_linear_cuda"] = record
            try:
                model(x, mask)
            finally:
                globals()["_x3_linear_cuda"] = original
            row = {"shape": index, "geometries": [], "algorithms": {}}
            winners = {}
            for dims, (a, w, inv) in operands.items():
                descriptors = extension.algorithms(a, w, 64)
                ref = original(a, w, None, inv, False)
                oracle = a[:8].double() @ w.double().t()
                calls = {"torch_mm": lambda xx, mm, aa=a, ww=w: original(aa, ww, None, 1., False)}
                errors, failures = {}, {}
                for algo, descriptor in enumerate(descriptors):
                    if descriptor[0] != 0:
                        continue
                    try:
                        out = extension.matmul(a, w, algo)
                        maximum = 0.
                        for start in range(0, len(a), 8192):
                            check = compare_outputs(ref[start:start + 8192] * inv,
                                                    out[start:start + 8192] * inv,
                                                    rtol=.0001, atol=.00005)
                            assert check.passed, check
                            maximum = max(maximum, check.max_abs_error)
                        oracle_check = compare_outputs((oracle * inv).float(), out[:8] * inv,
                                                        rtol=.0001, atol=.00005)
                        assert check.passed and oracle_check.passed, (check, oracle_check)
                        errors[str(algo)] = {"vs_torch_max_abs": maximum,
                                            "vs_fp64_max_abs": oracle_check.max_abs_error}
                        calls[str(algo)] = lambda xx, mm, aa=a, ww=w, ai=algo: extension.matmul(aa, ww, ai)
                        del out
                    except Exception as exc:
                        failures[str(algo)] = str(exc)[:1000]
                times = next_paired(calls, x, mask, 10)
                winner = min(times, key=lambda name: times[name]["wall_ms"])
                # Isolated GEMM wins must clear a noise margin, then still win in
                # public forward with all safety checks. No input/output memoization.
                improvement = times["torch_mm"]["wall_ms"] / times[winner]["wall_ms"]
                if winner != "torch_mm" and improvement > 1.03:
                    winners[dims] = int(winner)
                probe = {"geometry": dims, "descriptors": descriptors, "errors": errors,
                         "failures": failures, "timing": times, "winner": winner,
                         "improvement": improvement, "selected": dims in winners}
                row["geometries"].append(probe)
                print("LT_GEOMETRY " + json.dumps(probe), flush=True)
                del ref, oracle, calls

            def tuned(a, w, bias, inv, apply_scale=True):
                choice = winners.get(geometry(a, w))
                if choice is None:
                    return original(a, w, bias, inv, apply_scale)
                out = extension.matmul(a, w, choice)
                if apply_scale and inv != 1.:
                    out *= inv
                if bias is not None:
                    out += bias
                return out

            def call_original(xx, mm):
                globals()["_x3_linear_cuda"] = original
                return model(xx, mm)

            def call_tuned(xx, mm):
                globals()["_x3_linear_cuda"] = tuned
                return model(xx, mm)

            row["max_abs"] = 0.
            for trial in range(3):
                xx, mm = generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                              20261009 + trial, 0., 1.)
                ref = base(xx, mm)
                out = call_tuned(xx, mm)
                check = compare_outputs(ref, out, rtol=.02, atol=.002)
                assert check.passed, (index, check)
                row["max_abs"] = max(row["max_abs"], check.max_abs_error)
                del xx, mm, ref, out
            operands.clear()
            gc.collect()
            torch.cuda.empty_cache()
            row["selected_algorithms"] = {str(k): v for k, v in winners.items()}
            row["end_to_end"] = next_paired({"current": call_original, "tuned": call_tuned}, x, mask, 20)
            peaks = {}
            for name, call in (("current", call_original), ("tuned", call_tuned)):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                out = call(x, mask)
                torch.cuda.synchronize()
                peaks[name] = {"total_bytes": torch.cuda.max_memory_allocated(),
                               "extra_bytes": torch.cuda.max_memory_allocated() - before}
                del out
            row["memory"] = peaks
            row["workspace_note"] = "zero workspace in this revision; descriptors cached per geometry and alignment"
            globals()["_x3_linear_cuda"] = original
        rows.append(row)
        next_save("next_gemm", rows)
        extension.clear()
        del model, base, x, mask, a, w, operands, call_original, call_tuned, tuned
        gc.collect()
        torch.cuda.empty_cache()


def next_native_keys():
    """Bulk native metadata reads; preserve identity/version/storage/dtype/device."""
    from torch.utils.cpp_extension import load_inline
    os.environ["MAX_JOBS"] = "1"
    extension = load_inline("track3_lt_probe", cpp_sources=NEXT_LT_SOURCE,
                            extra_cflags=["-O2"], extra_ldflags=["-lcublasLt"],
                            with_cuda=True, verbose=True)

    class NativeKeysTransformer(UserOptimizedTransformer):
        def _refresh_x3(self):
            params = list(self.parameters())  # re-read to detect replacement
            keys = extension.tensor_keys(params)
            self._native_keys = {id(param): key for param, key in zip(params, keys)}
            try:
                super()._refresh_x3()
            finally:
                self._native_keys = None

        def _tensor_key(self, tensor):
            keys = getattr(self, "_native_keys", None)
            if tensor is not None and keys is not None and id(tensor) in keys:
                return keys[id(tensor)]
            return UserOptimizedTransformer._tensor_key(tensor)

    rows = []
    for index in (2, 3, 12):
        os.environ.update(T3_COMPILE="0", T3_CUDAGRAPH="1")
        cfg = TransformerConfig(*NEXT_SHAPES[index], True)
        torch.manual_seed(141009 + index)
        base = BaselineTransformer(cfg).cuda().eval()
        current = UserOptimizedTransformer(cfg).cuda().eval()
        candidate = NativeKeysTransformer(cfg).cuda().eval()
        for model in (current, candidate):
            copy_model_weights(base, model, strict=True)
        x, mask = generate_random_case(cfg, torch.device("cuda"), torch.float32, 141009, 0., 1.)
        with torch.inference_mode():
            for model in (current, candidate):
                model(x, mask)
            row = {"shape": index, "max_abs": 0., "host_us": {}}
            for seed in range(3):
                xx = torch.randn_like(x) * (1 + seed * .25)
                result = compare_outputs(base(xx, mask), candidate(xx, mask), rtol=.02, atol=.002)
                assert result.passed, result
                row["max_abs"] = max(row["max_abs"], result.max_abs_error)
            # Hold old output across a replay and exercise mutation/invalidation.
            old = candidate(x, mask)
            snapshot = old.clone()
            candidate(x * .5, mask)
            assert torch.equal(old, snapshot)
            mask[0, 64:] = False
            assert compare_outputs(base(x, mask), candidate(x, mask), rtol=.02, atol=.002).passed
            mask.fill_(True)
            candidate.layers[0].norm1.weight.add_(.001)
            base.layers[0].norm1.weight.add_(.001)
            assert compare_outputs(base(x, mask), candidate(x, mask), rtol=.02, atol=.002).passed
        lin = candidate.layers[0].attention.out_proj
        replacement = torch.nn.Parameter(torch.randn_like(lin.weight) * .1)
        with torch.no_grad():
            while replacement._version < lin.weight._version:
                replacement.add_(0)
        assert replacement._version == lin.weight._version
        lin.weight = replacement
        base.layers[0].attention.out_proj.weight = torch.nn.Parameter(replacement.detach().clone())
        with torch.inference_mode():
            assert compare_outputs(base(x, mask), candidate(x, mask), rtol=.02, atol=.002).passed
            for model in (current, candidate):
                copy_model_weights(base, model, strict=True)
                model(x, mask)
                assert model._capture_graph(x, mask, lambda fn, xx, mm, av: fn(xx, mm, True, av))
            for name, model in (("current", current), ("native_keys", candidate)):
                start = time.perf_counter()
                for _ in range(3000):
                    model._refresh_x3()
                row["host_us"][name] = (time.perf_counter() - start) * 1e6 / 3000
            row["timing"] = next_paired({"current": current, "native_keys": candidate}, x, mask, 100)
            row["contracts"] = "PASS (mutable masks/weights, same-version Parameter replacement, owned output)"
        rows.append(row)
        next_save("next_native_keys", rows)
        del base, current, candidate, x, mask, old, snapshot, xx, model, lin, replacement
        gc.collect()
        torch.cuda.empty_cache()


FLASH_TURING_REVISION = "9ef98fcb506bb1e2fe3cece50935e2935bf6b124"


def next_lt_integrated():
    """Public-forward comparison including default eager/compile/graph tuning."""
    os.environ.pop("T3_COMPILE", None)
    os.environ.pop("T3_CUDAGRAPH", None)
    cfg = TransformerConfig(*NEXT_SHAPES[8], True)
    torch.manual_seed(141017)
    base = BaselineTransformer(cfg).cuda().eval()
    current = UserOptimizedTransformer(cfg).cuda().eval()
    candidate = UserOptimizedTransformer(cfg).cuda().eval()
    for model in (current, candidate):
        copy_model_weights(base, model, strict=True)
    x, mask = generate_random_case(cfg, torch.device("cuda"), torch.float32, 141017, 0., 1.)

    def original(xx, mm):
        globals()["_X3_BLAS"] = "torch"
        return current(xx, mm)

    def tuned(xx, mm):
        globals()["_X3_BLAS"] = "lt"
        return candidate(xx, mm)

    row = {"shape": 8, "max_abs": 0.}
    with torch.inference_mode():
        for trial in range(3):
            xx, mm = generate_random_case(cfg, torch.device("cuda"), torch.float32, 20261009 + trial, 0., 1.)
            ref = base(xx, mm)
            for call in (original, tuned):
                out = call(xx, mm)
                check = compare_outputs(ref, out, rtol=.02, atol=.002)
                assert check.passed, check
                row["max_abs"] = max(row["max_abs"], check.max_abs_error)
            del xx, mm, ref, out
        old = tuned(x, mask)
        snap = old.clone()
        tuned(x * .5, mask)
        assert torch.equal(old, snap)
        del old, snap
        row["timing"] = next_paired({"current": original, "lt_opt_in": tuned}, x, mask, 30)
        row["mode"] = {"current_graph": current._graph is not None, "lt_graph": candidate._graph is not None,
                       "current_compiled": current._compiled is not None, "lt_compiled": candidate._compiled is not None}
        row["algorithm_search"] = _LT_RESULTS
        row["total_peak_bytes"] = torch.cuda.max_memory_allocated()
        row["ownership"] = "PASS"
    next_save("next_lt_integrated", row)
    del base, current, candidate, x, mask
    gc.collect()
    torch.cuda.empty_cache()


def next_flash_build():
    """Fetch a pinned optional dependency; do not redistribute unlicensed code."""
    from pathlib import Path
    if os.environ.get("T3_TURING_PREBUILT") == "1":
        # Reuse only this account's attached private experiment, with its pinned
        # checkout verified. No compiled binary is copied into the public repo.
        for binary in Path("/kaggle/input").rglob("flash_attn_turing*.so"):
            repository = next((parent for parent in binary.parents if (parent / ".git/HEAD").is_file()), None)
            if repository is not None and (repository / ".git/HEAD").read_text().strip() == FLASH_TURING_REVISION:
                sys.path.insert(0, str(binary.parent))
                import flash_attn_turing
                print("TURING_PREBUILT_PIN_VERIFIED " + str(binary), flush=True)
                return flash_attn_turing
        raise RuntimeError("no pinned private prebuilt artifact found; refusing an unverified binary")
    root = Path("/tmp/track3-flash-attention-turing")
    subprocess.run(["git", "clone", "https://github.com/ssiu/flash-attention-turing.git", str(root)], check=True)
    subprocess.run(["git", "checkout", FLASH_TURING_REVISION], cwd=root, check=True)
    subprocess.run(["git", "submodule", "update", "--init", "--depth", "1", "csrc/cutlass"], cwd=root, check=True)
    os.environ["MAX_JOBS"] = "2"
    build = subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
                            "--no-deps", "-v", str(root)], text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    Path("flash_turing_build.log").write_text(build.stdout, encoding="utf-8")
    print(build.stdout[-18000:], flush=True)
    assert build.returncode == 0, "pinned Turing extension failed to build; see flash_turing_build.log"
    import flash_attn_turing
    return flash_attn_turing


def next_flash():
    extension = next_flash_build()
    real_sdpa = F.scaled_dot_product_attention
    counters = {"flash": 0, "fallback": 0}

    def checked_flash(q, k, v, scale, causal=True):
        # Upstream assumes contiguous B,S,H,D and launches on the default stream.
        # Refuse unsupported tensors instead of silently interpreting fp32 as half.
        supported = (q.is_cuda and q.device == k.device == v.device and
                     q.dtype == k.dtype == v.dtype == torch.float16 and
                     q.shape == k.shape == v.shape and q.shape[-1] in (64, 96, 128) and
                     torch.cuda.get_device_capability(q.device) == (7, 5) and
                     torch.cuda.current_stream(q.device) == torch.cuda.default_stream(q.device) and
                     not torch.cuda.is_current_stream_capturing())
        if not supported:
            counters["fallback"] += 1
            return real_sdpa(q, k, v, attn_mask=None, is_causal=causal, scale=scale)
        counters["flash"] += 1
        # A transpose with stride(-1)==1 is still not contiguous: upstream's
        # Python maybe_contiguous is insufficient for the C++ kernel's layout.
        out, _lse = extension.fwd(q.transpose(1, 2).contiguous(),
                                  k.transpose(1, 2).contiguous(),
                                  v.transpose(1, 2).contiguous(), float(scale), causal)
        return out.transpose(1, 2)

    class FlashCandidateTransformer(UserOptimizedTransformer):
        def _attention(self, attn, xx, mm, causal, all_valid):
            if not all_valid:
                return super()._attention(attn, xx, mm, causal, all_valid)
            b, s, d = xx.shape
            if self._fused_qkv and getattr(attn, "_qkv_w", None) is not None:
                q, k, v = F.linear(xx, attn._qkv_w, attn._qkv_b).split(d, dim=-1)
            else:
                q, k, v = attn.q_proj(xx), attn.k_proj(xx), attn.v_proj(xx)
            q, k, v = [t.view(b, s, attn.num_heads, attn.head_dim).transpose(1, 2) for t in (q, k, v)]
            out = checked_flash(q, k, v, attn.scale, causal)
            return attn.out_proj(out.transpose(1, 2).contiguous().view(b, s, d))

    use_adapter = os.environ.get("T3_FLASH_USE_ADAPTER") == "1"
    rows = {"external_revision": FLASH_TURING_REVISION,
            "implementation": "production_adapter" if use_adapter else "isolated_candidate",
            "external_repository": "https://github.com/ssiu/flash-attention-turing",
            "license_note": "no top-level LICENSE at pinned revision; fetched only for private experiment, not vendored",
            "attention": [], "full": {}}
    if use_adapter:
        adapter_rows = []
        with torch.inference_mode():
            for hd in (64, 96, 128):
                for seed in range(3):
                    torch.manual_seed(141009 + hd + seed)
                    packed = torch.randn(1, 8192, 3, 16, hd, device="cuda", dtype=torch.float16)
                    q, k, v = [packed[:, :, part].transpose(1, 2) for part in range(3)]
                    before = _TURING_CALLS
                    out = turing_attention(q, k, v, hd ** -.5, True)
                    ref = real_sdpa(q, k, v, is_causal=True, scale=hd ** -.5)
                    check = compare_outputs(ref, out, rtol=.02, atol=.002)
                    assert check.passed and _TURING_CALLS == before + 1, check
                    adapter_rows.append({"head_dim": hd, "seed": seed, "max_abs": check.max_abs_error,
                                         "failed": check.failed_elements})
                    del packed, q, k, v, out, ref
            # Non-causal requests are intentionally outside the adapter's scope.
            q = torch.randn(1, 4, 8192, 64, device="cuda", dtype=torch.float16)
            before = _TURING_CALLS
            out = turing_attention(q, q, q, .125, False)
            assert _TURING_CALLS == before
            assert compare_outputs(real_sdpa(q, q, q, is_causal=False, scale=.125),
                                   out, rtol=.02, atol=.002).passed
            del q, out
        rows["production_adapter_contracts"] = adapter_rows
        next_save("next_flash", rows)
    with torch.inference_mode():
        cases = ((128, 64), (1024, 64), (8192, 64), (100000, 64),
                 (128, 96), (128, 128), (128, 32), (128, 256))
        for seq, hd in (() if os.environ.get("T3_FLASH_SKIP_ATTN") == "1" else cases):
            torch.manual_seed(141009 + seq + hd)
            # Q/K/V views from interleaved storage reproduce a packed QKV projection.
            packed = torch.randn(1, seq, 3, 16, hd, device="cuda", dtype=torch.float16)
            q, k, v = [packed[:, :, part].transpose(1, 2) for part in range(3)]
            scale = hd ** -.5
            ref = real_sdpa(q, k, v, is_causal=True, scale=scale)
            out = checked_flash(q, k, v, scale)
            check = compare_outputs(ref, out, rtol=.02, atol=.002)
            item = {"seq": seq, "head_dim": hd, "max_abs": check.max_abs_error,
                    "failed": check.failed_elements, "passed": check.passed}
            assert check.passed, item
            if seq <= 1024:
                # Independent fp32 math attention oracle, not merely backend agreement.
                oracle = (q.float() @ k.float().transpose(-1, -2)) * scale
                causal_mask = torch.ones(seq, seq, device="cuda", dtype=torch.bool).triu(1)
                oracle.masked_fill_(causal_mask, float("-inf"))
                oracle = torch.softmax(oracle, -1) @ v.float()
                check32 = compare_outputs(oracle, out, rtol=.02, atol=.002)
                assert check32.passed, check32
                item["fp32_oracle_max_abs"] = check32.max_abs_error
                del oracle, causal_mask
            item["timing"] = next_paired({
                "sdpa": lambda xx, mm: real_sdpa(q, k, v, is_causal=True, scale=scale),
                "flash_guarded_copies_included": lambda xx, mm: checked_flash(q, k, v, scale)},
                q, None, 3 if seq == 100000 else 20)
            rows["attention"].append(item)
            next_save("next_flash", rows)
            del packed, q, k, v, ref, out
            gc.collect()
            torch.cuda.empty_cache()

        # Stream and dtype rejection are correctness requirements, not tunables.
        q = torch.randn(1, 4, 128, 64, device="cuda")
        before = counters["fallback"]
        checked_flash(q, q, q, 1 / 8.)
        half = q.half()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            out = checked_flash(half, half, half, 1 / 8.)
            ref = real_sdpa(half, half, half, is_causal=True, scale=1 / 8.)
            assert compare_outputs(ref, out, rtol=.02, atol=.002).passed
        torch.cuda.current_stream().wait_stream(side)
        assert counters["fallback"] == before + 2
        rows["contracts"] = "PASS (strided QKV, unsupported heads, fp32, non-default stream)"
        del q, half, out, ref

        # Keep both candidates eager: this isolates the attention backend and
        # avoids unsupported default-stream launches inside CUDA graph capture.
        os.environ.update(T3_COMPILE="0", T3_CUDAGRAPH="0", T3_CHUNK_BS="1")
        cfg = TransformerConfig(32, 100000, 1024, 16, 1024, 2, True)
        torch.manual_seed(141009)
        base = BaselineTransformer(cfg).eval()
        os.environ["T3_ATTN"] = "sdpa"
        current = UserOptimizedTransformer(cfg).eval()
        if use_adapter:
            os.environ["T3_ATTN"] = "turing"
            candidate = UserOptimizedTransformer(cfg).eval()
        else:
            candidate = FlashCandidateTransformer(cfg).eval()
        os.environ["T3_ATTN"] = "sdpa"
        copy_model_weights(base, current, strict=True)
        copy_model_weights(base, candidate, strict=True)
        current.cuda().half()
        candidate.cuda().half()
        x, mask = generate_random_case(cfg, torch.device("cuda"), torch.float16, 141009, 0., 1.)
        # Full-sized first calls are logged separately and excluded from rounds.
        full = {name: {"warmup_seconds": None, "round_seconds": [], "peak_bytes": []}
                for name in ("sdpa", "flash")}
        saved_reference = None
        rounds = int(os.environ.get("T3_FLASH_ROUNDS", "3"))
        for turn in range(1 + rounds):
            order = (("sdpa", current), ("flash", candidate))
            if turn % 2:
                order = tuple(reversed(order))
            for name, model in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                out = model(x, mask)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                finite = all(bool(torch.isfinite(out[i:i + 1]).all()) for i in range(32))
                assert finite and out.shape == x.shape and out.dtype == x.dtype
                if turn == 0:
                    full[name]["warmup_seconds"] = elapsed
                else:
                    full[name]["round_seconds"].append(elapsed)
                    full[name]["peak_bytes"].append(torch.cuda.max_memory_allocated())
                print("FLASH_FULL_PROGRESS " + json.dumps({"round": turn, "model": name, "seconds": elapsed}), flush=True)
                if turn == 0 and name == "sdpa":
                    saved_reference = out.cpu()
                if turn == 0 and name == "flash":
                    failed, maximum = 0, 0.
                    # CPU tiles bound comparison memory well below the 6.55 GB
                    # output. Full output equivalence is against native-fp16 SDPA.
                    for batch in range(32):
                        for token in range(0, 100000, 8192):
                            check = compare_outputs(saved_reference[batch, token:token + 8192],
                                                    out[batch, token:token + 8192].cpu(), rtol=.02, atol=.002)
                            failed += check.failed_elements
                            maximum = max(maximum, check.max_abs_error)
                    assert failed == 0, ("full native-fp16 backend equivalence", failed, maximum)
                    rows["full_equivalence"] = {"failed": failed, "max_abs": maximum,
                                                "elements": x.numel(), "reference": "native-fp16 SDPA model, not original full fp32"}
                    saved_reference = None
                    prefix = x[:1, :512].float().contiguous()
                    base.cuda()
                    ref = base(prefix, None)
                    check = compare_outputs(ref, out[:1, :512], rtol=.02, atol=.002)
                    assert check.passed, check
                    rows["fp32_causal_prefix"] = {"tokens": 512, "batches": 1,
                                                  "max_abs": check.max_abs_error, "failed": check.failed_elements}
                    base.cpu()
                    del prefix, ref
                del out
                rows["full"] = full
                rows["counters"] = counters.copy()
                if use_adapter:
                    rows["production_adapter_calls"] = _TURING_CALLS
                next_save("next_flash", rows)
                gc.collect()
                torch.cuda.empty_cache()
        if rounds:
            for value in full.values():
                value["median_seconds"] = statistics.median(value["round_seconds"])
            rows["full"]["speedup"] = full["sdpa"]["median_seconds"] / full["flash"]["median_seconds"]
        else:
            rows["verification_only"] = "cold pair only, not a steady-state performance claim"
        if use_adapter:
            assert rows["production_adapter_calls"] >= 64, "production adapter did not actually execute"
        next_save("next_flash", rows)


def next_main():
    assert torch.cuda.is_available(), "GPU required; CPU timings are not evidence"
    phase = os.environ.get("T3_NEXT_PHASE", "profile_cpp")
    print("NEXT_START " + json.dumps(next_metadata()), flush=True)
    if phase == "profile_cpp":
        next_profile()
        next_cpp()
    elif phase == "cpp":
        next_cpp()
    elif phase == "gemm":
        next_gemm()
    elif phase == "flash":
        next_flash()
    elif phase == "native_keys":
        next_native_keys()
    elif phase == "followup":
        next_cpp()
        next_native_keys()
        next_gemm()
    elif phase == "regression":
        next_regression()
    elif phase == "integrated":
        next_lt_integrated()
        next_native_keys()
        globals()["_X3_BLAS"] = "lt"
        next_regression()
    elif phase == "final":
        next_lt_integrated()
        globals()["_X3_BLAS"] = "lt"
        os.environ["T3_ATTN"] = "turing"
        next_regression()
    else:
        raise ValueError("unknown next phase: " + phase)

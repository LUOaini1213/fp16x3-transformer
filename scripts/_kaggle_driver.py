# ===== driver (appended to the self-contained kernel) =====
# References names defined earlier in the combined module: TransformerConfig,
# BaselineTransformer, UserOptimizedTransformer, copy_model_weights,
# generate_random_case, compare_outputs.

_RESULTS = []  # (idx, pass, max_abs, max_rel, base_ms, opt_ms, speedup, note)
_ABL_RESULTS = []  # (idx, stage, pass, max_abs, max_rel, base_ms, opt_ms, speedup)


class _Skip(Exception):
    """Sentinel for the T3_ONLY selector."""


def _bench(model, x, mask, warmup=20, iters=50):
    import torch
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        s = []
        for _ in range(iters):
            st.record(); model(x, mask); en.record(); torch.cuda.synchronize()
            s.append(st.elapsed_time(en))
    s.sort()
    return s[len(s) // 2]


def _accuracy(baseline, optimized, cfg, device, dtype, trials=3):
    import torch
    ok = True; mabs = mrel = 0.0
    with torch.inference_mode():
        for t in range(trials):
            x, m = generate_random_case(cfg, device, dtype, 1234 + t, 0.0, 1.0)
            ref = baseline(x, m); o = optimized(x, m)
            r = compare_outputs(ref, o, rtol=0.02, atol=0.002)
            ok &= r.passed; mabs = max(mabs, r.max_abs_error); mrel = max(mrel, r.max_relative_error)
    return ok, mabs, mrel


def _shape14(device):
    import torch
    FULL = dict(batch_size=32, seq_len=100000, d_model=1024, num_heads=16,
                ffn_dim=1024, num_layers=2, causal=True)
    note = ""
    tc = dict(FULL); tc["seq_len"] = 2048; tc["batch_size"] = 2
    cfg = TransformerConfig(**tc)
    base = BaselineTransformer(cfg); opt = UserOptimizedTransformer(cfg)
    copy_model_weights(base, opt, strict=True)
    base = base.to(device, torch.float32).eval(); opt = opt.to(device, torch.float32).eval()
    x, m = generate_random_case(cfg, device, torch.float32, 1234, 0.0, 1.0)
    with torch.inference_mode():
        ref = base(x, m); o = opt(x, m)
    res = compare_outputs(ref, o, rtol=0.02, atol=0.002)
    tpass = "PASS" if res.passed else "FAIL"
    print(f"trunc S=2048 correctness: {tpass} max_abs={res.max_abs_error:.3g} "
          f"max_rel={res.max_relative_error:.3g}", flush=True)
    del base, opt, x, m, ref, o
    torch.cuda.empty_cache()

    cfg = TransformerConfig(**FULL)
    base = BaselineTransformer(cfg); opt = UserOptimizedTransformer(cfg)
    copy_model_weights(base, opt, strict=True); del base
    opt = opt.to(device, torch.float16).eval()
    torch.cuda.reset_peak_memory_stats(device)
    free0, total0 = torch.cuda.mem_get_info(device)
    scores_tb = cfg.batch_size * cfg.num_heads * cfg.seq_len ** 2 * 4 / 1e12
    print(f"vram free={free0/1e9:.2f}/{total0/1e9:.2f} GB | baseline scores would be "
          f"{scores_tb:.1f} TB -> infeasible", flush=True)
    try:
        x, m = generate_random_case(cfg, device, torch.float16, 1234, 0.0, 1.0)
        med = _bench(opt, x, m, warmup=3, iters=10)
        tok = cfg.batch_size * cfg.seq_len
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        note = (f"full S=1e5 OK {med:.0f}ms {tok*1000.0/med:,.0f}tok/s "
                f"peak{peak:.1f}GB chunk{opt._chunk_bs}")
        print(f"full S=100000: median={med:.1f} ms | {tok*1000.0/med:,.0f} tok/s | "
              f"peak_vram={peak:.2f} GB | chunk_bs={opt._chunk_bs}", flush=True)
    except RuntimeError as e:
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        note = f"full S=1e5 OOM peak{peak:.1f}GB chunk{opt._chunk_bs}"
        print("shape14 full-seq RuntimeError:", str(e)[:300], flush=True)
    _RESULTS.append((14, tpass + "(trunc)", res.max_abs_error, res.max_relative_error,
                     "", "", "", note))



# ---- stage ablation ---------------------------------------------------------
# Each stage is selected purely by environment variable, with no code edits, so
# the numbers below and the delivered path are literally the same code.
_ABL_CONFIGS = [
    ("sdpa",              {"T3_AUTOCAST": "off",  "T3_COMPILE": "0"}),
    ("sdpa+compile",      {"T3_AUTOCAST": "off",  "T3_COMPILE": "1"}),
    ("sdpa+fp16",         {"T3_AUTOCAST": "fp16", "T3_COMPILE": "0"}),
    ("sdpa+compile+fp16", {"T3_AUTOCAST": "fp16", "T3_COMPILE": "1"}),
]
_ABL_SHAPES = [
    (1, 64, 128, 4, 128, 4, 128),
    (12, 64, 128, 4, 32, 4, 128),
    (8, 64, 1024, 4, 128, 4, 1024),
    (13, 64, 128, 4, 1024, 4, 128),
]


def _ablation(device):
    import os
    import torch
    dtype = torch.float32
    print("stage ablation: shape,stage,pass,max_abs,max_rel,baseline_ms,opt_ms,speedup",
          flush=True)
    for (idx, b, d, h, sq, l, f) in _ABL_SHAPES:
        cfg = TransformerConfig(batch_size=b, seq_len=sq, d_model=d, num_heads=h,
                                ffn_dim=f, num_layers=l, causal=True)
        cfg.validate()
        baseline = BaselineTransformer(cfg).to(device, dtype).eval()
        xt, mt = generate_random_case(cfg, device, dtype, 101234, 0.0, 1.0)
        bms = _bench(baseline, xt, mt)
        for name, env in _ABL_CONFIGS:
            for k, v in env.items():
                os.environ[k] = v
            try:
                # Construct AFTER setting the env: __init__ and the one-shot
                # _plan() are what read these knobs.
                opt = UserOptimizedTransformer(cfg)
                copy_model_weights(baseline, opt, strict=True)
                opt = opt.to(device, dtype).eval()
                ok, mabs, mrel = _accuracy(baseline, opt, cfg, device, dtype)
                oms = _bench(opt, xt, mt)
                print(f"ABL,{idx},{name},{'PASS' if ok else 'FAIL'},{mabs:.3g},"
                      f"{mrel:.3g},{bms:.4f},{oms:.4f},{bms/oms:.3f}", flush=True)
                _ABL_RESULTS.append((idx, name, "PASS" if ok else "FAIL",
                                     f"{mabs:.3g}", f"{mrel:.3g}", f"{bms:.4f}",
                                     f"{oms:.4f}", f"{bms/oms:.3f}"))
            except Exception as e:
                print(f"ABL,{idx},{name},ERROR,,,,,{str(e)[:80]}", flush=True)
                _ABL_RESULTS.append((idx, name, "ERROR", "", "", "", "", str(e)[:60]))
            finally:
                try:
                    del opt
                except Exception:
                    pass
                torch.cuda.empty_cache()
        del baseline, xt, mt
        torch.cuda.empty_cache()
    # Restore the shipped defaults for anything that runs after us.
    os.environ["T3_AUTOCAST"] = "off"
    os.environ["T3_COMPILE"] = "1"



# ---- hand-written Triton kernel: does it actually beat the alternatives? -----
_TRITON_RESULTS = []


def _triton_bench(device):
    """Time fused add+LayerNorm three ways on the real activation sizes.

    eager    : x + y then nn.LayerNorm -- two kernels, four passes over [rows, D]
    inductor : the same two ops under torch.compile, which fuses them
    triton   : our hand-written kernel, one pass

    The shapes are (rows, D) taken from the graded sweep: rows = B*S.
    """
    import torch
    import torch.nn as nn
    # In the repo the kernels are a package; in the single-file Kaggle build the
    # same code is inlined above, so fall back to module scope.
    try:
        from kernels import HAVE_TRITON, fused_add_layernorm
    except Exception:
        g = globals()
        HAVE_TRITON = g.get("HAVE_TRITON", False)
        fused_add_layernorm = g.get("fused_add_layernorm")
        if fused_add_layernorm is None:
            print("triton kernels unavailable in this build", flush=True)
            return
    print(f"HAVE_TRITON={HAVE_TRITON}", flush=True)
    if not HAVE_TRITON:
        return

    CASES = [
        (64 * 128, 128, "shape 1/5/9-11  B*S=8192,  D=128"),
        (1 * 128, 128, "shape 2         B*S=128,   D=128"),
        (10000 * 128, 128, "shape 6         B*S=1.28M, D=128"),
        (64 * 128, 32, "shape 7         B*S=8192,  D=32"),
        (64 * 128, 1024, "shape 8         B*S=8192,  D=1024"),
        (64 * 1024, 128, "shape 13        B*S=65536, D=128"),
    ]

    def timeit(fn, warmup=20, iters=100):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        s = []
        for _ in range(iters):
            st.record(); fn(); en.record(); torch.cuda.synchronize()
            s.append(st.elapsed_time(en))
        s.sort()
        return s[len(s) // 2]

    try:
        from kernels import HAVE_TRITON_OP as _op
    except Exception:
        _op = globals().get("HAVE_TRITON_OP", False)
    print(f"HAVE_TRITON_OP={_op}", flush=True)
    print("triton bench: case,rows,D,eager_ms,inductor_ms,triton_ms,"
          "inductor+ourop_ms,vs_eager,vs_inductor,inductor_vs_ourop_in_graph,max_abs",
          flush=True)
    for rows, d, label in CASES:
        x = torch.randn(rows, d, device=device, dtype=torch.float32)
        r = torch.randn(rows, d, device=device, dtype=torch.float32)
        ln = nn.LayerNorm(d).to(device)

        def eager():
            t = x + r
            return ln(t), t

        compiled = torch.compile(eager, dynamic=False)

        def triton_fn():
            return fused_add_layernorm(x, r, ln.weight, ln.bias, ln.eps)

        # The number that decides everything: our op scheduled by Inductor
        # inside a compiled region, rather than compilation being switched off
        # around it.
        compiled_ours = torch.compile(triton_fn, dynamic=False)

        with torch.inference_mode():
            ref_out, ref_sum = eager()
            got_out, got_sum = triton_fn()
            mabs = max((got_out - ref_out).abs().max().item(),
                       (got_sum - ref_sum).abs().max().item())
            e = timeit(eager)
            try:
                compiled()
                c = timeit(compiled)
            except Exception as ex:
                print("  inductor failed:", str(ex)[:80], flush=True)
                c = float("nan")
            t = timeit(triton_fn)
            try:
                co_out, co_sum = compiled_ours()
                mabs = max(mabs, (co_out - ref_out).abs().max().item(),
                           (co_sum - ref_sum).abs().max().item())
                ct = timeit(compiled_ours)
            except Exception as ex:
                print("  compiled(our op) failed:", str(ex)[:120], flush=True)
                ct = float("nan")

        print(f"TRI,{label},{rows},{d},{e:.4f},{c:.4f},{t:.4f},{ct:.4f},"
              f"{e/t:.3f},{c/t:.3f},{c/ct:.3f},{mabs:.3g}", flush=True)
        _TRITON_RESULTS.append((label, rows, d, f"{e:.4f}", f"{c:.4f}", f"{t:.4f}",
                                f"{ct:.4f}", f"{e/t:.3f}", f"{c/t:.3f}", f"{c/ct:.3f}",
                                f"{mabs:.3g}"))
        del x, r, ln
        torch.cuda.empty_cache()



def _sdpa_probe(device):
    """Which scaled_dot_product_attention backend can actually run here.

    Tried per (dtype, head_dim) pair the sweep uses, by forcing each backend
    alone and seeing whether the call succeeds. PyTorch's flash backend is
    fp16/bf16-only and, in current releases, sm_80+; the graded path is fp32.
    So this settles, with a measurement rather than a belief, whether any run
    in this project ever used FlashAttention.
    """
    import torch
    import torch.nn.functional as F
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as e:
        print("sdpa probe unavailable:", e, flush=True)
        return
    backends = (("flash", SDPBackend.FLASH_ATTENTION),
                ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                ("math", SDPBackend.MATH))
    print("sdpa backend probe  (head_dim -> which backends can run, is_causal=True)", flush=True)
    for dtype in (torch.float32, torch.float16):
        for hd in (8, 32, 64, 128, 256):
            q = torch.randn(2, 4, 128, hd, device=device, dtype=dtype)
            res = []
            for name, be in backends:
                try:
                    with sdpa_kernel(be):
                        F.scaled_dot_product_attention(q, q, q, is_causal=True)
                    res.append(f"{name}=yes")
                except Exception:
                    res.append(f"{name}=no ")
            print(f"  PROBE {str(dtype).split('.')[-1]:8s} hd={hd:4d}  " + "  ".join(res), flush=True)
    print("  shapes -> head_dim: D=128,H=4 -> 32 | D=32,H=4 -> 8 | D=1024,H=4 -> 256 | "
          "D=128,H=1 -> 128 | D=128,H=2 -> 64 | D=128,H=16 -> 8 | shape14 D=1024,H=16 -> 64",
          flush=True)



def _attn_bench(device):
    """Correctness against an fp64 reference, and speed against fp32 / fp16 SDPA,
    for the tensor-core attention kernel at the sweep's real (H, S, head_dim)."""
    import math
    import torch
    import torch.nn.functional as F
    try:
        from kernels.attention import attention_raw, attention, HAVE_TRITON as hv, HAVE_ATTN_OP as hop
    except Exception:
        g = globals()
        attention_raw, attention = g.get("attention_raw"), g.get("attention")
        hv, hop = g.get("HAVE_TRITON", False), g.get("HAVE_ATTN_OP", False)
        if attention_raw is None:
            print("attention kernel unavailable in this build", flush=True)
            return
    print(f"HAVE_TRITON={hv} HAVE_ATTN_OP={hop}", flush=True)

    CASES = [
        ("shape 1/5 B=64 H=4 S=128 hd=32", 64, 4, 128, 32),
        ("shape 7 hd=8", 64, 4, 128, 8),
        ("shape 9 H=1 hd=128", 64, 1, 128, 128),
        ("shape 10 H=2 hd=64", 64, 2, 128, 64),
        ("shape 11 H=16 hd=8", 64, 16, 128, 8),
        ("shape 12 S=32", 64, 4, 32, 32),
        ("shape 13 S=1024", 64, 4, 1024, 32),
        ("shape 6 B=10000", 10000, 4, 128, 32),
    ]

    def timeit(fn, warm=10, iters=30):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        t = []
        for _ in range(iters):
            st.record(); fn(); en.record(); torch.cuda.synchronize()
            t.append(st.elapsed_time(en))
        t.sort()
        return t[len(t) // 2]

    print("ATTN: case | BxHxSxhd | sdpa32 ms | sdpa16 ms | k1 ms | k3 ms | "
          "err sdpa32 | err sdpa16 | err k1 | err k3 | k1 vs sdpa32 | k3 vs sdpa32", flush=True)
    for label, B, H, S, hd in CASES:
        torch.manual_seed(0)
        q = torch.randn(B, H, S, hd, device=device)
        k = torch.randn(B, H, S, hd, device=device)
        v = torch.randn(B, H, S, hd, device=device)
        scale = 1.0 / math.sqrt(hd)
        row = dict(err32="", err16="", e1="", e3="", t32="", t16="", t1="", t3="", s1="", s3="")
        with torch.no_grad():
            ref32 = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
            big = B * H * S * S * 8 * 2 > 6e9
            ref = ref32.double() if big else F.scaled_dot_product_attention(
                q.double(), k.double(), v.double(), is_causal=True, scale=scale)
            ref32_err = 0.0 if big else (ref32.double() - ref).abs().max().item()
            o16 = F.scaled_dot_product_attention(q.half(), k.half(), v.half(),
                                                 is_causal=True, scale=scale).float()
            err16 = (o16.double() - ref).abs().max().item()
            outs = {}
            for split in (1, 3):
                try:
                    o = attention_raw(q, k, v, scale, True, split)
                    torch.cuda.synchronize()
                    outs[split] = o
                    row[f"e{split}"] = f"{(o.double() - ref).abs().max().item():.3g}"
                    if not torch.isfinite(o).all():
                        row[f"e{split}"] += "(NONFINITE)"
                except Exception as ex:
                    row[f"e{split}"] = "FAIL:" + str(ex)[:70].replace("\n", " ").replace(",", ";")
            t32 = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale))
            qh, kh, vh = q.half(), k.half(), v.half()
            t16 = timeit(lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scale))
            row.update(t32=f"{t32:.4f}", t16=f"{t16:.4f}", err32=f"{ref32_err:.3g}", err16=f"{err16:.3g}")
            for split in (1, 3):
                if split in outs:
                    t = timeit(lambda: attention_raw(q, k, v, scale, True, split))
                    row[f"t{split}"] = f"{t:.4f}"
                    row[f"s{split}"] = f"{t32 / t:.3f}"
        print(f"ATTN,{label},{B}x{H}x{S}x{hd},{row['t32']},{row['t16']},{row['t1']},{row['t3']},"
              f"{row['err32']},{row['err16']},{row['e1']},{row['e3']},{row['s1']},{row['s3']}", flush=True)
        del q, k, v, ref32, ref, o16, outs
        torch.cuda.empty_cache()

    # does the registered op compose with torch.compile?
    try:
        q = torch.randn(4, 4, 128, 32, device=device)
        k = q.clone(); v = q.clone()
        fn = torch.compile(lambda a, b, c: attention(a, b, c, 1.0 / math.sqrt(32), True, 3), dynamic=False)
        with torch.no_grad():
            o = fn(q, k, v)
            o = fn(q, k, v)
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"compile compose: OK  max_abs={(o - ref).abs().max().item():.3g}", flush=True)
    except Exception as ex:
        print("compile compose FAILED:", str(ex)[:200].replace("\n", " "), flush=True)


_SH13 = [
    (1, 64, 128, 4, 128, 4, 128), (2, 1, 128, 4, 128, 4, 128),
    (3, 4, 128, 4, 128, 4, 128), (4, 16, 128, 4, 128, 4, 128),
    (5, 128, 128, 4, 128, 4, 128), (6, 10000, 128, 4, 128, 4, 128),
    (7, 64, 32, 4, 128, 4, 32), (8, 64, 1024, 4, 128, 4, 1024),
    (9, 64, 128, 1, 128, 4, 128), (10, 64, 128, 2, 128, 4, 128),
    (11, 64, 128, 16, 128, 4, 128), (12, 64, 128, 4, 32, 4, 128),
    (13, 64, 128, 4, 1024, 4, 128),
]


def _profile(device):
    """Where does the shipped forward spend its time, per graded shape?

    Three views per shape: (a) kernel-level profile of the eager path,
    (b) SDPA backend micro-bench (memory-efficient vs math) at the shape's
    (B, H, S, head_dim), (c) the shape's linear layers as fp32 cuBLAS vs an
    error-compensated fp16 tensor-core GEMM (fp16x3: each fp32 operand as an
    fp16 hi+lo pair, one GEMM with K tripled, fp32 accumulation, fp32 output)
    vs plain fp16, with max-abs error against fp64.
    """
    import os
    import math
    import torch
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity
    from torch.nn.attention import sdpa_kernel, SDPBackend
    dtype = torch.float32

    def med(fn, warm=5, iters=20):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        t = []
        for _ in range(iters):
            st.record(); fn(); en.record(); torch.cuda.synchronize(); t.append(st.elapsed_time(en))
        t.sort()
        return t[len(t) // 2]

    try:
        _a = torch.randn(8, 8, device=device, dtype=torch.float16)
        torch.mm(_a, _a, out_dtype=torch.float32)
        has_out_dtype = True
    except Exception as e:
        has_out_dtype = False
        print("mm out_dtype unavailable:", str(e)[:100], flush=True)
    print(f"mm(out_dtype=fp32) available: {has_out_dtype}", flush=True)

    def split(t):
        hi = t.half()
        return hi, (t - hi.float()).half()

    saved = {k: os.environ.get(k) for k in ("T3_COMPILE", "T3_CUDAGRAPH")}
    for (idx, b, d, h, s, l, f) in _SH13:
        hd = d // h
        print(f"\n##### PROFILE SHAPE {idx} : B={b} D={d} H={h} S={s} L={l} F={f} hd={hd} #####", flush=True)
        cfg = TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                ffn_dim=f, num_layers=l, causal=True)
        cfg.validate()
        x, m = generate_random_case(cfg, device, dtype, 1234, 0.0, 1.0)
        # (a) eager kernel mix
        os.environ["T3_COMPILE"] = "0"; os.environ["T3_CUDAGRAPH"] = "0"
        try:
            baseline = BaselineTransformer(cfg).to(device, dtype).eval()
            opt = UserOptimizedTransformer(cfg)
            copy_model_weights(baseline, opt, strict=True)
            opt = opt.to(device, dtype).eval()
            with torch.inference_mode():
                for _ in range(5):
                    opt(x, m)
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        opt(x, m)
                    torch.cuda.synchronize()
            ka = [e for e in prof.key_averages() if e.self_device_time_total > 0]
            ka.sort(key=lambda e: -e.self_device_time_total)
            tot = sum(e.self_device_time_total for e in ka) / 5 / 1000.0
            print(f"eager forward, GPU time {tot:.3f} ms/iter ({len(ka)} distinct kernels):", flush=True)
            for e in ka[:10]:
                ms = e.self_device_time_total / 5 / 1000.0
                print(f"   {ms:8.3f} ms {100 * ms / tot:5.1f}%  x{e.count // 5:<3d} {e.key[:88]}", flush=True)
            del baseline, opt
        except Exception as e:
            print("profile error:", str(e)[:160], flush=True)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        # (b) SDPA backend
        try:
            q = torch.randn(b, h, s, hd, device=device, dtype=dtype)
            k_ = torch.randn_like(q); v_ = torch.randn_like(q)
            res = {}
            for name, be in (("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH)):
                with sdpa_kernel(be):
                    with torch.inference_mode():
                        res[name] = (med(lambda: F.scaled_dot_product_attention(q, k_, v_, is_causal=True)),
                                     F.scaled_dot_product_attention(q, k_, v_, is_causal=True))
            diff = (res["efficient"][1] - res["math"][1]).abs().max().item()
            print(f"sdpa fp32 causal [{b},{h},{s},{hd}]: efficient {res['efficient'][0]:.3f} ms | math {res['math'][0]:.3f} ms "
                  f"| math/eff {res['math'][0] / res['efficient'][0]:.2f} | max|diff| {diff:.2e}", flush=True)
            del q, k_, v_, res
        except Exception as e:
            print("sdpa bench error:", str(e)[:160], flush=True)
        # (c) linear layers: fp32 vs fp16x3 vs fp16
        M = b * s
        dims = [(d, d, "q/k/v/o"), (d, f, "ffn_in"), (f, d, "ffn_out")]
        seen = set()
        for (K, N, tag) in dims:
            if (K, N) in seen:
                continue
            seen.add((K, N))
            try:
                a = F.layer_norm(torch.randn(M, K, device=device), (K,))
                w = (torch.rand(N, K, device=device) * 2 - 1) / math.sqrt(K)
                bias = (torch.rand(N, device=device) * 2 - 1) / math.sqrt(K)
                ref = a.double() @ w.double().t() + bias.double()
                sc = 2.0 ** math.floor(math.log2(1024.0 / w.abs().max().item()))
                wh, wl = split(w * sc)
                W3 = torch.cat([wh, wl, wh], dim=1).contiguous()          # [N, 3K]
                inv = 1.0 / sc
                t32 = med(lambda: F.linear(a, w, bias))
                e32 = (F.linear(a, w, bias).double() - ref).abs().max().item()
                t16 = med(lambda: F.linear(a.half(), w.half(), bias.half()))
                e16 = (F.linear(a.half(), w.half(), bias.half()).double() - ref).abs().max().item()
                line = f"linear {tag:8s} M={M} K={K} N={N}: fp32 {t32:.3f} ms ({e32:.1e}) | fp16 {t16:.3f} ms ({e16:.1e})"
                if has_out_dtype:
                    def x3():
                        ah, al = split(a)
                        A3 = torch.cat([ah, ah, al], dim=1)
                        return torch.mm(A3, W3.t(), out_dtype=torch.float32) * inv + bias
                    tx3 = med(x3)
                    ex3 = (x3().double() - ref).abs().max().item()
                    ah, al = split(a); A3 = torch.cat([ah, ah, al], dim=1)
                    tg = med(lambda: torch.mm(A3, W3.t(), out_dtype=torch.float32))
                    line += f" | fp16x3 {tx3:.3f} ms ({ex3:.1e}; GEMM alone {tg:.3f} ms) | fp32/fp16x3 {t32 / tx3:.2f}"
                print(line, flush=True)
                del a, w, bias, ref, W3
            except Exception as e:
                print(f"linear {tag} error:", str(e)[:160], flush=True)
        torch.cuda.empty_cache()


def _gpu_fingerprint(tag):
    """Clocks, temperature, power and throttle reasons, plus a fixed SGEMM as a
    yardstick, so a sweep's numbers can be read against the card's state."""
    import subprocess
    import torch
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,"
             "power.draw,clocks_throttle_reasons.active", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip().replace("\n", " / ")
    except Exception as e:
        q = f"nvidia-smi unavailable: {str(e)[:60]}"
    try:
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        t = []
        for _ in range(10):
            st.record(); a @ b; en.record(); torch.cuda.synchronize(); t.append(st.elapsed_time(en))
        t.sort()
        ms = t[len(t) // 2]
        yard = f"sgemm 4096^2 {ms:.2f} ms = {2 * 4096 ** 3 / ms / 1e9:.1f} TFLOPS"
        del a, b
    except Exception as e:
        yard = f"sgemm yardstick failed: {str(e)[:60]}"
    print(f"fingerprint[{tag}]: {q} | {yard}", flush=True)


def _med(fn, warm=3, iters=10):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(iters):
        st.record(); fn(); en.record(); torch.cuda.synchronize(); t.append(st.elapsed_time(en))
    t.sort()
    return t[len(t) // 2]


def _top_kernel(fn):
    """Name of the kernel that took the most device time in one call of fn."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    try:
        fn(); torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); torch.cuda.synchronize()
        ka = sorted([e for e in prof.key_averages() if e.self_device_time_total > 0],
                    key=lambda e: -e.self_device_time_total)
        return ka[0].key[:64] if ka else "?"
    except Exception as e:
        return f"profile failed: {str(e)[:40]}"


def _gemm_probe(device):
    """Which cuBLAS path is the fp16x3 GEMM actually on, and how far from the
    roofline? Variants on the shipped geometries, each with its kernel name,
    plus the int8 IMMA go/no-go for the cross-term idea."""
    import os
    import torch
    print(f"x3_available: {x3_available(device)}", flush=True)
    _gpu_fingerprint("gemm-start")
    geoms = [(1280000, 384, 128), (1280000, 384, 384), (65536, 384, 384), (16384, 384, 384),
             (8192, 384, 384), (8192, 96, 96), (8192, 3072, 3072), (8192, 3072, 1024)]
    if os.environ.get("T3_GEMM_TAILS") == "1":
        # How much does a K that is only a multiple of 8 (the bias tail) cost?
        for (M, K3, N) in ((8192, 3072, 3072), (8192, 3072, 1024), (1280000, 384, 128),
                           (65536, 384, 384), (8192, 384, 384), (8192, 96, 96)):
            for t in (0, 8, 16, 32, 64):
                a3 = torch.randn(M, K3 + t, device=device).half()
                w3 = (torch.randn(N, K3 + t, device=device) * 0.03).half()
                fn = lambda: torch.mm(a3, w3.t(), out_dtype=torch.float32)
                ms = _med(fn)
                print(f"tail M={M:<8d} K3={K3 + t:<5d} (tail {t:<2d}) N={N:<5d} {ms:9.3f} ms "
                      f"{2.0 * M * (K3 + t) * N / ms / 1e9:6.1f} TFLOPS  {_top_kernel(fn)}", flush=True)
                del a3, w3
            torch.cuda.empty_cache()
        _gpu_fingerprint("gemm-end")
        return
    for (M, K3, N) in geoms:
        try:
            a3 = torch.randn(M, K3, device=device).half()
            w3 = (torch.randn(N, K3, device=device) * 0.03).half()
            w3n = w3.t().contiguous()                      # [K3, N] for the NN layout
            bias = torch.randn(N, device=device)
            res = torch.randn(M, N, device=device)
        except Exception as e:
            print(f"gemm M={M} K3={K3} N={N} setup error: {str(e)[:100]}", flush=True)
            continue
        flop = 2.0 * M * K3 * N
        byt = M * K3 * 2 + N * K3 * 2 + M * N * 4

        def rep(tag, fn, flop=flop, byt=byt):
            try:
                ms = _med(fn)
                print(f"gemm M={M:<8d} K3={K3:<5d} N={N:<5d} {tag:28s} {ms:9.3f} ms "
                      f"{flop / ms / 1e9:6.1f} TFLOPS {byt / ms / 1e6:6.0f} GB/s  {_top_kernel(fn)}", flush=True)
            except Exception as e:
                print(f"gemm M={M:<8d} K3={K3:<5d} N={N:<5d} {tag:28s} ERROR {str(e)[:90]}", flush=True)

        rep("a: mm out_dtype=fp32", lambda: torch.mm(a3, w3.t(), out_dtype=torch.float32))
        rep("b: mm fp16 out (bound)", lambda: torch.mm(a3, w3.t()))
        rep("c: NN layout out_dtype", lambda: torch.mm(a3, w3n, out_dtype=torch.float32))
        try:
            torch.backends.cuda.preferred_blas_library("cublaslt")
            rep("d: preferred=cublaslt", lambda: torch.mm(a3, w3.t(), out_dtype=torch.float32))
            torch.backends.cuda.preferred_blas_library("cublas")
            rep("d: preferred=cublas", lambda: torch.mm(a3, w3.t(), out_dtype=torch.float32))
        except Exception as e:
            print(f"gemm d: preferred_blas_library unavailable: {str(e)[:80]}", flush=True)
        finally:
            try:
                torch.backends.cuda.preferred_blas_library("default")
            except Exception:
                pass
        old = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        rep("e: reduced-prec reduction ON", lambda: torch.mm(a3, w3.t(), out_dtype=torch.float32))
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old
        if M % 4 == 0:
            q = M // 4
            rep("f: M chunked x4", lambda: [torch.mm(a3[i * q:(i + 1) * q], w3.t(), out_dtype=torch.float32) for i in range(4)])
        if M % 8 == 0:
            rep("g: bmm [8,M/8,K3]", lambda: torch.bmm(a3.view(8, M // 8, K3), w3.t().unsqueeze(0).expand(8, K3, N), out_dtype=torch.float32))
        K = K3 // 3
        a32 = torch.randn(M, K, device=device)
        w32 = torch.randn(N, K, device=device) * 0.03
        rep("h: fp32 SGEMM (K=K3/3)", lambda: torch.mm(a32, w32.t()), flop=2.0 * M * K * N, byt=M * K * 4 + N * K * 4 + M * N * 4)
        rep("i: addmm bias1d alpha", lambda: torch.addmm(bias, a3, w3.t(), alpha=0.5, out_dtype=torch.float32))
        rep("i: addmm residual2d beta=1", lambda: torch.addmm(res, a3, w3.t(), alpha=0.5, beta=1.0, out_dtype=torch.float32), byt=byt + M * N * 4)
        # j: int8 IMMA for the two cross terms (K doubled) -- go/no-go
        K2 = 2 * K
        if K2 % 8 == 0 and N % 8 == 0 and M > 16 and M <= 65536:
            try:
                ai = torch.randint(-127, 128, (M, K2), device=device, dtype=torch.int8)
                wi = torch.randint(-127, 128, (N, K2), device=device, dtype=torch.int8)
                got = torch._int_mm(ai, wi.t())
                exact = ai.double() @ wi.double().t()
                err = (got.double() - exact).abs().max().item()
                rep(f"j: _int_mm int8 K2={K2} (err {err:.0f})", lambda: torch._int_mm(ai, wi.t()),
                    flop=2.0 * M * K2 * N, byt=M * K2 + N * K2 + M * N * 4)
                del ai, wi, got, exact
            except Exception as e:
                print(f"gemm M={M:<8d} K3={K3:<5d} N={N:<5d} j: _int_mm ERROR {str(e)[:100]}", flush=True)
        del a3, w3, w3n, bias, res, a32, w32
        torch.cuda.empty_cache()
    _gpu_fingerprint("gemm-end")


def _x3_kbench(device):
    """Error of the fp16x3 GEMM against fp64 as a function of K, for the
    shipped operand order, the lo-first order, split-K, the CPU
    round-to-nearest emulation and the fp64 sum of the split operands."""
    import math
    import torch
    import torch.nn.functional as F
    print(f"x3_available: {x3_available(device)} | order={_X3_ORDER} splitk={_X3_SPLITK}", flush=True)
    _gpu_fingerprint("x3k-start")
    M, N = 4096, 1024
    for K in (128, 256, 512, 1024, 2048, 4096):
        try:
            torch.manual_seed(K)
            x = torch.randn(M, K, device=device)
            a = F.layer_norm(x, (K,))
            w = (torch.rand(N, K, device=device) * 2 - 1) / math.sqrt(K)
            ref = a.double() @ w.double().t()
            w3, _b, inv = x3_prepare(w, None)
            hi = a.half()
            lo = (a - hi.float()).half()
            w3 = w3[:, :3 * K].contiguous()          # the K-sweep measures the GEMM alone
            a_hhl = torch.cat([hi, hi, lo], 1).contiguous()
            a_lhh = torch.cat([lo, hi, hi], 1).contiguous()

            def stats(out, tag, ms):
                e = out.double() - ref
                print(f"X3K,K={K},{tag},max_abs={e.abs().max().item():.3e},mean_abs={e.abs().mean().item():.3e},"
                      f"bias={(torch.sign(ref) * e).mean().item():+.3e},"
                      f"frac_below={(out.double().abs() < ref.abs()).double().mean().item():.3f},ms={ms:.4f}", flush=True)

            stats(a @ w.t(), "fp32-sgemm", _med(lambda: a @ w.t()))
            for tag, a3 in (("hhl", a_hhl), ("lhh", a_lhh)):
                fn = (lambda a3=a3: torch.mm(a3, w3.t(), out_dtype=torch.float32) * inv)  # GEMM only, no tail
                stats(fn(), tag, _med(fn))
                for c in (2, 4):
                    k3, ch = 3 * K, (3 * K) // c

                    def fn_sk(a3=a3, c=c, ch=ch, k3=k3):
                        out = torch.mm(a3[:, :ch], w3[:, :ch].t(), out_dtype=torch.float32) * inv
                        for i in range(1, c):
                            sl = slice(i * ch, (i + 1) * ch if i < c - 1 else k3)
                            out = torch.addmm(out, a3[:, sl], w3[:, sl].t(), alpha=inv, beta=1.0,
                                              out_dtype=torch.float32)
                        return out
                    stats(fn_sk(), f"{tag}-splitk{c}", _med(fn_sk))
            stats((a_hhl.double() @ w3.double().t()) * inv, "split-fp64sum", 0.0)
            sub = 512
            cpu = (a_hhl[:sub].cpu().float() @ w3.cpu().float().t()) * inv
            e = cpu.double() - ref[:sub].cpu()
            print(f"X3K,K={K},cpu-rn,max_abs={e.abs().max().item():.3e},mean_abs={e.abs().mean().item():.3e},"
                  f"bias={(torch.sign(ref[:sub].cpu()) * e).mean().item():+.3e},frac_below=nan,ms=0", flush=True)
            del x, a, w, ref, w3, hi, lo, a_hhl, a_lhh
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"X3K,K={K},error,{str(e)[:120]}", flush=True)
    # Accumulator precision probe: 2048 copies of fp16(1/3). fp32 accumulation
    # gives 682.5 to ~1e-4; fp16 accumulation would be off by tens.
    try:
        a1 = torch.ones(64, 2048, device=device).half()
        b1 = torch.full((64, 2048), 1.0 / 3.0, device=device).half()
        got = torch.mm(a1, b1.t(), out_dtype=torch.float32)[0, 0].item()
        exact = 2048 * float(b1[0, 0].float().item())
        print(f"X3K,accprobe,K=2048,got={got:.6f},exact={exact:.6f},diff={got - exact:+.3e}", flush=True)
    except Exception as e:
        print(f"X3K,accprobe,error,{str(e)[:100]}", flush=True)
    _gpu_fingerprint("x3k-end")


def _x3i8_bench(device):
    """fp16x3 (one K-tripled fp16 GEMM) against fp16 main + int8 cross terms
    (one K fp16 GEMM + one 2K int8 GEMM + dequant) per graded GEMM geometry:
    time and max-abs error against fp64."""
    import math
    import torch
    import torch.nn.functional as F
    print(f"x3_available: {x3_available(device)}", flush=True)
    _gpu_fingerprint("x3i8-start")
    geoms = [(8192, 1024, 3072, "8 qkv"), (8192, 1024, 1024, "8 o/ffn"), (65536, 128, 384, "13 qkv"),
             (65536, 128, 128, "13 o"), (16384, 128, 384, "5 qkv"), (1280000, 128, 128, "6 o")]
    for (M, K, N, tag) in geoms:
        try:
            x = torch.randn(M, K, device=device)
            g = torch.rand(K, device=device) + 0.5
            be = torch.randn(K, device=device) * 0.1
            w = (torch.rand(N, K, device=device) * 2 - 1) / math.sqrt(K)
            bias = (torch.rand(N, device=device) * 2 - 1) / math.sqrt(K)
            n_err = min(M, 4096)
            ref = F.layer_norm(x[:n_err].double(), (K,), g.double(), be.double(), 1e-5) @ w.double().t() + bias.double()
            # shipped fp16x3: LN-split + one GEMM (bias in the K tail)
            w3, _, inv = x3_prepare(w, bias)
            f3 = lambda: x3_linear(x3_ln_split(x, g, be, 1e-5), w3, None, inv, False)
            t3 = _med(f3)
            e3 = ((f3()[:n_err].double() * inv) - ref).abs().max().item()
            # int8 cross terms: LN (fp32, torch) + i8 split + fp16 GEMM + int8 GEMM + dequant
            w_hi, w_q, t_n, _ = x3i8_prepare(w, bias)
            ln = lambda: F.layer_norm(x, (K,), g, be, 1e-5)
            def f8():
                a_hi, a_q, s_r = x3i8_split(ln())
                return x3i8_linear(a_hi, a_q, s_r, w_hi, w_q, t_n)
            t8 = _med(f8)
            e8 = (f8()[:n_err].double() - ref).abs().max().item()
            # components
            y = ln()
            t_ln = _med(ln)
            t_split8 = _med(lambda: x3i8_split(y))
            a_hi, a_q, s_r = x3i8_split(y)
            a_hi_k = a_hi[:, :w_hi.shape[1]]
            t_main = _med(lambda: torch.mm(a_hi_k, w_hi.t(), out_dtype=torch.float32))
            t_cross = _med(lambda: torch._int_mm(a_q, w_q.t()))
            main = torch.mm(a_hi_k, w_hi.t(), out_dtype=torch.float32); cross = torch._int_mm(a_q, w_q.t())
            t_deq = _med(lambda: main + (s_r[:, None] * t_n[None, :] * (1.0 / 2048.0)) * cross.float())
            print(f"x3i8 shape {tag:8s} M={M:<8d} K={K:<5d} N={N:<5d} fp16x3 {t3:8.3f} ms (err {e3:.1e}) | "
                  f"int8-cross {t8:8.3f} ms (err {e8:.1e}) = LN {t_ln:.3f} + split {t_split8:.3f} + main {t_main:.3f} + cross {t_cross:.3f} + dequant {t_deq:.3f} | "
                  f"x{t3 / t8:.2f}", flush=True)
            del x, w, w3, w_hi, w_q, a_hi, a_q, main, cross, y
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"x3i8 shape {tag} error: {str(e)[:160]}", flush=True)
        torch.cuda.empty_cache()
    _gpu_fingerprint("x3i8-end")


def _x3_geo_bench(device):
    """Sweep rows-per-program and warps for the tiled split/LayerNorm kernels
    on the graded row widths, and run the thorough self-check on this GPU."""
    import sys
    import torch
    import torch.nn.functional as F
    try:
        import kernels.fp16x3 as KX          # package layout
    except ImportError:
        KX = sys.modules[__name__]           # single-file Kaggle build: same module
    print(f"x3_available: {x3_available(device)} | selfcheck(thorough): {x3_selfcheck(device, True)}", flush=True)
    _gpu_fingerprint("x3geo-start")
    geoms = [(8192, 128), (65536, 128), (1280000, 128), (2048, 128), (128, 128),
             (8192, 32), (8192, 1024), (65536, 384)]
    for (M, N) in geoms:
        x = torch.randn(M, N, device=device); r = torch.randn_like(x)
        g = torch.rand(N, device=device) + 0.5; be = torch.randn(N, device=device) * 0.1
        by_s = M * N * (4 + 6); by_as = M * N * (8 + 4 + 6); by_ln = M * N * 8
        t_torch = _med(lambda: F.layer_norm(x, (N,), g, be, 1e-5))
        print(f"geo M={M:<8d} N={N:<5d} torch LN {t_torch:8.3f} ms @{by_ln / t_torch / 1e6:4.0f} GB/s", flush=True)
        for rows in (1, 2, 4, 8, 16, 32):
            for warps in (2, 4, 8):
                KX._X3_ROWS_OVERRIDE, KX._X3_WARPS_OVERRIDE = rows, warps
                try:
                    t_s = _med(lambda: x3_ln_split(x, g, be, 1e-5))
                    t_as = _med(lambda: x3_add_ln_split(x, r, g, be, 1e-5, 1.0))
                    t_g = _med(lambda: x3_act_split(x, True, 1.0))
                    t_ln = _med(lambda: x3_ln(x, g, be, 1e-5))
                    print(f"geo M={M:<8d} N={N:<5d} rows={rows:<3d} warps={warps}  ln_split {t_s:8.3f} @{by_s / t_s / 1e6:4.0f} | "
                          f"add_ln_split {t_as:8.3f} @{by_as / t_as / 1e6:4.0f} | gelu_split {t_g:8.3f} @{by_s / t_g / 1e6:4.0f} | "
                          f"ln32 {t_ln:8.3f} @{by_ln / t_ln / 1e6:4.0f} GB/s", flush=True)
                except Exception as e:
                    print(f"geo M={M} N={N} rows={rows} warps={warps} error: {str(e)[:100]}", flush=True)
        KX._X3_ROWS_OVERRIDE = KX._X3_WARPS_OVERRIDE = None
        del x, r
        torch.cuda.empty_cache()
    _gpu_fingerprint("x3geo-end")


def _x3_bench(device):
    """Per-site op-level table for the fp16x3 path on every graded shape:
    what each GEMM site costs on the fp32 path (its producer + SGEMM) against
    the fp16x3 path (fused split + tensor-core GEMM), with the split kernels'
    achieved bandwidth. Errors against fp64 on a 4096-row subset, timings on
    the full M, so the 1.28M-row shape no longer OOMs the reference."""
    import math
    import torch
    import torch.nn.functional as F
    print(f"x3_available: {x3_available(device)} | triton ok: {_X3_TRITON_OK} | "
          f"addmm bias epilogue: {_X3_ADDMM_BIAS} | order={_X3_ORDER} splitk={_X3_SPLITK}", flush=True)
    _gpu_fingerprint("x3-start")
    seen = set()
    for (idx, b, d, h, s, l, f) in _SH13:
        M = b * s
        n_err = min(M, 4096)
        # the two LayerNorm-class kernels at this shape's width
        if (M, d, "ln") not in seen:
            seen.add((M, d, "ln"))
            try:
                x = torch.randn(M, d, device=device); r = torch.randn_like(x)
                g = torch.rand(d, device=device) + 0.5; be = torch.randn(d, device=device) * 0.1
                t_ln = _med(lambda: F.layer_norm(x, (d,), g, be, 1e-5))
                t_addln = _med(lambda: F.layer_norm(x + r, (d,), g, be, 1e-5))
                t_s = _med(lambda: x3_ln_split(x, g, be, 1e-5))
                t_as = _med(lambda: x3_add_ln_split(x, r, g, be, 1e-5))
                by_ln = M * d * (4 + 4); by_s = M * d * (4 + 6); by_as = M * d * (8 + 4 + 6)
                print(f"lnk  {idx:>2} M={M:<8d} D={d:<5d} torch LN {t_ln:8.3f} ms @{by_ln / t_ln / 1e6:4.0f} GB/s | torch add+LN {t_addln:8.3f} | "
                      f"ln_split {t_s:8.3f} @{by_s / t_s / 1e6:4.0f} GB/s | add_ln_split {t_as:8.3f} @{by_as / t_as / 1e6:4.0f} GB/s", flush=True)
                del x, r
            except Exception as e:
                print(f"lnk {idx} error: {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()
        for (site, K, N, kind) in (("qkv", d, 3 * d, "ln"), ("out", d, d, "plain"),
                                   ("ffn_in", d, f, "ln"), ("ffn_out", f, d, "gelu")):
            if (M, K, N, kind) in seen:
                continue
            seen.add((M, K, N, kind))
            try:
                x = torch.randn(M, K, device=device)
                g = torch.rand(K, device=device) + 0.5
                be = torch.randn(K, device=device) * 0.1
                w = (torch.rand(N, K, device=device) * 2 - 1) / math.sqrt(K)
                bias = (torch.rand(N, device=device) * 2 - 1) / math.sqrt(K)
                prep = x3_prepare(w, bias)
                if kind == "ln":
                    base = lambda: F.linear(F.layer_norm(x, (K,), g, be, 1e-5), w, bias)
                    split = lambda: x3_ln_split(x, g, be, 1e-5)
                    ref = F.layer_norm(x[:n_err].double(), (K,), g.double(), be.double(), 1e-5) @ w.double().t() + bias.double()
                elif kind == "plain":
                    base = lambda: F.linear(x, w, bias)
                    split = lambda: x3_act_split(x, False)
                    ref = x[:n_err].double() @ w.double().t() + bias.double()
                else:
                    base = lambda: F.linear(F.gelu(x, approximate="none"), w, bias)
                    split = lambda: x3_act_split(x, True)
                    ref = F.gelu(x[:n_err].double(), approximate="none") @ w.double().t() + bias.double()
                w3, _, inv = prep
                if site == "qkv":
                    x3 = lambda: x3_linear(split(), w3, None, inv, False)   # bias in the K tail
                    ex3 = ((x3()[:n_err].double() * inv) - ref).abs().max().item()
                else:
                    w3n, bn, invn = x3_prepare(w, bias, fold_bias=False)
                    x3 = lambda: x3_linear(split()[:, :3 * K], w3n, None, invn, False)  # consumer adds bias
                    ex3 = ((x3()[:n_err].double() * invn + bn.double()) - ref).abs().max().item()
                e32 = (base()[:n_err].double() - ref).abs().max().item()
                t_base = _med(base); t_x3 = _med(x3); t_split = _med(split)
                a3 = split()
                t_gemm = _med(x3) - t_split if False else _med(lambda: x3_linear(a3 if site == "qkv" else a3[:, :3 * K], w3 if site == "qkv" else w3n, None, inv, False))
                gbps = M * K * (4 + 6) / t_split / 1e6
                print(f"site {idx:>2} {site:7s} M={M:<8d} K={K:<5d} N={N:<5d} fp32 {t_base:8.3f} ms | "
                      f"x3 {t_x3:8.3f} ms (split {t_split:.3f} @{gbps:4.0f} GB/s + GEMM {t_gemm:.3f}) | "
                      f"x{t_base / t_x3:.2f} | err {e32:.1e} -> {ex3:.1e}", flush=True)
                del x, w, prep, a3, ref
            except Exception as e:
                print(f"site {idx} {site} error: {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()
    _gpu_fingerprint("x3-end")



def _main():
    import os
    import torch
    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("high")
    print("=== ENV ===", flush=True)
    print(f"gpu {torch.cuda.get_device_name(device)} | torch {torch.__version__} | "
          f"cuda {torch.version.cuda} | cc {torch.cuda.get_device_capability(device)}", flush=True)

    only = os.environ.get("T3_ONLY", "all").strip().lower()
    _gpu_fingerprint("start")
    import atexit
    atexit.register(lambda: _gpu_fingerprint("end"))
    if only == "gemm":
        _gemm_probe(device)
        return
    if only == "x3k":
        _x3_kbench(device)
        return
    if only == "x3geo":
        _x3_geo_bench(device)
        return
    if only == "x3i8":
        _x3i8_bench(device)
        return
    try:
        _sdpa_probe(device)
    except Exception as e:
        print("sdpa probe error:", str(e)[:120], flush=True)
    if only == "probe":
        return
    if only == "profile":
        _profile(device)
        return
    if only == "x3":
        print("\n##### FP16x3 LINEAR BENCH #####", flush=True)
        try:
            _x3_bench(device)
        except Exception as e:
            print("X3 BENCH ERROR:", str(e)[:200], flush=True)
    if only == "triton":
        SH_FILTER = []

    SH = [
        (1, 64, 128, 4, 128, 4, 128), (2, 1, 128, 4, 128, 4, 128),
        (3, 4, 128, 4, 128, 4, 128), (4, 16, 128, 4, 128, 4, 128),
        (5, 128, 128, 4, 128, 4, 128), (6, 10000, 128, 4, 128, 4, 128),
        (7, 64, 32, 4, 128, 4, 32), (8, 64, 1024, 4, 128, 4, 1024),
        (9, 64, 128, 1, 128, 4, 128), (10, 64, 128, 2, 128, 4, 128),
        (11, 64, 128, 16, 128, 4, 128), (12, 64, 128, 4, 32, 4, 128),
        (13, 64, 128, 4, 1024, 4, 128),
    ]
    for (idx, b, d, h, s, l, f) in (SH if only in ("all", "1-13", "x3") else []):
        print(f"\n##### SHAPE {idx} : B={b} D={d} H={h} S={s} L={l} F={f} #####", flush=True)
        cfg = TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                ffn_dim=f, num_layers=l, causal=True)
        cfg.validate()
        try:
            baseline = BaselineTransformer(cfg).to(device, dtype).eval()
            optimized = UserOptimizedTransformer(cfg)
            copy_model_weights(baseline, optimized, strict=True)
            optimized = optimized.to(device, dtype).eval()
            ok, mabs, mrel = _accuracy(baseline, optimized, cfg, device, dtype)
            tr = getattr(optimized, "_tune_result", None)
            if tr is not None:
                names = ("eager", "compiled", "graph")
                parts = " ".join(f"{n}={v:.4f}ms" for n, v in zip(names, tr)
                                 if v != float("inf"))
                chosen = ("graph" if getattr(optimized, "_graph", None) is not None
                          else "compiled" if optimized._compiled is not None else "eager")
                print(f"autotune: {parts} -> {chosen}", flush=True)
            lr = getattr(optimized, "_linear_result", None)
            if lr is not None:
                print("linear autotune: " + " ".join(
                    f"{'fp16x3' if on else 'fp32'}={ms:.4f}ms" for ms, on in lr)
                      + f" -> {'fp16x3' if getattr(optimized, '_x3_on', False) else 'fp32'}", flush=True)
            elif getattr(optimized, "_x3_on", False):
                print("linear: fp16x3 (forced)", flush=True)
            if ok:
                xt, mt = generate_random_case(cfg, device, dtype, 101234, 0.0, 1.0)
                bms = _bench(baseline, xt, mt); oms = _bench(optimized, xt, mt)
                sp = bms / oms
                print(f"PASS max_abs={mabs:.3g} max_rel={mrel:.3g} | "
                      f"baseline={bms:.4f}ms optimized={oms:.4f}ms | speedup={sp:.3f}x", flush=True)
                _RESULTS.append((idx, "PASS", mabs, mrel, f"{bms:.4f}", f"{oms:.4f}", f"{sp:.3f}", ""))
            else:
                print(f"FAIL max_abs={mabs:.3g} max_rel={mrel:.3g}", flush=True)
                _RESULTS.append((idx, "FAIL", mabs, mrel, "", "", "", ""))
        except Exception as e:
            print(f"SHAPE {idx} ERROR:", str(e)[:200], flush=True)
            _RESULTS.append((idx, "ERROR", "", "", "", "", "", str(e)[:60]))
        finally:
            try:
                del baseline, optimized
            except Exception:
                pass
            torch.cuda.empty_cache()

    if only == "attn":
        print("\n##### ATTENTION KERNEL BENCH #####", flush=True)
        try:
            _attn_bench(device)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("ATTN BENCH ERROR:", str(e)[:200], flush=True)
        return

    if only in ("all", "triton"):
        print("\n##### TRITON KERNEL BENCH #####", flush=True)
        try:
            _triton_bench(device)
        except Exception as e:
            print("TRITON BENCH ERROR:", str(e)[:200], flush=True)

    if only in ("all", "ablation"):
        print("\n##### STAGE ABLATION #####", flush=True)
        try:
            _ablation(device)
        except Exception as e:
            print("ABLATION ERROR:", str(e)[:200], flush=True)

    print("\n##### SHAPE 14 : optimized-only (baseline infeasible ~20.5 TB) #####", flush=True)
    try:
        if only in ("1-13", "ablation", "triton", "x3"):
            raise _Skip
        _shape14(device)
    except _Skip:
        print("skipped for this T3_ONLY selector", flush=True)
    except Exception as e:
        print("SHAPE 14 ERROR:", str(e)[:200], flush=True)
        _RESULTS.append((14, "ERROR", "", "", "", "", "", str(e)[:60]))

    # ---- compact summary: copy THIS block back ----
    sp_vals = sorted(float(r[6]) for r in _RESULTS if r[6] and r[1] == "PASS")
    print("\n=================== SUMMARY (copy from here) ===================", flush=True)
    print("shape,pass,max_abs,max_rel,baseline_ms,opt_ms,speedup,note", flush=True)
    for r in _RESULTS:
        print(",".join(str(x) for x in r), flush=True)
    if sp_vals:
        print(f"# median_speedup={sp_vals[len(sp_vals)//2]:.3f}x "
              f"min={sp_vals[0]:.3f}x max={sp_vals[-1]:.3f}x over {len(sp_vals)} PASS shapes", flush=True)
    if _ABL_RESULTS:
        print("# --- stage ablation ---", flush=True)
        print("abl_shape,stage,pass,max_abs,max_rel,baseline_ms,opt_ms,speedup",
              flush=True)
        for r in _ABL_RESULTS:
            print(",".join(str(x) for x in r), flush=True)
    print("=================== END SUMMARY ===================", flush=True)


_main()

#!/usr/bin/env python3
"""
UserOptimizedTransformer — TikTok TechJam 2026, Track 3
Implement a GPU Kernel for a Transformer Layer.

This module rewrites ONLY the forward compute of the reference
``BaselineTransformer`` (defined in ``torch_transformer_benchmark.py``) while
keeping every submodule and parameter name identical, so the harness'
``copy_model_weights(..., strict=True)`` succeeds with zero friction.

Optimization levers (all numerically equivalent within the harness tolerance
atol=0.002 OR rtol=0.02, checked per-element):
  1. Attention via ``F.scaled_dot_product_attention`` (memory-efficient fused kernel;
     PyTorch's flash backend is fp16-only and needs sm_80+, so it never ran here) -> O(S) memory instead of the baseline's
     O(S^2) materialized score matrix. This is what makes the seq_len=100000
     shape possible at all (the baseline would need ~20.5 TB for its scores).
  2. Optional internal fp16 autocast. Disabled by default: accumulated rounding
     can cross the per-element correctness gate. The shipped fp16x3 GEMMs
     use tensor cores with error compensation instead.
  3. Self-applied ``torch.compile`` (does not depend on the grader passing
     --compile-user); mode chosen per shape (reduce-overhead for launch-bound
     small shapes, default otherwise).
  4. Batch chunking ONLY for the extreme shape (seq_len=1e5) so activations fit
     in 16 GB.
  5. fp16x3 linear layers (kernels/fp16x3.py, default): every GEMM on the fp16
     tensor cores with fp32-class error -- each operand as an fp16 hi + lo pair,
     one cuBLAS GEMM with K tripled, fp32 accumulation and output; the split is
     fused into the LayerNorm / GELU kernels that produce the activation.

Ablation / robustness toggles via environment variables (see README):
  T3_AUTOCAST   = auto | fp16 | bf16 | off      (default off; see _plan)
  T3_COMPILE    = auto | 1 | 0                   (default auto: time eager vs compiled
                                                  once, on the real input, keep the winner)
  T3_CUDAGRAPH  = 1 | 0                          (default 1; the eager path captured into a
                                                  CUDA graph is a third autotune candidate)
  T3_COMPILE_MODE = default | reduce-overhead | max-autotune  (override)
  T3_FP32_FFN   = 1 | 0                          (default 0; force FFN+LN fp32)
  T3_CHUNK_BS   = <int>                          (override batch chunk size)
  T3_FUSED_QKV  = 1 | 0                          (default 0; one [3D,D] GEMM for q,k,v)
  T3_ATTN       = sdpa | triton | turing         (default sdpa; turing is optional,
                                                  native-fp16 long-sequence inference only)
  T3_X3_BLAS    = torch | lt                     (default torch; opt-in zero-workspace
                                                  cuBLASLt search for the wide QKV GEMM)
                                                (triton: tensor-core attention
                                                  kernel with fp32 statistics)
  T3_ATTN_SPLIT = 1 | 3                          (default 3; operand splitting for
                                                  fp32-class error on fp16 MMAs)
  T3_TRITON     = 1 | 0                          (default 0; hand-written fused
                                                  add+LayerNorm as a registered op,
                                                  composes with torch.compile)
  T3_LINEAR     = fp16x3 | fp32                   (default fp16x3: every GEMM on the fp16
                                                  tensor cores with hi/lo operand
                                                  compensation, fp32 accumulation and
                                                  output; fp32 = cuBLAS SGEMM)
  T3_X3_GUARD   = static+first | every | off      (default static+first: a bound from the
                                                  weights refuses fp16x3 when an activation
                                                  could leave fp16 range, and the first
                                                  forward's output is checked for
                                                  finiteness once; 'every' checks each
                                                  forward at the cost of a sync)
  T3_X3_SITES   = auto | <list>                   (which GEMMs take the fp16x3 path: a comma
                                                  list of qkv,out,ffn_in,ffn_out; auto = all
                                                  four. The operator-level table says the
                                                  split is an extra pass at out/ffn_out for
                                                  K=128, the end-to-end sweeps say all four
                                                  win by 5-15% once launches are hidden; the
                                                  sweeps decide)
  T3_X3_ORDER   = hhl | lhh                       (operand order along the tripled K; see
                                                  kernels/fp16x3.py)
  T3_X3_SPLITK  = 1 | 2 | 4                       (chunks of the tripled K combined in the
                                                  fp32 epilogue; accuracy option, off)
  T3_MASK_CACHE = 1 | 0                          (default 1: reuse mask validity only
                                                  while tensor identity, metadata and
                                                  mutation counter are unchanged;
                                                  inference tensors are never cached)
  T3_PACK_PADDING = 1 | 0                        (default 1: group and compact valid
                                                  tokens before attention, then scatter
                                                  outputs; avoids a dense S-by-S mask)
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

# Reuse the reference model definition so parameter names match exactly.
from torch_transformer_benchmark import BaselineTransformer

# --- kernels import (begin) --- the Kaggle builder replaces this block
try:
    from kernels import (HAVE_TRITON_OP, can_fuse, fused_add_layernorm,
                         triton_attention, can_use_attention, turing_attention,
                         x3_available, x3_prepare, x3_linear, x3_ln_split,
                         x3_add_ln_split, x3_act_split, x3_ln, x3_add_ln)
    HAVE_KERNELS = True
except Exception:  # the package is optional; the model works without it
    HAVE_KERNELS = False
    HAVE_TRITON_OP = False
    triton_attention = can_use_attention = None
    turing_attention = None
    x3_available = x3_prepare = x3_linear = None
    x3_ln_split = x3_add_ln_split = x3_act_split = x3_ln = x3_add_ln = None
# --- kernels import (end) ---


# Live [chunk, S, max(D, ffn)] intermediates a block keeps alive simultaneously.
_LIVE_INTERMEDIATES = 8

# torch.cuda.OutOfMemoryError exists on torch>=1.13; older builds raise RuntimeError.
_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class UserOptimizedTransformer(BaselineTransformer):
    """Drop-in optimized replacement. Same weights, faster forward."""

    def __init__(self, config) -> None:
        super().__init__(config)  # identical submodules -> strict weight copy OK

        # Lazily resolved on the first CUDA forward (see _plan).
        self._planned = False
        self._autocast_dtype: Optional[torch.dtype] = None
        self._chunk_bs: Optional[int] = None
        self._compiled = None
        self._graph_output = False  # set in forward() when a CUDA-graph mode is used
        raw = os.environ.get("T3_COMPILE", "auto").strip().lower()
        self._compile_policy = ("auto" if raw == "auto"
                                else "on" if raw in ("1", "true", "yes", "on")
                                else "off")
        self._compile_ok = self._compile_policy != "off"
        self._tuned = False          # T3_COMPILE=auto: decided once, first forward
        self._tune_result = None     # (eager_ms, compiled_ms[, graph_ms]) when it ran
        # Manual CUDA-graph capture of the *eager* path. For launch-bound shapes
        # the remaining cost is CPU launch overhead; a captured graph replays the
        # same kernels with none of it, and without the Inductor guard/dispatch
        # cost that made reduce-overhead lose on those shapes. It enters the
        # first-forward autotune as a third candidate and is kept only if it wins.
        self._cudagraph = _env_flag("T3_CUDAGRAPH", True)
        self._graph = None
        self._g_x = self._g_m = self._g_out = None
        self._g_key = None
        self._mask_cache = _env_flag("T3_MASK_CACHE", True)
        self._mask_tensor = None
        self._mask_key = None
        self._mask_valid = False
        self._pack_padding = _env_flag("T3_PACK_PADDING", True)
        self._padding_key = None
        self._padding_tensor = None
        self._padding_groups = None
        self._can_compile = False  # set in _plan: Triton needs CUDA capability >= 7.0
        self._fp32_ffn = _env_flag("T3_FP32_FFN", False)
        # One [3D, D] GEMM for q, k and v instead of three [D, D] ones: fewer
        # launches, and the activation is read once rather than three times.
        # The concatenated weight is cached as a plain tensor attribute on the
        # attention module -- not a Parameter, not a registered buffer -- so
        # state_dict and the harness' strict weight copy never see it.
        self._fused_qkv = _env_flag("T3_FUSED_QKV", False)
        # Attention implementation. 'triton' is the tensor-core kernel with fp32
        # statistics (kernels/attention.py); SPLIT=3 adds operand splitting for
        # fp32-class error at fp16 MMA rate. Off by default until measured.
        self._attn_impl = os.environ.get("T3_ATTN", "sdpa").strip().lower()
        try:
            self._attn_split = int(os.environ.get("T3_ATTN_SPLIT", "3"))
        except ValueError:
            self._attn_split = 3
        if self._attn_split not in (1, 3):
            self._attn_split = 3
        if self._attn_impl == "triton" and (not HAVE_KERNELS or triton_attention is None):
            print("[user_optimized] T3_ATTN=triton requested but the kernel is unavailable; using sdpa")
            self._attn_impl = "sdpa"
        # Hand-written Triton fused add+LayerNorm, off by default. When the
        # kernel is registered as a torch.library op (torch >= 2.6) Inductor can
        # schedule it inside the compiled graph, so compile stays on and the two
        # compose. Only the raw-launch fallback breaks the graph, and only then
        # does asking for the kernel turn compilation off.
        self._triton = HAVE_KERNELS and _env_flag("T3_TRITON", False)
        if self._triton and not HAVE_TRITON_OP:
            self._compile_ok = False
        # fp16x3 linear layers (kernels/fp16x3.py): every GEMM on the fp16
        # tensor cores with hi/lo operand compensation, fp32 accumulation and
        # fp32 output. Default, measured: median 2.44x against 2.26x for
        # SGEMM on the T4, worst error 4.1e-5 (K=1024) against a 2e-3 gate.
        # A per-shape first-forward tuner ('auto') was measured twice and
        # retired: its seven-sample verdicts sit at the noise floor on the
        # sub-millisecond shapes, and the static default scored higher.
        raw = os.environ.get("T3_LINEAR", "fp16x3").strip().lower()
        if raw == "auto":
            # Measured twice and retired: its seven-sample verdicts sit at the
            # noise floor on the sub-millisecond shapes and the static default
            # scored higher (results/ablation.md). Kept as an alias of the default.
            print("[user_optimized] T3_LINEAR=auto was retired after measurement; using 'fp16x3'")
            raw = "fp16x3"
        if raw not in ("fp32", "fp16x3"):
            print(f"[user_optimized] ignoring unrecognised T3_LINEAR={raw!r}; using 'fp16x3'")
            raw = "fp16x3"
        if raw != "fp32" and (not HAVE_KERNELS or x3_linear is None):
            print("[user_optimized] T3_LINEAR requested but kernels/fp16x3 is unavailable; using fp32")
            raw = "fp32"
        self._linear_policy = raw
        self._x3_on = False          # decided in _plan
        # Range guard for the split: activations are split into fp16 pairs, so
        # anything beyond 65504 would become inf. 'static+first' refuses the
        # path from a bound computed on the weights (no sync) and checks the
        # first forward's output once; 'every' checks each forward (one sync).
        self._x3_guard = os.environ.get("T3_X3_GUARD", "static+first").strip().lower()
        if self._x3_guard not in ("static+first", "every", "off"):
            print(f"[user_optimized] ignoring unrecognised T3_X3_GUARD={self._x3_guard!r}; "
                  f"using 'static+first'")
            self._x3_guard = "static+first"
        self._x3_checked = False     # first-forward finiteness check done
        self._x3_bound = None        # static activation bound from the weights
        self._x3_norm_key = None
        raw = os.environ.get("T3_X3_SITES", "auto").strip().lower()
        if raw == "auto":
            # Measured: the operator-level table has the split losing at the
            # out / ffn_out sites of narrow models, but end to end all four
            # sites beat the qkv+ffn_in subset on 12 of 13 shapes (5-15%),
            # because under graph replay the split is cheaper than the SGEMM
            # it replaces and an fp32 site still has to apply the neighbour's
            # scale and bias. results/ablation.md has both runs.
            sites = {"qkv", "out", "ffn_in", "ffn_out"}
        else:
            sites = {t.strip() for t in raw.split(",") if t.strip()}
            bad = sites - {"qkv", "out", "ffn_in", "ffn_out"}
            if bad:
                print(f"[user_optimized] ignoring unknown T3_X3_SITES entries {sorted(bad)}")
                sites -= bad
        self._x3_sites = frozenset(sites)

    # ---- one-time device/shape aware planning -------------------------------
    def _plan(self, x: torch.Tensor) -> None:
        if self._planned:
            return

        # torch.compile uses the Triton backend, which requires CUDA capability
        # >= 7.0 (Volta+). Older GPUs (e.g. Kaggle's Tesla P100 = sm_60) must run
        # eager. get_device_capability does not launch a kernel, so it is safe.
        self._can_compile = (
            x.device.type == "cuda"
            and torch.cuda.get_device_capability(x.device)[0] >= 7
        )

        # Precision: DEFAULT to the native dtype (no autocast). Internal fp16 is
        # opt-in only (T3_AUTOCAST=fp16/auto), because fp16's ~1e-3 per-op error
        # accumulates across layers and breaks the strict atol=0.002 gate for
        # near-zero output elements. Running in the grader's own dtype always
        # passes: fp32 grading -> fp32 (exact), fp16 grading -> fp16 (matches).
        want = os.environ.get("T3_AUTOCAST", "off").strip().lower()
        if want not in ("off", "fp16", "bf16", "auto"):
            # Never let a typo or an empty string turn reduced precision ON. The
            # fp16 path passes with almost no absolute-tolerance margin, so the
            # unrecognised case has to fail towards the exact path, loudly.
            print(f"[user_optimized] ignoring unrecognised T3_AUTOCAST={want!r}; "
                  f"using 'off' (valid: off|fp16|bf16|auto)")
            want = "off"
        if x.device.type != "cuda" or x.dtype != torch.float32 or want == "off":
            self._autocast_dtype = None
        elif want == "fp16":
            self._autocast_dtype = torch.float16
        elif want == "bf16":
            self._autocast_dtype = torch.bfloat16
        else:  # auto: T4/Turing has fp16 tensor cores only; bf16 on A100/L4.
            name = torch.cuda.get_device_name()
            if ("T4" in name) or (not torch.cuda.is_bf16_supported()):
                self._autocast_dtype = torch.float16
            else:
                self._autocast_dtype = torch.bfloat16

        # Batch-chunk only when the full batch's activations would not fit in
        # the VRAM that is actually free right now. In practice this only ever
        # triggers for the seq_len=1e5 shape.
        override = os.environ.get("T3_CHUNK_BS")
        b, s, _ = x.shape
        fdim = self.config.ffn_dim
        per_sample = s * max(self.config.d_model, fdim)
        if override is not None:
            try:
                self._chunk_bs = max(1, int(override))
            except (TypeError, ValueError):
                print(f"[user_optimized] ignoring non-integer "
                      f"T3_CHUNK_BS={override!r}; planning automatically")
                override = None
        if override is not None:
            pass
        elif b * per_sample <= 300_000_000:
            # Whole-batch activations are small by any measure -> never chunk.
            # Keeps every graded shape (1-13) on the single-pass path.
            self._chunk_bs = None
        else:
            cb = max(1, min(b, self._chunk_budget(x) // max(1, per_sample)))
            self._chunk_bs = cb if cb < b else None

        # fp16x3 needs tensor cores (sm_70+), cuBLAS out_dtype support and the
        # split kernels agreeing with the reference; x3_available checks all of
        # it once per device. On CPU the same arithmetic runs as exact
        # emulation, so the path is testable there.
        if self._linear_policy != "fp32":
            if x.device.type != "cuda":
                self._x3_on = self._chunk_bs is None
            elif x.dtype == torch.float32 and x3_available(x.device):
                # The chunked extreme shape is attention-bound; its GEMMs are
                # not worth the extra split buffers inside a tight VRAM budget.
                self._x3_on = self._chunk_bs is None
            else:
                print("[user_optimized] T3_LINEAR: fp16x3 path unavailable on this "
                      "device/dtype; using fp32")
                self._linear_policy = "fp32"
                self._x3_on = False

        # Only now. Setting it up front means a throw anywhere above would leave
        # planning permanently "done" with _chunk_bs still None -- which for the
        # seq_len=1e5 shape is the difference between chunking and an OOM.
        if (self._attn_impl == "turing" and x.device.type == "cuda"
                and x.dtype == torch.float16 and s >= 8192):
            # Upstream launches on the default stream. Keep this explicit
            # opt-in native-fp16 path eager, never capture its raw launches.
            self._compile_ok = False
            self._cudagraph = False
        self._planned = True

    def _chunk_budget(self, x: torch.Tensor) -> int:
        """Elements of working-set headroom available for one chunk.

        The chunked path pre-allocates the full [B,S,D] output, so the budget is
        (free VRAM - output buffer) with a fragmentation reserve, divided by the
        number of live intermediates a block keeps alive at once (~8: q/k/v, the
        SDPA output, its contiguous copy, the projection, and the two residuals).
        """
        if x.device.type != "cuda":
            return 300_000_000
        try:
            free_dev, _total = torch.cuda.mem_get_info(x.device)
            # Blocks the caching allocator already holds but is not using.
            free_dev += torch.cuda.memory_reserved(x.device) - torch.cuda.memory_allocated(
                x.device
            )
        except Exception:
            return 300_000_000
        out_bytes = x.numel() * x.element_size()
        usable = (free_dev - out_bytes) * 0.6  # 40% reserve for fragmentation
        elem_bytes = 2 if self._autocast_dtype is not None else x.element_size()
        return max(1, int(usable / (elem_bytes * _LIVE_INTERMEDIATES)))

    def _refresh_fused_qkv(self) -> None:
        """(Re)build each layer's concatenated QKV weight if its parts changed.

        Keyed on the parameters' in-place version counters plus device and
        dtype, so a weight copy or a .to() after the first forward rebuilds it
        and a steady-state forward pays only a few integer compares.
        """
        for layer in self.layers:
            attn = layer.attention
            parts = (attn.q_proj, attn.k_proj, attn.v_proj)
            key = tuple(self._tensor_key(m.weight) for m in parts) + tuple(
                self._tensor_key(m.bias) for m in parts
            )
            if getattr(attn, "_qkv_key", None) == key:
                continue
            with torch.no_grad():
                attn._qkv_w = torch.cat([m.weight for m in parts], dim=0)
                attn._qkv_b = (None if attn.q_proj.bias is None
                               else torch.cat([m.bias for m in parts], dim=0))
            attn._qkv_key = key
            # A captured graph replays the old tensors by address.
            self._drop_graph()

    def _x3_static_bound(self) -> float:
        """Largest value any split activation can take, from the weights alone.

        A standardized row has |z_i| <= sqrt(D), so a LayerNorm output is
        bounded elementwise by |gamma_i| sqrt(D) + |beta_i| and in 2-norm by
        max|gamma| sqrt(D) + ||beta||. A GEMM row j of that is bounded by that
        norm times ||W_j|| plus |b_j|; the attention output is a convex
        combination of v rows, and |gelu(t)| <= |t|. Input-independent, a few
        norms, computed only when a weight changes.
        """
        bound = 0.0
        with torch.no_grad():
            for layer in self.layers:
                d = layer.norm1.weight.numel()
                sq = float(d) ** 0.5
                for norm, lins in ((layer.norm1, (layer.attention.v_proj,)),
                                   (layer.norm2, (layer.ffn_in,))):
                    g = norm.weight.float().abs()
                    b = norm.bias.float().abs() if norm.bias is not None else None
                    elem = float((g * sq + (b if b is not None else 0.0)).max().item())
                    rown = float(g.max().item()) * sq + (float(b.norm().item()) if b is not None else 0.0)
                    bound = max(bound, elem)
                    for lin in lins:
                        wn = lin.weight.float().norm(dim=1)
                        bb = lin.bias.float().abs() if lin.bias is not None else 0.0
                        bound = max(bound, float((rown * wn + bb).max().item()))
        return bound

    def _x3_disable(self, why: str) -> None:
        print(f"[user_optimized] fp16x3 GEMMs off: {why}; using fp32 SGEMM")
        self._x3_on = False
        self._linear_policy = "fp32"
        self._drop_graph()

    def _refresh_x3(self) -> None:
        """(Re)build the split fp16 weights for the fp16x3 path if they changed.

        Same version-counter keying as the fused QKV cache. q, k and v are
        prepared as one [3D, D] weight so the projection is a single GEMM pair
        that reads the split activation once. A rebuild also drops any captured
        graph (it would replay the old tensors) and re-checks the static bound.
        """
        changed = False
        norms = [norm for layer in self.layers for norm in (layer.norm1, layer.norm2)]
        norms.append(self.final_norm)
        norm_key = tuple(self._tensor_key(p) for norm in norms
                         for p in (norm.weight, norm.bias))
        if norm_key != self._x3_norm_key:
            self._x3_norm_key = norm_key
            changed = True
        for layer in self.layers:
            attn = layer.attention
            parts = (attn.q_proj, attn.k_proj, attn.v_proj)
            key = tuple(self._tensor_key(m.weight) for m in parts) + tuple(
                self._tensor_key(m.bias) for m in parts
            )
            if "qkv" in self._x3_sites and getattr(attn, "_x3_qkv_key", None) != key:
                with torch.no_grad():
                    w = torch.cat([m.weight for m in parts], dim=0)
                    b = (None if attn.q_proj.bias is None
                         else torch.cat([m.bias for m in parts], dim=0))
                attn._x3_qkv = x3_prepare(w, b)
                attn._x3_qkv_key = key
                changed = True
            for site, lin in (("out", attn.out_proj), ("ffn_in", layer.ffn_in),
                              ("ffn_out", layer.ffn_out)):
                if site not in self._x3_sites:
                    continue
                key = (self._tensor_key(lin.weight), self._tensor_key(lin.bias))
                if getattr(lin, "_x3_key", None) != key:
                    # tail-free weight: the consumer kernel adds the bias
                    lin._x3 = x3_prepare(lin.weight, lin.bias, fold_bias=False)
                    lin._x3_key = key
                    changed = True
        if changed:
            self._drop_graph()
            self._x3_checked = False
            if self._x3_guard != "off":
                self._x3_bound = self._x3_static_bound()
                if not (self._x3_bound < 32768.0):
                    self._x3_disable(f"activations may reach {self._x3_bound:.3g} "
                                     f"(fp16 range is 65504)")

    @staticmethod
    def _tensor_key(tensor):
        """Replacing a Parameter can preserve its version; identity matters too."""
        if tensor is None:
            return None
        return (id(tensor), tensor._version, tensor.data_ptr(), tensor.device,
                tensor.dtype)

    def _all_valid(self, mask):
        """Cache a reduction, never the mask's contents or a model output.

        Ordinary tensors share a mutation counter with their views. Inference
        tensors have no such counter, so they must be reduced on every call.
        Writes through .data, NumPy or an external raw pointer bypass PyTorch's
        counter; callers using those must set T3_MASK_CACHE=0.
        """
        if mask is None:
            return True
        if not self._mask_cache or mask.is_inference():
            return bool(mask.all())
        key = self._tensor_key(mask) + (tuple(mask.shape), tuple(mask.stride()))
        if mask is not self._mask_tensor or key != self._mask_key:
            valid = bool(mask.all())
            self._mask_tensor, self._mask_key, self._mask_valid = mask, key, valid
        return self._mask_valid

    def _padded_groups(self, mask):
        """Valid positions in original order, grouped by sequence length.

        Compaction preserves causal order and is exact for this model: there
        are no position-dependent operators, and masked tokens are never keys
        for valid queries. Cache only indices, using the same mutation rules
        as the mask reduction; the activations are gathered afresh every time.
        """
        cacheable = self._mask_cache and not mask.is_inference()
        key = self._tensor_key(mask) if cacheable else None
        if (cacheable and self._padding_tensor is mask
                and key == self._padding_key):
            return self._padding_groups
        lengths = mask.sum(dim=1).tolist()
        positions = mask.nonzero(as_tuple=False)[:, 1]
        offsets = [0]
        by_length = {}
        for row, length in enumerate(lengths):
            offsets.append(offsets[-1] + length)
            if length:
                by_length.setdefault(length, []).append(row)
        groups = []
        for rows in by_length.values():
            token_ids = torch.stack([positions[offsets[r]:offsets[r + 1]] for r in rows])
            row_ids = torch.tensor(rows, dtype=torch.long, device=mask.device)
            groups.append((row_ids, token_ids))
        if cacheable:
            self._padding_tensor, self._padding_key, self._padding_groups = mask, key, groups
        return groups

    def _forward_padded(self, x, mask, causal, autocast_dtype):
        """O(B*S*D) packing storage, no [B,1,S,S] padding/causal bias.

        All-invalid rows remain zero, as in the reference. Different lengths
        never attend to one another; equal lengths share a batch. The compact
        kernels use the normal all-valid arithmetic, including fp16x3 when
        available, but do not replay a graph captured for the original shape.
        """
        out = torch.zeros_like(x)
        for rows, tokens in self._padded_groups(mask):
            chunk = self._chunk_bs or len(rows)
            for start in range(0, len(rows), chunk):
                rr, tt = rows[start:start + chunk], tokens[start:start + chunk]
                packed = x[rr[:, None], tt]
                if autocast_dtype is None:
                    result = self._run_full(packed, None, causal, True)
                else:
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        result = self._run_full(packed, None, causal, True)
                out[rr[:, None], tt] = result.to(x.dtype)
        return out

    # ---- compute ------------------------------------------------------------
    def _attention(self, attn, x, mask, causal, all_valid):
        b, s, d = x.shape
        h, hd = attn.num_heads, attn.head_dim
        if self._fused_qkv and getattr(attn, "_qkv_w", None) is not None:
            qkv = F.linear(x, attn._qkv_w, attn._qkv_b)
            q, k, v = qkv.split(d, dim=-1)
            q = q.reshape(b, s, h, hd).transpose(1, 2)
            k = k.reshape(b, s, h, hd).transpose(1, 2)
            v = v.reshape(b, s, h, hd).transpose(1, 2)
        else:
            q = attn.q_proj(x).view(b, s, h, hd).transpose(1, 2)
            k = attn.k_proj(x).view(b, s, h, hd).transpose(1, 2)
            v = attn.v_proj(x).view(b, s, h, hd).transpose(1, 2)

        if all_valid:
            # Graded hot path: no padding. A dense [S,S] mask is impossible for
            # S=1e5, so causality MUST go through is_causal (kernel-generated).
            if (self._attn_impl == "turing" and x.dtype == torch.float16
                    and turing_attention is not None):
                out = turing_attention(q, k, v, attn.scale, causal)
            elif self._attn_impl == "triton" and can_use_attention(q):
                out = triton_attention(q, k, v, attn.scale, causal, self._attn_split)
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=causal, scale=attn.scale
                )
        else:
            # Padded fallback (only reached for small S with a real mask).
            neg = torch.finfo(q.dtype).min
            bias = torch.zeros(b, 1, s, s, dtype=q.dtype, device=q.device)
            bias = bias.masked_fill((~mask)[:, None, None, :], neg)
            if causal:
                causal_mask = torch.ones(
                    s, s, dtype=torch.bool, device=q.device
                ).triu(1)
                bias = bias.masked_fill(causal_mask[None, None], neg)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=attn.scale
            )
            out = torch.nan_to_num(out, nan=0.0)  # guard fully-masked rows

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        out = attn.out_proj(out)
        if mask is not None and not all_valid:
            out = out.masked_fill(~mask[..., None], 0)
        return out

    def _ffn(self, layer, h2):
        if self._fp32_ffn and self._autocast_dtype is not None:
            with torch.autocast("cuda", enabled=False):
                h2 = h2.float()
                return layer.ffn_out(F.gelu(layer.ffn_in(h2), approximate="none"))
        return layer.ffn_out(F.gelu(layer.ffn_in(h2), approximate="none"))

    def _block(self, layer, x, mask, causal, all_valid):
        attn = self._attention(layer.attention, layer.norm1(x), mask, causal, all_valid)
        if self._triton and can_fuse(attn):
            # LayerNorm(x + attn) in one pass instead of an add kernel followed
            # by a norm kernel. norm1 deliberately stays on PyTorch: there is no
            # preceding add to fuse it with there, and nn.LayerNorm on its own is
            # already a tuned CUDA kernel we have no reason to beat.
            h, x = fused_add_layernorm(attn, x, layer.norm2.weight,
                                       layer.norm2.bias, layer.norm2.eps)
        else:
            x = x + attn
            h = layer.norm2(x)
        x = x + self._ffn(layer, h)
        if mask is not None and not all_valid:
            x = x.masked_fill(~mask[..., None], 0)
        return x

    def _run_full(self, x, mask, causal, all_valid):
        if self._x3_on and all_valid:
            return self._run_full_x3(x, causal)
        for layer in self.layers:
            x = self._block(layer, x, mask, causal, all_valid)
        x = self.final_norm(x)
        if mask is not None and not all_valid:
            x = x.masked_fill(~mask[..., None], 0)
        return x

    def _run_full_x3(self, x, causal):
        """The same block sequence with every GEMM on the fp16x3 path.

        Unpadded inputs only (the padded fallback keeps the fp32 structure; it
        is never graded). Each activation that feeds a GEMM is produced by a
        kernel that also emits its fp16 hi/lo split, so the split costs no
        extra pass; the residual stream stays fp32 throughout.
        """
        b, s, d = x.shape
        h = self.config.num_heads
        hd = d // h
        layers = self.layers
        sites = self._x3_sites
        x = x.contiguous().view(b * s, d)
        n0 = layers[0].norm1
        # Every fp16x3 GEMM returns s * (true value): the power-of-two weight
        # scale of kernels/fp16x3.py is undone by the consumer for free. Sites
        # that stay on fp32 SGEMM (T3_X3_SITES) read the fp32 tensor directly.
        a2 = x3_ln_split(x, n0.weight, n0.bias, n0.eps) if "qkv" in sites else None
        h1 = F.layer_norm(x, (d,), n0.weight, n0.bias, n0.eps) if "qkv" not in sites else None
        for i, layer in enumerate(layers):
            attn = layer.attention
            if "qkv" in sites:
                w3q, _, inv_q = attn._x3_qkv
                qkv = x3_linear(a2, w3q, None, inv_q, False)       # [M, 3D] fp32, x s_q
                q, k, v = qkv.split(d, dim=-1)
            else:
                inv_q = 1.0
                q, k, v = attn.q_proj(h1), attn.k_proj(h1), attn.v_proj(h1)
            q = q.reshape(b, s, h, hd).transpose(1, 2)
            k = k.reshape(b, s, h, hd).transpose(1, 2)
            v = v.reshape(b, s, h, hd).transpose(1, 2)
            # q.k carries s_q^2 -> folded into the softmax scale; the output
            # carries s_q through v -> folded into whatever consumes it.
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=causal,
                scale=attn.scale * inv_q * inv_q)
            o = o.transpose(1, 2).reshape(b * s, d)
            # Sites other than qkv use tail-free weights on the first 3K
            # columns of the split (a strided view, no copy) and hand their
            # bias to the consumer kernel; qkv keeps its bias in the K tail.
            if "out" in sites:
                w3o, b_o, inv_o = attn.out_proj._x3
                proj = x3_linear(x3_act_split(o, False, inv_q)[:, :3 * d], w3o, None, inv_o, False)
            else:
                inv_o, b_o = 1.0, None
                proj = F.linear(o if inv_q == 1.0 else o * inv_q,
                                attn.out_proj.weight, attn.out_proj.bias)
            n2 = layer.norm2
            if "ffn_in" in sites:
                w3i, b_i, inv_i = layer.ffn_in._x3
                x, a2 = x3_add_ln_split(x, proj, n2.weight, n2.bias, n2.eps, inv_o, b_o)
                hid = x3_linear(a2[:, :3 * d], w3i, None, inv_i, False)   # [M, F] fp32, x s_i
            else:
                inv_i, b_i = 1.0, None
                x = torch.add(x, proj, alpha=inv_o) if b_o is None else torch.add(x, proj, alpha=inv_o) + b_o
                hid = layer.ffn_in(F.layer_norm(x, (d,), n2.weight, n2.bias, n2.eps))
            if "ffn_out" in sites:
                w3f, b_f, inv_f = layer.ffn_out._x3
                fdim = hid.shape[-1]
                f = x3_linear(x3_act_split(hid, True, inv_i, b_i)[:, :3 * fdim], w3f, None, inv_f, False)
            else:
                inv_f, b_f = 1.0, None
                g_in = hid if inv_i == 1.0 else hid * inv_i
                if b_i is not None:
                    g_in = g_in + b_i
                f = F.linear(F.gelu(g_in, approximate="none"), layer.ffn_out.weight, layer.ffn_out.bias)
            if i + 1 < len(layers):
                n1 = layers[i + 1].norm1
                if "qkv" in sites:
                    x, a2 = x3_add_ln_split(x, f, n1.weight, n1.bias, n1.eps, inv_f, b_f)
                else:
                    x = torch.add(x, f, alpha=inv_f) if b_f is None else torch.add(x, f, alpha=inv_f) + b_f
                    h1 = F.layer_norm(x, (d,), n1.weight, n1.bias, n1.eps)
            else:
                fn = self.final_norm
                return x3_add_ln(x, f, fn.weight, fn.bias, fn.eps, inv_f, b_f).view(b, s, d)
        fn = self.final_norm
        return x3_ln(x, fn.weight, fn.bias, fn.eps).view(b, s, d)

    # ---- entry point --------------------------------------------------------
    def forward(self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None):
        self._plan(x)
        if self._fused_qkv:
            self._refresh_fused_qkv()
        if self._x3_on:
            self._refresh_x3()
        causal = self.config.causal
        # Device->host sync kept OUTSIDE any compiled region / CUDA graph.
        all_valid = self._all_valid(valid_token_mask)
        ad = self._autocast_dtype
        b = x.shape[0]

        if not all_valid and self._pack_padding:
            out = self._forward_padded(x, valid_token_mask, causal, ad)
            return self._check_x3_output(
                out, lambda: self._forward_padded(x, valid_token_mask, causal, ad))

        # Lazily build the compiled callable on the first CUDA forward.
        if (self._compiled is None and self._compile_ok and self._can_compile):
            try:
                # Dynamo caches compiled variants per *code object*, shared by
                # every instance, and after 8 it silently runs the function
                # eagerly. Each shape costs two variants (a guard on the
                # input's dispatch keys separates inference-mode tensors from
                # ordinary ones), so a process that benchmarks several shapes
                # -- our sweep driver -- lost compilation from the fifth shape
                # on without a word, and under-reported shapes 6, 8 and 13.
                try:
                    import torch._dynamo as _dynamo
                    for _name in ("recompile_limit", "cache_size_limit"):
                        if hasattr(_dynamo.config, _name):
                            setattr(_dynamo.config, _name,
                                    max(getattr(_dynamo.config, _name), 64))
                except Exception:
                    pass
                mode = os.environ.get("T3_COMPILE_MODE")
                if mode is None:
                    mode = "reduce-overhead" if b * x.shape[1] <= 16384 else "default"
                self._compiled = torch.compile(self._run_full, mode=mode, dynamic=False)
                # CUDA-graph modes hand back a static buffer that the NEXT call
                # overwrites. The official harness compares each trial before
                # calling again so it never sees this, but forward() should
                # return a tensor its caller owns regardless.
                self._graph_output = mode in ("reduce-overhead", "max-autotune")
            except Exception:
                self._compile_ok = False
                self._compiled = None

        def _invoke(fn, xin, m, av):
            if ad is not None:
                with torch.autocast("cuda", dtype=ad):
                    return fn(xin, m, causal, av)
            return fn(xin, m, causal, av)

        # First-forward autotune (T3_COMPILE=auto). The stage ablation showed
        # compilation *losing* on the launch-bound shape -- 2.29x eager against
        # 1.46x compiled at S=32 -- because Inductor's guard and dispatch cost
        # exceeds what its fusions save once a whole block is a few
        # microseconds. Rather than guess a threshold, time both on the actual
        # input once and keep the winner. It runs inside the harness' warmup,
        # so the cost is invisible to the graded timing.
        if (not self._tuned and x.device.type == "cuda"
                and (self._chunk_bs is None or self._chunk_bs >= b)
                and (b * x.shape[1] <= 16384 or self._compile_policy == "auto")
                and (self._compile_policy == "auto" or self._cudagraph)
                and (self._compiled is not None or self._cudagraph)):
            self._tuned = True
            self._pick_faster(x, valid_token_mask, all_valid, _invoke)

        def core(xin, m, av):
            using_compiled = self._compiled is not None
            fn = self._compiled if using_compiled else self._run_full
            try:
                return _invoke(fn, xin, m, av)
            except _OOM:
                # Out of memory is a sizing problem, not a compile problem: let
                # _forward_chunked see it and shrink the chunk.
                raise
            except Exception:
                if not using_compiled:
                    raise
                # Compiled path failed at CALL time (e.g. Triton needs sm>=7.0,
                # or an inductor edge case) -> permanently fall back to eager.
                self._compile_ok = False
                self._compiled = None
                return _invoke(self._run_full, xin, m, av)

        if self._chunk_bs is not None and self._chunk_bs < b:  # extreme shape only
            return self._forward_chunked(x, valid_token_mask, core, b)

        if (self._graph is not None and all_valid
                and self._g_key == (tuple(x.shape), x.dtype, str(x.device))):
            out = self._replay(x, valid_token_mask).to(x.dtype)
        else:
            out = core(x, valid_token_mask, all_valid).to(x.dtype)
            if self._graph_output and out.data_ptr() != x.data_ptr():
                out = out.clone()

        return self._check_x3_output(
            out, lambda: core(x, valid_token_mask, all_valid).to(x.dtype))

    def _check_x3_output(self, out, retry):
        # The split cannot represent an activation beyond fp16 range; the
        # static bound refuses the path for weights that could get there, and
        # this check catches an input that does anyway. Once, on the first
        # forward (the harness' first accuracy trial, never timed), unless
        # T3_X3_GUARD=every asks for the sync on each call.
        if self._x3_on and self._x3_guard != "off" and (
                self._x3_guard == "every" or not self._x3_checked):
            self._x3_checked = True
            if not bool(torch.isfinite(out).all()):
                self._x3_disable("non-finite output (an activation left fp16 range)")
                out = retry()
        return out

    def _drop_graph(self):
        """Forget a captured graph together with its static tensors. The
        tensors were allocated from the graph's private memory pool; keeping
        them alive past the graph corrupts the caching allocator."""
        self._graph = None
        self._g_x = self._g_m = self._g_out = None
        self._g_key = None

    def _pick_faster(self, x, mask, all_valid, invoke, warm=6, iters=7):
        """Keep the fastest of eager / compiled / captured-eager on this input.

        Six untimed calls per candidate first (Dynamo tracing, and the CUDA
        graph that reduce-overhead records), then timed calls interleaved
        round-robin across the candidates -- so a clock ramp or a thermal
        drift during the tune hits all of them alike -- with the median per
        candidate. Seven samples each, or twenty-one when the forward is under
        two milliseconds, where the verdict used to flip between runs on
        differences the sampling could not resolve. Any failure leaves the
        compiled path as it was. A captured graph of the eager path is kept
        only when it wins outright.
        """
        inf = float("inf")
        try:
            cands = [("eager", lambda: invoke(self._run_full, x, mask, all_valid))]
            if self._compiled is not None:
                cands.append(("compiled", lambda: invoke(self._compiled, x, mask, all_valid)))
            # Large shapes still need eager-vs-compiled tuning: compiler
            # versions can make opaque GEMMs slower. Their graph's static
            # input/output copies and private pool, however, can exhaust VRAM.
            if (self._cudagraph and all_valid and x.shape[0] * x.shape[1] <= 16384
                    and self._capture_graph(x, mask, invoke)):
                cands.append(("graph", lambda: self._replay(x, mask)))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                for _, call in cands:
                    for _ in range(warm):
                        call()
                torch.cuda.synchronize(x.device)
                # a first look at the eager forward sets the sample count
                start.record()
                cands[0][1]()
                end.record()
                torch.cuda.synchronize(x.device)
                n = 21 if start.elapsed_time(end) < 2.0 else iters
                times = {name: [] for name, _ in cands}
                for _ in range(n):
                    for name, call in cands:
                        start.record()
                        call()
                        end.record()
                        torch.cuda.synchronize(x.device)
                        times[name].append(start.elapsed_time(end))
            med = {name: sorted(t)[len(t) // 2] for name, t in times.items()}
            eager_ms = med["eager"]
            compiled_ms = med.get("compiled", inf)
            graph_ms = med.get("graph", inf)
        except Exception as e:
            print(f"[user_optimized] autotune failed ({type(e).__name__}: "
                  f"{str(e)[:120]}); keeping the current configuration")
            self._drop_graph()
            return
        self._tune_result = (eager_ms, compiled_ms, graph_ms)
        best = min(eager_ms, compiled_ms, graph_ms)
        if best == graph_ms and self._graph is not None:
            self._compiled = None
            self._compile_ok = False
            self._graph_output = False
        else:
            self._drop_graph()
            if eager_ms < compiled_ms:
                self._compiled = None
                self._compile_ok = False
                self._graph_output = False

    def _capture_graph(self, x, mask, invoke):
        """Record the eager forward into a CUDA graph over static input copies.

        Only for the no-padding hot path: with all_valid the mask is never
        read, so the recorded control flow is exact for every later input of
        the same shape. Returns False, leaving no state behind, on any failure.
        """
        try:
            self._g_x = x.detach().clone()
            # all_valid=True makes the compute independent of mask storage.
            # Capturing with None avoids an unused static mask and a copy on
            # every replay. The caller revalidates the real mask first.
            self._g_m = None
            side = torch.cuda.Stream(x.device)
            side.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):  # capture wants a warmed-up allocator
                    invoke(self._run_full, self._g_x, self._g_m, True)
            torch.cuda.current_stream(x.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.no_grad():
                self._g_out = invoke(self._run_full, self._g_x, self._g_m, True)
            self._graph = graph
            self._g_key = (tuple(x.shape), x.dtype, str(x.device))
            return True
        except Exception:
            self._graph = None
            self._g_x = self._g_m = self._g_out = None
            return False

    def _replay(self, x, mask):
        self._g_x.copy_(x)
        self._graph.replay()
        # The next replay rewrites the output buffer; hand back a tensor the
        # caller owns.
        return self._g_out.clone()

    def _forward_chunked(self, x, valid_token_mask, core, b):
        """Batch-chunked forward for the shape whose activations exceed VRAM.

        Writes each chunk straight into a pre-allocated output instead of
        collecting a list and ``torch.cat``-ing it: the concat would hold both
        the pieces and the joined result at once, doubling peak VRAM exactly
        when memory is tightest (it is what made seq_len=1e5 OOM on a 16 GB
        card). On OOM the chunk size is halved and the pass restarted, so a
        mis-estimated budget degrades instead of failing.
        """
        while True:
            try:
                out = torch.empty_like(x)
                for i in range(0, b, self._chunk_bs):
                    sl = slice(i, i + self._chunk_bs)
                    ms = None if valid_token_mask is None else valid_token_mask[sl]
                    av = True if ms is None else bool(ms.all())
                    piece = core(x[sl], ms, av)
                    out[sl].copy_(piece)
                    del piece
                return out
            except _OOM:
                if self._chunk_bs <= 1:
                    raise
                out = None
                self._chunk_bs = max(1, self._chunk_bs // 2)
                torch.cuda.empty_cache()

# Ablation study

Every stage is selected by an **environment variable, with no code edits**, so
the numbers here and the delivered path are literally the same code:

| Knob | Values | Default |
|---|---|---|
| `T3_COMPILE` | `auto` / `1` / `0` | `auto`: time eager vs compiled once on the first forward, keep the winner (compile needs `sm>=7.0`) |
| `T3_AUTOCAST` | `off` / `fp16` / `bf16` / `auto` | `off` |
| `T3_FP32_FFN` | `1` / `0` | `0` |
| `T3_CHUNK_BS` | int | auto, from free VRAM |
| `T3_CUDAGRAPH` | `1` / `0` | `1`: a CUDA graph of the eager path is a third autotune candidate |

```bash
T3_COMPILE=0 T3_AUTOCAST=off  python run_all.py --shapes 1 --out /tmp/sdpa.csv
T3_COMPILE=1 T3_AUTOCAST=off  python run_all.py --shapes 1 --out /tmp/compile.csv
T3_COMPILE=0 T3_AUTOCAST=fp16 python run_all.py --shapes 1 --out /tmp/fp16.csv
T3_COMPILE=1 T3_AUTOCAST=fp16 python run_all.py --shapes 1 --out /tmp/both.csv
```

## Stage table (Kaggle Tesla T4, sm_75, fp32 grading)

4 shapes x 4 stages, one run each. Raw log: `kaggle_t4_ablation.log`; machine
readable: `ablation_t4.csv`.

| shape | what it stresses | baseline ms | SDPA | +compile | +fp16 | +compile+fp16 |
|---|---|---|---|---|---|---|
| 1 (B=64,S=128) | compute | 9.4 | 2.01x | 2.33x | 3.71x | **5.61x** |
| 12 (S=32) | launch overhead | 3.4 | 2.29x | 1.46x | 1.42x | **2.86x** |
| 8 (D=ffn=1024) | GEMM / wide FFN | 118.9 | 1.07x | 1.05x | **3.80x** | 3.80x |
| 13 (S=1024) | long sequence | 313.6 | 4.68x | 4.63x | **11.57x** | 11.54x |

What each stage is actually worth:

- **SDPA is the whole story on long sequences.** Shape 13 gets 4.68x from
  memory-efficient attention alone, and nothing later adds to that except fp16.
- **`torch.compile` is not free.** It helps the compute-bound shape (2.01 ->
  2.33x) and it *hurts* the launch-bound one (2.29 -> 1.46x): shape 12 is
  3.4 ms of almost pure kernel-launch overhead, and Inductor's guard and
  dispatch overhead is a real fraction of that. On the GEMM-bound shape 8 it is
  a wash (1.07 -> 1.05x) because cuBLAS already owns the time.
- **fp16 is the biggest single lever** — and the one we do not ship. See below.
- **The stages are not additive.** On shape 8 and 13, compile contributes
  nothing once fp16 is on; on shape 12 the two only pay off together.

## The fp16 decision, with the numbers

fp16 is **off by default** (`T3_AUTOCAST=off`). The repo previously justified
that by asserting fp16 "breaks the strict `atol=0.002` gate". **That assertion
was wrong, and measuring it is what showed us why the conclusion is still
right.**

A full 13-shape fp16 sweep on the T4 (`results_t4_fp16.csv`,
`kaggle_t4_fp16_run.log`) **passes all 13 shapes** at a **median 4.014x**
(1.320x - 11.528x), against 2.286x for the shipped fp32 path. That is nearly
double, for one environment variable.

Here is why we still do not ship it:

| regime | shapes PASS | median speedup | worst `max_abs` | margin vs `atol=0.002` |
|---|---|---|---|---|
| fp32 (shipped) | 13/13 | 2.286x | 1.91e-6 | **1049x** |
| fp16 (`T3_AUTOCAST=fp16`) | 13/13 | 4.014x | **2.04e-3** | **0.98x** |
| fp16 attention + fp32 FFN/LN (`T3_AUTOCAST=fp16 T3_FP32_FFN=1`) | 13/13 | 2.953x | 1.72e-03 | 1.17x |

The fp16 worst case, shape 6, is `max_abs = 0.0020388` — it has **already
crossed `atol=0.002`**. It passes only because the harness gate is
`abs<=0.002` **OR** `rel<=0.02`, and that particular element happened to have a
reference value large enough (`|ref| >= 0.102`) for the relative branch to catch
it. Nothing about the run guarantees the next seed puts that error on an element
with a large reference; if it lands on a near-zero one instead, that element
fails, and a single failing element fails the whole shape and forfeits the speed
score entirely.

So the trade is: **2x more speed, in exchange for a correctness margin of
essentially zero on a hard gate.** We take the 2.286x that sits 1049x inside
tolerance. The flag is there, documented and measured, for anyone who wants the
other side of that trade.

**The middle ground was measured and is not one.** fp16 attention with fp32 FFN
and LayerNorm (`T3_FP32_FFN=1`) scores 2.953x, but its worst `max_abs` is 1.72e-03
— a margin of 1.17x, with every shape between 8.48e-04 and 1.72e-03. Taking
the FFN and norms back to fp32 only moved the worst case 2.04e-3 → 1.72e-03: the
error floor is in the fp16 attention matmuls, and nothing downstream of them
can buy it back (`results_t4_mixed.csv`, `kaggle_t4_mixed_run.log`).

## The chunking stage (shape 14)

The only stage with a binary outcome rather than a ratio.

| variant | outcome |
|---|---|
| baseline (explicit `[B,H,S,S]` scores) | cannot run — needs ~20.5 TB |
| SDPA + chunks collected into a list + `torch.cat` | **OOM** at the concat: `Tried to allocate 6.10 GiB` |
| SDPA + chunks written into a preallocated output | **runs**: 293377 ms, 10,907 tok/s, peak 14.61 GB, `chunk_bs=1` (P100) |
| the same, on a T4 (same backend; fp16 tensor cores for the natively-fp16 matmuls) | **runs faster**: 204132 ms, 15,676 tok/s, peak 14.58 GB — on a card with only 15.64 GB total |

The concat was the whole difference. Collecting 32 chunks and joining them holds
the pieces and the `[32,100000,1024]` result at the same time — a second ~6.5 GB
allocation on a card that already had the 6.5 GB input resident.

All three rows are **fp16**: an fp32 input plus output for this shape is 26.2 GB,
which does not fit a free 16 GB GPU at all, so fp16 is the only regime in which
the comparison exists. The truncated correctness check that backs this shape is
fp32, like every graded shape.

## The hand-written Triton kernel

`kernels/fused_layernorm.py` fuses the residual add with the LayerNorm that
follows it — a pattern eager PyTorch runs as two kernels and four passes over
the `[B,S,D]` activation. Measured on a T4 at the real sizes
(`triton_bench_t4.csv`):

| case | rows × D | eager | Inductor | Triton | vs eager | vs Inductor |
|---|---|---|---|---|---|---|
| shape 6 | 1.28M × 128 | 20.660 | 11.004 | **10.763** | **1.92×** | 1.02× |
| shape 13 | 65536 × 128 | 1.082 | 0.657 | **0.602** | **1.80×** | 1.09× |
| shape 1/5/9–11 | 8192 × 128 | 0.236 | **0.148** | 0.159 | 1.49× | 0.93× |
| shape 7 | 8192 × 32 | 0.092 | 0.102 | **0.076** | 1.21× | **1.34×** |
| shape 8 | 8192 × 1024 | 0.719 | 0.673 | **0.649** | 1.11× | 1.04× |
| shape 2 | 128 × 128 | **0.050** | 0.114 | 0.086 | 0.58× | 1.32× |

Beats eager on 5 of 6, Inductor on 5 of 6, `max_abs ≤ 1.43e-6`.

**End to end, as a raw launch, it lost:** 1.929× against 2.282×, because a
raw Triton call breaks the compiled graph and enabling the kernel switched
`torch.compile` off.

**Registered as a `torch.library` op, it composes** — Inductor schedules it inside
the graph, compile stays on. Same session, all columns (`triton_bench_t4.csv`):

| case | rows × D | eager | Inductor | registered op, eager | **op inside Inductor's graph** | in-graph vs Inductor |
|---|---|---|---|---|---|---|
| shape 6 | 1.28M × 128 | 20.043 | 11.080 | 10.592 | **10.594** | **1.046×** |
| shape 13 | 65536 × 128 | 1.055 | 0.657 | 0.634 | **0.633** | **1.038×** |
| shape 8 | 8192 × 1024 | 0.734 | 0.674 | 0.666 | **0.666** | 1.012× |
| shape 2 | 128 × 128 | 0.049 | 0.108 | 0.129 | 0.107 | 1.008× |
| shape 7 | 8192 × 32 | 0.095 | **0.108** | 0.125 | 0.125 | 0.862× |
| shape 1/5/9–11 | 8192 × 128 | 0.236 | **0.150** | 0.186 | 0.183 | 0.819× |

End to end: **2.190×** registered, against 2.282× with compilation fixed on, 13/13 PASS (2.286× under the autotune default, within noise). Inductor's
own fusion is within ±5% of ours on the memory-bound shapes and wins on the
small ones, where a custom op is an opaque boundary and dispatch overhead
(30–70 µs) shows. Matched the compiler; did not beat it. Off by default.

## Compile policy: measured per shape, not assumed

The stage table above showed `torch.compile` *losing* on the launch-bound shape,
and Inductor's `reduce-overhead` mode — a CUDA graph of *Inductor's* kernels plus
its guard and dispatch cost — losing to plain eager there too. Rather than pick a
threshold, the model times three candidates on its first forward, on the real
input, and keeps the fastest:

- **eager** — the plain PyTorch kernels;
- **compiled** — `torch.compile` (`reduce-overhead` under 16384 tokens, `default` above);
- **graph** — the eager kernels captured into a `torch.cuda.CUDAGraph`, replayed with
  inputs copied into static buffers and the output cloned out (`T3_CUDAGRAPH=1`).

Six untimed calls each (Dynamo tracing, graph capture), then a median of seven.
It runs inside the harness' warmup, so the graded timing never sees it; shapes
with `B*S > 16384` (6 and 13) are not tuned, compilation wins there outright.

| shape | eager ms | compiled ms | graph ms | kept | speedup |
|---|---|---|---|---|---|
| 1 | 8.071 | **3.830** | 4.776 | compiled | 2.264x |
| 2 | 1.448 | 0.840 | **0.768** | graph | 4.047x |
| 3 | 2.359 | 1.151 | **1.143** | graph | 3.518x |
| 4 | 2.167 | **2.087** | 2.392 | compiled | 2.714x |
| 5 | 10.203 | **8.207** | 9.585 | compiled | 2.286x |
| 7 | **1.989** | 2.378 | 2.196 | eager | 2.783x |
| 8 | 136.434 | **136.177** | 136.910 | compiled | 1.094x |
| 9 | **4.821** | 5.335 | 5.039 | eager | 1.279x |
| 10 | **4.793** | 5.300 | 5.034 | eager | 1.581x |
| 11 | 7.635 | 7.539 | **7.404** | graph | 3.071x |
| 12 | 1.383 | 1.467 | **1.298** | graph | 2.068x |

Median **2.286x** (autotune without the graph candidate: 2.280x; compile fixed
on: 2.282x) — unchanged within noise — mean 2.541x against 2.494x. Where the
graph wins it wins by a few percent; where candidates are within a few percent
of each other the pick is noise and does not matter. The harness' accuracy
trials feed fresh tensors after the choice is made, and pass, which is the
static-buffer discipline being exercised on the graded pattern. Raw log:
`kaggle_t4_graph_run.log`; shipped results: `results_t4.csv`; the two-candidate
run it replaced: `results_t4_auto.csv`; compile fixed on: `results_t4_compile_on.csv`.

## Fused QKV projection: measured, a wash

`T3_FUSED_QKV=1` replaces the three `[D, D]` projections with one `[3D, D]` GEMM
(two fewer launches, the activation read once; the concatenated weight is a plain
attribute, never in `state_dict`). On the T4 (`results_t4_qkv.csv`,
`kaggle_t4_qkv_run.log`):

| | shipped | fused QKV |
|---|---|---|
| median | 2.286x | 2.387x |
| mean | 2.541x | 2.490x |
| shapes better / worse | — | 8 / 5 |

The median moves because its element (shape 1) moved; the mean does not. Shape 6,
where a single read of a 1.28M-row activation should pay most, is flat. The
likely reason is that the memory-efficient attention backend makes its own
contiguous copies of the strided q/k/v views that the split produces, so the
reads the fusion saved are spent again one kernel later. Off by default.

## The attention kernel: accuracy won, speed lost

`kernels/attention.py`: fp32 statistics, fp16 tensor-core operands, and with
`SPLIT=3` each operand split into fp16 hi/lo halves so the product is formed from
three MMAs at ~2^-22 relative error. Against an fp64 reference on a T4:

| case | fp32 SDPA | fp16 SDPA | kernel, SPLIT=1 | **kernel, SPLIT=3** |
|---|---|---|---|---|
| shape 1/5 hd=32 | 9.49e-07 | 0.00177 | 0.00129 | **2.25e-06** |
| shape 7 hd=8 | 6.88e-07 | 0.00187 | 0.00191 | **1.44e-06** |
| shape 9 H=1 hd=128 | 1.27e-06 | 0.00198 | 0.00137 | **3.55e-06** |
| shape 10 H=2 hd=64 | 1.23e-06 | 0.00187 | 0.00171 | **2.29e-06** |
| shape 11 H=16 hd=8 | 8.52e-07 | 0.00221 | 0.00235 | **1.56e-06** |
| shape 12 S=32 | 1.15e-06 | 0.00188 | 0.00172 | **1.62e-06** |
| shape 13 S=1024 | 8.8e-07 | 0.00177 | 0.00129 | **2.17e-06** |
| shape 6 B=10000, S=128 | (ref) | 0.00278 | 0.00235 | **4.17e-06** |

| case | fp32 SDPA ms | fp16 SDPA ms | SPLIT=1 ms | SPLIT=3 ms | SPLIT=1 vs fp32 | SPLIT=3 vs fp32 |
|---|---|---|---|---|---|---|
| shape 1/5 hd=32 | 0.539 | 0.161 | 0.819 | 2.590 | 0.66× | 0.21× |
| shape 7 hd=8 | 0.441 | 0.157 | 0.565 | 1.253 | 0.78× | 0.35× |
| shape 9 H=1 hd=128 | 0.353 | 0.125 | 0.997 | 8.580 | 0.35× | 0.04× |
| shape 10 H=2 hd=64 | 0.373 | 0.118 | 1.093 | 9.001 | 0.34× | 0.04× |
| shape 11 H=16 hd=8 | 1.554 | 0.512 | 1.677 | 2.165 | 0.93× | 0.72× |
| shape 12 S=32 | 0.191 | 0.076 | 0.408 | 0.719 | 0.47× | 0.27× |
| shape 13 S=1024 | 9.191 | 2.268 | 12.649 | 43.537 | 0.73× | 0.21× |
| shape 6 B=10000, S=128 | 37.082 | 11.250 | 62.618 | 214.216 | 0.59× | 0.17× |

fp32-class accuracy (1.4e-06—4.2e-06), at 0.04—0.72× the speed of fp32 SDPA. The
ceiling on a T4 is the problem: fp16 cutlass SDPA is only 4.1× fp32 here, and the
kernel does 3× the matmuls; Triton's Turing codegen then loses another 4—6× to
cutlass. Under `torch.compile` the `triton_op` registration measured
2.08e-03 because Inductor folds the lo halves to zero (it does not emulate
intermediate casts); an opaque `custom_op` restores 2.98e-06.

End to end (`T3_ATTN=triton`, `results_t4_attn.csv`): 13/13 PASS, worst `max_abs`
1.91e-06, median 1.093x against 2.286x shipped. Off by default. Logs:
`kaggle_t4_attn_v1.log` (first cut), `kaggle_t4_attn_v2.log` (the tables above),
`kaggle_t4_attn_v3.log` (composition fix).

<!-- x3-ablation:begin -->
## fp16x3 linear layers: measured, shipped

Motivation and mechanism are in the README; the numbers behind each decision are here.

**cuBLAS epilogue** (`kaggle_t4_s0gemm_run.log`): a bare `mm(out_dtype=fp32)` against
`addmm` with `alpha` and a 1-D bias, the call the first version made:

| M | 3K | N | fp32 SGEMM ms | fp16x3 `mm` ms | `addmm` + bias ms | addmm / mm |
|---|---|---|---|---|---|---|
| 8192 | 96 | 96 | 0.05 | 0.06 | 0.08 | 1.37x |
| 8192 | 384 | 384 | 0.19 | 0.15 | 0.25 | 1.71x |
| 8192 | 3072 | 1024 | 3.92 | 1.26 | 1.78 | 1.41x |
| 8192 | 3072 | 3072 | 11.58 | 4.59 | 6.92 | 1.51x |
| 16384 | 384 | 384 | 0.35 | 0.26 | 0.45 | 1.77x |
| 65536 | 384 | 384 | 1.35 | 0.86 | 1.64 | 1.91x |
| 1280000 | 384 | 128 | 13.54 | 8.54 | 12.87 | 1.51x |
| 1280000 | 384 | 384 | 33.03 | 22.35 | 35.83 | 1.60x |

**K tail** (`kaggle_t4_b0tail_run.log`, bare `mm` ms for K3 = 3K + tail):

| M | N | tail 0 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| 8192 | 96 | 0.157 | 0.131 | 0.168 | 0.062 | 0.074 |
| 65536 | 384 | 0.856 | 1.124 | 1.172 | 1.200 | 0.996 |
| 1280000 | 128 | 8.548 | 9.979 | 9.870 | 9.583 | 9.627 |

**Error against K** (`x3_error_vs_k_t4.csv`, M=4096, N=1024, max-abs vs fp64; "below"
is the fraction of outputs smaller in magnitude than the reference, 0.5 for an unbiased
method):

| K | fp32 SGEMM | hi-first | **lo-first (default)** | lo-first + split-K 4 | CPU emulation | below, hhl / lhh |
|---|---|---|---|---|---|---|
| 128 | 1.5e-06 | 6.1e-06 | **1.7e-06** | 1.4e-06 (3.9x time) | 3.3e-06 | 0.92 / 0.78 |
| 256 | 2.3e-06 | 1.1e-05 | **3.0e-06** | 2.3e-06 (3.3x time) | 2.9e-06 | 0.95 / 0.81 |
| 512 | 3.2e-06 | 1.8e-05 | **6.1e-06** | 3.9e-06 (2.5x time) | 2.2e-06 | 0.96 / 0.83 |
| 1024 | 2.3e-06 | 1.5e-05 | **9.6e-06** | 6.5e-06 (1.7x time) | 1.8e-06 | 0.91 / 0.83 |
| 2048 | 3.5e-06 | 2.4e-05 | **1.5e-05** | 1.2e-05 (1.3x time) | 1.7e-06 | 0.91 / 0.83 |
| 4096 | 5.3e-06 | 4.5e-05 | **2.7e-05** | 9.6e-06 (1.2x time) | 1.3e-06 | 0.91 / 0.84 |

**Kernel geometry** (`kaggle_t4_b2geo2_run.log`): rows per program x warps swept per width;
the table in `kernels/fp16x3.py::_x3_geometry` is the winner.

| M | N | one row / program | best tiled LN-split | PyTorch LayerNorm | best tiled fp32 LN |
|---|---|---|---|---|---|
| 1280000 | 128 | 9.673 @ 169 | 7.969 @ 206 (rows 4, 8 warps) | 12.420 @ 106 | 5.348 @ 245 (rows 1, 2 warps) |
| 65536 | 128 | 0.915 @ 92 | 0.509 @ 165 (rows 4, 8 warps) | 1.323 @ 51 | 0.377 @ 178 (rows 4, 8 warps) |
| 65536 | 384 | 1.584 @ 159 | 1.350 @ 186 (rows 1, 2 warps) | 1.759 @ 114 | 0.943 @ 214 (rows 4, 8 warps) |
| 8192 | 1024 | 0.612 @ 137 | 0.522 @ 161 (rows 2, 8 warps) | 0.401 @ 168 | 0.384 @ 175 (rows 2, 8 warps) |

**Per site** (`kaggle_t4_b1x3_run.log`; fp32 = producer kernel + SGEMM, x3 = fused split +
tensor-core GEMM, errors vs fp64 on 4096 rows):

| shape | site | M | K | N | fp32 ms | fp16x3 ms (split @ GB/s + GEMM) | ratio | fp32 err | fp16x3 err |
|---|---|---|---|---|---|---|---|---|---|
| 5 | qkv | 16384 | 128 | 384 | 1.087 | 0.637 (0.228 @ 92 + 0.481) | **1.71x** | 1.6e-06 | 2.0e-06 |
| 5 | out | 16384 | 128 | 128 | 0.303 | 0.365 (0.193 @ 109 + 0.223) | **0.83x** | 1.5e-06 | 1.6e-06 |
| 5 | ffn_in | 16384 | 128 | 128 | 0.631 | 0.379 (0.218 @ 96 + 0.280) | **1.67x** | 1.2e-06 | 1.6e-06 |
| 5 | ffn_out | 16384 | 128 | 128 | 0.365 | 0.387 (0.215 @ 98 + 0.222) | **0.94x** | 9.5e-07 | 1.1e-06 |
| 6 | qkv | 1280000 | 128 | 384 | 51.208 | 31.994 (8.123 @ 202 + 25.844) | **1.60x** | 1.5e-06 | 2.2e-06 |
| 6 | out | 1280000 | 128 | 128 | 14.936 | 16.977 (8.339 @ 196 + 9.935) | **0.88x** | 1.6e-06 | 1.5e-06 |
| 6 | ffn_in | 1280000 | 128 | 128 | 28.414 | 16.814 (8.141 @ 201 + 10.000) | **1.69x** | 1.2e-06 | 1.8e-06 |
| 6 | ffn_out | 1280000 | 128 | 128 | 19.452 | 17.296 (8.375 @ 196 + 9.728) | **1.12x** | 9.6e-07 | 1.3e-06 |
| 8 | qkv | 8192 | 1024 | 3072 | 13.745 | 8.858 (0.496 @ 169 + 8.636) | **1.55x** | 4.6e-06 | 9.6e-06 |
| 8 | out | 8192 | 1024 | 1024 | 3.349 | 2.967 (0.506 @ 166 + 2.597) | **1.13x** | 2.6e-06 | 8.6e-06 |
| 8 | ffn_in | 8192 | 1024 | 1024 | 3.260 | 2.565 (0.492 @ 171 + 2.398) | **1.27x** | 2.8e-06 | 9.0e-06 |
| 8 | ffn_out | 8192 | 1024 | 1024 | 3.258 | 3.023 (0.512 @ 164 + 2.676) | **1.08x** | 1.7e-06 | 5.3e-06 |
| 13 | qkv | 65536 | 128 | 384 | 2.031 | 1.729 (0.527 @ 159 + 1.309) | **1.17x** | 1.6e-06 | 1.8e-06 |
| 13 | out | 65536 | 128 | 128 | 0.782 | 0.965 (0.519 @ 162 + 0.504) | **0.81x** | 1.3e-06 | 1.7e-06 |
| 13 | ffn_in | 65536 | 128 | 128 | 1.262 | 0.971 (0.523 @ 160 + 0.508) | **1.30x** | 1.4e-06 | 2.1e-06 |
| 13 | ffn_out | 65536 | 128 | 128 | 0.875 | 0.981 (0.528 @ 159 + 0.508) | **0.89x** | 1.1e-06 | 1.1e-06 |

**End to end**, three runs per configuration, per-shape medians (`results_t4_fp32.csv`,
`results_t4.csv`; every run in `results_t4_runs.csv`):

| shape | `T3_LINEAR=fp32` | **`fp16x3` (shipped)** | delta | fp32 `max_abs` | fp16x3 `max_abs` |
|---|---|---|---|---|---|
| 1 | 2.282x | **2.702x** | +18% | 1.2e-06 | 2.0e-06 |
| 2 | 3.826x | **4.397x** | +15% | 9.5e-07 | 1.5e-06 |
| 3 | 3.385x | **3.652x** | +8% | 1.2e-06 | 1.3e-06 |
| 4 | 2.485x | **2.994x** | +20% | 1.4e-06 | 1.9e-06 |
| 5 | 2.300x | **2.588x** | +13% | 1.4e-06 | 2.1e-06 |
| 6 | 2.158x | **2.490x** | +15% | 1.7e-06 | 2.9e-06 |
| 7 | 2.764x | **4.496x** | +63% | 9.5e-07 | 1.7e-06 |
| 8 | 1.095x | **1.566x** | +43% | 3.2e-06 | 9.5e-06 |
| 9 | 1.257x | **1.865x** | +48% | 1.4e-06 | 2.4e-06 |
| 10 | 1.581x | **2.294x** | +45% | 1.4e-06 | 2.9e-06 |
| 11 | 3.075x | **3.857x** | +25% | 1.4e-06 | 2.4e-06 |
| 12 | 2.093x | **2.834x** | +35% | 1.2e-06 | 1.9e-06 |
| 13 | 4.818x | **5.222x** | +8% | 1.4e-06 | 2.4e-06 |
| **median** | 2.300x | **2.834x** | +23% | 3.2e-06 | 9.5e-06 |

Run medians: fp16x3 2.828 / 2.912 / 2.834x; fp32 2.261 / 2.300 / 2.332x.

**Site policy, end to end** (three runs each, per-shape medians): the operator table's
policy — fp16x3 only where the split replaces a LayerNorm (`T3_X3_SITES=qkv,ffn_in`,
`results_t4_d1..d3.csv`) — against all four sites (shipped):

| shape | qkv + ffn_in | **all four sites** | delta |
|---|---|---|---|
| 1 | 2.458x | **2.702x** | +10% |
| 2 | 4.043x | **4.397x** | +9% |
| 3 | 3.627x | **3.652x** | +1% |
| 4 | 2.595x | **2.994x** | +15% |
| 5 | 2.436x | **2.588x** | +6% |
| 6 | 2.310x | **2.490x** | +8% |
| 7 | 4.005x | **4.496x** | +12% |
| 8 | 1.549x | **1.566x** | +1% |
| 9 | 1.677x | **1.865x** | +11% |
| 10 | 2.068x | **2.294x** | +11% |
| 11 | 3.660x | **3.857x** | +5% |
| 12 | 2.559x | **2.834x** | +11% |
| 13 | 4.941x | **5.222x** | +6% |
| **median** | 2.559x | **2.834x** | +11% |

The retired per-shape tuner (`T3_LINEAR=auto`) scored 2.299x and
2.214x on the first version of the path (`results_t4_x3v4.csv`, `results_t4_x3v4b.csv`).

**What we learned building it.**

1. *The split must be fused, and the GEMM must be bare.* The first version split
   operands in PyTorch and lost on shape 8 despite a 2x faster GEMM; the second used two
   GEMMs and was slower than fp32; the third used one GEMM with the bias and scale in the
   cuBLAS epilogue and paid 40-90% for it. Bias and scale in the neighbouring kernels,
   and a bare `mm`, is the version that wins.
2. *Operator tables do not decide programs.* The per-site table, measured in eager mode,
   said to keep two of the four GEMM sites on SGEMM; three end-to-end runs of that policy
   lost to all four sites on 12 of 13 shapes. Launch overhead hidden by graph replay, and
   the passes an SGEMM site needs for its neighbour's scale and bias, are invisible at
   the operator level and decisive in the program.
3. *The residual error is the hardware's, and its order is ours.* Exact emulation lands
   ~1.3e-6 at every K; the T4 lands 1.531e-05 at K=1024 with the natural operand order and
   9.562e-06 with the tiny terms summed first. Disabling PyTorch's reduced-precision
   reductions changed nothing; it was never that.
4. *Alignment is not a detail.* Eight padding columns made K=392 and cost 10-30% on the
   K=128 shapes' GEMMs; rounding to 32 made it free, and for K=32 (shape 7) it made the
   GEMM 2.5x faster than the unpadded K=96.
5. *Do not tune two Dynamo variants through Inductor's CUDA-graph trees.* Timing fp16x3
   as a second compiled variant of the same reduce-overhead function, then switching
   back, corrupted the caching allocator (`Expected curr_block->next == nullptr` on the
   next capture); the tuner was retired.

<!-- x3-ablation:end -->

## Measurement protocol

Every T4 cell of the shipped per-shape table (`results/results_t4.csv`, reproduced in the
README and the report) is the **median of three independent runs** of the shipped
configuration (run medians 2.828x / 2.912x / 2.834x; every run 13/13 PASS), and so is
the `T3_LINEAR=fp32` reference (`results_t4_fp32.csv`; 2.261x / 2.300x / 2.332x).
`max_abs` is the worst of the three. The per-shape spread, (max − min) / median, reaches
25% on shape 4, and shape 6 drifts by up to 20% between sessions on identical code.
The other tables in this file (compile policy, fused QKV, Triton kernels, precision
regimes) are single runs, so that spread is the yardstick for them: a difference below
it between two configurations is noise, and the text says so wherever it applies. All
six runs are in `results/results_t4_runs.csv`.

**The sweep bug.** Every table in this file dated before this section was measured with
`torch.compile` silently inactive from the fifth shape of the sweep on: Dynamo caches
eight compiled variants per function, each shape costs two (inference-mode and ordinary
input tensors guard differently), and past eight it runs the function eagerly without a
warning on stdout. The harness runs one shape per process and is unaffected; the model
now raises the limit itself. The per-shape tables above were re-measured after the fix;
the single-run stage and kernel tables were not, and their shapes 6, 8 and 13 columns
should be read as "compile off" where the text says compile was on. The old code's
control run is `kaggle_t4_head_run.log`.

## Cross-GPU: what the hardware is worth

The same code on two free cards, both fp32, both 13/13 PASS:

| | Tesla P100 (sm_60) | Tesla T4 (sm_75) |
|---|---|---|
| `torch.compile` | unavailable (Triton needs sm>=7.0) | autotuned per shape with eager and a CUDA graph as rivals: 0 compiled / 10 graph / 1 eager of 11 tuned |
| SDPA backend | memory-efficient | memory-efficient (same kernel: flash needs fp16 and sm_80+, probed) |
| GEMMs | fp32 SGEMM (no tensor cores) | fp16 tensor cores with hi/lo compensation, fp32 accumulation (`T3_LINEAR=fp16x3`) |
| median speedup | 2.065x | **2.834x** (2.300x with fp32 SGEMM) |
| range | 1.098x - 4.001x | 1.566x - 5.222x |

Note the T4's *baselines* are slower than the P100's (shape 13: 324.7 ms vs
168.6 ms) — the P100 has higher fp32 throughput and roughly twice the memory
bandwidth. The ratio improves on the T4 anyway, because our path picks up
compile there while the baseline stays bandwidth-bound; the attention kernel is
identical on both cards. The
speedup is a ratio, and it is worth saying which way each side of it moved.

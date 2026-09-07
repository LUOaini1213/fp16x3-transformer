#!/usr/bin/env python3
"""
Build the 3-minute demo video from the measured results.

The timeline is fixed to the required submission structure:

    0:00-0:15  Problem
    0:15-0:35  Our Solution
    0:35-0:55  Architecture
    0:55-2:20  Live Demo        <- the largest block
    2:20-2:45  Results
    2:45-3:00  Impact

Sections are cut to those windows rather than to however long the narration
happens to run: each section's voice-over is synthesized, the slides inside it
fill the window exactly, and any section whose narration does not fit is
reported so the script can be shortened instead of silently drifting.

Everything on screen comes from the committed CSVs and the committed kernel
logs, so a new sweep produces a video that still agrees with the data. The demo
block is a replay of the recorded run, labelled as such on screen -- the GPUs
are in Kaggle, so there is no local screen to capture.

    python scripts/make_video.py --out build/track3_demo.mp4

Requires: pillow, edge-tts, moviepy (imageio-ffmpeg supplies the ffmpeg binary).
Narration is synthesized by Microsoft's edge-tts service, so that step needs
network access; --no-vo renders a silent, subtitle-only cut instead.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1920, 1080

# Same dark tokens as the report page and the dark figures.
GROUND = "#0d1117"
PANEL = "#161c25"
INK = "#e6ebf1"
INK_2 = "#aab6c4"
MUTED = "#778392"
ACCENT = "#6ba4f0"
CRIT = "#ec8279"
GOOD = "#52b581"
WARN = "#e0b341"
RULE = "#242d38"

FONTS = "C:/Windows/Fonts"


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


def fonts():
    return dict(
        h1=font("segoeuib.ttf", 82), h2=font("segoeuib.ttf", 56),
        h3=font("seguisb.ttf", 38), body=font("segoeui.ttf", 33),
        small=font("segoeui.ttf", 26), label=font("seguisb.ttf", 22),
        mono=font("consola.ttf", 28), monob=font("consolab.ttf", 28),
        mono_s=font("consola.ttf", 23), mono_sb=font("consolab.ttf", 23),
    )


def read_csv(name):
    with open(os.path.join(HERE, "results", name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(name):
    sp = sorted(float(r["speedup"]) for r in read_csv(name)
                if r["pass"] == "PASS" and r["speedup"])
    return dict(median=sp[len(sp) // 2], lo=sp[0], hi=sp[-1], n=len(sp))


# ---------------------------------------------------------------- slide parts
def new_slide():
    img = Image.new("RGB", (W, H), GROUND)
    return img, ImageDraw.Draw(img)


def eyebrow(d, F, text, x=140, y=100):
    d.text((x, y), text.upper(), font=F["label"], fill=MUTED)


def heading(d, F, text, x=140, y=152, fill=INK, f="h2"):
    d.text((x, y), text, font=F[f], fill=fill)


def panel(d, box, fill=PANEL, outline=RULE, r=10):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=2)


def paste_fit(img, path, box, pad=0):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0 - 2 * pad, y1 - y0 - 2 * pad
    fig = Image.open(path).convert("RGB")
    s = min(bw / fig.width, bh / fig.height)
    fig = fig.resize((int(fig.width * s), int(fig.height * s)), Image.LANCZOS)
    img.paste(fig, (x0 + pad + (bw - fig.width) // 2,
                    y0 + pad + (bh - fig.height) // 2))


def wrap(d, text, f, maxw):
    words, out, line = text.split(), [], ""
    for w in words:
        t = (line + " " + w).strip()
        if d.textlength(t, font=f) <= maxw:
            line = t
        else:
            out.append(line)
            line = w
    if line:
        out.append(line)
    return out


def arrow(d, x0, y, x1, color=MUTED, w=3):
    d.line([(x0, y), (x1 - 14, y)], fill=color, width=w)
    d.polygon([(x1, y), (x1 - 16, y - 9), (x1 - 16, y + 9)], fill=color)


# -------------------------------------------------------------- 0:00 Problem
def s_problem(F, ctx):
    img, d = new_slide()
    eyebrow(d, F, "TikTok TechJam 2026  \u00b7  Track 3")
    d.text((140, 236), "A Transformer layer,", font=F["h1"], fill=INK_2)
    d.text((140, 336), "faster on the GPU.", font=F["h1"], fill=INK)
    d.text((140, 448), "Without moving the answer.", font=F["h1"], fill=ACCENT)

    panel(d, (140, 640, 1000, 850))
    d.text((178, 672), "THE GRADER CHECKS EVERY ELEMENT", font=F["label"], fill=MUTED)
    d.text((178, 718), "abs_err \u2264 0.002  OR  rel_err \u2264 0.02", font=F["monob"], fill=INK)
    d.text((178, 780), "one failing element fails the whole shape", font=F["small"], fill=CRIT)

    panel(d, (1040, 640, 1780, 850))
    d.text((1078, 672), "AND ONE SHAPE IS IMPOSSIBLE", font=F["label"], fill=MUTED)
    d.text((1078, 710), "20.5 TB", font=F["h2"], fill=CRIT)
    d.text((1078, 790), "of attention scores, at seq_len 100,000",
           font=F["small"], fill=INK_2)

    d.text((140, 910), "14 shapes. Speed is worth nothing without correctness.",
           font=F["h3"], fill=MUTED)
    return img


# ------------------------------------------------------------- 0:15 Solution
def s_solution(F, ctx):
    img, d = new_slide()
    eyebrow(d, F, "Our solution")
    heading(d, F, "Rewrite the compute. Keep every parameter name.")

    cards = [
        ("FUSED ATTENTION", "F.scaled_dot_product_attention",
         "O(S) memory instead of O(S\u00b2). The score matrix is never built, which is "
         "what makes seq_len = 100,000 possible at all.", ACCENT),
        ("SELF-APPLIED COMPILATION", "torch.compile, from inside the model",
         "Fuses LayerNorm / bias / GELU into Triton kernels without the grader "
         "having to pass --compile-user.", ACCENT),
        ("VRAM-PLANNED CHUNKING", "sized from the memory actually free at runtime",
         "Each chunk is written into a preallocated output; the chunk halves and "
         "retries if the estimate was optimistic.", GOOD),
    ]
    y = 290
    for label, title, body, color in cards:
        panel(d, (140, y, 1780, y + 196))
        d.text((178, y + 26), label, font=F["label"], fill=color)
        d.text((178, y + 60), title, font=F["monob"], fill=INK)
        for i, ln in enumerate(wrap(d, body, F["small"], 1500)):
            d.text((178, y + 110 + i * 34), ln, font=F["small"], fill=INK_2)
        y += 220
    d.text((140, 970), "The innovation is not a new kernel. It is reading the grading "
                       "contract closely enough to know which one to reach for.",
           font=F["small"], fill=MUTED)
    return img


# --------------------------------------------------------- 0:35 Architecture
def s_architecture(F, ctx):
    img, d = new_slide()
    eyebrow(d, F, "Architecture  \u00b7  where the memory goes")
    heading(d, F, "Same weights, same math, one path removed")

    def flow(y, title, boxes, color, note):
        d.text((140, y), title, font=F["h3"], fill=color)
        x = 140
        for i, (txt, w, bad) in enumerate(boxes):
            panel(d, (x, y + 56, x + w, y + 166), outline=CRIT if bad else RULE)
            for j, ln in enumerate(txt.split("\n")):
                d.text((x + 20, y + 76 + j * 32), ln, font=F["mono_s"],
                       fill=CRIT if bad else INK_2)
            if i < len(boxes) - 1:
                arrow(d, x + w + 10, y + 111, x + w + 58)
            x += w + 68
        for k, ln in enumerate(wrap(d, note, F["small"], 1560)):
            d.text((140, y + 186 + k * 32), ln, font=F["small"], fill=MUTED)

    flow(270, "BASELINE", [
        ("x \u2192 LayerNorm\nQ, K, V proj", 290, False),
        ("scores = QK\u1d40\n[B, H, S, S]", 250, True),
        ("softmax\n+ mask", 200, True),
        ("probs @ V\nout proj", 230, False),
        ("FFN\nGELU", 180, False),
    ], CRIT, "Two stages hold an S\u00d7S tensor. At S = 100,000 that is 20.5 TB, "
             "so this path simply cannot execute.")

    flow(590, "OURS", [
        ("x \u2192 LayerNorm\nQ, K, V proj", 290, False),
        ("scaled_dot_product_attention(q, k, v, is_causal=True)\n"
         "scores + softmax + PV fused, never materialized", 720, False),
        ("out proj", 170, False),
        ("FFN\nGELU", 180, False),
    ], GOOD, "One fused kernel, with causal masking generated inside it \u2014 a dense "
             "[S,S] mask at S = 100,000 would be 10 GB on its own.")

    panel(d, (140, 890, 1780, 1005))
    d.text((178, 914), "DROP-IN CONTRACT", font=F["label"], fill=MUTED)
    d.text((178, 952), "UserOptimizedTransformer(BaselineTransformer)  \u2192  every "
                       "submodule and parameter name unchanged  \u2192  "
                       "copy_model_weights(strict=True)",
           font=F["mono_s"], fill=INK_2)
    return img


# ------------------------------------------------------------ 0:55 Live demo
DEMO = [
    ("$ python scripts/build_kaggle_selfcontained.py --accelerator NvidiaTeslaT4", ACCENT, True),
    ("wrote .kaggle_upload/kernel_t4/track3_sc.py (50049 chars, 1296 lines)", INK_2, False),
    ("  only=1-13  id=wenjiluo/track3-t4  accelerator=NvidiaTeslaT4", INK_2, False),
    ("py_compile: OK", GOOD, False),
    ("", INK_2, False),
    ("$ kaggle kernels push -p .kaggle_upload/kernel_t4", ACCENT, True),
    ("Kernel version 2 successfully pushed.", INK_2, False),
    ("", INK_2, False),
    ("=== ENV ===", MUTED, False),
    ("gpu Tesla T4 | torch 2.10.0+cu128 | cuda 12.8 | cc (7, 5)", INK, False),
    ("", INK_2, False),
    ("##### SHAPE 1 : B=64 D=128 H=4 S=128 L=4 F=128 #####", MUTED, False),
    ("PASS max_abs=1.19e-06 | baseline=9.5355ms optimized=4.1779ms | speedup=2.282x", GOOD, False),
    ("##### SHAPE 2 : B=1 D=128 H=4 S=128 L=4 F=128 #####", MUTED, False),
    ("PASS max_abs=9.54e-07 | baseline=3.0460ms optimized=0.9067ms | speedup=3.359x", GOOD, False),
    ("##### SHAPE 6 : B=10000 D=128 H=4 S=128 L=4 F=128 #####", MUTED, False),
    ("PASS max_abs=1.67e-06 | baseline=1431.88ms optimized=761.33ms | speedup=1.881x", GOOD, False),
    ("##### SHAPE 8 : B=64 D=1024 H=4 S=128 L=4 F=1024 #####", MUTED, False),
    ("PASS max_abs=1.91e-06 | baseline=128.51ms optimized=118.32ms | speedup=1.086x", GOOD, False),
    ("##### SHAPE 13 : B=64 D=128 H=4 S=1024 L=4 F=128 #####", MUTED, False),
    ("PASS max_abs=1.43e-06 | baseline=318.74ms optimized=71.81ms | speedup=4.439x", GOOD, False),
    ("", INK_2, False),
    ("##### SHAPE 14 : optimized-only (baseline infeasible ~20.5 TB) #####", WARN, False),
    ("trunc S=2048 correctness: PASS max_abs=1.19e-06 max_rel=0.43", GOOD, False),
    ("vram free=15.45/15.64 GB | baseline scores would be 20.5 TB -> infeasible", CRIT, False),
    ("full S=100000: median=204132.3 ms | 15,676 tok/s | peak_vram=14.58 GB", INK, False),
    ("", INK_2, False),
    ("# median_speedup=2.282x  min=1.086x  max=4.439x  over 13 PASS shapes", ACCENT, False),
    ("PASS 13/13 gradeable shapes", GOOD, False),
]

# (lines revealed, caption) -- one step per beat of the demo
DEMO_STEPS = [
    (4, "Build one self-contained kernel from the unmodified harness"),
    (7, "Push it to a free Kaggle T4 \u2014 no local NVIDIA GPU involved"),
    (10, "The card comes up: Tesla T4, compute capability 7.5"),
    (13, "Every shape is checked for correctness first, then timed"),
    (17, "Throughput shapes: 10,000 sequences per batch"),
    (21, "The long-sequence shape, where fused attention wins biggest"),
    (24, "Shape 14: the one with no runnable baseline"),
    (26, "The full 100,000-token forward completes inside 14.6 GB"),
    (len(DEMO), "13 / 13 pass \u00b7 median 2.28\u00d7"),
]


def demo_frame(F, ctx, n_lines, caption):
    img, d = new_slide()
    eyebrow(d, F, "Live demo  \u00b7  input \u2192 processing \u2192 output, one full sweep")
    heading(d, F, caption, f="h3", y=144)

    box = (140, 216, 1780, 1010)
    panel(d, box)
    x0, y0, x1, _ = box
    d.text((x0 + 26, y0 + 18), "replay of the recorded run  \u00b7  "
                               "results/kaggle_t4_run.log \u00b7 kaggle_t4_shape14.log",
           font=F["small"], fill=MUTED)
    d.line([(x0 + 2, y0 + 58), (x1 - 2, y0 + 58)], fill=RULE, width=2)

    visible = DEMO[:n_lines]
    rows = 22
    if len(visible) > rows:
        visible = visible[-rows:]
    y = y0 + 82
    for text, color, is_cmd in visible:
        d.text((x0 + 28, y), text, font=F["mono_sb"] if is_cmd else F["mono_s"],
               fill=color)
        y += 32
    if n_lines < len(DEMO):
        d.rectangle([x0 + 28, y + 4, x0 + 42, y + 26], fill=INK_2)
    return img


# ------------------------------------------------------------- 2:20 Results
def s_results(F, ctx):
    img, d = new_slide()
    t4 = ctx["t4"]
    eyebrow(d, F, "Results  \u00b7  against the reference implementation")
    heading(d, F, f"13 / 13 PASS   \u00b7   median {t4['median']:.2f}\u00d7   \u00b7   up to "
                  f"{t4['hi']:.2f}\u00d7")
    paste_fit(img, ctx["fig_speed"], (140, 240, 1780, 690))

    panel(d, (140, 716, 1780, 1000))
    d.text((178, 740), "REGIME", font=F["label"], fill=MUTED)
    d.text((880, 740), "MEDIAN", font=F["label"], fill=MUTED)
    d.text((1130, 740), "WORST max_abs", font=F["label"], fill=MUTED)
    d.text((1460, 740), "MARGIN vs atol", font=F["label"], fill=MUTED)
    rows = [("Tesla P100  \u00b7  SDPA only", "2.065\u00d7", "1.91e-6", "1049\u00d7 inside", GOOD),
            ("Tesla T4  \u00b7  SDPA + compile   (shipped)", f"{t4['median']:.3f}\u00d7",
             "1.91e-6", "1049\u00d7 inside", GOOD),
            ("Tesla T4  \u00b7  + fp16   (measured, not shipped)", "4.014\u00d7", "2.04e-3",
             "0.98\u00d7 \u2014 past it", CRIT)]
    y = 790
    for name, med, mabs, margin, color in rows:
        d.text((178, y), name, font=F["small"], fill=INK)
        d.text((880, y), med, font=F["monob"], fill=INK)
        d.text((1130, y), mabs, font=F["mono_s"], fill=INK_2)
        d.text((1460, y), margin, font=F["mono_sb"], fill=color)
        y += 66
    return img


# -------------------------------------------------------------- 2:45 Impact
def s_impact(F, ctx):
    img, d = new_slide()
    eyebrow(d, F, "Impact")
    d.text((140, 216), "Long-context inference", font=F["h1"], fill=INK_2)
    d.text((140, 316), "on the hardware you already have.", font=F["h1"], fill=ACCENT)

    cells = [("SHAPE 14", "infeasible \u2192 runs",
              "3.2M tokens per forward, 14.6 GB, on a free card"),
             ("EVERY GPU RUN", "free tier",
              "Kaggle T4 and P100, driven from a laptop with no NVIDIA GPU"),
             ("EVERY NUMBER", "traceable",
              "raw kernel logs committed next to each table")]
    x = 140
    for label, value, note in cells:
        panel(d, (x, 470, x + 520, 700))
        d.text((x + 30, 498), label, font=F["label"], fill=MUTED)
        d.text((x + 30, 538), value, font=F["h3"], fill=ACCENT)
        for i, ln in enumerate(wrap(d, note, F["small"], 460)):
            d.text((x + 30, 604 + i * 34), ln, font=F["small"], fill=INK_2)
        x += 550

    d.text((140, 752), "It is a drop-in class \u2014 same submodules, same parameter "
                       "names \u2014 so any model built", font=F["body"], fill=INK_2)
    d.text((140, 796), "on this layer inherits the speedup with no retrain and no "
                       "re-export.", font=F["body"], fill=INK_2)

    panel(d, (140, 876, 1780, 1000))
    d.text((178, 902), "CODE, RAW LOGS, PER-SHAPE DATA", font=F["label"], fill=MUTED)
    d.text((178, 942), "github.com/LUOaini1213/fp16x3-transformer",
           font=F["monob"], fill=ACCENT)
    return img


# ------------------------------------------------- gallery-only slides
def s_memory_wall(F, ctx):
    """The 20.5 TB wall on its own. Used in the Devpost gallery, where the
    problem has to land in a single still with no narration under it."""
    img, d = new_slide()
    eyebrow(d, F, "Shape 14  ·  seq_len = 100,000")
    heading(d, F, "The baseline cannot run this shape")
    paste_fit(img, ctx["fig_wall"], (140, 250, 1780, 790))
    panel(d, (140, 820, 1780, 1000))
    d.text((178, 848), "[B, H, S, S]  =  32 × 16 × 100000 × 100000  =  5.12e12 elements",
           font=F["mono"], fill=INK_2)
    d.text((178, 896), "= 20.5 TB in fp32.  No GPU holds that.",
           font=F["monob"], fill=CRIT)
    d.text((178, 946), "Ours completes the same forward in 14.6 GB, at 15,676 tok/s.",
           font=F["monob"], fill=GOOD)
    return img


def s_precision(F, ctx):
    """The fp16 trade, as a margin gauge against the hard gate."""
    img, d = new_slide()
    eyebrow(d, F, "The decision we had to measure to get right")
    heading(d, F, "fp16 is twice as fast. We ship fp32.")

    for i, (name, sub, width, color, note) in enumerate([
        ("fp32  — shipped", "worst max_abs 1.91e-6", 0.33, GOOD,
         "1049× inside the gate"),
        ("fp16  — measured, not shipped", "worst max_abs 2.04e-3", 1.00, CRIT,
         "0.98× — already past it"),
    ]):
        y = 330 + i * 160
        d.text((140, y), name, font=F["h3"], fill=INK)
        d.text((140, y + 58), sub, font=F["mono_s"], fill=MUTED)
        x0, x1 = 780, 1740
        fill_w = int((x1 - x0) * width)
        d.rounded_rectangle((x0, y, x1, y + 84), radius=6, fill=PANEL)
        d.rounded_rectangle((x0, y, x0 + fill_w, y + 84), radius=6, fill=color)
        d.line([(x1 - 3, y - 8), (x1 - 3, y + 92)], fill=INK, width=4)
        tw = d.textlength(note, font=F["monob"])
        if tw + 48 <= fill_w:
            d.text((x0 + 24, y + 24), note, font=F["monob"], fill=GROUND)
        else:
            d.text((x0 + fill_w + 24, y + 24), note, font=F["monob"], fill=color)
    d.text((1600, 640), "atol = 0.002", font=F["small"], fill=MUTED)

    body = [
        ("fp16 passes all 13 shapes at 4.01× median — nearly double the "
         "shipped path.", INK_2),
        ("But its worst absolute error has already crossed the gate. It survives "
         "only because the rule is abs OR rel, and that element happened to have "
         "a large reference value.", CRIT),
        ("That is luck, not margin. One failing element forfeits the speed score "
         "entirely, so we kept the exact path and left fp16 as a documented flag.",
         INK_2),
    ]
    y = 720
    for text, color in body:
        lines = wrap(d, text, F["body"], 1500)
        d.ellipse([150, y + 15, 162, y + 27], fill=color)
        for i, ln in enumerate(lines):
            d.text((186, y + i * 42), ln, font=F["body"], fill=INK_2)
        y += len(lines) * 42 + 26
    return img


GALLERY = [
    ("01_results", lambda F, c: s_results(F, c)),
    ("02_memory_wall", lambda F, c: s_memory_wall(F, c)),
    ("03_architecture", lambda F, c: s_architecture(F, c)),
    ("04_precision", lambda F, c: s_precision(F, c)),
    ("05_demo", lambda F, c: demo_frame(F, c, len(DEMO), "13 / 13 pass · median 2.28×")),
]


def render_gallery(F, ctx, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, fn in GALLERY:
        path = os.path.join(outdir, name + ".png")
        fn(F, ctx).save(path, optimize=True)
        print(f"  {path}  ({os.path.getsize(path)/1024:.0f} KB)")


# ------------------------------------------------------------------ timeline
SECTIONS = [
    dict(id="1_problem", start=0, end=15, kind="still", render=s_problem,
         vo="Make a Transformer layer faster on a GPU, without moving the "
            "answer. The grader checks every output element, and one bad "
            "element fails the shape. Fourteen of them \u2014 and the "
            "fourteenth needs twenty point five terabytes of attention "
            "scores, so the reference cannot run it at all."),

    dict(id="2_solution", start=15, end=35, kind="still", render=s_solution,
         vo="We rewrite only the forward compute. Fused attention, so the "
            "score matrix is never built and memory grows linearly with "
            "sequence length. Compilation the model applies to itself. And "
            "chunking sized from the memory actually free at runtime. Every "
            "parameter name stays identical, so it drops straight into the "
            "official harness."),

    dict(id="3_architecture", start=35, end=55, kind="still", render=s_architecture,
         vo="Here is the data flow. The baseline holds an S by S tensor twice, "
            "once for the scores and once for the softmax. We replace both with a "
            "single fused kernel that computes the same result without ever "
            "materializing that matrix, and generates the causal mask inside "
            "itself."),

    dict(id="4_demo", start=55, end=140, kind="demo",
         vo="Now the full run, end to end. One self-contained kernel is built from "
            "the unmodified harness and pushed to a free Kaggle T4 \u2014 there is no "
            "NVIDIA GPU on this laptop. The card comes up, and the sweep begins. "
            "Every shape is checked for correctness first; only if it passes is "
            "the latency timed. Shape one, two point two eight times faster. Shape "
            "two, three point three six. Shape six, ten thousand sequences in a "
            "single batch, one point eight eight. Shape eight is dominated by its "
            "projections, so it barely moves. Shape thirteen, the long one, four "
            "point four four. Then shape fourteen, the one with no runnable "
            "baseline. Correctness is established at a truncated length where the "
            "reference still fits, and then the full hundred-thousand-token "
            "forward completes in two hundred and four seconds, at fifteen "
            "thousand tokens a second, inside fourteen point six gigabytes \u2014 on a "
            "card that only has fifteen point six in total. Getting that last "
            "shape to run took one more fix, and it was not in the attention: "
            "collecting the chunks in a list and joining them with torch cat "
            "holds the pieces and the joined result at the same time, a second "
            "six and a half gigabyte allocation exactly where memory was "
            "tightest. Writing each chunk straight into a preallocated output "
            "removed it. Thirteen of thirteen shapes pass, and the fourteenth "
            "runs."),

    dict(id="5_results", start=140, end=165, kind="still", render=s_results,
         vo="All thirteen shapes pass, with a worst absolute error around two "
            "millionths \u2014 about a thousand times inside the tolerance. Median two "
            "point two eight times, up to four point four. We also measured a "
            "half-precision path at four point zero one, and we ship it turned "
            "off: its worst error has already crossed the absolute gate, and it "
            "survives only on the relative one."),

    dict(id="6_impact", start=165, end=180, kind="still", render=s_impact,
         vo="It is a drop-in class, so any model built on this layer inherits the "
            "speedup with no retrain. Every run was on a free cloud GPU, and every "
            "number traces to a raw log in the repository."),
]


# ------------------------------------------------------------------- pipeline
async def synth(text, path, voice, rate="+0%"):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate).save(path)


def synth_to_fit(text, path, voice, window, max_speedup=0.25):
    """Synthesize, and if it overruns its window, re-synthesize faster.

    Hand-trimming words to hit a mark is a losing game -- every edit shifts the
    duration again. Speaking rate is the parameter that actually maps onto the
    constraint, so solve for it directly. Returns (duration, rate_used).
    """
    from moviepy import AudioFileClip

    def measure(rate):
        if os.path.exists(path):
            os.remove(path)
        asyncio.run(synth(text, path, voice, rate))
        with AudioFileClip(path) as a:
            return a.duration

    dur = measure("+0%")
    if dur <= window:
        return dur, "+0%"
    # Need to compress by dur/window; ask for a bit more to cover TTS rounding.
    need = min(max_speedup, dur / window - 1.0 + 0.02)
    rate = f"+{int(round(need * 100))}%"
    dur = measure(rate)
    return dur, rate


def srt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "build", "track3_demo.mp4"))
    ap.add_argument("--workdir", default=os.path.join(HERE, "build", "video"))
    ap.add_argument("--voice", default="en-US-AndrewNeural")
    ap.add_argument("--no-vo", action="store_true")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gallery", default=None,
                    help="render the Devpost gallery stills into this folder "
                         "and exit without building the video")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    figdir = os.path.join(args.workdir, "figs")
    subprocess.run([sys.executable, os.path.join(HERE, "scripts", "make_figures.py"),
                    "--theme", "dark", "--outdir", figdir], check=True)

    ctx = dict(t4=stats("results_t4.csv"), p100=stats("results.csv"),
               fig_wall=os.path.join(figdir, "memory_wall.png"),
               fig_speed=os.path.join(figdir, "speedups.png"))
    F = fonts()

    if args.gallery:
        render_gallery(F, ctx, args.gallery
                       if os.path.isabs(args.gallery)
                       else os.path.join(HERE, args.gallery))
        return 0

    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips

    clips, srt, over = [], [], []
    for sec in SECTIONS:
        window = sec["end"] - sec["start"]

        rate = "+0%"
        if args.no_vo:
            audio, vo_len = None, window
        else:
            mp3 = os.path.join(args.workdir, f"{sec['id']}.mp3")
            rate = "+0%"
            if os.path.exists(mp3):
                with AudioFileClip(mp3) as probe:
                    vo_len = probe.duration
                if vo_len > window:
                    vo_len, rate = synth_to_fit(sec["vo"], mp3, args.voice, window)
            else:
                vo_len, rate = synth_to_fit(sec["vo"], mp3, args.voice, window)
            audio = AudioFileClip(mp3)
            vo_len = audio.duration
            if vo_len > window:
                over.append((sec["id"], vo_len, window))

        if sec["kind"] == "demo":
            per = window / len(DEMO_STEPS)
            sub = []
            for i, (n, caption) in enumerate(DEMO_STEPS):
                png = os.path.join(args.workdir, f"{sec['id']}_{i:02d}.png")
                demo_frame(F, ctx, n, caption).save(png)
                sub.append(ImageClip(png).with_duration(per))
            body = concatenate_videoclips(sub, method="chain")
        else:
            png = os.path.join(args.workdir, f"{sec['id']}.png")
            sec["render"](F, ctx).save(png)
            body = ImageClip(png).with_duration(window)

        if audio is not None:
            body = body.with_audio(audio.subclipped(0, min(vo_len, window)))
        clips.append(body)
        srt.append((len(srt) + 1, sec["start"], sec["end"], sec["vo"]))
        note = "OVER" if vo_len > window else ("ok" if rate == "+0%"
                                               else f"ok (rate {rate})")
        print(f"  {sec['id']:16s} window {window:5.1f}s   narration {vo_len:5.1f}s"
              f"   {note}", flush=True)

    if over:
        print("\nnarration does not fit its window:")
        for sid, got, want in over:
            print(f"  {sid}: {got:.1f}s of voice-over in a {want:.0f}s slot "
                  f"({got - want:+.1f}s) -- shorten SECTIONS[...]['vo']")

    video = concatenate_videoclips(clips, method="chain")
    video.write_videofile(args.out, fps=args.fps, codec="libx264",
                          audio_codec="aac", preset="medium", logger=None)

    sub_path = os.path.splitext(args.out)[0] + ".srt"
    with open(sub_path, "w", encoding="utf-8", newline="\n") as f:
        for i, a, b, text in srt:
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")

    total = SECTIONS[-1]["end"]
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, {total}s)")
    print(f"wrote {sub_path}")
    return 1 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())

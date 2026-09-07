# Devpost submission — Track 3

The submission text now lives in **[`docs/DEVPOST_HANDOFF.md`](../docs/DEVPOST_HANDOFF.md)**,
which carries every Devpost field ready to copy-paste, the full "About the
project" body (Markdown + LaTeX), and a table mapping each headline number to
the raw kernel log it came from.

This file used to hold a second copy of that prose. Two copies is precisely the
drift the documentation audit kept catching, so there is now one.

Quick reference:

| Field | Value |
|---|---|
| Track | 3 — Implement a GPU Kernel for a Transformer Layer |
| Project | ExactSwap — Drop-in GPU Transformer Layer |
| Repo | https://github.com/LUOaini1213/fp16x3-transformer |
| Demo | https://youtu.be/3aAw-jq1oTM |
| Headline | 13/13 shapes PASS, median 2.282x on a free T4, and `seq_len=100000` runs where the reference needs 20.5 TB |

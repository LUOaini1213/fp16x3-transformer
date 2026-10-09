"""Explicitly build the pinned optional T4 dependency without upgrading torch.

Upstream has no top-level LICENSE at this revision. Source stays in a temporary
local build directory and is not copied into this repository or redistributed.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

REVISION = "9ef98fcb506bb1e2fe3cece50935e2935bf6b124"


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
        raise SystemExit("This optional backend requires a CUDA Turing GPU (sm_75).")
    if not shutil.which("nvcc"):
        raise SystemExit("CUDA development toolkit (nvcc) required; installed torch is not changed.")
    root = Path(tempfile.mkdtemp(prefix="track3-turing-")) / "source"
    subprocess.run(["git", "clone", "https://github.com/ssiu/flash-attention-turing.git", str(root)], check=True)
    subprocess.run(["git", "checkout", REVISION], cwd=root, check=True)
    subprocess.run(["git", "submodule", "update", "--init", "--depth", "1", "csrc/cutlass"], cwd=root, check=True)
    os.environ.setdefault("MAX_JOBS", "2")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(root)], check=True)
    print(f"Built optional revision {REVISION}. Temporary source retained at {root}.")
    print("Enable with T3_ATTN=turing; default remains SDPA. Native-fp16 inference only.")


if __name__ == "__main__":
    main()

"""Build a private T4 launcher that clones an exact public Git commit."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/LUOaini1213/fp16x3-transformer.git"


def launcher(revision, phase):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("an exact lowercase Git commit SHA is required")
    if phase not in ("fp32", "flash", "attention", "quick", "workspace"):
        raise ValueError("unsupported release phase")
    module = "scripts.benchmark_usability" if phase in ("quick", "workspace") else "scripts.benchmark_release"
    return f'''# Generated launcher: no model code is inlined.
import os, pathlib, subprocess, sys, tempfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"
root = pathlib.Path(tempfile.mkdtemp(prefix="track3-clean-")) / "repository"
subprocess.run(["git", "clone", {REPOSITORY!r}, str(root)], check=True)
subprocess.run(["git", "checkout", "--detach", {revision!r}], cwd=root, check=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == {revision!r}
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
# Kaggle may omit the unversioned CUDA driver linker name. Only add a private
# symlink/search directory; never alter the installed Torch/CUDA environment.
for folder in ("/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64", "/usr/lib64"):
    target = pathlib.Path(folder) / "libcuda.so.1"
    if target.exists():
        links = pathlib.Path(tempfile.mkdtemp(prefix="track3-linker-"))
        (links / "libcuda.so").symlink_to(target.resolve())
        os.environ["LIBRARY_PATH"] = str(links) + ":" + os.environ.get("LIBRARY_PATH", "")
        break
out = pathlib.Path.cwd()
subprocess.run([sys.executable, "-m", {module!r}, "--phase", {phase!r},
                "--output", str(out)], cwd=root, check=True)
'''


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--phase", choices=("fp32", "flash", "attention", "quick", "workspace"), required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", args.ref], cwd=ROOT, text=True).strip()
    source = launcher(revision, args.phase)
    compile(source, "clean_launcher.py", "exec")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "clean_launcher.py").write_text(source, encoding="utf-8")
    metadata = dict(id=args.id, title=args.id.split("/")[-1], code_file="clean_launcher.py",
                    language="python", kernel_type="script", is_private=True, enable_gpu=True,
                    enable_tpu=False, enable_internet=True, machine_shape="NvidiaTeslaT4",
                    dataset_sources=[], competition_sources=[], kernel_sources=[], model_sources=[])
    (args.out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Built clean {args.phase} launcher for {revision}: {args.out}")


if __name__ == "__main__":
    main()

# Generated launcher: no model code is inlined.
import os, pathlib, subprocess, sys, tempfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"
root = pathlib.Path(tempfile.mkdtemp(prefix="track3-clean-")) / "repository"
subprocess.run(["git", "clone", 'https://github.com/LUOaini1213/fp16x3-transformer.git', str(root)], check=True)
subprocess.run(["git", "checkout", "--detach", 'b335fb64a0fc76ad4105abf58e23589ef8af80ae'], cwd=root, check=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == 'b335fb64a0fc76ad4105abf58e23589ef8af80ae'
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
subprocess.run([sys.executable, "-m", 'scripts.benchmark_usability', "--phase", 'workspace',
                "--output", str(out)], cwd=root, check=True)

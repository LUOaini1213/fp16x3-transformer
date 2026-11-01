# Generated launcher: no model code is inlined.
import os, pathlib, subprocess, sys, tempfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUNBUFFERED"] = "1"
root = pathlib.Path(tempfile.mkdtemp(prefix="track3-clean-")) / "repository"
subprocess.run(["git", "clone", 'https://github.com/LUOaini1213/fp16x3-transformer.git', str(root)], check=True)
subprocess.run(["git", "checkout", "--detach", 'c8fe1cf7807d5633a1b75773d2ab03461318cb22'], cwd=root, check=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() == 'c8fe1cf7807d5633a1b75773d2ab03461318cb22'
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
worker = '''
import pathlib, sys
from scripts import benchmark_improvements as driver
original = driver.release.paired
driver.release.paired = lambda calls, x, mask, repeats: original(calls, x, mask, 200)
driver.pair_worker(2, pathlib.Path(sys.argv[1]))
'''
import json
rows = []
for index in range(3):
    directory = out / f"worker_{index}"
    directory.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-c', worker, str(directory)], cwd=root, check=True)
    rows.append(json.loads((directory / 'next_improvements_pair_2.json').read_text()))
payload = {'metadata': rows[0]['metadata'], 'results': {
    'shape': 2, 'repetitions_per_round': 200, 'fresh_processes': 3,
    'scope': 'one new T4 session; three fresh workers, shared compiler cache; original timing helpers, only repetition count increased',
    'workers': [row['results'] for row in rows]}}
payload['metadata']['protocol'] = payload['results']['scope']
(out / 'next_improvements_shape2_repeat.json').write_text(json.dumps(payload, indent=2))
print('IMPROVEMENTS_SHAPE2_REPEAT ' + json.dumps(payload), flush=True)

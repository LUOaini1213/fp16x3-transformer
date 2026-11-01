"""Invalid CLI requests must fail before environment changes, Torch or inference."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run as cli


ROOT = Path(__file__).resolve().parents[1]
ENTRY = """
import os, runpy, sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('invalid CLI request imported Torch or probed a device')
sys.meta_path.insert(0, NoTorch())
os.environ['T3_CLI_INPUT_SENTINEL'] = 'preserve'
sys.argv = ['scripts.run', *sys.argv[1:]]
try:
    runpy.run_module('scripts.run', run_name='__main__')
except SystemExit:
    assert os.environ.get('T3_CLI_INPUT_SENTINEL') == 'preserve', 'configure ran on invalid input'
    assert 'torch' not in sys.modules
    raise
"""


@pytest.mark.parametrize("arguments", [
    ["--shapes", ""], ["--shapes", "   "], ["--shapes", ","],
    ["--shapes", "1,,2"], ["--shapes", "1-100000000000000000000"],
    ["--shapes", "0-13"], ["--shapes", "1-14"], ["--shapes", "4-2"],
    ["--shape", "0"], ["--shape", "14"],
    ["--shape", "2", "--shapes", "1-3"],
    ["--shapes", "", "--check-only"], ["--repeats", "0"],
])
def test_invalid_requests_fail_in_actual_entry_before_torch_or_configure(arguments, tmp_path):
    output = tmp_path / "result.json"
    process = subprocess.run([sys.executable, "-c", ENTRY, *arguments, "--device", "cuda",
                              "--output", str(output)], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
    assert process.returncode == 2, process.stdout + process.stderr
    assert "error:" in process.stderr
    assert "Traceback" not in process.stderr and "Install a suitable" not in process.stderr
    assert not process.stdout and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", ["1-100000000000000000000", "0-13", "14"])
def test_out_of_bounds_endpoints_never_allocate_a_range(monkeypatch, value):
    def forbidden_range(*args):
        raise AssertionError("an invalid range was expanded before endpoint validation")
    monkeypatch.setattr(cli, "range", forbidden_range, raising=False)
    with pytest.raises(ValueError, match="1-13"):
        cli.parse_shapes(value)


def test_invalid_shapes_do_not_call_probe_benchmark_or_output_writer(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached a side effect")
    for name in ("configure", "environment_report", "run_case", "write_json"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(cli.subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["run", "--shapes", "", "--output", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arguments", [[], ["--shape", "2"]])
def test_implicit_and_explicit_default_shape_remain_two(monkeypatch, tmp_path, arguments):
    calls, outputs = [], []
    monkeypatch.setattr(cli, "configure", lambda *args: None)
    monkeypatch.setattr(cli, "environment_report", lambda *args: {"device": "cpu"})

    def run_case(shape, *args):
        calls.append(shape)
        return {"shape": shape}

    monkeypatch.setattr(cli, "run_case", run_case)
    monkeypatch.setattr(cli, "write_json", lambda path, payload: outputs.append(payload))
    monkeypatch.setattr(sys, "argv", ["run", *arguments, "--output", str(tmp_path / "result.json")])
    assert cli.main() == 0
    assert calls == [2] and outputs == [{"environment": {"device": "cpu"}, "results": [{"shape": 2}]}]

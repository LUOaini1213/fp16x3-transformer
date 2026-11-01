"""A failed capacity demonstration must not masquerade as a successful run."""

import sys

import pytest

from scripts import shape14_optimized_only as runner


@pytest.mark.parametrize("correct,full,skip,exit_code", [
    (False, True, False, 2), (True, False, False, 1),
    (True, True, False, 0), (True, False, True, 0),
])
def test_shape14_exit_gate(monkeypatch, correct, full, skip, exit_code):
    monkeypatch.setattr(sys, "argv", ["shape14"] + (["--skip-full"] if skip else []))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda device: "test GPU")
    monkeypatch.setattr(runner, "correctness_truncated", lambda args, device: correct)
    timed = []

    def capacity(args, device):
        timed.append(True)
        return full

    monkeypatch.setattr(runner, "timing_full", capacity)
    assert runner.main() == exit_code
    assert bool(timed) == (correct and not skip)

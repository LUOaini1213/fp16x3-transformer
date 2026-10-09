"""A reviewer must not certify a complete GPU run from a partial checkout."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.verify_next_evidence import (MAINTAINED_ARTIFACTS, ROOT, verify_artifacts,
                                         verify_maintained_runtime)


@pytest.fixture
def evidence_copy(tmp_path):
    for folder in MAINTAINED_ARTIFACTS:
        shutil.copytree(ROOT / "results/next" / folder, tmp_path / folder)
    for name in ("improvements_summary.md", "improvements_pilot_summary.md", "portfolio_runtime_summary.md"):
        shutil.copy(ROOT / "results/next" / name, tmp_path)
    return tmp_path


def edit_receipt(root, transform):
    path = root / "improvements-full/artifact_sha256.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    transform(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_legacy_subset_check_is_preserved_but_strict_audit_rejects_missing_receipt(evidence_copy):
    (evidence_copy / "improvements-full/artifact_sha256.json").unlink()
    assert verify_artifacts(evidence_copy) == 43
    with pytest.raises(ValueError, match="improvements-full/artifact_sha256.json"):
        verify_maintained_runtime(evidence_copy)


def test_strict_audit_lists_every_missing_collection_receipt(evidence_copy):
    for folder in ("improvements-full", "improvements-pilot"):
        (evidence_copy / folder / "artifact_sha256.json").unlink()
    with pytest.raises(ValueError) as failure:
        verify_maintained_runtime(evidence_copy)
    assert "improvements-full/artifact_sha256.json" in str(failure.value)
    assert "improvements-pilot/artifact_sha256.json" in str(failure.value)


@pytest.mark.parametrize("name", ["next_improvements_pair_13.json", "next_improvements_full.json", "cloud_script.py"])
def test_receipt_cannot_omit_required_artifact(evidence_copy, name):
    edit_receipt(evidence_copy, lambda receipt: receipt.pop(name))
    with pytest.raises(ValueError, match=name):
        verify_maintained_runtime(evidence_copy)


def test_strict_audit_rejects_unreceipted_measurements(evidence_copy):
    (evidence_copy / "improvements-full/next_extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unreceipted.*next_extra.json"):
        verify_maintained_runtime(evidence_copy)


def test_strict_audit_names_missing_recorded_file(evidence_copy):
    (evidence_copy / "improvements-full/next_improvements_pair_13.json").unlink()
    with pytest.raises(ValueError, match="missing evidence files.*next_improvements_pair_13.json"):
        verify_maintained_runtime(evidence_copy)


def test_strict_audit_rejects_changed_measurements_even_if_json_still_parses(evidence_copy):
    path = evidence_copy / "improvements-full/next_improvements_full.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_maintained_runtime(evidence_copy)


def test_strict_audit_requires_raw_log_even_if_removed_from_receipt_and_disk(evidence_copy):
    directory = evidence_copy / "improvements-full"
    logs = list(directory.glob("*.log"))
    assert logs
    for path in logs:
        path.unlink()
    edit_receipt(evidence_copy, lambda receipt: [receipt.pop(p.name) for p in logs])
    with pytest.raises(ValueError, match="missing raw execution log"):
        verify_maintained_runtime(evidence_copy)


@pytest.mark.parametrize("change", ["empty", "bad_hash", "duplicate", "outside"])
def test_strict_audit_rejects_invalid_receipt_structure(evidence_copy, change):
    path = evidence_copy / "improvements-full/artifact_sha256.json"
    if change == "empty":
        path.write_text("{}", encoding="utf-8")
    elif change == "bad_hash":
        edit_receipt(evidence_copy, lambda r: r.update({"cloud_script.py": "not-a-sha256"}))
    elif change == "duplicate":
        digest = "0" * 64
        path.write_text('{"cloud_script.py":"' + digest + '","cloud_script.py":"' + digest + '"}',
                        encoding="utf-8")
    else:
        edit_receipt(evidence_copy, lambda r: r.update({"../outside.json": "0" * 64}))
    with pytest.raises(ValueError):
        verify_maintained_runtime(evidence_copy)


@pytest.mark.parametrize("name", ["improvements_summary.md", "improvements_pilot_summary.md",
                                  "portfolio_runtime_summary.md"])
def test_strict_audit_does_not_regenerate_missing_or_changed_report(evidence_copy, name):
    report = evidence_copy / name
    report.write_text("altered report", encoding="utf-8")
    with pytest.raises(ValueError, match="report differs"):
        verify_maintained_runtime(evidence_copy)
    assert report.read_text(encoding="utf-8") == "altered report"
    report.unlink()
    with pytest.raises(ValueError, match="missing maintained-runtime report"):
        verify_maintained_runtime(evidence_copy)
    assert not report.exists()


def test_strict_audit_verifies_current_core_and_leaves_evidence_unchanged(evidence_copy, tmp_path):
    before = {p.relative_to(evidence_copy): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in evidence_copy.rglob("*") if p.is_file()}
    result = verify_maintained_runtime(evidence_copy)
    assert result["artifact_hashes"] == 61
    assert result["core_source_hashes"] == 11
    assert result["cohorts"] == [
        {"name": "balanced-full", "profile_input_checks": 234, "balanced_gates_passed": 12, "shapes": 13},
        {"name": "portfolio-full", "profile_input_checks": 234, "balanced_gates_passed": 13, "shapes": 13}]
    assert result["shape2_repeat_passed"] == 2 and result["shape2_repeat_workers"] == 3
    after = {p.relative_to(evidence_copy): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in evidence_copy.rglob("*") if p.is_file()}
    assert before == after
    changed = tmp_path / "changed-repository"
    changed.mkdir()
    (changed / "user_optimized.py").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="final tested core does not match user_optimized.py"):
        verify_maintained_runtime(evidence_copy, repository=changed)


def test_audit_cli_succeeds_without_importing_torch():
    code = ("import sys; from scripts.verify_next_evidence import main; "
            "sys.argv=['audit','--maintained-runtime']; main(); assert 'torch' not in sys.modules")
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "complete maintained-runtime receipts" in process.stdout
    assert "balanced-full: 234 profile/input checks; 12/13 balanced gates" in process.stdout
    assert "portfolio-full: 234 profile/input checks; 13/13 balanced gates" in process.stdout
    assert "shape-2 repeat: 2/3" in process.stdout


def test_audit_cli_missing_receipt_exits_two_with_actionable_error(evidence_copy):
    (evidence_copy / "improvements-full/artifact_sha256.json").unlink()
    virtual_repository = evidence_copy / "cli-root"
    measurements = virtual_repository / "results/next"
    measurements.mkdir(parents=True)
    for folder in MAINTAINED_ARTIFACTS:
        shutil.copytree(evidence_copy / folder, measurements / folder)
    for name in ("improvements_summary.md", "improvements_pilot_summary.md", "portfolio_runtime_summary.md"):
        shutil.copy(evidence_copy / name, measurements)
    code = ("import sys; from pathlib import Path; import scripts.verify_next_evidence as audit; "
            f"audit.ROOT=Path({str(virtual_repository)!r}); "
            "sys.argv=['audit','--maintained-runtime']; audit.main()")
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert process.returncode == 2
    assert "ERROR:" in process.stderr and "improvements-full/artifact_sha256.json" in process.stderr
    assert "Evidence was not certified" in process.stderr and "Traceback" not in process.stderr
    assert "PASS:" not in process.stdout


@pytest.mark.parametrize("optimization", ["-O", "-OO", "environment"])
def test_strict_audit_refuses_disabled_assertions_in_real_process(optimization):
    env = os.environ.copy()
    command = [sys.executable]
    if optimization == "environment":
        env["PYTHONOPTIMIZE"] = "1"
    else:
        command.append(optimization)
    command += ["-m", "scripts.verify_next_evidence", "--maintained-runtime"]
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert process.returncode == 2
    assert "requires assertions" in process.stderr and "PYTHONOPTIMIZE" in process.stderr
    assert "Traceback" not in process.stderr and "PASS:" not in process.stdout


def test_legacy_audit_keeps_working_in_optimized_python():
    process = subprocess.run([sys.executable, "-O", "-m", "scripts.verify_next_evidence"],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    assert "PASS: 248 artifact hashes" in process.stdout


@pytest.mark.parametrize("folder", list(MAINTAINED_ARTIFACTS))
def test_neither_full_session_nor_repeat_can_lose_its_receipt(evidence_copy, folder):
    (evidence_copy / folder / "artifact_sha256.json").unlink()
    with pytest.raises(ValueError, match=folder + "/artifact_sha256.json"):
        verify_maintained_runtime(evidence_copy)

"""Resolve an immutable measured runner separately from the maintained CLI."""
import json
from pathlib import Path


MEASURED_RUNNER_COMMIT = "c8fe1cf7807d5633a1b75773d2ab03461318cb22"
RUNNER_ARCHIVE_FOLDER = "runtime-source-c8fe1cf"
RUNNER_SOURCE = {"schema": 1, "git_commit": MEASURED_RUNNER_COMMIT,
                 "source_path": "scripts/run.py", "snapshot": "run.py.snapshot"}


def measured_source(repository, metadata, name, evidence_root=None):
    """The frozen c8 runner proves only that historical session's source identity."""
    repository = Path(repository)
    if name == "scripts/run.py" and metadata["git_commit"] == MEASURED_RUNNER_COMMIT:
        directory = (Path(evidence_root) if evidence_root is not None else
                     repository / "results/next") / RUNNER_ARCHIVE_FOLDER
        provenance = json.loads((directory / "source.json").read_text(encoding="utf-8"))
        if provenance != RUNNER_SOURCE:
            raise ValueError("frozen runner provenance does not match the measured c8 source")
        return directory / "run.py.snapshot"
    return repository / name

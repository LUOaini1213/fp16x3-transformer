# Historical runner source

`run.py.snapshot` is the exact `scripts/run.py` blob from measured Git commit
`c8fe1cf7807d5633a1b75773d2ab03461318cb22`. Its SHA-256 matches the unchanged
source manifests in the balanced and portfolio GPU sessions. `source.json`
records that identity; the separate receipt hashes both files.

This file is frozen evidence, not an executable entry point. The maintained
CLI is `python -m scripts.run`. The snapshot does not establish correctness or
performance for the new CLI parser. The GPU reports continue to describe their
original measured source, with current model/kernel hashes verified separately.

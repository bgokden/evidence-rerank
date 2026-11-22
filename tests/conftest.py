"""Make the harnesses in `examples/` importable from the tests.

The package under test is `src/evrank`, and `examples/` holds the measurement
harnesses that go with it. Most of that is scripts and is exercised by running them,
but the corpus is not: it is a generator with constraints whose failures are silent.
A duplicated distractor still produces a slate, and a leaked held-out category still
produces a number -- a wrong one. Those are worth a test.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))

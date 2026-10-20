"""Pytest configuration for the VQC test suite.

Makes the ``src`` package importable from the repository root without
requiring an editable install, and registers shared fixtures here as the
suite grows.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

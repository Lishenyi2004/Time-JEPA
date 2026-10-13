"""Repo-root sys.path bootstrap shared by all probe modules.

Importing this module (anywhere, first) guarantees the V-JEPA repo root is on
sys.path so that ``app.*`` and ``src.*`` imports resolve regardless of cwd.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

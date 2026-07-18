"""Repository-wide filesystem locations.

All experiment scripts are executable from any working directory.  Inputs are
read from ``data/`` and generated artifacts are written below ``results/``.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"

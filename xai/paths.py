"""Project folders. XAI_DATA_DIR, if set, points the data folder elsewhere
(for example a read-only copy of data/ in another checkout)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("XAI_DATA_DIR") or PROJECT_ROOT / "data")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint"
RESULTS_DIR = PROJECT_ROOT / "results"

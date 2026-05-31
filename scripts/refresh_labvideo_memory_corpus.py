from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from key_action_indexer.experiment_action_ledger import main_refresh_corpus


if __name__ == "__main__":
    raise SystemExit(main_refresh_corpus())

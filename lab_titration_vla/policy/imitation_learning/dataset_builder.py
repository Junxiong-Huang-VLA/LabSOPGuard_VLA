from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def build_il_dataset_index(trajectories_dir: str, out_csv: str) -> Dict[str, int]:
    tdir = Path(trajectories_dir)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for p in sorted(tdir.glob("*.json")):
        rows.append({"episode_id": p.stem, "trajectory_path": str(p).replace("\\", "/")})
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode_id", "trajectory_path"])
        w.writeheader()
        w.writerows(rows)
    return {"episodes": len(rows)}


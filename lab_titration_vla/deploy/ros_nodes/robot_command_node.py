from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    cmd_path = Path("outputs/predictions/robot_action_command.json")
    if not cmd_path.exists():
        print(f"robot command not found: {cmd_path}")
        return 1
    cmd = json.loads(cmd_path.read_text(encoding="utf-8"))
    print("ROS node stub received command:")
    print(json.dumps(cmd, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


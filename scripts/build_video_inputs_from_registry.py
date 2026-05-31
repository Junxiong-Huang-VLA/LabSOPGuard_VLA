from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.device_registry import write_video_inputs_from_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build video_inputs.json from a camera registry.")
    parser.add_argument("--registry", default=str(PROJECT_ROOT / "configs" / "devices" / "camera_registry.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "device_registry" / "video_inputs.json"))
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_inputs = write_video_inputs_from_registry(args.registry, args.output, strict=not args.non_strict)
    print(json.dumps({"count": len(video_inputs), "output": args.output, "video_inputs": video_inputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

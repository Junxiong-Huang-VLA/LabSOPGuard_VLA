from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth_m: Optional[np.ndarray]
    rgb_path: str
    depth_path: Optional[str]
    depth_meta: Dict[str, str]


class RGBDLoader:
    def __init__(self, depth_unit: str = "auto") -> None:
        self.depth_unit = depth_unit
        self.depth_candidates = [".npy", ".png", ".tiff", ".tif", ".exr"]

    def resolve_depth_path(self, rgb_path: str, depth_path: str | None = None) -> str | None:
        if depth_path and Path(depth_path).exists():
            return depth_path
        stem = Path(rgb_path).with_suffix("")
        for ext in self.depth_candidates:
            c = str(stem) + ext
            if Path(c).exists():
                return c
        return None

    def load_rgb(self, rgb_path: str) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("cv2 is required")
        img = self._imread_unicode(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"rgb not found or decode failed: {rgb_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _imread_unicode(self, path: str, flags: int) -> Optional[np.ndarray]:
        # Windows fallback for non-ASCII path decoding issues.
        img = cv2.imread(path, flags)
        if img is not None:
            return img
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, flags)
        except Exception:
            return None

    def _to_meter(self, depth: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.depth_unit == "m":
            return depth.astype(np.float32), "m"
        if self.depth_unit == "mm":
            return depth.astype(np.float32) / 1000.0, "mm"

        d = depth.astype(np.float32)
        valid = d[np.isfinite(d) & (d > 0)]
        if valid.size == 0:
            return d, "unknown"
        if float(np.median(valid)) > 10:
            return d / 1000.0, "mm(auto)"
        return d, "m(auto)"

    def _denoise_depth(self, depth_m: np.ndarray) -> np.ndarray:
        # Remove obvious spikes and invalid holes for reflective/transparent noise robustness.
        d = np.where(np.isfinite(depth_m), depth_m, 0.0)
        d = np.where(d > 0, d, 0.0)
        valid = d[d > 0]
        if valid.size == 0:
            return d
        q1, q99 = np.quantile(valid, 0.01), np.quantile(valid, 0.99)
        d = np.where((d >= q1) & (d <= q99), d, 0.0)
        return d

    def load_depth(self, depth_path: str | None) -> Tuple[Optional[np.ndarray], Dict[str, str]]:
        if depth_path is None:
            return None, {"status": "missing_depth"}
        p = Path(depth_path)
        if not p.exists():
            return None, {"status": "missing_depth_file", "path": depth_path}

        if p.suffix.lower() == ".npy":
            depth = np.load(str(p))
        else:
            if cv2 is None:
                return None, {"status": "cv2_missing"}
            arr = self._imread_unicode(str(p), cv2.IMREAD_UNCHANGED)
            if arr is None:
                return None, {"status": "decode_failed", "path": depth_path}
            depth = arr

        depth_m, unit = self._to_meter(depth)
        depth_m = self._denoise_depth(depth_m)
        return depth_m, {"status": "ok", "unit": unit}

    def load(self, rgb_path: str, depth_path: str | None = None) -> RGBDFrame:
        rgb = self.load_rgb(rgb_path)
        depth_resolved = self.resolve_depth_path(rgb_path, depth_path)
        depth_m, meta = self.load_depth(depth_resolved)
        return RGBDFrame(rgb=rgb, depth_m=depth_m, rgb_path=rgb_path, depth_path=depth_resolved, depth_meta=meta)

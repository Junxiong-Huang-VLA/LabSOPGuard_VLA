from __future__ import annotations

from typing import Any, Dict, Optional

from project_name.common.schemas import PerceptionResult
from project_name.perception.center_depth_extractor import CenterDepthExtractor
from project_name.perception.object_parser import ObjectParser
from project_name.perception.region_depth_stats import RegionDepthStats
from project_name.perception.rgbd_loader import RGBDLoader
from project_name.perception.target_exporter import build_target_representation
from project_name.perception.xyz_converter import XYZConverter
from project_name.utils.spatial import bbox_center_xyxy


class PerceptionEngine:
    """Perception module for target parsing + robust depth statistics."""

    def __init__(self, depth_unit: str = "auto") -> None:
        self.loader = RGBDLoader(depth_unit=depth_unit)
        self.parser = ObjectParser()
        self.center_depth_extractor = CenterDepthExtractor()
        self.region_stats = RegionDepthStats()
        self.xyz_converter = XYZConverter()

    def infer(
        self,
        rgb_path: str,
        depth_path: str | None = None,
        raw_target: Optional[Dict[str, Any]] = None,
        camera_intrinsics: Optional[Dict[str, float]] = None,
        sample_id: str = "unknown",
    ) -> PerceptionResult:
        frame = self.loader.load(rgb_path=rgb_path, depth_path=depth_path)
        h, w = frame.rgb.shape[:2]
        if hasattr(self.parser, "parse"):
            parsed = self.parser.parse(image_shape=(h, w), raw_target=raw_target)
        else:
            fallback_obj = raw_target or {
                "label": "sample_container",
                "score": 0.6,
                "bbox": [int(w * 0.3), int(h * 0.25), int(w * 0.7), int(h * 0.8)],
            }
            parsed_obj = self.parser.parse_object(fallback_obj, image_shape=(h, w))
            parsed = type(
                "ParsedTarget",
                (),
                {
                    "target_name": parsed_obj.class_name,
                    "bbox": parsed_obj.bbox or [0, 0, w - 1, h - 1],
                    "segmentation": None,
                    "region_reference": parsed_obj.region,
                    "flags": parsed_obj.quality_flags,
                },
            )()

        center = bbox_center_xyxy(parsed.bbox)
        stats = self.region_stats.compute(
            depth_m=frame.depth_m,
            bbox=parsed.bbox,
            segmentation=parsed.segmentation,
        )
        center_info = self.center_depth_extractor.extract(
            depth_m=frame.depth_m,
            center_point=center,
            region_fallback_depth=float(stats.get("region_depth_median", 0.0)),
        )
        center_depth = float(center_info["center_depth"])

        xyz, xyz_status = self.xyz_converter.to_xyz(
            center_point=center,
            center_depth=center_depth,
            camera_intrinsics=camera_intrinsics,
        )

        flags = list(parsed.flags)
        if float(stats.get("valid_depth_ratio", 0.0)) < 0.15:
            flags.append("low_valid_depth_ratio")
        if str(center_info.get("center_depth_source", "")) != "center_pixel":
            flags.append("center_depth_fallback_used")
        if xyz_status != "ok":
            flags.append(f"xyz_unavailable:{xyz_status}")

        target_representation = build_target_representation(
            sample_id=sample_id,
            target_name=parsed.target_name,
            bbox=parsed.bbox,
            center_point=center,
            center_depth=center_depth,
            region_depth_mean=float(stats.get("region_depth_mean", 0.0)),
            region_depth_median=float(stats.get("region_depth_median", 0.0)),
            valid_depth_ratio=float(stats.get("valid_depth_ratio", 0.0)),
            region_reference=parsed.region_reference,
            xyz=xyz,
            flags=flags,
        )

        depth_info: Dict[str, float] = {
            "center_depth": center_depth,
            "region_depth_mean": float(stats.get("region_depth_mean", 0.0)),
            "region_depth_median": float(stats.get("region_depth_median", 0.0)),
            "valid_depth_ratio": float(stats.get("valid_depth_ratio", 0.0)),
        }
        confidence = min(1.0, max(0.1, depth_info["valid_depth_ratio"] + 0.2))

        return PerceptionResult(
            target_name=parsed.target_name,
            bbox=parsed.bbox,
            center_point=center,
            depth_info=depth_info,
            confidence=confidence,
            segmentation=parsed.segmentation,
            region_reference=parsed.region_reference,
            xyz=xyz,
            target_representation=target_representation,
        )

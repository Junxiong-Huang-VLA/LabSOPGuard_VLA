from __future__ import annotations

from typing import Any, Dict, Iterable, List


class StructuredOutputBuilder:
    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    def build(
        self,
        experiment: Any,
        timeline: Any,
        steps: Iterable[Any],
        preprocessing_artifact: Any,
    ) -> Dict[str, Any]:
        video_assets = getattr(experiment, "video_assets", []) or []
        shared_video_refs = [
            {
                "video_asset_id": getattr(asset, "asset_id", None),
                "filename": getattr(asset, "filename", ""),
                "file_path": getattr(asset, "file_path", None),
                "duration_sec": getattr(asset, "duration_sec", None),
                "fps": getattr(asset, "fps", None),
            }
            for asset in video_assets
        ] or [{"video_asset_id": getattr(timeline, "video_asset_id", None)}]

        step_payload: List[Dict[str, Any]] = []
        for step in steps:
            evidence_refs = [ref.to_dict() if hasattr(ref, "to_dict") else dict(ref) for ref in getattr(step, "evidence_refs", [])]
            image_refs = [ref for ref in evidence_refs if ref.get("evidence_type") == "video_frame"]
            text_refs = [evt for evt in getattr(step, "linked_context_events", [])]
            event_refs = [evt for evt in getattr(step, "linked_physical_events", [])]
            params = [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in getattr(step, "parameters", [])]
            metadata = getattr(step, "metadata", {}) or {}
            step_payload.append(
                {
                    "step_id": getattr(step, "step_id", ""),
                    "step_name": getattr(step, "step_name", ""),
                    "stage_name": metadata.get("stage_name", "execution"),
                    "start_time": getattr(step, "start_time_sec", 0.0),
                    "end_time": getattr(step, "end_time_sec", getattr(step, "start_time_sec", 0.0)),
                    "status": self._enum_value(getattr(step, "status", "")),
                    "completion_type": "inferred" if getattr(step, "completed_by_inference", False) else "observed",
                    "image_refs": image_refs,
                    "video_refs": shared_video_refs,
                    "text_refs": text_refs,
                    "event_refs": event_refs,
                    "summary": getattr(step, "step_description", ""),
                    "inference_result": {
                        "completed_by_inference": getattr(step, "completed_by_inference", False),
                        "inference_method": getattr(step, "inference_method", None),
                        "inference_model": getattr(step, "inference_model", None),
                    },
                    "parameters": params,
                    "confidence": getattr(step, "confidence", 0.0),
                    "provenance": getattr(step, "provenance", None).to_dict() if getattr(step, "provenance", None) else None,
                    "evidence_notes": getattr(step, "evidence_notes", "") or metadata.get("evidence_notes", ""),
                }
            )

        return {
            "experiment_id": getattr(experiment, "experiment_id", ""),
            "title": getattr(experiment, "title", ""),
            "status": self._enum_value(getattr(experiment, "status", "")),
            "processing_stage": self._enum_value(getattr(experiment, "processing_stage", "")),
            "models_used": getattr(experiment, "models_used", []),
            "timeline": {
                "timeline_id": getattr(timeline, "timeline_id", ""),
                "total_steps": getattr(timeline, "total_steps", 0),
                "confirmed_steps": getattr(timeline, "confirmed_steps", 0),
                "candidate_steps": getattr(timeline, "candidate_steps", 0),
                "inferred_steps": getattr(timeline, "inferred_steps", 0),
                "video_duration_sec": getattr(timeline, "video_duration_sec", 0.0),
                "video_coverage_ratio": getattr(timeline, "video_coverage_ratio", 0.0),
            },
            "input_layer": {
                "context_summary": getattr(timeline, "context_summary", None),
                "protocol_text": getattr(timeline, "protocol_text", None),
                "video_inputs": shared_video_refs,
            },
            "preprocessing_layer": {
                "aligned_text": [item.__dict__ for item in getattr(preprocessing_artifact, "aligned_text", [])],
                "key_timestamps": getattr(preprocessing_artifact, "key_timestamps", []),
                "video_index": getattr(preprocessing_artifact, "video_index", []),
                "detected_changes": getattr(preprocessing_artifact, "detected_changes", []),
                "video_streams": getattr(preprocessing_artifact, "video_streams", []),
                "key_frames": getattr(preprocessing_artifact, "key_frames", []),
                "key_clips": getattr(preprocessing_artifact, "key_clips", []),
                "time_anchored_material_stream": getattr(preprocessing_artifact, "time_anchored_material_stream", []),
                "alignment_summary": getattr(preprocessing_artifact, "alignment_summary", {}),
            },
            "steps": step_payload,
        }

# Repository Agent Notes

- Keep `src/key_action_indexer` independent from the existing LabSOPGuard application.
- The project mainline is physical-evidence extraction from long dual-view experiment videos, multimodal time alignment, text descriptions, vector indexing, and query.
- Current key action work must prioritize YOLO-backed key action segments, hand-object interaction evidence, micro-segments, multiview clip alignment, metadata, and retrieval.
- Do not expand complex frontend work, cloud PTZ, five-camera orchestration, camera port mapping, or other infrastructure outside this mainline.
- PTZ is intentionally split out as a separate D-drive project at `D:\PtzTracker`; this repository should not reintroduce PTZ pages, routes, services, launchers, or MQTT tooling.
- Multi-camera and wireless-video monitoring are intentionally split out at `D:\MultiCameraMonitor`; LabCapability keeps only key-action dual-view upload/input metadata and should not reintroduce `/api/v1/cameras`, wireless video SDKs, or multi-monitor recording endpoints as core scope.
- Do not treat YOLO bounding-box rendering as the final deliverable; it must feed segment/micro-segment metadata and retrieval.
- Dry-run mode must remain runnable without real video files or ffmpeg.
- All multi-thread, multi-agent, and parallel development tasks default to model GPT-5.5 with xhigh reasoning. If the active Codex environment does not support xhigh, fall back to high, but keep gpt5.5xhigh as the project default development standard.
- Every change should preserve `pytest -q`, frontend `npm run build`, backend Python compilation, existing key-actions pages, and existing segment-level retrieval.

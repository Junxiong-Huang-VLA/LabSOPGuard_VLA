# Project Scope

## Mainline

This repository's key-action work is limited to experiment process understanding:

- extract physical evidence from long dual-view or multiview experiment videos;
- align video, transcript, user text, AI replies, uploads, and history onto one session timeline;
- produce key action segments, micro-segments, state evidence, step reasoning, and retrieval metadata;
- write reusable JSON/JSONL artifacts into a standard session directory.

## Non Goals

The key-action mainline does not include complex frontend expansion, cloud PTZ control, five-camera orchestration, camera port mapping, storage platform integration, or standalone YOLO box rendering as a final deliverable.

PTZ tracking/control has been split out of this repository. The working copy for that capability lives at `D:\PtzTracker`; LabEmbodied should stay focused on YOLO-backed key action segments, hand-object evidence, micro-segments, multiview alignment, metadata, and retrieval.

Multi-camera and wireless-video monitoring have also been split out. The working copy lives at `D:\MultiCameraMonitor`; LabEmbodied keeps dual-view video input through upload/session metadata and does not own camera orchestration, wireless video SDKs, or multi-monitor recording services.

Run `scripts/check_project_scope.ps1` before merging key-action changes. The guard fails if PTZ, wireless-video, camera proxy/streaming, or multi-monitor orchestration code appears in the LabEmbodied core paths.

YOLO output is an intermediate physical-evidence source. It must feed segment metadata, micro-segment metadata, evidence references, and retrieval text.

## Independence Boundary

`src/key_action_indexer` must remain usable without importing the LabSOPGuard application. It may read local model inventories, labels, or exported files when available, but those are optional inputs. Dry-run mode must remain runnable without real videos, ffmpeg, OpenCV video access, YOLO weights, or external model services.

## Development Gate

Every change should preserve:

- `pytest -q`;
- backend Python import and compilation;
- existing key-action pages and artifacts;
- existing segment-level and micro-segment retrieval.

Each task should map to at least one of: timeline alignment, key clips, physical evidence, step reasoning, evidence chain, structured output, or retrieval.

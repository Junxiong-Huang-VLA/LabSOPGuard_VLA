# Architecture

## End-to-End Flow

1. Video Capture: ingest camera/video stream.
2. Multi-level Detection: PPE/object/action signals.
3. SOP Engine: map signals to required steps and violations.
4. Alerting: realtime alert stream and persistence.
5. Reporting: auto-generate PDF/text compliance report.

## Core Modules

- `video.capture`: unified stream reader for webcam/file.
- `detection.multi_level_detector`: CV + VLM hybrid detector abstraction.
- `monitoring.sop_engine`: rule-based compliance state machine.
- `alerting.notifier`: console/file alert emitter.
- `reporting.pdf_report`: compliance report generator.
- `pipelines.sop_monitor_pipeline`: orchestration for runtime monitoring.

## Output Contracts

- Structured JSON/JSONL for detections/violations/status.
- CSV summary for analytics.
- PDF/TXT report for compliance audit.

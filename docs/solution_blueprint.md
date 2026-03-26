# Solution Blueprint (From Real SOP Draft)

This document maps your provided SOP planning text into executable engineering settings for LabSOPGuard.

## 1. Global Pipeline

采集 -> 检测 -> 告警 -> 归档 -> 报告

- Video ingestion: RTSP/USB
- Preprocess: decode, batching, ROI crop
- Multi-layer AI detection:
  - Layer1 YOLO26-pose
  - Layer2 Action analysis (SkateFormer/ST-GCN++)
  - Layer3 VLM semantic analysis (Qwen3-VL-8B)
  - Layer4 Step anomaly prediction (PREGO)
- Realtime alert channels: WebSocket/MQTT/Webhook
- Report generation: keyframes + timeline + LLM summary + PDF

## 2. Recommended Runtime Targets

- Primary hardware: RTX 3060 12GB
- Camera scale target: 4-8 streams
- Realtime screening: high-frequency lightweight detector
- Semantic analysis: low-frequency VLM on suspicious clips

## 3. Implemented Config Mapping

- `configs/sop/rules.yaml`: SOP steps + violation rules + severity mapping
- `configs/model/vla_model.yaml`: 4-layer model stack and runtime strategy
- `configs/alerts/alerting.yaml`: low-latency multi-channel alert policy
- `configs/report/report.yaml`: report sections and input/output contract

## 4. Delivery Phases

- Phase 1: MVP (single camera + basic violations + report)
- Phase 2: multi-camera + VLM semantic reasoning + dashboard
- Phase 3: DeepStream production pipeline + enterprise integration

## 5. Next Integration Actions

1. Replace placeholder rules with lab-specific SOP clauses.
2. Provide real sample videos and violation annotations.
3. Validate end-to-end output schema and report quality.
4. Tune thresholds per camera/environment.

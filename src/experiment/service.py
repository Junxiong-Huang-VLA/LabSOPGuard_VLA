"""
实验过程理解与步骤推理服务
Experiment Process Understanding Service

主处理链路：
1. 材料摄入 (Ingestion) - 接收视频、对话上下文、protocol
2. 视频理解 (Video Understanding) - Qwen-VL 帧分析
3. 上下文整合 (Context Integration) - 融合多源信息
4. 步骤推理 (Step Reasoning) - 生成 StepRecords
5. 证据关联 (Evidence Linking) - 关联多模态证据
6. 输出生成 (Output Generation) - 生成 Timeline + JSON
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from labsopguard.semantic_events import SemanticEventDetector
from labsopguard.semantic_sync import MultimodalSemanticSyncEngine
from labsopguard.time_sync import CameraSyncProfile, TimeSyncCalibrator
from labsopguard.video_input_schema import VideoInputValidationError, normalize_video_input
from labsopguard.detectors import build_yolo26_detector
from labsopguard.observability import StageTimer
from labsopguard.event_preprocessing.activity_presegmenter import ActivityPreSegmenter, PresegmentConfig
from labsopguard.event_preprocessing.experiment_segmenter import ExperimentSegmenter, SegmentationConfig
from labsopguard.event_preprocessing.gated_output import legacy_physical_event_candidate_rows
from labsopguard.event_preprocessing.physical_event_gate import gate_object_move

# 实验数据模型
from experiment.models import (
    Experiment, ExperimentTimeline, ExperimentStatus, StepRecord, StepStatus,
    StepConfidence, ContextEvent, MediaAsset, PhysicalEvent,
    MultimodalMaterialStreamItem, EvidenceRef, EvidenceType, ProvenanceInfo,
    ProcessStage, MediaType, ContextSource, StepParameter,
    make_confirmed_step, make_inferred_step, make_inferred_parameter,
    _now_iso, _uuid,
)

# DashScope VLM 客户端
try:
    from experiment.vlm_client import DashScopeVLClient, default_vlm_model, get_vlm_client, set_vlm_client
except ImportError:
    DashScopeVLClient = None
    default_vlm_model = None
    get_vlm_client = None
    set_vlm_client = None


# ---------------------------------------------------------------------------
# 默认提示词
# ---------------------------------------------------------------------------

DEFAULT_STEP_INFERENCE_PROMPT = """
你是一个实验室操作步骤推理专家。根据以下信息，推断实验过程步骤。

【视频帧描述】
{frames_context}

【对话上下文】
{conversation_context}

【Protocol】
{protocol_text}

请以 JSON 格式输出步骤列表：
{{
  "steps": [
    {{
      "step_name": "步骤名称",
      "step_description": "详细描述",
      "start_time_sec": 0.0,
      "end_time_sec": 5.0,
      "confidence": 0.8,
      "inference_level": "confirmed/candidate/inferred",
      "reasoning": "推断理由",
      "evidence_frame_ids": [0, 1, 2]
    }}
  ]
}}

注意：
- confirmed: 视频帧中明确观察到的步骤
- candidate: 有一定证据支持但不够明确的步骤（置信度 0.5-0.8）
- inferred: 基于上下文和 protocol 推断的步骤（置信度 < 0.5）
"""

DEFAULT_FRAME_ANALYSIS_PROMPT = """
分析这张实验室操作图像，返回 JSON：
{
  "description": "场景描述",
  "detected_activities": ["活动列表"],
  "object_labels": ["物体列表"],
  "step_indicators": ["步骤关键词"],
  "ppe_status": {"gloves": false, "goggles": false, "lab_coat": false},
  "confidence": 0.8
}
"""


# ---------------------------------------------------------------------------
# 视频帧提取器
# ---------------------------------------------------------------------------

class VideoFrameExtractor:
    """从视频中提取关键帧。"""

    def __init__(self, sample_interval_sec: float = 1.0, max_frames: int = 60):
        self.sample_interval_sec = sample_interval_sec
        self.max_frames = max_frames

    def extract_frames(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        提取视频帧。

        Returns:
            List of frame info dicts: {frame_id, timestamp_sec, path}
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = self._open_capture(video_path, source_type="file")
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / fps if fps > 0 else 0.0

            adaptive_target = max(6, min(self.max_frames, int(duration_sec / max(self.sample_interval_sec, 1.0)) + 2))
            interval_frames = max(1, total_frames // adaptive_target) if total_frames else 1
            frames = []
            saved_count = 0

            for i in range(0, total_frames, interval_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = i / fps
                frame_id = i

                frame_info = {
                    "frame_id": frame_id,
                    "timestamp_sec": round(timestamp, 3),
                    "width": width,
                    "height": height,
                    "fps": fps,
                }

                # 保存帧文件
                if output_dir:
                    out_path = Path(output_dir) / f"frame_{frame_id:06d}.jpg"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), frame)
                    frame_info["path"] = str(out_path)

                frames.append(frame_info)
                saved_count += 1

                if saved_count >= self.max_frames:
                    break

            return frames, {"fps": fps, "total_frames": total_frames, "duration_sec": duration_sec}
        finally:
            cap.release()

    @staticmethod
    def _open_capture(source: Any, source_type: str = "file") -> cv2.VideoCapture:
        normalized_source = source
        if source_type == "usb":
            try:
                normalized_source = int(source)
            except (TypeError, ValueError):
                normalized_source = source
        cap = cv2.VideoCapture(normalized_source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def extract_stream_frames(
        self,
        source: str,
        source_type: str = "rtsp",
        output_dir: Optional[str] = None,
        capture_duration_sec: Optional[float] = None,
        record_output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        从实时流或非本地文件源中采样关键帧。

        对于 RTSP/USB，按 sample_interval_sec 近似采样，最多提取 max_frames。
        """
        cap = self._open_capture(source, source_type=source_type)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1e-6:
                fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            effective_duration = capture_duration_sec
            if effective_duration is None or effective_duration <= 0:
                if total_frames > 0:
                    effective_duration = total_frames / fps if fps > 0 else 0.0
                else:
                    effective_duration = self.sample_interval_sec * max(self.max_frames - 1, 1)

            interval_frames = max(1, int(round(max(self.sample_interval_sec, 0.1) * fps)))
            frames: List[Dict[str, Any]] = []
            frame_index = 0
            saved_count = 0
            last_timestamp = 0.0
            writer = None
            recorded_frame_count = 0
            read_failures = 0
            total_read_failures = 0
            reconnect_attempts = 0
            max_consecutive_read_failures = 3 if source_type in {"rtsp", "usb", "http", "rtmp", "udp"} else 0
            recorded_path = Path(record_output_path) if record_output_path else None
            if recorded_path:
                recorded_path.parent.mkdir(parents=True, exist_ok=True)

            while saved_count < self.max_frames:
                ok, frame = cap.read()
                if not ok:
                    read_failures += 1
                    total_read_failures += 1
                    if read_failures <= max_consecutive_read_failures:
                        reconnect_attempts += 1
                        cap.release()
                        cap = self._open_capture(source, source_type=source_type)
                        if cap.isOpened():
                            continue
                    break
                read_failures = 0
                if recorded_path is not None:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(recorded_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (frame.shape[1], frame.shape[0]),
                        )
                    if writer is not None and writer.isOpened():
                        writer.write(frame)
                        recorded_frame_count += 1

                timestamp = frame_index / fps if fps > 0 else saved_count * self.sample_interval_sec
                if total_frames <= 0 and effective_duration and timestamp > effective_duration:
                    break

                should_capture = frame_index == 0 or frame_index % interval_frames == 0
                if should_capture:
                    frame_info = {
                        "frame_id": frame_index,
                        "timestamp_sec": round(timestamp, 3),
                        "width": width,
                        "height": height,
                        "fps": fps,
                    }
                    if output_dir:
                        out_path = Path(output_dir) / f"frame_{frame_index:06d}.jpg"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_path), frame)
                        frame_info["path"] = str(out_path)
                    frames.append(frame_info)
                    saved_count += 1
                    last_timestamp = timestamp

                frame_index += 1
                if total_frames > 0 and frame_index >= total_frames:
                    break

            if writer is not None:
                writer.release()

            duration_sec = round(
                (total_frames / fps) if total_frames > 0 else (last_timestamp if frames else effective_duration or 0.0),
                3,
            )
            stream_health = {
                "opened": True,
                "source_type": source_type,
                "frames_read": frame_index,
                "frames_sampled": saved_count,
                "frames_recorded": recorded_frame_count,
                "read_failures": total_read_failures,
                "consecutive_read_failures": read_failures,
                "reconnect_attempts": reconnect_attempts,
                "configured_fps": round(float(fps), 4),
                "observed_sample_interval_sec": round(
                    duration_sec / max(saved_count - 1, 1),
                    4,
                ) if saved_count > 1 else None,
                "recorded_file_path": str(recorded_path) if recorded_path and recorded_path.exists() else None,
            }
            return frames, {
                "fps": fps,
                "total_frames": total_frames if total_frames > 0 else frame_index,
                "duration_sec": duration_sec,
                "source_type": source_type,
                "is_live_source": source_type in {"rtsp", "usb", "http", "rtmp", "udp"},
                "recorded_file_path": str(recorded_path) if recorded_path and recorded_path.exists() else None,
                "recorded_frame_count": recorded_frame_count,
                "stream_health": stream_health,
            }
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# 步骤推理器（使用 Qwen-VL）
# ---------------------------------------------------------------------------

class StepReasoner:
    """
    步骤推理器。将帧分析结果 + 上下文 + protocol 转化为步骤列表。
    """

    def __init__(
        self,
        vlm_client: Optional[DashScopeVLClient] = None,
        default_confidence: float = 0.65,
    ):
        self.vlm_client = vlm_client
        self.default_confidence = default_confidence
        self._vlm_available = (
            vlm_client is not None
            and DashScopeVLClient is not None
        )

    def analyze_frame(self, frame_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """分析单帧图像。"""
        if self._vlm_available and self.vlm_client:
            try:
                print(f"[StepReasoner] Calling VLM for frame: {frame_path}")
                desc = self.vlm_client.describe_scene(
                    image_path=frame_path,
                    prompt=prompt or "Analyze this laboratory scene. Identify activities, objects, and safety equipment.",
                )
                print(f"[StepReasoner] VLM success: confidence={desc.confidence}, model={desc.model}")
                return {
                    "description": desc.description,
                    "detected_activities": desc.detected_activities,
                    "object_labels": desc.object_labels,
                    "step_indicators": desc.step_indicators,
                    "ppe_status": desc.ppe_status,
                    "confidence": desc.confidence,
                    "model": desc.model,
                    "inference_time_ms": desc.inference_time_ms,
                    "inferred": False,
                }
            except Exception as e:
                print(f"[StepReasoner] VLM failed: {e}")
                import traceback
                traceback.print_exc()
                return {"error": str(e), "inferred": True, "confidence": 0.0}
        else:
            print(f"[StepReasoner] VLM not available, using fallback")
            return self._fallback_frame_analysis(frame_path)

    def _fallback_frame_analysis(self, frame_path: str) -> Dict[str, Any]:
        """Fallback when VLM is not available."""
        cap = cv2.VideoCapture(frame_path)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok:
                h, w = frame.shape[:2]
                return {
                    "description": "Fallback: no VLM available",
                    "detected_activities": [],
                    "object_labels": [],
                    "step_indicators": [],
                    "ppe_status": {},
                    "confidence": 0.3,
                    "model": "fallback_heuristic",
                    "inference_time_ms": 0,
                    "inferred": True,
                }
        return {
            "description": "",
            "detected_activities": [],
            "object_labels": [],
            "step_indicators": [],
            "ppe_status": {},
            "confidence": 0.0,
            "model": "fallback_heuristic",
            "inference_time_ms": 0,
            "inferred": True,
        }

    def infer_steps_from_frames(
        self,
        frames: List[Dict[str, Any]],
        frame_analyses: List[Dict[str, Any]],
        context: str,
        protocol: str,
        context_events: Optional[List[ContextEvent]] = None,
    ) -> List[StepRecord]:
        """
        根据帧分析结果和上下文推断步骤。

        Args:
            frames: 视频帧信息列表
            frame_analyses: 帧分析结果列表
            context: 对话上下文文本
            protocol: 实验 protocol 文本
            context_events: ContextEvent 列表（关键：参与步骤推理）

        Returns:
            StepRecord 列表
        """
        # 1. 构建帧上下文摘要
        frames_context = self._build_frames_context(frames, frame_analyses)

        # 2. 构建上下文事件摘要（关键：让 ContextEvent 参与推理）
        context_summary = self._build_context_summary(context_events)

        # 3. 调用 VLM 进行步骤推理（如果可用）
        if self._vlm_available and self.vlm_client:
            steps = self._vlm_step_inference(
                frames_context, context, protocol, frames, frame_analyses,
                context_summary=context_summary, context_events=context_events
            )
        else:
            steps = self._fallback_step_inference(
                frames, frame_analyses,
                context_events=context_events
            )

        return steps

    def _build_frames_context(
        self,
        frames: List[Dict[str, Any]],
        analyses: List[Dict[str, Any]],
    ) -> str:
        """构建帧上下文摘要文本。"""
        lines = []
        for frame, analysis in zip(frames, analyses):
            if "error" in analysis:
                continue
            ts = frame.get("timestamp_sec", 0)
            desc = analysis.get("description", "")
            activities = analysis.get("detected_activities", [])
            objects = analysis.get("object_labels", [])
            ppe = analysis.get("ppe_status", {})
            ppe_str = f"PPE: gloves={ppe.get('gloves', False)}, goggles={ppe.get('goggles', False)}"
            lines.append(
                f"[@{ts:.1f}s] {desc} | Activities: {activities} | Objects: {objects} | {ppe_str}"
            )
        return "\n".join(lines)

    def _build_context_summary(self, context_events: Optional[List[ContextEvent]] = None) -> str:
        """
        构建上下文事件摘要，用于步骤推理。
        ContextEvent 在此处真正参与推理链路。
        """
        if not context_events:
            return ""
        lines = []
        for evt in context_events:
            src = evt.context_source.value if isinstance(evt.context_source, ContextSource) else str(evt.context_source)
            ts_str = f"@{evt.timestamp_sec:.1f}s" if evt.timestamp_sec is not None else ""
            lines.append(f"[{src}]{ts_str} {evt.event_type}: {evt.content[:200]}")
        return "\n".join(lines)

    def _vlm_step_inference(
        self,
        frames_context: str,
        context: str,
        protocol: str,
        frames: List[Dict[str, Any]],
        analyses: List[Dict[str, Any]],
        context_summary: str = "",
        context_events: Optional[List[ContextEvent]] = None,
    ) -> List[StepRecord]:
        """使用 Qwen-VL 推断步骤（ContextEvent 参与推理）。"""
        # 将 context_summary 融入推理 prompt
        enriched_context = context or ""
        if context_summary:
            enriched_context += f"\n\n【上下文事件】\n{context_summary}"

        prompt = DEFAULT_STEP_INFERENCE_PROMPT.format(
            frames_context=frames_context or "[No frames available]",
            conversation_context=enriched_context or "[No conversation context]",
            protocol_text=protocol or "[No protocol]",
        )

        try:
            # 用最后一个有效帧进行分析
            valid_frames = [f for f in frames if f.get("path") and Path(f["path"]).exists()]
            if valid_frames:
                last_frame = valid_frames[-1]
                desc = self.vlm_client.describe_scene(
                    last_frame["path"],
                    prompt=(
                        f"基于以下帧上下文和实验信息，推断实验步骤。\n\n"
                        f"帧上下文摘要：\n{frames_context}\n\n"
                        f"对话上下文：{context}\n\n"
                        f"Protocol：{protocol}\n\n"
                        "请输出一系列步骤，包含步骤名、描述、时间范围和置信度。"
                    ),
                )
                return self._parse_vlm_steps(desc, frames)
            return self._fallback_step_inference(frames, analyses, context_events=context_events)
        except Exception as e:
            print(f"[StepReasoner] VLM inference failed: {e}")
            return self._fallback_step_inference(frames, analyses, context_events=context_events)

    def _parse_vlm_steps(
        self,
        desc: Any,
        frames: List[Dict[str, Any]],
    ) -> List[StepRecord]:
        """解析 VLM 返回的步骤描述。"""
        # 从 VLMSceneDescription 提取步骤
        steps = []
        raw_text = desc.description if hasattr(desc, "description") else str(desc)

        # 简单解析：如果没有明确的步骤结构，则创建一个综合步骤
        if frames:
            first_ts = frames[0].get("timestamp_sec", 0.0)
            last_ts = frames[-1].get("timestamp_sec", 10.0)

            step = make_confirmed_step(
                experiment_id="",
                step_index=0,
                step_name="实验操作",
                start_time_sec=first_ts,
                end_time_sec=last_ts,
                description=raw_text[:500],
            )
            steps.append(step)

        return steps

    def _fallback_step_inference(
        self,
        frames: List[Dict[str, Any]],
        analyses: List[Dict[str, Any]],
        context_events: Optional[List[ContextEvent]] = None,
    ) -> List[StepRecord]:
        """
        Fallback 步骤推理：当 VLM 不可用时，基于帧分析生成步骤。

        策略：
        1. 将活动变化点作为步骤分割点
        2. 相似活动聚合为同一步骤
        3. 无活动帧跳过
        4. ContextEvent 提供步骤先验和语义提示
        """
        # 从 ContextEvent 提取步骤先验（protocol 步骤列表）
        protocol_steps = self._extract_protocol_steps(context_events)

        steps = []
        current_step_frames = []
        prev_activities = set()

        for frame, analysis in zip(frames, analyses):
            activities = set(analysis.get("detected_activities", []))

            # 活动发生变化，创建新步骤
            if activities != prev_activities and activities:
                if current_step_frames:
                    step = self._aggregate_frames_to_step(
                        current_step_frames, len(steps),
                        context_events=context_events,
                        protocol_steps=protocol_steps,
                    )
                    steps.append(step)
                current_step_frames = [frame]
                prev_activities = activities
            elif activities:
                current_step_frames.append(frame)

        # 处理最后一个步骤
        if current_step_frames:
            step = self._aggregate_frames_to_step(
                current_step_frames, len(steps),
                context_events=context_events,
                protocol_steps=protocol_steps,
            )
            steps.append(step)

        # 如果没有步骤，创建默认步骤并关联所有 context_events
        if not steps and frames:
            # 使用 context_events 中的 protocol 信息丰富描述
            desc = "基于视频帧分析推断的实验操作"
            linked_ctx_ids = []
            if context_events:
                for evt in context_events:
                    linked_ctx_ids.append(evt.event_id)
                    if evt.context_source == ContextSource.PROTOCOL:
                        desc = f"基于 Protocol 推断: {evt.content[:100]}"

            step = make_inferred_step(
                experiment_id="",
                step_index=0,
                step_name="实验操作",
                start_time_sec=frames[0].get("timestamp_sec", 0.0),
                end_time_sec=frames[-1].get("timestamp_sec", 0.0) if frames else None,
                confidence=0.5,
                inference_method="fallback_activity_clustering",
                inference_model="heuristic",
                description=desc,
                linked_context=linked_ctx_ids,
            )
            step.metadata["context_participated"] = bool(linked_ctx_ids)
            step.metadata["protocol_step_matched"] = any(
                e.context_source == ContextSource.PROTOCOL for e in (context_events or [])
            )
            steps.append(step)

        return steps

    def _extract_protocol_steps(self, context_events: Optional[List[ContextEvent]] = None) -> List[str]:
        """从 ContextEvent 中提取 protocol 步骤列表作为先验。"""
        if not context_events:
            return []
        protocol_steps = []
        for evt in context_events:
            if evt.context_source == ContextSource.PROTOCOL and evt.content:
                # 简单解析：按行分割，提取编号步骤
                for line in evt.content.split("\n"):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("-") or line.startswith("Step")):
                        protocol_steps.append(line)
        return protocol_steps

    def _aggregate_frames_to_step(
        self,
        frames: List[Dict[str, Any]],
        step_index: int,
        context_events: Optional[List[ContextEvent]] = None,
        protocol_steps: Optional[List[str]] = None,
    ) -> StepRecord:
        """将一组帧聚合为一个步骤。ContextEvent 参与步骤命名和语义丰富。"""
        ts_start = frames[0].get("timestamp_sec", 0.0)
        ts_end = frames[-1].get("timestamp_sec", ts_start)

        # 合并所有帧的活动
        all_activities = []
        all_objects = []
        ppe_status = {"gloves": False, "goggles": False, "lab_coat": False}
        total_conf = 0.0
        count = 0

        for frame in frames:
            all_activities.append(f"frame_{frame['frame_id']}")
            total_conf += frame.get("confidence", 0.5)
            count += 1

        avg_conf = total_conf / count if count > 0 else 0.5

        # 使用 protocol_steps 作为步骤名先验
        step_name = all_activities[0] if all_activities else "未知操作"
        if len(all_activities) > 1:
            step_name = f"{all_activities[0]} -> ..."

        # 如果有 protocol 步骤先验，尝试匹配
        linked_ctx_ids = []
        context_hint = ""
        if protocol_steps and step_index < len(protocol_steps):
            step_name = protocol_steps[step_index]
            context_hint = f"(来自 Protocol 步骤 {step_index + 1})"

        # 关联时间范围内的 ContextEvent
        if context_events:
            for evt in context_events:
                if evt.timestamp_sec is not None:
                    if ts_start <= evt.timestamp_sec <= ts_end:
                        linked_ctx_ids.append(evt.event_id)
                        if evt.context_source == ContextSource.CONVERSATION:
                            context_hint += f" | 对话: {evt.content[:80]}"

        description = f"包含 {len(frames)} 帧的操作片段"
        if context_hint:
            description += f" {context_hint}"

        return StepRecord(
            experiment_id="",
            step_index=step_index,
            step_name=step_name,
            step_description=description,
            status=StepStatus.CONFIRMED if avg_conf >= 0.7 else StepStatus.CANDIDATE,
            start_time_sec=ts_start,
            end_time_sec=ts_end,
            duration_sec=round(ts_end - ts_start, 3),
            confidence=round(avg_conf, 4),
            step_confidence=(
                StepConfidence.HIGH if avg_conf >= 0.8
                else StepConfidence.MEDIUM if avg_conf >= 0.5
                else StepConfidence.LOW
            ),
            completed_by_inference=False,
            parameters=[],
            evidence_refs=[
                EvidenceRef(
                    evidence_type=EvidenceType.VIDEO_FRAME,
                    source="video",
                    frame_id=f["frame_id"],
                    timestamp_sec=f.get("timestamp_sec", 0.0),
                    confidence=avg_conf,
                )
                for f in frames
            ],
            linked_context_events=linked_ctx_ids,
            provenance=ProvenanceInfo(
                source="video",
                confidence=avg_conf,
                is_inferred=False,
            ),
            metadata={
                "context_participated": bool(linked_ctx_ids),
                "protocol_step_matched": bool(protocol_steps and step_index < len(protocol_steps)),
            },
        )


# ---------------------------------------------------------------------------
# 实验服务主类
# ---------------------------------------------------------------------------

class ExperimentService:
    """
    实验过程理解服务。

    用法：
    ```python
    service = ExperimentService()
    service.set_video("/path/to/video.mp4")
    service.set_context("操作员在第5分钟更换了移液管...")
    service.set_protocol("Protocol: 1. 准备样本 2. 稀释 3. 加样...")
    result = service.process()

    experiment = result["experiment"]
    timeline = result["timeline"]
    print(timeline.step_summary())
    ```
    """

    def __init__(
        self,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        vlm_model: Optional[str] = None,
        frame_sample_interval: float = 2.0,
        max_frames: int = 30,
        yolo26_weights_path: Optional[str] = None,
        detector_device: Optional[str] = None,
    ):
        # 视频信息
        self._video_path: Optional[str] = None
        self._video_paths: List[str] = []
        self._video_inputs: List[Dict[str, Any]] = []
        self._video_frames: List[Dict[str, Any]] = []
        self._video_info: Dict[str, Any] = {}
        self._frame_analyses: List[Dict[str, Any]] = []

        # 上下文
        self._context_text: str = ""
        self._protocol_text: str = ""
        self._context_events: List[ContextEvent] = []
        self._context_inputs: List[Dict[str, Any]] = []

        # VLM 客户端
        self._vlm_client: Optional[DashScopeVLClient] = None
        resolved_vlm_model = (
            vlm_model
            or (default_vlm_model() if default_vlm_model is not None else None)
            or os.environ.get("KEY_ACTION_VLM_MODEL")
            or os.environ.get("QWEN_VL_MODEL")
            or os.environ.get("VLM_MODEL")
            or "qwen3.6-plus"
        )
        if vlm_api_key and DashScopeVLClient:
            try:
                self._vlm_client = DashScopeVLClient(
                    api_key=vlm_api_key,
                    base_url=vlm_base_url,
                    model=resolved_vlm_model,
                )
                print(f"[ExperimentService] VLM client initialized: {resolved_vlm_model}")
            except Exception as e:
                print(f"[ExperimentService] VLM init failed: {e}")

        # 子组件
        self._frame_extractor = VideoFrameExtractor(
            sample_interval_sec=frame_sample_interval,
            max_frames=max_frames,
        )
        self._step_reasoner = StepReasoner(vlm_client=self._vlm_client)
        self._detector = build_yolo26_detector(
            weights_path=yolo26_weights_path,
            device=detector_device,
            confidence_threshold=float(os.getenv("YOLO26_CONFIDENCE_THRESHOLD", "0.25")),
            iou_threshold=float(os.getenv("YOLO26_IOU_THRESHOLD", "0.45")),
            max_detections=int(os.getenv("YOLO26_MAX_DETECTIONS", "50")),
        )
        if self._detector is not None:
            print(f"[ExperimentService] detector status: {self._detector.status.to_dict()}")

        # 实验状态
        self._experiment: Optional[Experiment] = None
        self._timeline: Optional[ExperimentTimeline] = None
        self._material_stream: List[MultimodalMaterialStreamItem] = []
        self._semantic_sync: Dict[str, Any] = {}
        self._artifact_root: Optional[Path] = None

    # ------------------------------------------------------------------
    # 输入设置
    # ------------------------------------------------------------------

    def set_video(self, video_path: str) -> "ExperimentService":
        """设置视频路径。"""
        self._video_path = video_path
        self._video_paths = [video_path] if video_path else []
        self._video_inputs = []
        return self

    def set_videos(self, video_paths: List[str]) -> "ExperimentService":
        """设置多视频路径。"""
        cleaned_paths = [str(path) for path in video_paths if path]
        self._video_paths = cleaned_paths
        self._video_path = cleaned_paths[0] if cleaned_paths else None
        self._video_inputs = []
        return self

    def set_video_inputs(self, video_inputs: List[Dict[str, Any]]) -> "ExperimentService":
        """设置多视频输入描述，支持显式时间偏移。"""
        normalized_inputs: List[Dict[str, Any]] = []
        resolved_paths: List[str] = []
        for index, item in enumerate(video_inputs or []):
            if not isinstance(item, dict):
                continue
            try:
                normalized, _ = normalize_video_input(item, index=index, strict=False)
            except VideoInputValidationError:
                continue
            path = normalized.get("video_path")
            if not path:
                continue
            normalized_inputs.append(normalized)
            resolved_paths.append(str(path))
        self._video_inputs = normalized_inputs
        self._video_paths = resolved_paths
        self._video_path = resolved_paths[0] if resolved_paths else None
        return self

    def set_context(self, context_text: str) -> "ExperimentService":
        """设置对话上下文。"""
        self._context_text = context_text
        return self

    def set_context_inputs(self, context_inputs: List[Dict[str, Any]]) -> "ExperimentService":
        """设置带时间锚点的上下文输入。"""
        self._context_inputs = [dict(item) for item in (context_inputs or []) if isinstance(item, dict)]
        if not self._context_text:
            texts = [str(item.get("text", "")).strip() for item in self._context_inputs if str(item.get("text", "")).strip()]
            self._context_text = "\n".join(texts).strip()
        return self

    def set_protocol(self, protocol_text: str) -> "ExperimentService":
        """设置实验 Protocol。"""
        self._protocol_text = protocol_text
        return self

    def set_context_events(self, events: List[ContextEvent]) -> "ExperimentService":
        """设置上下文事件列表。"""
        self._context_events = events
        return self

    # ------------------------------------------------------------------
    # 处理流程
    # ------------------------------------------------------------------

    def process(
        self,
        experiment_id: Optional[str] = None,
        experiment_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行完整处理流程。

        Returns:
            {"experiment": Experiment, "timeline": ExperimentTimeline, "steps": List[StepRecord]}
        """
        timer = StageTimer()
        exp_id = experiment_id or _uuid()
        title = experiment_title or f"实验 {exp_id[:8]}"
        self._artifact_root = Path("outputs") / "experiments" / exp_id / "artifacts"
        if self._artifact_root.exists():
            shutil.rmtree(self._artifact_root, ignore_errors=True)
        self._artifact_root.mkdir(parents=True, exist_ok=True)

        # 创建 Experiment
        experiment = Experiment(
            experiment_id=exp_id,
            title=title,
            status=ExperimentStatus.PROCESSING,
            context_inputs=self._context_inputs or ([{"text": self._context_text}] if self._context_text else []),
            protocol_text=self._protocol_text,
            processing_stage=ProcessStage.INGESTION,
        )

        # Stage 1: 视频帧提取
        experiment.processing_stage = ProcessStage.INGESTION
        with timer.measure("ingestion"):
            self._extract_video(experiment)

        # Stage 1.5: 实验边界检测（长视频自动拆分）
        with timer.measure("experiment_segmentation"):
            segmentation = self._detect_experiment_boundaries(experiment)

        # Stage 1.7: Audio transcription (if available)
        with timer.measure("audio_transcription"):
            transcript_segments = self._transcribe_audio(experiment)
            if transcript_segments:
                experiment.metadata["transcript_segments"] = transcript_segments

        if segmentation and segmentation.total_segments > 1:
            sub_results = self._process_sub_experiments(experiment, segmentation, timer)
            timer.save(self._artifact_root / "processing_timing.json")
            return sub_results

        # Stage 2: 视频理解
        experiment.processing_stage = ProcessStage.VIDEO_UNDERSTANDING
        with timer.measure("video_understanding"):
            self._analyze_frames(experiment)
            semantic_sync = self._run_multimodal_semantic_sync(experiment)
            self._semantic_sync = semantic_sync

        # Stage 2.5: 生成多模态素材流（MultimodalMaterialStreamItem）
        with timer.measure("material_stream_generation"):
            material_stream = self._generate_material_stream(experiment)
            self._material_stream = material_stream

        # Stage 2.6: 生成物理事件（PhysicalEvent）
        with timer.measure("physical_events"):
            physical_events = self._generate_physical_events(experiment)
            experiment.physical_events = physical_events

        # Stage 3: 上下文整合
        experiment.processing_stage = ProcessStage.CONTEXT_INTEGRATION
        with timer.measure("context_integration"):
            self._integrate_context(experiment)

        # Stage 4: 步骤推理
        experiment.processing_stage = ProcessStage.STEP_REASONING
        with timer.measure("step_reasoning"):
            timeline = self._reason_steps(experiment)

        # Stage 5: 证据关联 + PhysicalEvent/MaterialStream 关联
        experiment.processing_stage = ProcessStage.EVIDENCE_LINKING
        with timer.measure("evidence_linking"):
            self._link_evidence(experiment, timeline)
            self._link_physical_events(experiment, timeline)
            self._link_material_stream(experiment, timeline, material_stream)

        # Stage 6: 完成
        experiment.processing_stage = ProcessStage.OUTPUT_GENERATION
        experiment.timeline = timeline
        experiment.sync_stats()
        experiment.status = ExperimentStatus.COMPLETED

        # 更新 models_used
        if self._vlm_client:
            experiment.models_used = [self._vlm_client.model]
        experiment.models_used.append("step_reasoner")
        if self._detector is not None and self._detector.status.available:
            experiment.models_used.append("ultralytics_yolo26")
            experiment.metadata["detector_status"] = self._detector.status.to_dict()

        self._experiment = experiment
        self._timeline = timeline
        self._material_stream = material_stream
        self._semantic_sync = semantic_sync

        # 保存 pipeline 各阶段耗时
        timer.save(self._artifact_root / "processing_timing.json")

        return {
            "experiment": experiment,
            "timeline": timeline,
            "steps": timeline.steps,
            "physical_events": physical_events,
            "material_stream": material_stream,
            "semantic_sync": semantic_sync,
        }

    def _detect_experiment_boundaries(self, experiment: "Experiment"):
        """Detect experiment boundaries in long videos using presegmentation + gap analysis."""
        from labsopguard.event_preprocessing.experiment_segmenter import ExperimentSegmenter, SegmentationConfig
        from labsopguard.event_preprocessing.activity_presegmenter import ActivityPreSegmenter, ActivitySegment, PresegmentConfig

        video_paths = [Path(f.get("frame_bgr_path") or f.get("video_path", ""))
                       for f in (self._video_descriptors() or [])
                       if f.get("video_path")]
        if not video_paths:
            video_paths = [Path(self._video_path)] if self._video_path else []

        if not video_paths:
            return None

        # Get video duration
        first_video = video_paths[0]
        if not first_video.exists():
            return None
        cap = cv2.VideoCapture(str(first_video))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        config = SegmentationConfig()

        # Run presegmentation on all streams and merge
        presegmenter = ActivityPreSegmenter(PresegmentConfig())
        all_segments = []
        for vpath in video_paths:
            if vpath.exists():
                segments = presegmenter.segment(vpath, stream_id=str(vpath.name))
                all_segments.extend(segments)

        if not all_segments:
            all_segments = [
                ActivitySegment(
                    start_sec=0.0,
                    end_sec=duration,
                    peak_score=1.0,
                    avg_score=1.0,
                    trigger="full_video_fallback",
                    stream_id=first_video.name,
                )
            ]

        # Merge overlapping segments from multiple streams
        merged = self._merge_multi_stream_segments(all_segments)

        # Detect experiment boundaries
        segmenter = ExperimentSegmenter(config, yolo_pipeline=self._detector)
        result = segmenter.segment(merged, video_path=first_video, video_duration_sec=duration)

        import json
        naming_manifest_path = None
        naming_enabled = os.environ.get("LABSOPGUARD_SEGMENT_NAMING_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
        if naming_enabled:
            try:
                from labsopguard.event_preprocessing.segment_naming import name_experiment_segments

                naming_manifest = name_experiment_segments(
                    video_path=first_video,
                    segmentation=result,
                    output_dir=self._artifact_root,
                )
                naming_manifest_path = self._artifact_root / "segment_naming.json"
                naming_manifest_path.write_text(
                    json.dumps(naming_manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Experiment segment naming failed: %s", exc)

        preview_manifest_path = None
        preview_error = None
        preview_enabled = os.environ.get("LABSOPGUARD_SEGMENTATION_PREVIEW_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
        if preview_enabled:
            try:
                from labsopguard.event_preprocessing.segmentation_preview import render_segmentation_preview

                segment_preview_dir = self._artifact_root / "segment_previews"
                segment_preview_dir.mkdir(parents=True, exist_ok=True)
                segment_preview_manifests = []
                annotated_preview_max_frames = int(os.environ.get("LABSOPGUARD_SEGMENT_ANNOTATED_PREVIEW_MAX_FRAMES", "180"))
                annotated_preview_interval = float(os.environ.get("LABSOPGUARD_SEGMENT_ANNOTATED_PREVIEW_INTERVAL_SEC", "2"))
                annotated_preview_yolo_enabled = os.environ.get("LABSOPGUARD_SEGMENT_ANNOTATED_PREVIEW_YOLO_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
                for seg in result.segments:
                    segment_preview_path = segment_preview_dir / f"{seg.segment_id}.mp4"
                    segment_manifest = render_segmentation_preview(
                        video_path=first_video,
                        segmentation=result,
                        output_path=segment_preview_path,
                        time_range=(seg.start_sec, seg.end_sec),
                        title=f"Experiment {seg.index + 1}",
                        detector=self._detector if annotated_preview_yolo_enabled else None,
                        yolo_overlay=annotated_preview_yolo_enabled,
                        sample_interval_sec=annotated_preview_interval,
                        max_frames=annotated_preview_max_frames,
                    )
                    seg.preview_video_path = str(segment_preview_path)
                    segment_preview_manifests.append({
                        **segment_manifest,
                        "segment_id": seg.segment_id,
                        "segment_index": seg.index,
                    })

                segment_manifest_path = segment_preview_dir / "manifest.json"
                segment_manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "segment_previews.v1",
                            "count": len(segment_preview_manifests),
                            "segments": segment_preview_manifests,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                preview_path = self._artifact_root / "segmentation_preview.mp4"
                preview_manifest = render_segmentation_preview(
                    video_path=first_video,
                    segmentation=result,
                    output_path=preview_path,
                )
                preview_manifest_path = self._artifact_root / "segmentation_preview.json"
                preview_manifest_path.write_text(json.dumps(preview_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                preview_error = str(exc)

        payload = result.to_dict()
        if naming_manifest_path is not None:
            payload["naming_manifest_path"] = str(naming_manifest_path)
        if preview_enabled and preview_manifest_path is not None:
            payload["preview_video_path"] = str(self._artifact_root / "segmentation_preview.mp4")
            payload["preview_manifest_path"] = str(preview_manifest_path)
            payload["preview_ready"] = True
            payload["segment_preview_ready_count"] = sum(1 for seg in result.segments if seg.preview_video_path)
        elif preview_enabled:
            payload["preview_ready"] = False
            payload["preview_error"] = preview_error or "Segmentation preview was not generated"

        seg_path = self._artifact_root / "experiment_segmentation.json"
        seg_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        return result

    @staticmethod
    def _merge_multi_stream_segments(segments):
        """Merge activity segments from multiple streams into unified timeline."""
        from labsopguard.event_preprocessing.activity_presegmenter import ActivitySegment
        if not segments:
            return []
        sorted_segs = sorted(segments, key=lambda s: s.start_sec)
        merged = [sorted_segs[0]]
        for seg in sorted_segs[1:]:
            if seg.start_sec <= merged[-1].end_sec + 2.0:
                merged[-1] = ActivitySegment(
                    start_sec=merged[-1].start_sec,
                    end_sec=max(merged[-1].end_sec, seg.end_sec),
                    peak_score=max(merged[-1].peak_score, seg.peak_score),
                    avg_score=(merged[-1].avg_score + seg.avg_score) / 2,
                    trigger="merged",
                    stream_id="multi",
                )
            else:
                merged.append(seg)
        return merged

    def _transcribe_audio(self, experiment) -> list:
        """Extract audio from video and run ASR transcription."""
        import logging
        logger = logging.getLogger(__name__)
        try:
            from labsopguard.asr import transcribe_audio_file
        except ImportError:
            logger.info("ASR module not available, skipping transcription")
            return []

        video_paths = [Path(f.get("video_path", ""))
                       for f in (self._video_descriptors() or [])
                       if f.get("video_path") and Path(f["video_path"]).exists()]
        if not video_paths and self._video_path:
            video_paths = [Path(self._video_path)]
        if not video_paths:
            return []

        try:
            result = transcribe_audio_file(str(video_paths[0]))
            segments = [seg.to_context_input() for seg in result.segments] if result and result.segments else []
            logger.info("ASR transcription completed: %d segments", len(segments))
            return segments
        except Exception as exc:
            logger.warning("ASR transcription failed: %s", exc)
            return []

    def _process_sub_experiments(self, parent_experiment, segmentation, timer):
        """Process each detected experiment segment as a sub-experiment."""
        import logging
        logger = logging.getLogger(__name__)

        parent_experiment.metadata["segmentation"] = segmentation.to_dict()
        parent_experiment.metadata["is_parent"] = True
        parent_experiment.metadata["sub_experiment_count"] = segmentation.total_segments

        sub_results = []
        for seg in segmentation.segments:
            sub_id = f"{parent_experiment.experiment_id}_sub{seg.index}"
            logger.info(
                "Processing sub-experiment %s: %.1fs - %.1fs",
                sub_id, seg.start_sec, seg.end_sec,
            )
            with timer.measure(f"sub_experiment_{seg.index}"):
                sub_result = self._run_single_segment(
                    parent_experiment, seg, sub_id
                )
                sub_results.append(sub_result)

        parent_experiment.status = ExperimentStatus.COMPLETED
        parent_experiment.metadata["sub_results_summary"] = [
            {
                "sub_id": r.get("experiment", {}).experiment_id if hasattr(r.get("experiment", {}), "experiment_id") else str(r.get("experiment", "")),
                "segment_index": i,
                "event_count": len(r.get("physical_events", [])),
            }
            for i, r in enumerate(sub_results)
        ]

        self._experiment = parent_experiment
        return {
            "experiment": parent_experiment,
            "segmentation": segmentation.to_dict(),
            "sub_experiments": sub_results,
            "timeline": sub_results[0].get("timeline") if sub_results else None,
            "steps": [],
            "physical_events": [],
            "material_stream": [],
            "semantic_sync": None,
        }

    def _run_single_segment(self, parent_experiment, segment, sub_id):
        """Run the full pipeline on a single time segment."""
        # Create a new service instance for this segment with time_range
        # We reuse the same video paths but limit processing to time_range
        from labsopguard.event_preprocessing.engine import EventPreprocessingEngine
        from labsopguard.config import load_runtime_settings

        time_range = (segment.start_sec, segment.end_sec)
        output_dir = Path("outputs") / "experiments" / parent_experiment.experiment_id / f"sub_{segment.index}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run event preprocessing with time_range
        video_paths = [Path(f.get("video_path", ""))
                       for f in (self._video_descriptors() or [])
                       if f.get("video_path") and Path(f["video_path"]).exists()]
        if not video_paths and self._video_path:
            video_paths = [Path(self._video_path)]

        if not video_paths:
            return {"experiment": None, "physical_events": [], "timeline": None}

        settings = load_runtime_settings(Path("."))
        engine = EventPreprocessingEngine(settings)

        source_video = str(video_paths[0])
        material_index_path = output_dir / "material_index.sqlite"

        result = engine.run(
            experiment_id=sub_id,
            experiment_name=f"{parent_experiment.title}_segment{segment.index}",
            source_video=source_video,
            output_dir=output_dir,
            material_index_path=material_index_path,
            time_range=time_range,
        )

        return {
            "experiment": {"experiment_id": sub_id, "time_range": time_range},
            "physical_events": result.get("events", []),
            "timeline": None,
            "segment": segment.to_dict(),
        }

    def _video_descriptors(self) -> List[Dict[str, Any]]:
        if self._video_inputs:
            return [dict(item) for item in self._video_inputs]
        return [
            {
                "video_index": index,
                "video_path": path,
            }
            for index, path in enumerate(self._video_paths or ([self._video_path] if self._video_path else []))
        ]

    @staticmethod
    def _compute_file_sha256(file_path: str) -> Optional[str]:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_sync_profile(
        self,
        descriptor: Dict[str, Any],
        camera_id: str,
        default_offset_sec: float,
    ) -> CameraSyncProfile:
        anchors = TimeSyncCalibrator.anchors_from_descriptor(camera_id, descriptor)
        if anchors:
            profile = TimeSyncCalibrator.fit_profile_from_anchors(
                camera_id=camera_id,
                anchors=anchors,
                reference_camera_id=str(descriptor.get("reference_camera_id") or "global"),
            )
        else:
            offset_source = str(descriptor.get("offset_source") or "").strip().lower()
            default_method = str(
                descriptor.get("sync_method")
                or ("explicit_offset" if offset_source == "explicit" else "sequential")
            )
            profile = CameraSyncProfile(
                camera_id=camera_id,
                reference_camera_id=str(descriptor.get("reference_camera_id") or "global"),
                offset_sec=round(default_offset_sec, 6),
                method=default_method,
                confidence=float(self._safe_float(descriptor.get("sync_confidence"), 0.7) or 0.7),
            )

        hardware_start = self._safe_float(descriptor.get("hardware_timecode_start_sec"))
        sync_board_offset = self._safe_float(descriptor.get("sync_board_offset_sec"))
        if hardware_start is not None:
            profile.offset_sec = round(hardware_start, 6)
            profile.method = "hardware_timecode"
            profile.confidence = max(profile.confidence, 0.95)
        if sync_board_offset is not None:
            profile.offset_sec = round(sync_board_offset, 6)
            profile.method = "sync_board"
            profile.confidence = max(profile.confidence, 0.9)

        clock_scale = self._safe_float(descriptor.get("clock_scale"))
        drift_ppm = self._safe_float(descriptor.get("clock_drift_ppm"))
        if clock_scale is None and drift_ppm is not None:
            clock_scale = 1.0 + drift_ppm / 1_000_000.0
        if clock_scale is not None:
            profile.clock_scale = round(clock_scale, 9)
            profile.drift_ppm = round((clock_scale - 1.0) * 1_000_000.0, 3)
            if "calibrated" not in profile.method:
                profile.method = f"{profile.method}+drift_corrected"
        return profile

    def _resolve_context_timestamp(
        self,
        item: Dict[str, Any],
        index: int,
        total_items: int,
    ) -> Optional[float]:
        direct_timestamp = self._safe_float(item.get("timestamp_sec"))
        start_time = self._safe_float(item.get("start_time_sec"))
        if direct_timestamp is not None:
            return round(max(direct_timestamp, 0.0), 3)
        if start_time is not None:
            return round(max(start_time, 0.0), 3)

        local_timestamp = self._safe_float(item.get("local_timestamp_sec"))
        if local_timestamp is not None:
            stream_index = item.get("video_index")
            video_asset_id = item.get("video_asset_id")
            for stream in self._video_info.get("streams", []):
                sync_profile_data = stream.get("sync_profile") if isinstance(stream.get("sync_profile"), dict) else {}
                offset = self._safe_float(sync_profile_data.get("offset_sec"), self._safe_float(stream.get("start_offset_sec"), 0.0)) or 0.0
                clock_scale = self._safe_float(sync_profile_data.get("clock_scale"), self._safe_float(stream.get("clock_scale"), 1.0)) or 1.0
                global_timestamp = round((local_timestamp * clock_scale) + offset, 3)
                if stream_index is not None and stream.get("video_index") == stream_index:
                    return global_timestamp
                if video_asset_id and stream.get("asset_id") == video_asset_id:
                    return global_timestamp

        total_duration = self._video_info.get("duration_sec", 0.0) or 0.0
        if total_duration <= 0:
            return None
        if total_items <= 1:
            return 0.0
        return round((index / max(total_items - 1, 1)) * total_duration, 3)

    @staticmethod
    def _auto_time_sync_enabled() -> bool:
        return str(os.getenv("LABSOPGUARD_AUTO_TIME_SYNC", "1")).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _descriptor_has_authoritative_sync(descriptor: Dict[str, Any]) -> bool:
        return bool(
            descriptor.get("sync_anchors")
            or descriptor.get("hardware_timecode_start_sec") is not None
            or descriptor.get("sync_board_offset_sec") is not None
        )

    @classmethod
    def _descriptor_has_fixed_offset(cls, descriptor: Dict[str, Any]) -> bool:
        offset_source = str(descriptor.get("offset_source") or "").strip().lower()
        sync_method = str(descriptor.get("sync_method") or "").strip().lower()
        return bool(
            offset_source in {"explicit", "shared_recording", "shared_recording_session", "calibrated", "hardware_timecode", "sync_board"}
            or sync_method in {"shared_recording", "shared_recording_session", "hardware_timecode", "sync_board"}
            or cls._descriptor_has_authoritative_sync(descriptor)
        )

    @staticmethod
    def _mean_frame_brightness(frame_path: Optional[str]) -> Optional[float]:
        if not frame_path:
            return None
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return float(np.mean(image)) / 255.0

    def _visual_sync_peaks(self, frames: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        if len(frames) < 2:
            return []
        means: List[Optional[float]] = [self._mean_frame_brightness(frame.get("path")) for frame in frames]
        samples: List[Dict[str, float]] = []
        previous_path: Optional[str] = None
        for index, frame in enumerate(frames):
            timestamp = self._safe_float(frame.get("timestamp_sec"))
            if timestamp is None:
                previous_path = frame.get("path")
                continue
            brightness_delta = 0.0
            if index > 0 and means[index] is not None and means[index - 1] is not None:
                brightness_delta = abs(float(means[index]) - float(means[index - 1]))
            change_score = self._frame_change_score(frame.get("path"), previous_path) if previous_path else 0.0
            score = max(brightness_delta, change_score)
            samples.append({"timestamp_sec": round(timestamp, 3), "score": round(float(score), 6)})
            previous_path = frame.get("path")
        if not samples:
            return []
        scores = np.array([sample["score"] for sample in samples], dtype=np.float32)
        dynamic_threshold = float(np.mean(scores) + np.std(scores) * 1.25)
        min_threshold = self._env_float("LABSOPGUARD_SYNC_PEAK_MIN_SCORE", 0.08, min_value=0.005, max_value=1.0)
        threshold = max(min_threshold, dynamic_threshold)
        peaks = [sample for sample in samples if sample["score"] >= threshold]
        if not peaks and float(np.max(scores)) >= min_threshold:
            peaks = [max(samples, key=lambda item: item["score"])]
        peaks.sort(key=lambda item: (-item["score"], item["timestamp_sec"]))
        return peaks[: int(self._env_float("LABSOPGUARD_SYNC_PEAK_MAX_COUNT", 8, min_value=1, max_value=32))]

    def _estimate_visual_sync_offset(
        self,
        reference_peaks: List[Dict[str, float]],
        candidate_peaks: List[Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        if not reference_peaks or not candidate_peaks:
            return None
        tolerance = self._env_float("LABSOPGUARD_SYNC_PEAK_MATCH_TOLERANCE_SEC", 1.25, min_value=0.05, max_value=10.0)
        best: Optional[Dict[str, Any]] = None
        for ref_peak in reference_peaks:
            for candidate_peak in candidate_peaks:
                offset = float(ref_peak["timestamp_sec"]) - float(candidate_peak["timestamp_sec"])
                residuals: List[float] = []
                weighted_score = 0.0
                for ref_item in reference_peaks:
                    shifted_candidates = [
                        abs(float(ref_item["timestamp_sec"]) - (float(item["timestamp_sec"]) + offset))
                        for item in candidate_peaks
                    ]
                    if not shifted_candidates:
                        continue
                    residual = min(shifted_candidates)
                    if residual <= tolerance:
                        residuals.append(residual)
                        weighted_score += float(ref_item["score"])
                if not residuals:
                    continue
                mean_residual = float(np.mean(residuals))
                confidence = min(0.9, max(0.35, 0.45 + min(len(residuals), 4) * 0.1 + weighted_score * 0.25 - mean_residual * 0.08))
                candidate = {
                    "offset_sec": round(offset, 6),
                    "confidence": round(confidence, 4),
                    "match_count": len(residuals),
                    "residual_error_sec": round(mean_residual, 6),
                    "reference_peak_sec": ref_peak["timestamp_sec"],
                    "candidate_peak_sec": candidate_peak["timestamp_sec"],
                }
                if best is None or (candidate["match_count"], candidate["confidence"]) > (best["match_count"], best["confidence"]):
                    best = candidate
        return best

    def _apply_auto_visual_time_sync(self, stream_records: List[Dict[str, Any]]) -> bool:
        if not self._auto_time_sync_enabled() or len(stream_records) < 2:
            return False

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in stream_records:
            descriptor = record.get("descriptor") or {}
            group_key = str(descriptor.get("sync_group") or record.get("stream_info", {}).get("source_group") or "default")
            grouped.setdefault(group_key, []).append(record)

        changed = False
        for records in grouped.values():
            if len(records) < 2:
                continue
            for record in records:
                record["visual_sync_peaks"] = self._visual_sync_peaks(record.get("frames") or [])

            reference = max(records, key=lambda item: len(item.get("visual_sync_peaks") or []))
            reference_peaks = reference.get("visual_sync_peaks") or []
            if not reference_peaks:
                continue
            reference_profile: CameraSyncProfile = reference["sync_profile"]
            group_changed = False

            for record in records:
                if record is reference:
                    continue
                descriptor = record.get("descriptor") or {}
                if self._descriptor_has_authoritative_sync(descriptor):
                    continue
                estimate = self._estimate_visual_sync_offset(reference_peaks, record.get("visual_sync_peaks") or [])
                if not estimate:
                    continue
                profile: CameraSyncProfile = record["sync_profile"]
                profile.offset_sec = round(reference_profile.offset_sec + estimate["offset_sec"], 6)
                profile.method = "auto_visual_peak"
                profile.confidence = max(float(profile.confidence or 0.0), float(estimate["confidence"]))
                profile.anchor_count = int(estimate["match_count"])
                profile.residual_error_sec = float(estimate["residual_error_sec"])
                record["auto_sync"] = estimate
                group_changed = True
                changed = True

            if group_changed:
                if not str(reference_profile.method or "").startswith(("calibrated", "hardware_timecode", "sync_board")):
                    reference_profile.method = "auto_visual_reference"
                    reference_profile.confidence = max(float(reference_profile.confidence or 0.0), 0.8)
                    reference_profile.anchor_count = max(int(reference_profile.anchor_count or 0), len(reference_peaks))
                offsets = [float(item["sync_profile"].offset_sec) for item in records]
                min_offset = min(offsets)
                if min_offset < 0:
                    shift = abs(min_offset)
                    for item in records:
                        profile: CameraSyncProfile = item["sync_profile"]
                        profile.offset_sec = round(profile.offset_sec + shift, 6)
                        if "normalized" not in profile.method:
                            profile.method = f"{profile.method}+normalized"
                        item.setdefault("auto_sync", {})["normalization_shift_sec"] = round(shift, 6)

        return changed

    @staticmethod
    def _semantic_sync_enabled() -> bool:
        return str(os.getenv("LABSOPGUARD_MULTIMODAL_SEMANTIC_SYNC", "1")).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _semantic_sync_apply_enabled() -> bool:
        return str(os.getenv("LABSOPGUARD_MULTIMODAL_SEMANTIC_SYNC_APPLY", "1")).strip().lower() not in {"0", "false", "no", "off"}

    def _semantic_sync_frame_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for frame, analysis in zip(self._video_frames, self._frame_analyses):
            if not isinstance(frame, dict) or not isinstance(analysis, dict):
                continue
            items.append(
                {
                    "camera_id": frame.get("camera_id"),
                    "view_type": frame.get("view_type"),
                    "source_group": frame.get("source_group"),
                    "local_timestamp_sec": frame.get("local_timestamp_sec", frame.get("timestamp_sec")),
                    "timestamp_sec": frame.get("timestamp_sec"),
                    "frame_id": frame.get("frame_id"),
                    "local_frame_id": frame.get("local_frame_id"),
                    "scene_description": analysis.get("scene_description") or analysis.get("description") or analysis.get("raw") or "",
                    "object_labels": analysis.get("object_labels") or [],
                    "detected_activities": analysis.get("detected_activities") or [],
                    "transcript_segment": analysis.get("transcript_segment"),
                    "confidence": analysis.get("confidence", 0.5),
                }
            )
        return items

    def _run_multimodal_semantic_sync(self, experiment: Experiment) -> Dict[str, Any]:
        frame_items = self._semantic_sync_frame_items()
        run_id = str(experiment.metadata.get("run_id") or f"{experiment.experiment_id}:semantic_sync")
        if not self._semantic_sync_enabled() or len({item.get("camera_id") for item in frame_items if item.get("camera_id")}) < 2:
            result = MultimodalSemanticSyncEngine.build(
                experiment_id=experiment.experiment_id,
                run_id=run_id,
                frame_items=frame_items,
            )
            result["status"] = "disabled" if not self._semantic_sync_enabled() else result.get("status", "insufficient_overlap")
            experiment.metadata["semantic_sync"] = {
                "status": result.get("status"),
                "anchor_count": len(result.get("sync_anchors") or []),
                "applied": False,
            }
            return result

        min_confidence = self._env_float("LABSOPGUARD_MULTIMODAL_SEMANTIC_SYNC_MIN_CONFIDENCE", 0.75, min_value=0.0, max_value=1.0)
        result = MultimodalSemanticSyncEngine.build(
            experiment_id=experiment.experiment_id,
            run_id=run_id,
            frame_items=frame_items,
            min_anchor_confidence=min_confidence,
        )
        applied = False
        if self._semantic_sync_apply_enabled() and result.get("sync_anchors"):
            applied = self._apply_multimodal_semantic_sync(experiment, result)
        experiment.metadata["semantic_sync"] = {
            "status": result.get("status"),
            "anchor_count": len(result.get("sync_anchors") or []),
            "semantic_event_count": len(result.get("semantic_events") or []),
            "reference_stream": result.get("reference_stream"),
            "applied": applied,
        }
        if applied:
            experiment.metadata["time_alignment_mode"] = "calibrated"
        return result

    def _apply_multimodal_semantic_sync(self, experiment: Experiment, result: Dict[str, Any]) -> bool:
        anchors = MultimodalSemanticSyncEngine.anchors_as_sync_anchors(result)
        anchors_by_camera: Dict[str, List[Any]] = {}
        for anchor in anchors:
            anchors_by_camera.setdefault(anchor.camera_id, []).append(anchor)
        if not anchors_by_camera:
            return False

        asset_by_id = {asset.asset_id: asset for asset in experiment.video_assets}
        profiles_by_camera: Dict[str, CameraSyncProfile] = {}
        for stream in self._video_info.get("streams", []):
            camera_id = str(stream.get("camera_id") or "")
            camera_anchors = anchors_by_camera.get(camera_id)
            if not camera_id or not camera_anchors:
                continue
            profile = TimeSyncCalibrator.fit_profile_from_anchors(
                camera_id,
                camera_anchors,
                reference_camera_id=str((result.get("reference_stream") or {}).get("camera_id") or "global"),
            )
            existing_confidence = float(self._safe_float(stream.get("alignment_confidence"), 0.0) or 0.0)
            existing_method = str(stream.get("alignment_method") or "").lower()
            if (
                any(marker in existing_method for marker in ("hardware_timecode", "sync_board", "explicit_offset"))
                and existing_confidence >= profile.confidence
            ):
                continue
            stream["start_offset_sec"] = round(profile.offset_sec, 3)
            stream["end_offset_sec"] = round(profile.local_to_global(float(stream.get("duration_sec") or 0.0)), 3)
            stream["clock_scale"] = profile.clock_scale
            stream["clock_drift_ppm"] = profile.drift_ppm
            stream["alignment_method"] = "multimodal_semantic"
            stream["alignment_confidence"] = profile.confidence
            stream["sync_profile"] = profile.to_dict()
            stream["semantic_sync_anchor_count"] = len(camera_anchors)
            stream["semantic_sync_event_ids"] = [anchor.metadata.get("event_id") for anchor in camera_anchors if anchor.metadata]
            profiles_by_camera[camera_id] = profile

            asset = asset_by_id.get(str(stream.get("asset_id") or ""))
            if asset is not None:
                asset.metadata["start_offset_sec"] = stream["start_offset_sec"]
                asset.metadata["clock_scale"] = profile.clock_scale
                asset.metadata["clock_drift_ppm"] = profile.drift_ppm
                asset.metadata["sync_profile"] = profile.to_dict()
                asset.metadata["alignment_method"] = "multimodal_semantic"
                asset.metadata["alignment_confidence"] = profile.confidence
                asset.metadata["offset_source"] = "calibrated"
                asset.metadata["semantic_sync_anchor_count"] = len(camera_anchors)

        if not profiles_by_camera:
            return False

        for frame in self._video_frames:
            camera_id = str(frame.get("camera_id") or "")
            profile = profiles_by_camera.get(camera_id)
            if profile is None:
                continue
            local_timestamp = self._safe_float(frame.get("local_timestamp_sec"), self._safe_float(frame.get("timestamp_sec"), 0.0)) or 0.0
            frame["timestamp_sec"] = round(profile.local_to_global(local_timestamp), 3)
            frame["alignment_method"] = "multimodal_semantic"
            frame["alignment_confidence"] = profile.confidence
            frame["start_offset_sec"] = round(profile.offset_sec, 3)
            frame["clock_scale"] = profile.clock_scale

        self._video_frames = sorted(self._video_frames, key=lambda frame: (frame.get("timestamp_sec", 0.0), frame.get("frame_id", 0)))
        self._video_info["duration_sec"] = max((stream.get("end_offset_sec") or 0.0) for stream in self._video_info.get("streams", [])) if self._video_info.get("streams") else 0.0
        self._video_info["alignment_mode"] = "calibrated"
        self._video_info["semantic_sync"] = {
            "status": result.get("status"),
            "anchor_count": len(result.get("sync_anchors") or []),
            "reference_stream": result.get("reference_stream"),
        }
        return True

    @staticmethod
    def _context_source_from_input(item: Dict[str, Any]) -> ContextSource:
        kind = str(item.get("kind", "")).lower()
        source = str(item.get("source_type", "")).lower()
        if "protocol" in kind or source == "protocol":
            return ContextSource.PROTOCOL
        if "sensor" in kind or source == "sensor":
            return ContextSource.SENSOR
        if "manual" in kind or source == "manual":
            return ContextSource.MANUAL
        return ContextSource.CONVERSATION

    @staticmethod
    def _frame_change_score(current_path: Optional[str], previous_path: Optional[str]) -> float:
        if not current_path or not previous_path:
            return 0.0
        current = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)
        previous = cv2.imread(str(previous_path), cv2.IMREAD_GRAYSCALE)
        if current is None or previous is None:
            return 0.0
        if current.shape != previous.shape:
            previous = cv2.resize(previous, (current.shape[1], current.shape[0]))
        current = cv2.GaussianBlur(current, (5, 5), 0)
        previous = cv2.GaussianBlur(previous, (5, 5), 0)
        diff = cv2.absdiff(current, previous).astype("float32") / 255.0
        mean_score = float(np.mean(diff))
        # Mean frame difference misses small hand/tool movements on a static
        # bench. Percentile and active-pixel terms raise localized changes
        # without making global brightness drift look like motion.
        active_ratio = float(np.mean(diff >= 0.06))
        p95 = float(np.percentile(diff, 95))
        p98 = float(np.percentile(diff, 98))
        localized_score = max(active_ratio * 0.65, p95 * 0.35, p98 * 0.25)
        return float(min(1.0, max(mean_score, localized_score)))

    @staticmethod
    def _env_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(min_value, min(max_value, value))

    @classmethod
    def _keyframe_base_change_threshold(cls) -> float:
        return cls._env_float("LABSOPGUARD_KEYFRAME_CHANGE_THRESHOLD", 0.055, min_value=0.005, max_value=0.5)

    @classmethod
    def _keyframe_min_change_threshold(cls) -> float:
        return cls._env_float("LABSOPGUARD_KEYFRAME_MIN_CHANGE_THRESHOLD", 0.025, min_value=0.001, max_value=0.2)

    @classmethod
    def _keyframe_max_change_threshold(cls) -> float:
        floor = cls._keyframe_min_change_threshold()
        return cls._env_float("LABSOPGUARD_KEYFRAME_MAX_CHANGE_THRESHOLD", 0.09, min_value=floor, max_value=0.8)

    @classmethod
    def _adaptive_keyframe_change_threshold(cls, scores: List[float]) -> float:
        base = cls._keyframe_base_change_threshold()
        floor = cls._keyframe_min_change_threshold()
        ceiling = cls._keyframe_max_change_threshold()
        values = np.array([float(score) for score in scores if float(score) > 0.001], dtype=np.float32)
        if values.size < 3:
            return round(max(floor, min(ceiling, base)), 4)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        p75 = float(np.percentile(values, 75))
        robust = max(median + 2.2 * mad, p75 * 0.85)
        adaptive = max(floor, max(base * 0.55, robust))
        return round(min(ceiling, adaptive), 4)

    @classmethod
    def _object_motion_change_threshold(cls, scene_threshold: float) -> float:
        base = cls._env_float("LABSOPGUARD_OBJECT_MOVE_CHANGE_THRESHOLD", 0.085, min_value=0.005, max_value=0.8)
        multiplier = cls._env_float("LABSOPGUARD_OBJECT_MOVE_THRESHOLD_MULTIPLIER", 1.35, min_value=1.0, max_value=5.0)
        return round(max(base, float(scene_threshold) * multiplier), 4)

    @staticmethod
    def _infer_view_type(camera_id: Any, source_type: Any, descriptor: Dict[str, Any]) -> str:
        explicit = str(descriptor.get("view_type") or descriptor.get("role") or "").strip().lower()
        if explicit:
            return explicit
        cam = str(camera_id or "").strip().lower()
        if cam in {"top", "top_view", "overview", "third_person", "side", "side_view"}:
            return "third_person"
        if cam in {"bottom", "bottom_view", "hand", "hand_view", "first_person"}:
            return "first_person"
        return "first_person"

    @staticmethod
    def _infer_source_group(camera_id: Any, source_type: Any, view_type: str, descriptor: Dict[str, Any]) -> str:
        explicit = str(descriptor.get("source_group") or "").strip().lower()
        if explicit:
            return explicit
        return "key_action_dual_view"

    @staticmethod
    def _normalized_label_set(values: Any) -> set[str]:
        if not isinstance(values, list):
            values = [values] if values else []
        return {str(value).strip().lower() for value in values if str(value).strip()}

    def _extract_video(self, experiment: Experiment) -> None:
        """提取视频帧。"""
        descriptors = self._video_descriptors()
        if not descriptors:
            experiment.processing_error = "No video path set"
            return

        try:
            all_frames: List[Dict[str, Any]] = []
            stream_infos: List[Dict[str, Any]] = []
            stream_records: List[Dict[str, Any]] = []
            current_offset = 0.0
            running_frame_id = 0
            explicit_offsets = any(self._descriptor_has_fixed_offset(item) for item in descriptors)
            calibrated_alignment = any(
                item.get("sync_anchors")
                or item.get("hardware_timecode_start_sec") is not None
                or item.get("sync_board_offset_sec") is not None
                or item.get("clock_drift_ppm") is not None
                or item.get("clock_scale") is not None
                for item in descriptors
            )

            for fallback_index, descriptor in enumerate(descriptors):
                source_ref = descriptor.get("video_path") or descriptor.get("file_path") or descriptor.get("path") or descriptor.get("source")
                if not source_ref:
                    continue
                source_type = str(descriptor.get("source_type") or descriptor.get("ingest_mode") or "file").lower()
                camera_id = descriptor.get("camera_id") or f"camera_{fallback_index:02d}"
                view_type = self._infer_view_type(camera_id, source_type, dict(descriptor))
                source_group = self._infer_source_group(camera_id, source_type, view_type, dict(descriptor))
                role = str(descriptor.get("role") or view_type)
                frame_dir = (
                    (self._artifact_root / "frames" / f"stream_{fallback_index:02d}")
                    if self._artifact_root
                    else Path(tempfile.mkdtemp())
                )
                frame_dir.mkdir(parents=True, exist_ok=True)
                source_path = Path(str(source_ref))
                is_local_file = source_type == "file" and source_path.exists()
                if is_local_file:
                    frames, info = self._frame_extractor.extract_frames(
                        str(source_ref),
                        output_dir=str(frame_dir),
                    )
                else:
                    record_output_path = None
                    if self._artifact_root:
                        safe_camera = "".join(ch if str(ch).isalnum() or ch in {"-", "_"} else "_" for ch in str(camera_id))
                        record_output_path = str(self._artifact_root / "recordings" / f"stream_{fallback_index:02d}_{safe_camera}.mp4")
                    frames, info = self._frame_extractor.extract_stream_frames(
                        str(source_ref),
                        source_type=source_type,
                        output_dir=str(frame_dir),
                        capture_duration_sec=self._safe_float(descriptor.get("capture_duration_sec")),
                        record_output_path=record_output_path,
                    )

                file_path = source_path
                size_bytes = file_path.stat().st_size if file_path.exists() else None
                offset_source = str(descriptor.get("offset_source") or "").strip().lower()
                start_offset = self._safe_float(descriptor.get("start_offset_sec")) if self._descriptor_has_fixed_offset(dict(descriptor)) else None
                offset_source = offset_source or ("explicit" if start_offset is not None else "sequential")
                if start_offset is None:
                    start_offset = current_offset
                start_offset = round(max(start_offset or 0.0, 0.0), 3)
                sync_profile = self._resolve_sync_profile(dict(descriptor), str(camera_id), start_offset)
                start_offset = round(sync_profile.offset_sec, 3)
                clock_scale = sync_profile.clock_scale

                asset = MediaAsset(
                    experiment_id=experiment.experiment_id,
                    media_type=MediaType.VIDEO,
                    file_path=str(source_ref),
                    filename=file_path.name if file_path.name else str(source_ref),
                    duration_sec=info.get("duration_sec"),
                    frame_count=info.get("total_frames"),
                    width=frames[0].get("width") if frames else None,
                    height=frames[0].get("height") if frames else None,
                    fps=info.get("fps"),
                    size_bytes=size_bytes,
                    hash_sha256=self._compute_file_sha256(str(source_ref)) if is_local_file else None,
                    metadata={
                        "video_index": descriptor.get("video_index", fallback_index),
                        "camera_id": camera_id,
                        "view_type": view_type,
                        "role": role,
                        "source_group": source_group,
                        "sync_group": descriptor.get("sync_group"),
                        "start_offset_sec": start_offset,
                        "offset_source": "calibrated" if calibrated_alignment else offset_source,
                        "clock_scale": clock_scale,
                        "clock_drift_ppm": sync_profile.drift_ppm,
                        "sync_profile": sync_profile.to_dict(),
                        "recorded_file_path": info.get("recorded_file_path"),
                        "ingest_mode": descriptor.get("ingest_mode", source_type),
                        "source_type": source_type,
                        "is_live_source": bool(info.get("is_live_source", False)),
                        "capture_duration_sec": self._safe_float(descriptor.get("capture_duration_sec")),
                        "stream_health": info.get("stream_health") or {},
                    },
                )
                experiment.video_assets.append(asset)

                stream_info = {
                    "video_index": descriptor.get("video_index", fallback_index),
                    "asset_id": asset.asset_id,
                    "video_path": str(source_ref),
                    "filename": file_path.name if file_path.name else str(source_ref),
                    "fps": info.get("fps"),
                    "total_frames": info.get("total_frames"),
                    "duration_sec": info.get("duration_sec"),
                    "start_offset_sec": start_offset,
                    "end_offset_sec": round(start_offset + ((info.get("duration_sec") or 0.0) * clock_scale), 3),
                    "clock_scale": clock_scale,
                    "clock_drift_ppm": sync_profile.drift_ppm,
                    "sync_profile": sync_profile.to_dict(),
                    "recorded_file_path": info.get("recorded_file_path"),
                    "camera_id": camera_id,
                    "view_type": view_type,
                    "role": role,
                    "source_group": source_group,
                    "sync_group": descriptor.get("sync_group"),
                    "source_type": source_type,
                    "is_live_source": bool(info.get("is_live_source", False)),
                    "stream_health": info.get("stream_health") or {},
                    "alignment_confidence": sync_profile.confidence,
                    "alignment_method": sync_profile.method,
                }
                stream_infos.append(stream_info)

                stream_records.append(
                    {
                        "descriptor": dict(descriptor),
                        "frames": frames,
                        "info": info,
                        "asset": asset,
                        "stream_info": stream_info,
                        "sync_profile": sync_profile,
                        "source_ref": str(source_ref),
                        "source_type": source_type,
                        "camera_id": camera_id,
                        "view_type": view_type,
                        "role": role,
                        "source_group": source_group,
                        "sync_group": descriptor.get("sync_group"),
                    }
                )

                if explicit_offsets:
                    current_offset = max(current_offset, stream_info["end_offset_sec"])
                else:
                    current_offset = stream_info["end_offset_sec"]

            auto_visual_alignment = self._apply_auto_visual_time_sync(stream_records)
            for record in stream_records:
                sync_profile = record["sync_profile"]
                stream_info = record["stream_info"]
                asset = record["asset"]
                info = record["info"]
                start_offset = round(sync_profile.offset_sec, 3)
                clock_scale = sync_profile.clock_scale
                stream_info["start_offset_sec"] = start_offset
                stream_info["end_offset_sec"] = round(start_offset + ((info.get("duration_sec") or 0.0) * clock_scale), 3)
                stream_info["clock_scale"] = clock_scale
                stream_info["clock_drift_ppm"] = sync_profile.drift_ppm
                stream_info["sync_profile"] = sync_profile.to_dict()
                stream_info["alignment_confidence"] = sync_profile.confidence
                stream_info["alignment_method"] = sync_profile.method
                if record.get("auto_sync"):
                    stream_info["auto_sync"] = record["auto_sync"]
                stream_info["visual_sync_peaks"] = record.get("visual_sync_peaks", [])

                asset.metadata["start_offset_sec"] = start_offset
                asset.metadata["offset_source"] = "calibrated" if (calibrated_alignment or auto_visual_alignment) else asset.metadata.get("offset_source")
                asset.metadata["clock_scale"] = clock_scale
                asset.metadata["clock_drift_ppm"] = sync_profile.drift_ppm
                asset.metadata["sync_profile"] = sync_profile.to_dict()
                asset.metadata["alignment_confidence"] = sync_profile.confidence
                asset.metadata["alignment_method"] = sync_profile.method
                if record.get("auto_sync"):
                    asset.metadata["auto_sync"] = record["auto_sync"]
                asset.metadata["visual_sync_peaks"] = record.get("visual_sync_peaks", [])

                for frame in record.get("frames") or []:
                    local_timestamp = round(frame.get("timestamp_sec", 0.0), 3)
                    global_timestamp = round(sync_profile.local_to_global(local_timestamp), 3)
                    frame["local_timestamp_sec"] = local_timestamp
                    frame["timestamp_sec"] = global_timestamp
                    frame["local_frame_id"] = frame.get("frame_id")
                    frame["frame_id"] = running_frame_id
                    frame["stream_id"] = asset.asset_id
                    frame["media_asset_id"] = asset.asset_id
                    frame["video_index"] = stream_info.get("video_index")
                    frame["video_path"] = record["source_ref"]
                    frame["source_type"] = record["source_type"]
                    frame["camera_id"] = record["camera_id"]
                    frame["view_type"] = record["view_type"]
                    frame["role"] = record["role"]
                    frame["source_group"] = record["source_group"]
                    frame["sync_group"] = record.get("sync_group")
                    frame["alignment_confidence"] = sync_profile.confidence
                    frame["alignment_method"] = sync_profile.method
                    frame["start_offset_sec"] = start_offset
                    frame["clock_scale"] = clock_scale
                    all_frames.append(frame)
                    running_frame_id += 1

            alignment_mode = (
                "calibrated"
                if calibrated_alignment or auto_visual_alignment
                else ("explicit_offsets" if explicit_offsets else "sequential")
            )
            self._video_frames = sorted(all_frames, key=lambda frame: (frame.get("timestamp_sec", 0.0), frame.get("frame_id", 0)))
            self._video_info = {
                "fps": stream_infos[0]["fps"] if stream_infos else 0.0,
                "total_frames": sum((stream.get("total_frames") or 0) for stream in stream_infos),
                "duration_sec": max((stream.get("end_offset_sec") or 0.0) for stream in stream_infos) if stream_infos else 0.0,
                "streams": stream_infos,
                "video_count": len(stream_infos),
                "alignment_mode": alignment_mode,
                "auto_visual_alignment": auto_visual_alignment,
            }
            experiment.video_asset_id = experiment.video_assets[0].asset_id if experiment.video_assets else None
            experiment.metadata["video_stream_count"] = len(stream_infos)
            experiment.metadata["time_alignment_mode"] = self._video_info["alignment_mode"]

        except Exception as e:
            experiment.processing_error = f"Video extraction failed: {e}"
            raise

    def _analyze_frames(self, experiment: Experiment) -> None:
        """分析视频帧（Qwen-VL）。"""
        self._frame_analyses = []

        for frame_info in self._video_frames:
            path = frame_info.get("path")
            if not path or not Path(path).exists():
                self._frame_analyses.append({"error": "no_path", "inferred": True})
                continue

            analysis = self._step_reasoner.analyze_frame(path)
            view_type = str(frame_info.get("view_type") or "first_person").lower()
            camera_id = frame_info.get("camera_id")
            detections = self._run_detector(path, view_type=view_type)
            if detections:
                detected_labels = [str(item.get("label")) for item in detections if item.get("label")]
                merge_detector_labels = view_type not in {"third_person", "overview"} or os.getenv("YOLO26_MERGE_THIRD_PERSON_LABELS", "0") == "1"
                if merge_detector_labels:
                    merged_labels = []
                    seen_labels = set()
                    for label in [*(analysis.get("object_labels") or []), *detected_labels]:
                        key = str(label).strip().lower()
                        if key and key not in seen_labels:
                            seen_labels.add(key)
                            merged_labels.append(str(label))
                    analysis["object_labels"] = merged_labels
                analysis["detected_objects"] = detections
                analysis["detector_model"] = "ultralytics_yolo26"
                analysis["detector_weights_path"] = self._detector.status.weights_path if self._detector else None
                analysis["detector_view_strategy"] = "candidate_only" if not merge_detector_labels else "merged_semantic_labels"
                analysis["model"] = (
                    f"{analysis.get('model', 'frame_analysis')}+ultralytics_yolo26"
                    if analysis.get("model")
                    else "ultralytics_yolo26"
                )
            elif self._detector is not None:
                analysis.setdefault("detected_objects", [])
                analysis["detector_model"] = "ultralytics_yolo26"
                analysis["detector_weights_path"] = self._detector.status.weights_path
                analysis["detector_view_strategy"] = "candidate_only" if view_type in {"third_person", "overview"} else "merged_semantic_labels"
            analysis["frame_id"] = frame_info["frame_id"]
            analysis["timestamp_sec"] = frame_info.get("timestamp_sec", 0.0)
            analysis["local_timestamp_sec"] = frame_info.get("local_timestamp_sec", frame_info.get("timestamp_sec", 0.0))
            analysis["stream_id"] = frame_info.get("stream_id")
            analysis["media_asset_id"] = frame_info.get("media_asset_id")
            analysis["camera_id"] = camera_id
            analysis["view_type"] = view_type
            analysis["source_group"] = frame_info.get("source_group")
            analysis["source_type"] = frame_info.get("source_type")
            analysis["alignment_confidence"] = frame_info.get("alignment_confidence")
            self._frame_analyses.append(analysis)

    def _run_detector(self, frame_path: str, *, view_type: str = "first_person") -> List[Dict[str, Any]]:
        if self._detector is None:
            return []
        try:
            detections = self._detector.detect_image_path(frame_path)
            if str(view_type).lower() in {"third_person", "overview"}:
                threshold = float(os.getenv("YOLO26_THIRD_PERSON_CONFIDENCE_THRESHOLD", "0.45"))
                detections = [item for item in detections if float(item.get("confidence", item.get("score", 0.0)) or 0.0) >= threshold]
            for item in detections:
                item.setdefault("optional_attributes", {})
                if isinstance(item["optional_attributes"], dict):
                    item["optional_attributes"]["view_type"] = view_type
            return detections
        except Exception as exc:
            print(f"[ExperimentService] detector failed for {frame_path}: {exc}")
            return []

    def _integrate_context(self, experiment: Experiment) -> None:
        """整合对话上下文和 Protocol。"""
        context_inputs = self._context_inputs or ([{"text": self._context_text}] if self._context_text else [])
        if context_inputs:
            valid_inputs = [item for item in context_inputs if str(item.get("text", "")).strip()]
            for index, item in enumerate(valid_inputs):
                timestamp_sec = self._resolve_context_timestamp(item, index, len(valid_inputs))
                metadata = {k: v for k, v in item.items() if k != "text"}
                event = ContextEvent(
                    experiment_id=experiment.experiment_id,
                    context_source=self._context_source_from_input(item),
                    event_type=str(item.get("kind") or "context_input"),
                    timestamp_sec=timestamp_sec,
                    duration_sec=self._safe_float(item.get("duration_sec")),
                    content=str(item.get("text", "")).strip(),
                    raw_content=str(item.get("raw_text")) if item.get("raw_text") else None,
                    confidence=1.0,
                    provenance=ProvenanceInfo(
                        source="conversation",
                        confidence=1.0,
                        is_inferred=False,
                    ),
                    metadata=metadata,
                )
                experiment.context_events.append(event)
        elif self._context_text:
            experiment.context_events.append(
                ContextEvent(
                    experiment_id=experiment.experiment_id,
                    context_source=ContextSource.CONVERSATION,
                    event_type="conversation_input",
                    content=self._context_text,
                    confidence=1.0,
                    provenance=ProvenanceInfo(
                        source="conversation",
                        confidence=1.0,
                        is_inferred=False,
                    ),
                )
            )

        # 从 protocol_text 创建 ContextEvent
        if self._protocol_text:
            event = ContextEvent(
                experiment_id=experiment.experiment_id,
                context_source=ContextSource.PROTOCOL,
                event_type="protocol_text",
                content=self._protocol_text,
                confidence=1.0,
                provenance=ProvenanceInfo(
                    source="protocol",
                    confidence=1.0,
                    is_inferred=False,
                ),
            )
            experiment.context_events.append(event)

        # 合并已有的 context_events
        experiment.context_events.extend(self._context_events)

    def _reason_steps(self, experiment: Experiment) -> ExperimentTimeline:
        """推理步骤，生成时间线。ContextEvent 真正参与推理。"""
        steps = self._step_reasoner.infer_steps_from_frames(
            frames=self._video_frames,
            frame_analyses=self._frame_analyses,
            context=self._context_text,
            protocol=self._protocol_text,
            context_events=experiment.context_events,  # 关键：传入 ContextEvent
        )

        # 为每个步骤填入 experiment_id
        for step in steps:
            step.experiment_id = experiment.experiment_id

        # 重新排序并更新 step_index
        steps.sort(key=lambda s: s.start_time_sec)
        for i, step in enumerate(steps):
            step.step_index = i

        # 构建 Timeline
        video_duration = self._video_info.get("duration_sec", 0.0)
        timeline = ExperimentTimeline(
            experiment_id=experiment.experiment_id,
            title=experiment.title,
            steps=steps,
            video_asset_id=(
                experiment.video_assets[0].asset_id
                if experiment.video_assets
                else None
            ),
            video_duration_sec=video_duration,
            protocol_text=self._protocol_text[:200] if self._protocol_text else None,
            models_used=[
                *([self._vlm_client.model] if self._vlm_client else ["fallback"]),
                *(["ultralytics_yolo26"] if self._detector is not None and self._detector.status.available else []),
            ],
            context_summary=self._context_text[:500] if self._context_text else None,
            media_assets=[asset.asset_id for asset in experiment.video_assets],
            context_events=[event.event_id for event in experiment.context_events],
            metadata={
                "video_stream_count": len(experiment.video_assets),
                "video_streams": self._video_info.get("streams", []),
                "time_alignment_mode": self._video_info.get("alignment_mode", "sequential"),
                "detector_status": self._detector.status.to_dict() if self._detector is not None else {"available": False},
            },
        )
        timeline.compute_stats()

        return timeline

    def _generate_physical_events(self, experiment: Experiment) -> List[PhysicalEvent]:
        """
        从帧分析结果中生成 PhysicalEvent。

        基于启发式规则检测物理事件：
        - 手部接触（hand_contact）：检测到手部关键点靠近物体
        - 物体移动（object_movement）：连续帧中物体位置变化
        - 容器状态变化（container_state_change）：检测到容器相关活动
        - 液体转移（liquid_transfer）：检测到倾倒/移液相关活动
        """
        events: List[PhysicalEvent] = []
        prev_objects_by_stream: Dict[str, set[str]] = {}
        asset_metadata_by_id = {
            asset.asset_id: dict(asset.metadata or {})
            for asset in experiment.video_assets
        }
        semantic_detector = SemanticEventDetector()

        for frame, analysis, item in zip(self._video_frames, self._frame_analyses, self._material_stream):
            has_error = "error" in analysis
            ts = frame.get("timestamp_sec", 0.0)
            local_ts = frame.get("local_timestamp_sec", ts)
            activities = analysis.get("detected_activities", []) if not has_error else []
            objects = set(analysis.get("object_labels", [])) if not has_error else set()
            description = analysis.get("description", "") if not has_error else ""
            conf = float(analysis.get("confidence", 0.3)) if not has_error else 0.3
            model = analysis.get("model", "fallback_heuristic") if not has_error else "fallback_heuristic"
            stream_id = frame.get("stream_id") or "workspace"
            prev_objects = prev_objects_by_stream.get(stream_id, set())
            asset_metadata = asset_metadata_by_id.get(frame.get("media_asset_id"), {})
            location = f"stream:{stream_id}"
            base_metadata = {
                "frame_id": frame["frame_id"],
                "local_timestamp_sec": local_ts,
                "global_timestamp_sec": ts,
                "video_asset_id": frame.get("media_asset_id"),
                "camera_id": asset_metadata.get("camera_id"),
                "view_type": asset_metadata.get("view_type") or frame.get("view_type"),
                "source_group": asset_metadata.get("source_group") or frame.get("source_group"),
                "source_type": asset_metadata.get("source_type") or frame.get("source_type"),
                "sync_group": asset_metadata.get("sync_group"),
                "alignment_confidence": asset_metadata.get("sync_profile", {}).get("confidence") if isinstance(asset_metadata.get("sync_profile"), dict) else frame.get("alignment_confidence"),
                "stream_id": stream_id,
                "clip_id": getattr(item, "clip_id", None),
                "material_item_id": getattr(item, "item_id", None),
            }
            keyframe_detection = (getattr(item, "analysis", {}) or {}).get("key_frame_detection", {})
            scene_change_threshold = self._safe_float(keyframe_detection.get("change_threshold"))
            if scene_change_threshold is None:
                scene_change_threshold = self._keyframe_base_change_threshold()
            object_motion_threshold = self._object_motion_change_threshold(scene_change_threshold)

            if not has_error:
                semantic_detections = analysis.get("detected_objects", [])
                if isinstance(semantic_detections, list) and semantic_detections:
                    for semantic_event in semantic_detector.update(ts, semantic_detections, frame_metadata=base_metadata):
                        sem_meta = {**base_metadata, **(semantic_event.get("metadata") or {})}
                        events.append(
                            PhysicalEvent(
                                experiment_id=experiment.experiment_id,
                                event_type=str(semantic_event.get("event_type", "semantic_event")),
                                timestamp_sec=float(semantic_event.get("timestamp_sec", ts)),
                                duration_sec=self._frame_extractor.sample_interval_sec,
                                location=location,
                                confidence=float(semantic_event.get("confidence", conf)),
                                provenance=ProvenanceInfo(
                                    source="video",
                                    source_id=f"frame_{frame['frame_id']}",
                                    confidence=float(semantic_event.get("confidence", conf)),
                                    is_inferred=True,
                                    inference_method="semantic_geometry_detector",
                                    model_used=model,
                                ),
                                metadata=sem_meta,
                            )
                        )

            if getattr(item, "change_score", 0.0) >= scene_change_threshold:
                events.append(
                    PhysicalEvent(
                        experiment_id=experiment.experiment_id,
                        event_type="scene_change",
                        timestamp_sec=ts,
                        duration_sec=self._frame_extractor.sample_interval_sec,
                        location=location,
                        confidence=max(conf, min(0.95, 0.35 + getattr(item, "change_score", 0.0))),
                        provenance=ProvenanceInfo(
                            source="video",
                            source_id=f"frame_{frame['frame_id']}",
                            confidence=conf,
                            is_inferred=True,
                            inference_method="frame_difference_detection",
                            model_used=model,
                        ),
                        metadata={
                            **base_metadata,
                            "change_score": round(getattr(item, "change_score", 0.0), 4),
                            "change_threshold": round(scene_change_threshold, 4),
                            "key_frame_reason": getattr(item, "key_frame_reason", None),
                        },
                    )
                )

            if not has_error:
                if getattr(item, "change_score", 0.0) >= object_motion_threshold:
                    motion_gate = gate_object_move(
                        event_candidate={
                            "event_type": "object_move",
                            "candidate_source": "legacy_visual_motion_proxy",
                            "time_start": ts,
                            "time_end": ts + self._frame_extractor.sample_interval_sec,
                        },
                        track=None,
                        scene_motion={"method": "none", "limitations": ["legacy_change_score_only"]},
                        jitter_profile={"source": "fallback"},
                        frame_evidence_list=[{"change_score": getattr(item, "change_score", 0.0)}],
                    )
                    events.append(
                        PhysicalEvent(
                            experiment_id=experiment.experiment_id,
                            event_type="object_move_candidate",
                            timestamp_sec=ts,
                            duration_sec=self._frame_extractor.sample_interval_sec,
                            location=location,
                            confidence=min(0.45, max(conf, float(motion_gate.get("confidence") or 0.0))),
                            provenance=ProvenanceInfo(
                                source="video",
                                source_id=f"frame_{frame['frame_id']}",
                                confidence=conf,
                                is_inferred=True,
                                inference_method="visual_motion_proxy_candidate",
                                model_used=model,
                            ),
                            metadata={
                                **base_metadata,
                                "status": motion_gate.get("status", "rejected"),
                                "hard_gate": motion_gate.get("hard_gate"),
                                "reject_reasons": list(
                                    motion_gate.get("reject_reasons") or ["change_score_only"]
                                ),
                                "evidence_detail": motion_gate.get("evidence") or {},
                                "limitations": list(
                                    motion_gate.get("limitations")
                                    or ["change_score alone cannot confirm object movement"]
                                ),
                                "legacy_candidate": True,
                                "change_score": round(getattr(item, "change_score", 0.0), 4),
                                "change_threshold": round(object_motion_threshold, 4),
                                "current_objects": sorted(objects),
                                "previous_objects": sorted(prev_objects),
                            },
                        )
                    )
                hand_keywords = ["hand", "grab", "hold", "pick", "touch", "grasp", "pour", "pipette"]
                for kw in hand_keywords:
                    if any(kw in act.lower() for act in activities) or kw in description.lower():
                        events.append(
                            PhysicalEvent(
                                experiment_id=experiment.experiment_id,
                                event_type="hand_contact",
                                timestamp_sec=ts,
                                duration_sec=self._frame_extractor.sample_interval_sec,
                                location=location,
                                confidence=conf,
                                provenance=ProvenanceInfo(
                                    source="video",
                                    source_id=f"frame_{frame['frame_id']}",
                                    confidence=conf,
                                    is_inferred=True,
                                    inference_method="heuristic_keyword_detection",
                                    model_used=model,
                                ),
                                metadata={
                                    **base_metadata,
                                    "trigger_keyword": kw,
                                    "activities": activities,
                                },
                            )
                        )
                        break

            new_objects = objects - prev_objects
            removed_objects = prev_objects - objects
            if not has_error and (new_objects or removed_objects):
                events.append(
                    PhysicalEvent(
                        experiment_id=experiment.experiment_id,
                        event_type="object_state_change",
                        timestamp_sec=ts,
                        duration_sec=self._frame_extractor.sample_interval_sec,
                        location=location,
                        confidence=conf,
                        parameters=[
                            StepParameter(name="new_objects", value=sorted(new_objects), source="observed"),
                            StepParameter(name="removed_objects", value=sorted(removed_objects), source="observed"),
                        ],
                        provenance=ProvenanceInfo(
                            source="video",
                            source_id=f"frame_{frame['frame_id']}",
                            confidence=conf,
                            is_inferred=True,
                            inference_method="object_diff_detection",
                            model_used=model,
                        ),
                        metadata={
                            **base_metadata,
                            "new_objects": sorted(new_objects),
                            "removed_objects": sorted(removed_objects),
                        },
                    )
                )

            if not has_error:
                liquid_keywords = ["pour", "transfer", "pipette", "dispense", "titrate", "add liquid"]
                for kw in liquid_keywords:
                    if any(kw in act.lower() for act in activities) or kw in description.lower():
                        events.append(
                            PhysicalEvent(
                                experiment_id=experiment.experiment_id,
                                event_type="liquid_transfer",
                                timestamp_sec=ts,
                                duration_sec=self._frame_extractor.sample_interval_sec,
                                location=location,
                                confidence=conf,
                                provenance=ProvenanceInfo(
                                    source="video",
                                    source_id=f"frame_{frame['frame_id']}",
                                    confidence=conf,
                                    is_inferred=True,
                                    inference_method="heuristic_keyword_detection",
                                    model_used=model,
                                ),
                                metadata={
                                    **base_metadata,
                                    "trigger_keyword": kw,
                                },
                            )
                        )
                        break

            prev_objects_by_stream[stream_id] = objects

        # Fallback: 如果所有帧分析都失败，从 context 文本生成物理事件
        if not events and self._context_text:
            events.extend(self._fallback_physical_events_from_context(experiment))

        return events

    def _fallback_physical_events_from_context(
        self, experiment: Experiment
    ) -> List[PhysicalEvent]:
        """Fallback: 从 context 文本中解析实验室操作关键词，生成 PhysicalEvent。"""
        events = []
        text = self._context_text.lower()

        # 关键词到事件类型的映射（支持中英文）
        keyword_map = [
            ("hand_contact", ["hand", "拿起", "握住", "抓住", "pick", "hold", "grab", "touch", "grasp", "吸取", "pipette"]),
            ("liquid_transfer", ["pour", "transfer", "pipette", "dispense", "titrate", "倾倒", "转移", "加液", "分液"]),
            ("object_state_change", ["move", "put", "place", "remove", "移动", "放置", "取出", "拿起"]),
            ("mixing", ["mix", "vortex", "stir", "shake", " vortex", "混合", "涡旋", "搅拌"]),
            ("incubation", ["incubate", "heat", "cool", "孵育", "加热", "冷却"]),
            ("centrifugation", ["centrifuge", "spin", "离心", "离心机"]),
        ]

        for event_type, keywords in keyword_map:
            for kw in keywords:
                if kw in text:
                    events.append(PhysicalEvent(
                        experiment_id=experiment.experiment_id,
                        event_type=event_type,
                        timestamp_sec=0.0,
                        duration_sec=5.0,
                        location="workspace",
                        confidence=0.6,
                        provenance=ProvenanceInfo(
                            source="context",
                            confidence=0.6,
                            is_inferred=True,
                            inference_method="context_keyword_inference",
                            model_used="heuristic",
                        ),
                        metadata={
                            "trigger_keyword": kw,
                            "source": "context_text_fallback",
                        },
                    ))
                    break  # 每个事件类型最多生成一个

        return events

    def _generate_material_stream(self, experiment: Experiment) -> List[MultimodalMaterialStreamItem]:
        """
        从帧分析结果生成 MultimodalMaterialStreamItem。

        每个采样帧生成一个 MultimodalMaterialStreamItem，
        桥接视频帧 / 关键帧 / 文本上下文 / physical event -> StepRecord / Timeline。
        """
        items: List[MultimodalMaterialStreamItem] = []
        change_score_by_frame_id: Dict[int, float] = {}
        change_scores_by_stream: Dict[str, List[float]] = {}
        previous_frame_path_by_stream: Dict[str, Optional[str]] = {}
        previous_objects_by_stream: Dict[str, set[str]] = {}
        previous_activities_by_stream: Dict[str, set[str]] = {}
        clip_counter_by_stream: Dict[str, int] = {}
        for frame in self._video_frames:
            stream_id = frame.get("stream_id") or "video"
            previous_frame_path = previous_frame_path_by_stream.get(stream_id)
            change_score = self._frame_change_score(frame.get("path"), previous_frame_path)
            frame_id = int(frame.get("frame_id", len(change_score_by_frame_id)))
            change_score_by_frame_id[frame_id] = change_score
            if previous_frame_path is not None:
                change_scores_by_stream.setdefault(stream_id, []).append(change_score)
            previous_frame_path_by_stream[stream_id] = frame.get("path")
        default_change_threshold = self._adaptive_keyframe_change_threshold(
            [score for scores in change_scores_by_stream.values() for score in scores]
        )
        change_threshold_by_stream = {
            stream_id: self._adaptive_keyframe_change_threshold(scores)
            for stream_id, scores in change_scores_by_stream.items()
        }
        previous_frame_path_by_stream = {}
        conversation_events = [evt for evt in self._context_events if evt.context_source == ContextSource.CONVERSATION]
        if not conversation_events and self._context_inputs:
            for index, item in enumerate(self._context_inputs):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                conversation_events.append(
                    ContextEvent(
                        context_source=self._context_source_from_input(item),
                        event_type=str(item.get("kind") or "context_input"),
                        timestamp_sec=self._resolve_context_timestamp(item, index, len(self._context_inputs)),
                        content=text,
                        metadata={k: v for k, v in item.items() if k != "text"},
                    )
                )

        for frame, analysis in zip(self._video_frames, self._frame_analyses):
            stream_id = frame.get("stream_id") or "video"
            previous_frame_path = previous_frame_path_by_stream.get(stream_id)
            local_timestamp = frame.get("local_timestamp_sec", frame.get("timestamp_sec", 0.0))
            object_labels = analysis.get("object_labels", []) if "error" not in analysis else []
            detected_activities = analysis.get("detected_activities", []) if "error" not in analysis else []
            object_set = self._normalized_label_set(object_labels)
            activity_set = self._normalized_label_set(detected_activities)
            change_score = change_score_by_frame_id.get(int(frame.get("frame_id", 0)), 0.0)
            change_threshold = change_threshold_by_stream.get(stream_id, default_change_threshold)
            object_changed = object_set != previous_objects_by_stream.get(stream_id, set())
            activity_changed = activity_set != previous_activities_by_stream.get(stream_id, set())
            is_boundary_frame = previous_frame_path is None
            visual_changed = change_score >= change_threshold
            is_key_frame = is_boundary_frame or visual_changed or object_changed or activity_changed
            key_frame_reason = None
            if is_boundary_frame:
                key_frame_reason = "stream_start"
            elif visual_changed:
                key_frame_reason = "visual_change"
            elif object_changed:
                key_frame_reason = "object_state_change"
            elif activity_changed:
                key_frame_reason = "activity_shift"

            linked_context_ids: List[str] = []
            transcript_segment: Optional[str] = None
            nearest_context = None
            for event in conversation_events:
                if event.context_source != ContextSource.CONVERSATION:
                    continue
                if event.timestamp_sec is None:
                    continue
                if abs(event.timestamp_sec - frame.get("timestamp_sec", 0.0)) <= max(self._frame_extractor.sample_interval_sec * 1.5, 2.0):
                    nearest_context = event
                    break
            if nearest_context is not None:
                linked_context_ids.append(nearest_context.event_id)
                transcript_segment = nearest_context.content[:200]

            if is_key_frame:
                clip_counter_by_stream[stream_id] = clip_counter_by_stream.get(stream_id, 0) + 1
            clip_id = f"{stream_id}:clip:{clip_counter_by_stream.get(stream_id, 0)}" if clip_counter_by_stream.get(stream_id, 0) else None

            item = MultimodalMaterialStreamItem(
                experiment_id=experiment.experiment_id,
                timestamp_sec=frame.get("timestamp_sec", 0.0),
                local_timestamp_sec=local_timestamp,
                global_timestamp_sec=frame.get("timestamp_sec", 0.0),
                camera_id=frame.get("camera_id"),
                view_type=frame.get("view_type"),
                source_group=frame.get("source_group"),
                source_type=frame.get("source_type"),
                sync_group=frame.get("sync_group"),
                alignment_confidence=frame.get("alignment_confidence"),
                media_asset_id=frame.get("media_asset_id"),
                stream_id=stream_id,
                frame_id=frame.get("frame_id", 0),
                local_frame_id=frame.get("local_frame_id"),
                frame_bgr_path=frame.get("path"),
                detected_objects=analysis.get("detected_objects", []) if "error" not in analysis else [],
                scene_description=analysis.get("description") if "error" not in analysis else "fallback_analysis",
                detected_activities=detected_activities,
                object_labels=object_labels,
                ppe_status=analysis.get("ppe_status", {}) if "error" not in analysis else {},
                transcript_segment=transcript_segment,
                conversation_context=self._context_text[:200] if self._context_text else None,
                linked_context_event_ids=linked_context_ids,
                inference_model=analysis.get("model", "fallback_heuristic"),
                confidence=float(analysis.get("confidence", 0.3)) if "error" not in analysis else 0.3,
                is_key_frame=is_key_frame,
                key_frame_reason=key_frame_reason,
                change_score=round(change_score, 4),
                clip_id=clip_id,
                analysis={
                    "key_frame_detection": {
                        "change_threshold": round(change_threshold, 4),
                        "base_change_threshold": round(self._keyframe_base_change_threshold(), 4),
                        "visual_changed": bool(visual_changed),
                        "object_changed": bool(object_changed),
                        "activity_changed": bool(activity_changed),
                    }
                },
                provenance=ProvenanceInfo(
                    source="video",
                    source_id=f"frame_{frame['frame_id']}",
                    confidence=float(analysis.get("confidence", 0.3)) if "error" not in analysis else 0.3,
                    is_inferred=True,
                    inference_method="frame_analysis",
                    model_used=analysis.get("model", "fallback_heuristic"),
                ),
            )
            items.append(item)
            previous_frame_path_by_stream[stream_id] = frame.get("path")
            previous_objects_by_stream[stream_id] = object_set
            previous_activities_by_stream[stream_id] = activity_set

        return items

    def _link_physical_events(self, experiment: Experiment, timeline: ExperimentTimeline) -> None:
        """将 PhysicalEvent 关联到对应的 StepRecord。"""
        for step in timeline.steps:
            step_start = step.start_time_sec
            step_end = step.end_time_sec or (step_start + 10.0)

            for pe in experiment.physical_events:
                if step_start <= pe.timestamp_sec <= step_end:
                    if pe.event_id not in step.linked_physical_events:
                        step.linked_physical_events.append(pe.event_id)

    def _link_material_stream(
        self,
        experiment: Experiment,
        timeline: ExperimentTimeline,
        material_stream: List[MultimodalMaterialStreamItem],
    ) -> None:
        """将 MultimodalMaterialStreamItem 关联到 Timeline 和 StepRecord。"""
        # 将 material_stream item IDs 存入 timeline metadata
        timeline.metadata["material_stream_ids"] = [item.item_id for item in material_stream]
        timeline.metadata["material_stream_count"] = len(material_stream)
        timeline.metadata["material_stream_key_frame_count"] = sum(1 for item in material_stream if item.is_key_frame)
        timeline.metadata["material_stream_stream_ids"] = sorted({item.stream_id for item in material_stream if item.stream_id})

        # 关联到步骤
        for step in timeline.steps:
            step_start = step.start_time_sec
            step_end = step.end_time_sec or (step_start + 10.0)

            for item in material_stream:
                if step_start <= item.timestamp_sec <= step_end:
                    # 将 material stream item 的 media asset 关联到步骤
                    if item.item_id not in step.linked_media_assets:
                        step.linked_media_assets.append(item.item_id)

    def _link_evidence(self, experiment: Experiment, timeline: ExperimentTimeline) -> None:
        """关联证据到步骤。确保 EvidenceRef.media_asset_id 正确关联到 MediaAsset。"""
        frame_asset_map = {
            frame.get("frame_id"): frame.get("media_asset_id")
            for frame in self._video_frames
        }

        for step in timeline.steps:
            # 处理已有的 evidence_refs（来自 StepReasoner），补全 media_asset_id
            for er in step.evidence_refs:
                if er.media_asset_id is None and er.frame_id in frame_asset_map:
                    er.media_asset_id = frame_asset_map.get(er.frame_id)
                if er.media_asset_id is None and er.timestamp_sec is not None:
                    nearest_frame = min(
                        self._video_frames,
                        key=lambda frame: abs((frame.get("timestamp_sec", 0.0) or 0.0) - er.timestamp_sec),
                        default=None,
                    )
                    if nearest_frame is not None:
                        er.media_asset_id = nearest_frame.get("media_asset_id")

            # 为没有证据的步骤添加默认证据引用（包含正确的 media_asset_id）
            if not step.evidence_refs:
                relevant_frames = [
                    f for f in self._video_frames
                    if f.get("timestamp_sec", 0) >= step.start_time_sec
                    and f.get("timestamp_sec", 0) <= (step.end_time_sec or step.start_time_sec + 10)
                ]
                for frame in relevant_frames[:3]:  # 最多3个证据帧
                    step.evidence_refs.append(
                        EvidenceRef(
                            evidence_type=EvidenceType.VIDEO_FRAME,
                            source="video",
                            frame_id=frame["frame_id"],
                            timestamp_sec=frame.get("timestamp_sec", 0.0),
                            media_asset_id=frame.get("media_asset_id"),
                            confidence=step.confidence,
                        )
                    )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_experiment(self) -> Optional[Experiment]:
        return self._experiment

    def get_timeline(self) -> Optional[ExperimentTimeline]:
        return self._timeline

    def save_outputs(self, output_dir: str = "outputs/experiments") -> Dict[str, str]:
        """保存输出到文件。"""
        if not self._experiment or not self._timeline:
            raise ValueError("No experiment processed yet. Call process() first.")

        out_dir = Path(output_dir) / self._experiment.experiment_id
        out_dir.mkdir(parents=True, exist_ok=True)

        exp_path = out_dir / "experiment.json"
        exp_payload = self._experiment.to_dict()
        exp_payload["video_inputs"] = list(self._video_inputs or [])
        exp_payload["video_metadata"] = list(self._video_inputs or [])
        exp_path.write_text(json.dumps(exp_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        timeline_path = out_dir / "timeline.json"
        timeline_path.write_text(self._timeline.to_json(), encoding="utf-8")

        # 步骤列表（简化版）
        steps_path = out_dir / "steps.json"
        steps_data = [s.to_dict() for s in self._timeline.steps]
        steps_path.write_text(
            json.dumps(steps_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 物理事件
        pe_path = out_dir / "legacy_physical_events.ungated.json"
        pe_data = legacy_physical_event_candidate_rows(
            self._experiment.physical_events,
            source="experiment_service_legacy_save_outputs",
        )
        pe_path.write_text(
            json.dumps(pe_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "physical_events_legacy_candidates.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in pe_data) + ("\n" if pe_data else ""),
            encoding="utf-8",
        )

        # 多模态素材流
        ms_path = out_dir / "material_stream.json"
        ms_data = [item.to_dict() for item in getattr(self, "_material_stream", [])]
        ms_path.write_text(
            json.dumps(ms_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        semantic_sync_path = out_dir / "semantic_sync_anchors.json"
        semantic_sync_path.write_text(
            json.dumps(getattr(self, "_semantic_sync", {}) or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        preprocessing_path = out_dir / "preprocessing.json"
        material_index_path = out_dir / "material_index.sqlite"
        try:
            from labsopguard.preprocessing import MultiModalPreprocessor
            from labsopguard.input_layer import TimeAnchoredText
            from labsopguard.retrieval import MaterialRetrievalIndex

            context_records = [
                TimeAnchoredText(
                    source_type=str(item.get("kind") or "context"),
                    content=str(item.get("text", "")).strip(),
                    timestamp_sec=self._safe_float(item.get("timestamp_sec")),
                    start_time_sec=self._safe_float(item.get("start_time_sec")),
                    end_time_sec=self._safe_float(item.get("end_time_sec")),
                    anchor_video_index=item.get("video_index"),
                    anchor_video_asset_id=item.get("video_asset_id"),
                    metadata={k: v for k, v in item.items() if k != "text"},
                )
                for item in self._context_inputs
                if str(item.get("text", "")).strip()
            ]

            artifact = MultiModalPreprocessor().build_artifact(
                duration_sec=self._video_info.get("duration_sec", 0.0) or 0.0,
                context_text=self._context_text,
                protocol_text=self._protocol_text,
                physical_events=self._experiment.physical_events,
                material_stream=getattr(self, "_material_stream", []),
                context_records=context_records,
                video_assets=self._experiment.video_assets,
                clip_output_dir=out_dir / "clips",
                clip_window_sec=max(self._frame_extractor.sample_interval_sec, 1.0),
            )
            preprocessing_payload = {
                "schema_version": "preprocessing.v1",
                "material_stream_schema_version": "material_stream.v1",
                "aligned_text": [item.__dict__ for item in artifact.aligned_text],
                "key_timestamps": artifact.key_timestamps,
                "video_index": artifact.video_index,
                "detected_changes": artifact.detected_changes,
                "video_streams": artifact.video_streams,
                "key_frames": artifact.key_frames,
                "key_clips": artifact.key_clips,
                "time_anchored_material_stream": artifact.time_anchored_material_stream,
                "alignment_summary": artifact.alignment_summary,
            }
            preprocessing_path.write_text(
                json.dumps(preprocessing_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            material_index = MaterialRetrievalIndex(material_index_path)
            material_index.reset()
            material_index.index_payloads(
                preprocessing_payload.get("time_anchored_material_stream", []),
                preprocessing=preprocessing_payload,
                experiment_id=self._experiment.experiment_id,
            )
            preprocessing_payload["material_index_health"] = material_index.health_check()
            material_index.close()
            try:
                from labsopguard.key_material_reference import (
                    append_reference_items_to_preprocessing,
                    material_stream_items_from_references,
                    write_experiment_reference_outputs,
                )

                reference_outputs = write_experiment_reference_outputs(
                    experiment_dir=out_dir,
                    experiment_record=exp_payload,
                    material_stream=ms_data,
                    preprocessing=preprocessing_payload,
                    steps=steps_data,
                    segmentation=json.loads((self._artifact_root / "experiment_segmentation.json").read_text(encoding="utf-8"))
                    if self._artifact_root and (self._artifact_root / "experiment_segmentation.json").exists()
                    else {},
                    formal_library_root=Path(output_dir).parent / "material_references",
                )
                references = reference_outputs.get("references", [])
                preprocessing_payload = append_reference_items_to_preprocessing(preprocessing_payload, references)
                reference_items = material_stream_items_from_references(
                    references,
                    existing_item_ids=[str(item.get("item_id") or "") for item in ms_data],
                )
                if reference_items:
                    ms_data = [*ms_data, *reference_items]
                    ms_path.write_text(json.dumps(ms_data, ensure_ascii=False, indent=2), encoding="utf-8")
                    material_index = MaterialRetrievalIndex(material_index_path)
                    material_index.reset()
                    material_index.index_payloads(
                        ms_data,
                        preprocessing=preprocessing_payload,
                        experiment_id=self._experiment.experiment_id,
                    )
                    preprocessing_payload["material_index_health"] = material_index.health_check()
                    material_index.close()
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Key material reference ledger generation failed: %s", exc)
            preprocessing_path.write_text(
                json.dumps(preprocessing_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            preprocessing_path.write_text(
                json.dumps({}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {
            "experiment": str(exp_path),
            "timeline": str(timeline_path),
            "steps": str(steps_path),
            "physical_events": str(pe_path),
            "material_stream": str(ms_path),
            "semantic_sync_anchors": str(semantic_sync_path),
            "semantic_sync_anchors_json": str(semantic_sync_path),
            "preprocessing": str(preprocessing_path),
            "material_index": str(material_index_path),
            "time_alignment": str(out_dir / "artifacts" / "time_alignment.json"),
            "key_material_references": str(out_dir / "artifacts" / "key_material_references.jsonl"),
            "key_material_references_sqlite": str(out_dir / "artifacts" / "key_material_references.sqlite"),
            "key_material_reference_manifest": str(out_dir / "artifacts" / "key_material_reference_manifest.json"),
            "physical_change_log": str(out_dir / "artifacts" / "physical_change_log.jsonl"),
        }

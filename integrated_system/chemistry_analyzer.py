"""
Chemistry-Aware Detection Layer for LabSOPGuard.

Two-stage AI analysis framework:
  Stage 1: Scene Understanding - identify experiment type, containers, tools
  Stage 2: Per-frame Operation Analysis - what operation is happening, is it compliant

Structured prompts optimized for chemistry lab scenarios.
When AI (Doubao/OpenAI-compatible) is unavailable, falls back to local visual analysis
with chemistry-specific heuristics.

Data models:
  - ChemistryObservation: structured per-frame analysis result
  - ChemistrySceneAnalysis: aggregate scene-level analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from integrated_system.scene_understander import (
    ChemistryEntity,
    ExperimentType,
    OperationType,
    SceneProfile,
    build_scene_profile_from_analysis,
    build_scene_profile_from_text,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ChemistryObservation:
    """Structured per-frame chemistry analysis result."""
    frame_index: int
    timestamp_sec: float
    image_name: str

    # What operation is happening
    operation_type: str  # OperationType value
    operation_description_zh: str
    operation_description_en: str

    # What objects are involved
    active_containers: List[ChemistryEntity] = field(default_factory=list)
    active_tools: List[ChemistryEntity] = field(default_factory=list)

    # Safety state
    ppe_detected: Dict[str, bool] = field(default_factory=lambda: {
        "gloves": False, "goggles": False, "lab_coat": False
    })
    safety_concerns: List[str] = field(default_factory=list)

    # Compliance
    expected_step_id: str = ""  # Which expected step this maps to
    step_compliance: str = "unknown"  # "compliant", "violation", "unknown"
    compliance_notes: str = ""

    # Visual quality
    image_quality: str = "ok"  # "ok", "blurry", "dark", "overexposed"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["active_containers"] = [asdict(c) for c in self.active_containers]
        d["active_tools"] = [asdict(t) for t in self.active_tools]
        return d


@dataclass
class ChemistrySceneAnalysis:
    """Aggregate scene-level analysis across all keyframes."""
    experiment_type: str
    experiment_type_zh: str
    total_frames_analyzed: int
    observations: List[ChemistryObservation]

    # Aggregated statistics
    operations_detected: Dict[str, int] = field(default_factory=dict)  # operation_type -> count
    safety_violation_count: int = 0
    compliance_ratio: float = 0.0  # compliant / total

    # Scene profile reference
    scene_profile: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["observations"] = [o.to_dict() for o in self.observations]
        return d

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage 1: Scene Understanding Prompt
# ---------------------------------------------------------------------------

STAGE1_PROMPT_ZH = """你是一位专业的化学实验室安全分析AI。请分析以下实验视频的关键帧，完成场景理解。

请以JSON格式返回以下信息：
{
  "experiment_type": "acid_base_titration | solution_preparation | pipetting | weighing | filtration | heating_reaction | ph_measurement | general_lab_operation",
  "confidence": 0.0-1.0,
  "description_zh": "中文场景描述",
  "description_en": "English scene description",
  "containers_detected": [
    {"name": "beaker", "name_zh": "烧杯", "confidence": 0.9, "attributes": {"contains_liquid": true}}
  ],
  "tools_detected": [
    {"name": "burette", "name_zh": "滴定管", "confidence": 0.85}
  ],
  "reagents_detected": [
    {"name": "hcl", "name_zh": "盐酸", "confidence": 0.7}
  ],
  "key_objects": ["burette", "erlenmeyer_flask", "indicator_bottle"],
  "safety_notes": ["must_have_gloves", "slow_addition_near_endpoint"],
  "initial_observation": "观察到的初始状态描述"
}

识别的容器类型包括：beaker(烧杯), volumetric_flask(容量瓶), erlenmeyer_flask(锥形瓶),
graduated_cylinder(量筒), burette(滴定管), pipette(移液管), reagent_bottle(试剂瓶),
wash_bottle(洗瓶), crucible(坩埚), test_tube(试管), funnel(漏斗), spatula(刮刀)

识别的工具类型包括：magnetic_stirrer(磁力搅拌器), hot_plate(加热板), balance(天平),
ph_meter(pH计), thermometer(温度计), timer(计时器)

请仔细观察画面中的实验器材、操作人员的PPE穿戴情况、以及台面布局。"""

STAGE1_PROMPT_EN = """You are a professional chemistry lab safety AI. Analyze these keyframes from an experiment video for scene understanding.

Return JSON with:
{
  "experiment_type": "acid_base_titration | solution_preparation | pipetting | weighing | filtration | heating_reaction | ph_measurement | general_lab_operation",
  "confidence": 0.0-1.0,
  "description_zh": "Chinese description",
  "description_en": "English description",
  "containers_detected": [{"name": "beaker", "name_zh": "烧杯", "confidence": 0.9}],
  "tools_detected": [{"name": "burette", "name_zh": "滴定管", "confidence": 0.85}],
  "reagents_detected": [{"name": "hcl", "name_zh": "盐酸", "confidence": 0.7}],
  "key_objects": ["burette", "erlenmeyer_flask"],
  "safety_notes": ["must_have_gloves"],
  "initial_observation": "Initial state description"
}

Observe containers, tools, PPE, and workspace layout carefully."""


# ---------------------------------------------------------------------------
# Stage 2: Per-Frame Operation Analysis Prompt
# ---------------------------------------------------------------------------

STAGE2_PROMPT_ZH_TEMPLATE = """你是一位专业的化学实验室安全分析AI。当前实验类型为：{experiment_type_zh}（{experiment_type_en}）。

期望的实验步骤顺序为：
{expected_steps}

请分析以下关键帧，判断当前正在进行的操作。

请以JSON格式返回：
{
  "operation_type": "pour | transfer | titrate | measure | weigh | stir | heat | filter | rinse | label | verify | dispose | clean | read_volume | unknown",
  "operation_description_zh": "中文操作描述",
  "operation_description_en": "English operation description",
  "active_containers": [{"name": "beaker", "name_zh": "烧杯", "confidence": 0.9}],
  "active_tools": [{"name": "pipette", "name_zh": "移液管", "confidence": 0.8}],
  "ppe_detected": {"gloves": true, "goggles": true, "lab_coat": false},
  "safety_concerns": ["未佩戴护目镜"],
  "expected_step_id": "execute_titration",
  "step_compliance": "compliant | violation | unknown",
  "compliance_notes": "操作符合预期步骤",
  "confidence": 0.85
}

请特别注意：
1. 操作人员是否佩戴了手套、护目镜、实验服
2. 容器的使用是否正确（如：是否使用了正确的容器进行转移）
3. 操作手法是否规范（如：滴定速度是否合适）
4. 是否存在安全隐患（如：试剂瓶未盖、废液处理不当）"""

STAGE2_PROMPT_EN_TEMPLATE = """You are a professional chemistry lab safety AI. Current experiment type: {experiment_type_en}.

Expected step sequence:
{expected_steps}

Analyze this keyframe and determine the current operation.

Return JSON:
{
  "operation_type": "pour | transfer | titrate | measure | weigh | stir | heat | filter | rinse | label | verify | dispose | clean | read_volume | unknown",
  "operation_description_zh": "Chinese description",
  "operation_description_en": "English description",
  "active_containers": [{"name": "beaker", "name_zh": "烧杯", "confidence": 0.9}],
  "active_tools": [{"name": "pipette", "name_zh": "移液管", "confidence": 0.8}],
  "ppe_detected": {"gloves": true, "goggles": true, "lab_coat": false},
  "safety_concerns": ["no goggles detected"],
  "expected_step_id": "execute_titration",
  "step_compliance": "compliant | violation | unknown",
  "compliance_notes": "Operation matches expected step",
  "confidence": 0.85
}

Pay special attention to PPE, container usage, technique, and safety concerns."""


# ---------------------------------------------------------------------------
# Local Visual Analysis (Chemistry-Enhanced Fallback)
# ---------------------------------------------------------------------------

def _chemistry_local_analysis(
    image_path: Path,
    frame_index: int,
    timestamp_sec: float,
    scene_profile: Optional[SceneProfile] = None,
) -> ChemistryObservation:
    """Enhanced local visual analysis with chemistry-specific heuristics.

    Uses OpenCV to extract visual features and map them to chemistry observations.
    This is the fallback when AI/VLM is unavailable.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return ChemistryObservation(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            image_name=image_path.name,
            operation_type="unknown",
            operation_description_zh="图像处理库不可用，无法进行视觉分析。",
            operation_description_en="Image processing library unavailable.",
            confidence=0.0,
            image_quality="unknown",
        )

    frame = cv2.imread(str(image_path))
    if frame is None:
        return ChemistryObservation(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            image_name=image_path.name,
            operation_type="unknown",
            operation_description_zh="图像读取失败。",
            operation_description_en="Failed to read image.",
            confidence=0.0,
            image_quality="unknown",
        )

    # Visual quality assessment
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 40:
        image_quality = "dark"
    elif brightness > 225:
        image_quality = "overexposed"
    elif sharpness < 30:
        image_quality = "blurry"
    else:
        image_quality = "ok"

    # Color analysis for chemistry-specific cues
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect liquid colors (common in chemistry)
    # Red/orange/yellow range (indicators, solutions)
    red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    orange_mask = cv2.inRange(hsv, (10, 50, 50), (25, 255, 255))
    yellow_mask = cv2.inRange(hsv, (25, 50, 50), (35, 255, 255))
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))

    color_cues = []
    total_pixels = frame.shape[0] * frame.shape[1]
    for color_name, mask in [("red", red_mask), ("orange", orange_mask),
                              ("yellow", yellow_mask), ("blue", blue_mask),
                              ("green", green_mask)]:
        ratio = float(np.count_nonzero(mask)) / total_pixels
        if ratio > 0.02:
            color_cues.append(f"{color_name}({ratio:.2%})")

    # Edge density for activity level
    edges = cv2.Canny(gray, 60, 140)
    edge_ratio = float(np.count_nonzero(edges) / edges.size) if edges.size > 0 else 0.0

    # Build observation based on visual cues
    desc_parts_zh = []
    desc_parts_en = []

    if color_cues:
        desc_parts_zh.append(f"检测到颜色特征：{', '.join(color_cues)}")
        desc_parts_en.append(f"Color features detected: {', '.join(color_cues)}")

    if edge_ratio > 0.15:
        desc_parts_zh.append("操作区域变化较大，可能正在进行动态操作")
        desc_parts_en.append("High activity in operation area, dynamic operation likely")
    elif edge_ratio < 0.03:
        desc_parts_zh.append("画面变化较小，可能处于静止或观察阶段")
        desc_parts_en.append("Low activity, possibly in observation phase")

    desc_parts_zh.append(f"图像质量：亮度{brightness:.0f}、清晰度{sharpness:.0f}")
    desc_parts_en.append(f"Image quality: brightness={brightness:.0f}, sharpness={sharpness:.0f}")

    # Infer operation type from scene context if available
    operation_type = "unknown"
    expected_step_id = ""
    if scene_profile and scene_profile.expected_steps:
        # Simple heuristic: map frame position to expected step
        step_idx = min(frame_index, len(scene_profile.expected_steps) - 1)
        step = scene_profile.expected_steps[step_idx]
        expected_step_id = step["step_id"]

    return ChemistryObservation(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        image_name=image_path.name,
        operation_type=operation_type,
        operation_description_zh="；".join(desc_parts_zh) if desc_parts_zh else "基础视觉分析完成。",
        operation_description_en="; ".join(desc_parts_en) if desc_parts_en else "Basic visual analysis complete.",
        expected_step_id=expected_step_id,
        step_compliance="unknown",
        confidence=0.3 if image_quality == "ok" else 0.15,
        image_quality=image_quality,
    )


# ---------------------------------------------------------------------------
# Two-Stage Analysis Pipeline
# ---------------------------------------------------------------------------

def run_stage1_scene_understanding(
    initial_keyframes: List[Path],
    ai_client: Any = None,
    ai_model: str = "",
    ai_base_url: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> SceneProfile:
    """Stage 1: Analyze initial keyframes to understand the experiment scene.

    Args:
        initial_keyframes: First few keyframes (3-5 frames) for scene understanding
        ai_client: OpenAI-compatible client (None if unavailable)
        ai_model: Model name for AI analysis
        ai_base_url: Base URL for AI API
        progress_callback: Progress reporting callback

    Returns:
        SceneProfile with experiment type, expected steps, and entity detection
    """
    if not initial_keyframes:
        return SceneProfile(
            experiment_type=ExperimentType.UNKNOWN,
            experiment_type_zh="未知实验",
            confidence=0.0,
            expected_steps=[],
            expected_step_ids=[],
            ai_analysis_ready=False,
            ai_analysis_unavailable_reason="no_keyframes",
        )

    # Try AI analysis first
    if ai_client is not None:
        try:
            import base64

            # Build multi-image content for scene understanding
            content_parts = [{"type": "text", "text": STAGE1_PROMPT_ZH}]
            for kf_path in initial_keyframes[:5]:  # Max 5 frames for scene understanding
                raw = kf_path.read_bytes()
                b64 = base64.b64encode(raw).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

            resp = ai_client.chat.completions.create(
                model=ai_model,
                messages=[
                    {"role": "system", "content": "You are a chemistry lab analysis AI. Return valid JSON only."},
                    {"role": "user", "content": content_parts},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            # Extract response text
            text = ""
            if resp.choices:
                msg = resp.choices[0].message
                if isinstance(msg.content, str):
                    text = msg.content
                elif isinstance(msg.content, list):
                    text = "\n".join(
                        getattr(p, "text", "") for p in msg.content if hasattr(p, "text")
                    )

            # Parse JSON from response
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            analysis = json.loads(text)

            scene_profile = build_scene_profile_from_analysis(analysis)
            if progress_callback:
                progress_callback(1, 1, f"场景理解完成：{scene_profile.experiment_type_zh}（置信度 {scene_profile.confidence:.0%}）")
            return scene_profile

        except Exception as exc:
            if progress_callback:
                progress_callback(0, 1, f"AI场景理解失败，使用规则回退：{exc}")

    # Fallback: use local analysis text to classify
    local_results = []
    for kf in initial_keyframes[:3]:
        obs = _chemistry_local_analysis(kf, 0, 0.0)
        local_results.append(obs.operation_description_en)

    combined_text = " ".join(local_results)
    scene_profile = build_scene_profile_from_text(combined_text)
    if progress_callback:
        progress_callback(1, 1, f"场景理解完成（规则回退）：{scene_profile.experiment_type_zh}")
    return scene_profile


def run_stage2_operation_analysis(
    keyframe_paths: List[Path],
    scene_profile: SceneProfile,
    ai_client: Any = None,
    ai_model: str = "",
    ai_base_url: str = "",
    timestamps: Optional[List[float]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[ChemistryObservation]:
    """Stage 2: Per-frame operation analysis.

    Args:
        keyframe_paths: All keyframe image paths
        scene_profile: Scene understanding result from Stage 1
        ai_client: OpenAI-compatible client (None if unavailable)
        ai_model: Model name
        ai_base_url: Base URL
        timestamps: Timestamp for each keyframe
        progress_callback: Progress callback

    Returns:
        List of ChemistryObservation per keyframe
    """
    observations: List[ChemistryObservation] = []
    ts_list = timestamps or [0.0] * len(keyframe_paths)

    # Build step description for prompt
    steps_text = "\n".join(
        f"  {i+1}. {s['name_zh']} ({s['name_en']}) - step_id: {s['step_id']}"
        for i, s in enumerate(scene_profile.expected_steps)
    )

    for idx, kf_path in enumerate(keyframe_paths):
        ts = ts_list[idx] if idx < len(ts_list) else float(idx)

        if ai_client is not None:
            try:
                import base64

                prompt = STAGE2_PROMPT_ZH_TEMPLATE.format(
                    experiment_type_zh=scene_profile.experiment_type_zh,
                    experiment_type_en=scene_profile.experiment_type.value.replace("_", " "),
                    expected_steps=steps_text,
                )

                raw = kf_path.read_bytes()
                b64 = base64.b64encode(raw).decode("utf-8")

                resp = ai_client.chat.completions.create(
                    model=ai_model,
                    messages=[
                        {"role": "system", "content": "You are a chemistry lab analysis AI. Return valid JSON only."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ]},
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                )

                text = ""
                if resp.choices:
                    msg = resp.choices[0].message
                    if isinstance(msg.content, str):
                        text = msg.content
                    elif isinstance(msg.content, list):
                        text = "\n".join(getattr(p, "text", "") for p in msg.content if hasattr(p, "text"))

                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
                result = json.loads(text)

                obs = ChemistryObservation(
                    frame_index=idx,
                    timestamp_sec=ts,
                    image_name=kf_path.name,
                    operation_type=result.get("operation_type", "unknown"),
                    operation_description_zh=result.get("operation_description_zh", ""),
                    operation_description_en=result.get("operation_description_en", ""),
                    ppe_detected=result.get("ppe_detected", {}),
                    safety_concerns=result.get("safety_concerns", []),
                    expected_step_id=result.get("expected_step_id", ""),
                    step_compliance=result.get("step_compliance", "unknown"),
                    compliance_notes=result.get("compliance_notes", ""),
                    confidence=float(result.get("confidence", 0.5)),
                )
                observations.append(obs)

            except Exception:
                # AI failed for this frame, use local fallback
                obs = _chemistry_local_analysis(kf_path, idx, ts, scene_profile)
                observations.append(obs)
        else:
            # No AI available, use local fallback
            obs = _chemistry_local_analysis(kf_path, idx, ts, scene_profile)
            observations.append(obs)

        if progress_callback:
            progress_callback(idx + 1, len(keyframe_paths),
                              f"操作分析中：第 {idx + 1}/{len(keyframe_paths)} 帧")

    return observations


def build_scene_analysis(
    observations: List[ChemistryObservation],
    scene_profile: SceneProfile,
) -> ChemistrySceneAnalysis:
    """Build aggregate scene analysis from per-frame observations."""
    # Count operations
    op_counts: Dict[str, int] = {}
    violation_count = 0
    compliant_count = 0

    for obs in observations:
        op_type = obs.operation_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1
        if obs.step_compliance == "violation":
            violation_count += 1
        elif obs.step_compliance == "compliant":
            compliant_count += 1

    total = len(observations)
    compliance_ratio = compliant_count / total if total > 0 else 0.0

    return ChemistrySceneAnalysis(
        experiment_type=scene_profile.experiment_type.value,
        experiment_type_zh=scene_profile.experiment_type_zh,
        total_frames_analyzed=total,
        observations=observations,
        operations_detected=op_counts,
        safety_violation_count=violation_count,
        compliance_ratio=compliance_ratio,
        scene_profile=scene_profile.to_dict(),
    )

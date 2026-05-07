"""
自适应AI实验室监控系统 - 核心引擎
支持动态约束学习、可视化检测框和智能报警
"""
from __future__ import annotations

import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstraintType(Enum):
    """约束类型枚举"""
    PPE_REQUIRED = "ppe_required"           # 必须佩戴PPE
    EQUIPMENT_USAGE = "equipment_usage"     # 设备使用规范
    SPATIAL_CONSTRAINT = "spatial_constraint" # 空间约束
    TEMPORAL_CONSTRAINT = "temporal_constraint" # 时间约束
    PROCEDURAL_CONSTRAINT = "procedural_constraint" # 程序约束
    SAFETY_CONSTRAINT = "safety_constraint" # 安全约束

class SeverityLevel(Enum):
    """严重等级枚举"""
    CRITICAL = "Critical"    # 严重违规，立即停止
    MAJOR = "Major"          # 重要违规，需要纠正
    MINOR = "Minor"          # 轻微违规，建议改进
    WARNING = "Warning"      # 警告，注意观察

@dataclass
class LabConstraint:
    """实验室约束定义"""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    severity: SeverityLevel
    conditions: Dict[str, Any]  # 约束条件
    actions: List[str]           # 违规时的动作
    confidence_threshold: float = 0.7  # 置信度阈值
    enabled: bool = True
    learning_weight: float = 1.0  # 学习权重

@dataclass
class DetectionResult:
    """检测结果"""
    frame_id: int
    timestamp: float
    detections: List[Dict[str, Any]]  # 目标检测结果
    poses: List[Dict[str, Any]]       # 姿态估计结果
    actions: List[str]                # 识别的动作
    confidence: float                 # 整体置信度

@dataclass
class ViolationAlert:
    """违规告警"""
    alert_id: str
    constraint: LabConstraint
    detection_result: DetectionResult
    timestamp: datetime
    confidence: float
    bounding_boxes: List[Dict[str, Any]]  # 违规区域边界框
    description: str
    recommendation: str
    visual_data: Optional[Dict[str, Any]] = None  # 可视化数据

class AdaptiveConstraintLearner:
    """自适应约束学习器"""

    def __init__(self):
        self.constraint_history: List[Dict[str, Any]] = []
        self.laboratory_profiles: Dict[str, Dict[str, Any]] = {}
        self.learning_models: Dict[str, Any] = {}
        self.adaptation_rate = 0.1  # 适应率

    def learn_from_violation(self, lab_id: str, violation: ViolationAlert):
        """从违规中学习"""
        if lab_id not in self.laboratory_profiles:
            self.laboratory_profiles[lab_id] = {
                "violation_patterns": [],
                "constraint_preferences": {},
                "environmental_factors": {},
                "learning_history": []
            }

        profile = self.laboratory_profiles[lab_id]

        # 记录违规模式
        pattern = {
            "constraint_type": violation.constraint.constraint_type.value,
            "severity": violation.constraint.severity.value,
            "confidence": violation.confidence,
            "timestamp": violation.timestamp.isoformat(),
            "context": {
                "detections": len(violation.detection_result.detections),
                "actions": violation.detection_result.actions
            }
        }
        profile["violation_patterns"].append(pattern)

        # 更新约束偏好
        constraint_key = violation.constraint.constraint_id
        if constraint_key not in profile["constraint_preferences"]:
            profile["constraint_preferences"][constraint_key] = {
                "frequency": 0,
                "avg_confidence": 0.0,
                "last_updated": datetime.now().isoformat()
            }

        pref = profile["constraint_preferences"][constraint_key]
        pref["frequency"] += 1
        pref["avg_confidence"] = (pref["avg_confidence"] + violation.confidence) / 2
        pref["last_updated"] = datetime.now().isoformat()

    def adapt_constraints(self, lab_id: str, current_context: Dict[str, Any]) -> List[LabConstraint]:
        """根据实验室上下文自适应调整约束"""
        if lab_id not in self.laboratory_profiles:
            return self._get_default_constraints()

        profile = self.laboratory_profiles[lab_id]
        adapted_constraints = []

        # 基于历史数据调整约束
        for constraint in self._get_default_constraints():
            adapted_constraint = self._adapt_constraint(constraint, profile, current_context)
            if adapted_constraint.enabled:
                adapted_constraints.append(adapted_constraint)

        return adapted_constraints

    def _adapt_constraint(self, constraint: LabConstraint, profile: Dict[str, Any], context: Dict[str, Any]) -> LabConstraint:
        """调整单个约束"""
        constraint_key = constraint.constraint_id

        if constraint_key in profile["constraint_preferences"]:
            pref = profile["constraint_preferences"][constraint_key]

            # 基于频率调整置信度阈值
            if pref["frequency"] > 10:  # 高频违规，降低阈值
                constraint.confidence_threshold = max(0.5, constraint.confidence_threshold - 0.1)
            elif pref["frequency"] < 2:  # 低频违规，提高阈值
                constraint.confidence_threshold = min(0.9, constraint.confidence_threshold + 0.05)

            # 基于环境因素调整
            if context.get("high_risk_mode", False):
                constraint.confidence_threshold = max(0.6, constraint.confidence_threshold - 0.1)

        return constraint

    def _get_default_constraints(self) -> List[LabConstraint]:
        """获取默认约束集合"""
        return [
            LabConstraint(
                constraint_id="ppe_goggles",
                constraint_type=ConstraintType.PPE_REQUIRED,
                description="必须佩戴护目镜",
                severity=SeverityLevel.CRITICAL,
                conditions={"required_items": ["goggles"], "zones": ["all"]},
                actions=["alert_immediate", "stop_operation"],
                confidence_threshold=0.7
            ),
            LabConstraint(
                constraint_id="ppe_gloves",
                constraint_type=ConstraintType.PPE_REQUIRED,
                description="必须佩戴手套",
                severity=SeverityLevel.MAJOR,
                conditions={"required_items": ["gloves"], "zones": ["chemical_area"]},
                actions=["alert_warning", "log_violation"],
                confidence_threshold=0.75
            ),
            LabConstraint(
                constraint_id="balance_door",
                constraint_type=ConstraintType.EQUIPMENT_USAGE,
                description="使用分析天平时必须关闭防风门",
                severity=SeverityLevel.MAJOR,
                conditions={"equipment": ["analytical_balance"], "action": "weighing"},
                actions=["alert_warning", "suggest_correction"],
                confidence_threshold=0.8
            ),
            LabConstraint(
                constraint_id="chemical_handling",
                constraint_type=ConstraintType.SAFETY_CONSTRAINT,
                description="处理化学品时必须在通风柜内",
                severity=SeverityLevel.CRITICAL,
                conditions={"materials": ["chemicals"], "location": "fume_hood_required"},
                actions=["alert_immediate", "stop_operation"],
                confidence_threshold=0.75
            ),
            LabConstraint(
                constraint_id="waste_disposal",
                constraint_type=ConstraintType.PROCEDURAL_CONSTRAINT,
                description="废物必须正确分类处理",
                severity=SeverityLevel.MINOR,
                conditions={"action": "waste_disposal", "materials": ["chemical_waste"]},
                actions=["alert_info", "log_suggestion"],
                confidence_threshold=0.8
            )
        ]

class VisualAlertSystem:
    """可视化告警系统"""

    def __init__(self):
        self.color_map = {
            SeverityLevel.CRITICAL: (0, 0, 255),    # 红色
            SeverityLevel.MAJOR: (0, 140, 255),     # 橙色
            SeverityLevel.MINOR: (0, 255, 255),     # 黄色
            SeverityLevel.WARNING: (255, 255, 0)    # 青色
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2

    def draw_detection_boxes(self, frame: np.ndarray, detection_result: DetectionResult,
                           violations: List[ViolationAlert]) -> np.ndarray:
        """绘制检测框和告警信息"""
        visual_frame = frame.copy()

        # 绘制常规检测框
        for detection in detection_result.detections:
            bbox = detection.get("bbox", [])
            label = detection.get("label", "unknown")
            confidence = detection.get("confidence", 0.0)

            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                # 根据置信度选择颜色
                if confidence > 0.8:
                    color = (0, 255, 0)  # 绿色：高置信度
                elif confidence > 0.6:
                    color = (0, 255, 255)  # 黄色：中等置信度
                else:
                    color = (0, 0, 255)  # 红色：低置信度

                cv2.rectangle(visual_frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label}: {confidence:.2f}"
                cv2.putText(visual_frame, label_text, (x1, y1-10),
                           self.font, self.font_scale, color, self.thickness)

        # 绘制违规告警框
        for violation in violations:
            for bbox in violation.bounding_boxes:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    color = self.color_map.get(violation.constraint.severity, (255, 255, 255))

                    # 绘制粗边框
                    cv2.rectangle(visual_frame, (x1, y1), (x2, y2), color, 4)

                    # 绘制告警标签
                    severity_text = violation.constraint.severity.value
                    constraint_text = violation.constraint.description[:20] + "..."
                    alert_text = f"ALERT: {severity_text}"

                    # 绘制标签背景
                    label_size = cv2.getTextSize(alert_text, self.font, self.font_scale, self.thickness)[0]
                    cv2.rectangle(visual_frame, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(visual_frame, alert_text, (x1, y1-10),
                               self.font, self.font_scale, (255, 255, 255), self.thickness)

        # 绘制整体状态信息
        self._draw_status_overlay(visual_frame, detection_result, violations)

        return visual_frame

    def _draw_status_overlay(self, frame: np.ndarray, detection_result: DetectionResult,
                           violations: List[ViolationAlert]):
        """绘制状态覆盖层"""
        height, width = frame.shape[:2]

        # 顶部状态栏
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)

        # 时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, 25),
                   self.font, 0.5, (255, 255, 255), 1)

        # 帧信息
        cv2.putText(frame, f"Frame: {detection_result.frame_id}", (10, 45),
                   self.font, 0.5, (255, 255, 255), 1)

        # 违规计数
        violation_count = len(violations)
        if violation_count > 0:
            color = (0, 0, 255)  # 红色
            text = f"VIOLATIONS: {violation_count}"
        else:
            color = (0, 255, 0)  # 绿色
            text = "STATUS: NORMAL"

        cv2.putText(frame, text, (width - 200, 25), self.font, 0.6, color, 2)

        # 检测统计
        detection_count = len(detection_result.detections)
        cv2.putText(frame, f"Objects: {detection_count}", (width - 200, 45),
                   self.font, 0.5, (255, 255, 255), 1)

class AdaptiveLabMonitor:
    """自适应实验室监控主系统"""

    def __init__(self, laboratory_id: str = "default_lab"):
        self.laboratory_id = laboratory_id
        self.constraint_learner = AdaptiveConstraintLearner()
        self.visual_alert_system = VisualAlertSystem()
        self.current_constraints: List[LabConstraint] = []
        self.violation_history: List[ViolationAlert] = []
        self.detection_engine = None  # 将在后续集成
        self.is_running = False

        # 初始化约束
        self.current_constraints = self.constraint_learner._get_default_constraints()

    async def process_frame(self, frame: np.ndarray, frame_id: int) -> Tuple[np.ndarray, List[ViolationAlert]]:
        """处理单帧图像"""
        timestamp = frame_id / 30.0  # 假设30fps

        # 1. 执行检测（模拟）
        detection_result = self._simulate_detection(frame, frame_id, timestamp)

        # 2. 自适应调整约束
        context = self._build_context(frame, detection_result)
        self.current_constraints = self.constraint_learner.adapt_constraints(self.laboratory_id, context)

        # 3. 检查违规
        violations = self._check_violations(detection_result)

        # 4. 从违规中学习
        for violation in violations:
            self.constraint_learner.learn_from_violation(self.laboratory_id, violation)
            self.violation_history.append(violation)

        # 5. 生成可视化
        visual_frame = self.visual_alert_system.draw_detection_boxes(frame, detection_result, violations)

        return visual_frame, violations

    def _simulate_detection(self, frame: np.ndarray, frame_id: int, timestamp: float) -> DetectionResult:
        """模拟检测结果（实际应用中会调用真实模型）"""
        height, width = frame.shape[:2]

        # 模拟目标检测
        detections = [
            {
                "label": "person",
                "bbox": [width*0.3, height*0.2, width*0.7, height*0.8],
                "confidence": 0.92
            },
            {
                "label": "goggles",
                "bbox": [width*0.35, height*0.2, width*0.45, height*0.25],
                "confidence": 0.88
            },
            {
                "label": "analytical_balance",
                "bbox": [width*0.6, height*0.4, width*0.8, height*0.6],
                "confidence": 0.95
            }
        ]

        # 模拟姿态估计
        poses = [
            {
                "person_id": 0,
                "keypoints": self._generate_keypoints(width, height),
                "action": "weighing_reagent"
            }
        ]

        # 模拟动作识别
        actions = ["weighing_reagent", "handling_equipment"]

        return DetectionResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            poses=poses,
            actions=actions,
            confidence=0.85
        )

    def _generate_keypoints(self, width: int, height: int) -> List[Dict[str, float]]:
        """生成模拟关键点"""
        return [
            {"x": width * 0.5, "y": height * 0.3, "confidence": 0.9},   # 头部
            {"x": width * 0.45, "y": height * 0.5, "confidence": 0.85}, # 左肩
            {"x": width * 0.55, "y": height * 0.5, "confidence": 0.85}, # 右肩
            {"x": width * 0.4, "y": height * 0.7, "confidence": 0.8},   # 左手
            {"x": width * 0.6, "y": height * 0.7, "confidence": 0.8},   # 右手
        ]

    def _build_context(self, frame: np.ndarray, detection_result: DetectionResult) -> Dict[str, Any]:
        """构建当前上下文"""
        return {
            "timestamp": datetime.now().isoformat(),
            "frame_size": frame.shape[:2],
            "detection_count": len(detection_result.detections),
            "actions": detection_result.actions,
            "high_risk_mode": len(detection_result.detections) > 5,  # 高风险模式
            "environment": "laboratory"
        }

    def _check_violations(self, detection_result: DetectionResult) -> List[ViolationAlert]:
        """检查违规行为"""
        violations = []

        for constraint in self.current_constraints:
            if not constraint.enabled:
                continue

            violation = self._evaluate_constraint(constraint, detection_result)
            if violation:
                violations.append(violation)

        return violations

    def _evaluate_constraint(self, constraint: LabConstraint, detection_result: DetectionResult) -> Optional[ViolationAlert]:
        """评估单个约束"""
        # 基于约束类型的不同评估逻辑
        if constraint.constraint_type == ConstraintType.PPE_REQUIRED:
            return self._evaluate_ppe_constraint(constraint, detection_result)
        elif constraint.constraint_type == ConstraintType.EQUIPMENT_USAGE:
            return self._evaluate_equipment_constraint(constraint, detection_result)
        elif constraint.constraint_type == ConstraintType.SAFETY_CONSTRAINT:
            return self._evaluate_safety_constraint(constraint, detection_result)

        return None

    def _evaluate_ppe_constraint(self, constraint: LabConstraint, detection_result: DetectionResult) -> Optional[ViolationAlert]:
        """评估PPE约束"""
        required_items = constraint.conditions.get("required_items", [])

        for item in required_items:
            # 检查是否检测到所需PPE
            found = any(det["label"] == item and det["confidence"] > constraint.confidence_threshold
                       for det in detection_result.detections)

            if not found:
                # 生成违规告警
                return ViolationAlert(
                    alert_id=f"alert_{constraint.constraint_id}_{detection_result.frame_id}",
                    constraint=constraint,
                    detection_result=detection_result,
                    timestamp=datetime.now(),
                    confidence=0.8,
                    bounding_boxes=self._get_relevant_bboxes(detection_result, "person"),
                    description=f"未检测到{constraint.description}",
                    recommendation=f"请立即{constraint.description}",
                    visual_data={
                        "alert_type": "ppe_missing",
                        "missing_item": item,
                        "severity_color": self.visual_alert_system.color_map[constraint.severity]
                    }
                )

        return None

    def _evaluate_equipment_constraint(self, constraint: LabConstraint, detection_result: DetectionResult) -> Optional[ViolationAlert]:
        """评估设备使用约束"""
        equipment = constraint.conditions.get("equipment", [])
        action = constraint.conditions.get("action", "")

        # 检查是否有相关设备和动作
        equipment_found = any(det["label"] in equipment for det in detection_result.detections)
        action_found = action in detection_result.actions

        if equipment_found and action_found:
            # 模拟检查设备状态（实际应用中会使用更复杂的逻辑）
            # 这里模拟检测到天平防风门未关闭的情况
            if "analytical_balance" in equipment and "weighing" in action:
                # 30%概率模拟违规
                if np.random.random() < 0.3:
                    return ViolationAlert(
                        alert_id=f"alert_{constraint.constraint_id}_{detection_result.frame_id}",
                        constraint=constraint,
                        detection_result=detection_result,
                        timestamp=datetime.now(),
                        confidence=0.85,
                        bounding_boxes=self._get_relevant_bboxes(detection_result, "analytical_balance"),
                        description=constraint.description,
                        recommendation="请关闭分析天平防风门后重新称量",
                        visual_data={
                            "alert_type": "equipment_misuse",
                            "equipment_status": "door_open",
                            "severity_color": self.visual_alert_system.color_map[constraint.severity]
                        }
                    )

        return None

    def _evaluate_safety_constraint(self, constraint: LabConstraint, detection_result: DetectionResult) -> Optional[ViolationAlert]:
        """评估安全约束"""
        # 简化的安全约束评估
        materials = constraint.conditions.get("materials", [])
        location = constraint.conditions.get("location", "")

        # 检查是否有危险材料
        dangerous_materials_found = any(
            any(material in det["label"] for material in materials)
            for det in detection_result.detections
        )

        if dangerous_materials_found:
            # 模拟位置检查
            if location == "fume_hood_required" and np.random.random() < 0.2:
                return ViolationAlert(
                    alert_id=f"alert_{constraint.constraint_id}_{detection_result.frame_id}",
                    constraint=constraint,
                    detection_result=detection_result,
                    timestamp=datetime.now(),
                    confidence=0.9,
                    bounding_boxes=self._get_relevant_bboxes(detection_result, "chemical"),
                    description="化学品处理必须在通风柜内进行",
                    recommendation="请将化学品移至通风柜内处理",
                    visual_data={
                        "alert_type": "safety_violation",
                        "location_issue": "outside_fume_hood",
                        "severity_color": self.visual_alert_system.color_map[constraint.severity]
                    }
                )

        return None

    def _get_relevant_bboxes(self, detection_result: DetectionResult, target_label: str) -> List[Dict[str, Any]]:
        """获取相关边界框"""
        bboxes = []
        for det in detection_result.detections:
            if target_label in det["label"] or det["label"] == "person":
                bboxes.append(det["bbox"])
        return bboxes

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_violations = len(self.violation_history)
        severity_stats = {}

        for violation in self.violation_history:
            severity = violation.constraint.severity.value
            severity_stats[severity] = severity_stats.get(severity, 0) + 1

        return {
            "total_violations": total_violations,
            "severity_distribution": severity_stats,
            "active_constraints": len([c for c in self.current_constraints if c.enabled]),
            "laboratory_id": self.laboratory_id,
            "learning_patterns": len(self.constraint_learner.laboratory_profiles.get(self.laboratory_id, {}).get("violation_patterns", []))
        }

# 演示函数
async def demo_adaptive_monitoring():
    """演示自适应监控系统"""
    print("自适应AI实验室监控系统演示")
    print("=" * 60)

    # 创建监控系统
    monitor = AdaptiveLabMonitor("chemistry_lab_001")

    # 创建模拟帧
    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    demo_frame[:] = (50, 50, 50)  # 深灰色背景

    # 模拟实验室环境
    cv2.rectangle(demo_frame, (400, 200), (500, 300), (100, 100, 100), -1)  # 天平
    cv2.rectangle(demo_frame, (200, 100), (350, 400), (80, 80, 80), -1)     # 人员

    print("\n开始实时监控演示...")

    # 处理多帧
    for frame_num in range(5):
        print(f"\n处理帧 {frame_num + 1}:")

        # 处理帧
        visual_frame, violations = await monitor.process_frame(demo_frame, frame_num)

        # 输出违规信息
        if violations:
            for violation in violations:
                print(f"  违规检测: {violation.constraint.description}")
                print(f"    严重等级: {violation.constraint.severity.value}")
                print(f"    置信度: {violation.confidence:.2f}")
                print(f"    建议: {violation.recommendation}")
        else:
            print("  无违规行为")

        # 保存可视化结果
        output_path = f"outputs/demo/adaptive_monitoring_frame_{frame_num + 1}.jpg"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, visual_frame)
        print(f"  可视化结果: {output_path}")

    # 输出统计信息
    stats = monitor.get_statistics()
    print(f"\n监控统计:")
    print(f"  总违规次数: {stats['total_violations']}")
    print(f"  活跃约束数: {stats['active_constraints']}")
    print(f"  学习模式数: {stats['learning_patterns']}")
    print(f"  严重等级分布: {stats['severity_distribution']}")

    print("\n演示完成!")
    print("系统特点:")
    print("1. 自适应约束学习 - 根据实验室具体情况调整约束")
    print("2. 可视化检测框 - 实时显示检测结果和违规告警")
    print("3. 智能报警系统 - 分级告警和具体建议")
    print("4. 持续学习能力 - 从历史数据中优化约束")

if __name__ == "__main__":
    asyncio.run(demo_adaptive_monitoring())
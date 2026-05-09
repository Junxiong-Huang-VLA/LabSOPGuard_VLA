"""
实验室SOP违规检测与报警演示
基于现有系统架构，演示如何实现"称量试剂"步骤的SOP合规监控
"""
from __future__ import annotations

import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 模拟检测结果
class MockDetectionEngine:
    """模拟多层级检测引擎"""

    def __init__(self):
        self.frame_count = 0

    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """模拟单帧检测结果"""
        self.frame_count += 1

        # 模拟YOLO检测结果
        # 在实际应用中，这里会调用真实的YOLO模型
        mock_detections = {
            "frame_id": self.frame_count,
            "timestamp_sec": self.frame_count / 30.0,  # 假设30fps
            "ppe_detection": {
                "goggles": {
                    "detected": True,
                    "confidence": 0.92,
                    "bbox": [120, 45, 180, 75]  # x1, y1, x2, y2
                },
                "gloves": {
                    "detected": False,  # 未检测到手套 - 违规!
                    "confidence": 0.15,
                    "bbox": None
                },
                "lab_coat": {
                    "detected": True,
                    "confidence": 0.88,
                    "bbox": [80, 80, 220, 300]
                }
            },
            "equipment_detection": {
                "analytical_balance": {
                    "detected": True,
                    "confidence": 0.95,
                    "bbox": [250, 100, 400, 250],
                    "door_status": "open"  # 防风门打开 - 违规!
                }
            },
            "pose_estimation": {
                "person_detected": True,
                "keypoints": self._generate_mock_keypoints(),
                "action": "weighing_reagent"  # 当前动作：称量试剂
            }
        }

        return mock_detections

    def _generate_mock_keypoints(self) -> List[Dict[str, float]]:
        """生成模拟的人体关键点"""
        # 模拟17个COCO关键点
        keypoints = [
            {"x": 160, "y": 30, "confidence": 0.95},   # nose
            {"x": 155, "y": 25, "confidence": 0.92},   # left_eye
            {"x": 165, "y": 25, "confidence": 0.93},   # right_eye
            {"x": 150, "y": 28, "confidence": 0.90},   # left_ear
            {"x": 170, "y": 28, "confidence": 0.91},   # right_ear
            {"x": 140, "y": 50, "confidence": 0.89},   # left_shoulder
            {"x": 180, "y": 50, "confidence": 0.88},   # right_shoulder
            {"x": 120, "y": 80, "confidence": 0.85},   # left_elbow
            {"x": 200, "y": 80, "confidence": 0.86},   # right_elbow
            {"x": 110, "y": 120, "confidence": 0.82},  # left_wrist
            {"x": 210, "y": 120, "confidence": 0.83},  # right_wrist
            {"x": 145, "y": 150, "confidence": 0.80},  # left_hip
            {"x": 175, "y": 150, "confidence": 0.81},  # right_hip
            {"x": 140, "y": 200, "confidence": 0.78},  # left_knee
            {"x": 180, "y": 200, "confidence": 0.79},  # right_knee
            {"x": 135, "y": 250, "confidence": 0.75},  # left_ankle
            {"x": 185, "y": 250, "confidence": 0.76},  # right_ankle
        ]
        return keypoints


class WeighingSOPAnalyzer:
    """称量试剂SOP分析器"""

    def __init__(self):
        # 称量试剂的SOP规则
        self.sop_rules = {
            "step_name": "称量试剂",
            "required_ppe": ["护目镜", "乳胶手套"],
            "equipment_rules": {
                "分析天平": {
                    "requirement": "使用时必须关闭防风门",
                    "violation_condition": "door_status == 'open'"
                }
            },
            "severity_mapping": {
                "missing_goggles": "Critical",  # 缺少护目镜 - 严重安全风险
                "missing_gloves": "Major",      # 缺少手套 - 化学品接触风险
                "balance_door_open": "Major",   # 天平防风门未关 - 影响称量精度
                "incorrect_weighing": "Minor"   # 称量操作不规范 - 程序性偏差
            }
        }

    def analyze_violations(self, detection_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析SOP违规行为"""
        violations = []

        # 1. 检查PPE合规性
        ppe_violations = self._check_ppe_compliance(detection_result)
        violations.extend(ppe_violations)

        # 2. 检查设备使用合规性
        equipment_violations = self._check_equipment_compliance(detection_result)
        violations.extend(equipment_violations)

        # 3. 检查操作流程合规性
        procedure_violations = self._check_procedure_compliance(detection_result)
        violations.extend(procedure_violations)

        return violations

    def _check_ppe_compliance(self, detection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查个人防护装备合规性"""
        violations = []
        ppe = detection["ppe_detection"]

        # 检查护目镜
        if not ppe["goggles"]["detected"]:
            violations.append({
                "violation": True,
                "type": "missing_goggles",
                "description": "操作人员未佩戴护目镜，在称量试剂过程中存在化学品飞溅风险",
                "severity": "Critical",
                "sop_ref": "SOP-LAB-COMPLIANCE-001-PPE-001",
                "recommendation": "立即停止操作，佩戴符合标准的护目镜后方可继续",
                "confidence": ppe["goggles"]["confidence"],
                "bbox": ppe["goggles"]["bbox"],
                "timestamp": detection["timestamp_sec"]
            })

        # 检查手套
        if not ppe["gloves"]["detected"]:
            violations.append({
                "violation": True,
                "type": "missing_gloves",
                "description": "操作人员未佩戴乳胶手套，直接接触试剂存在皮肤腐蚀和污染风险",
                "severity": "Major",
                "sop_ref": "SOP-LAB-COMPLIANCE-001-PPE-002",
                "recommendation": "立即佩戴乳胶手套，确保手套完整无破损",
                "confidence": ppe["gloves"]["confidence"],
                "bbox": ppe["gloves"]["bbox"],
                "timestamp": detection["timestamp_sec"]
            })

        return violations

    def _check_equipment_compliance(self, detection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查设备使用合规性"""
        violations = []
        equipment = detection["equipment_detection"]

        # 检查分析天平防风门状态
        if "analytical_balance" in equipment:
            balance = equipment["analytical_balance"]
            if balance["detected"] and balance.get("door_status") == "open":
                violations.append({
                    "violation": True,
                    "type": "balance_door_open",
                    "description": "使用分析天平时防风门未关闭，影响称量精度和实验结果准确性",
                    "severity": "Major",
                    "sop_ref": "SOP-LAB-COMPLIANCE-001-EQUIP-001",
                    "recommendation": "关闭分析天平防风门，等待读数稳定后重新称量",
                    "confidence": balance["confidence"],
                    "bbox": balance["bbox"],
                    "timestamp": detection["timestamp_sec"]
                })

        return violations

    def _check_procedure_compliance(self, detection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查操作流程合规性"""
        violations = []
        pose = detection["pose_estimation"]

        # 检查是否在进行称量操作
        if pose["action"] == "weighing_reagent":
            # 可以添加更多操作规范检查
            # 例如：检查手部位置是否正确、操作姿势是否规范等
            pass

        return violations

    def generate_alert_report(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成告警报告"""
        if not violations:
            return {
                "violation": False,
                "description": "操作符合SOP规范，未检测到违规行为",
                "severity": "None",
                "sop_ref": "SOP-LAB-COMPLIANCE-001",
                "recommendation": "继续按照标准操作流程进行"
            }

        # 按严重等级排序
        severity_order = {"Critical": 0, "Major": 1, "Minor": 2}
        violations.sort(key=lambda x: severity_order.get(x["severity"], 3))

        # 生成主要违规报告（最严重的）
        primary_violation = violations[0]

        return {
            "violation": True,
            "description": primary_violation["description"],
            "severity": primary_violation["severity"],
            "sop_ref": primary_violation["sop_ref"],
            "recommendation": primary_violation["recommendation"],
            "all_violations": violations,
            "total_violations": len(violations),
            "timestamp": datetime.now().isoformat()
        }


def create_demo_visualization(frame: np.ndarray, violations: List[Dict[str, Any]],
                            detection: Dict[str, Any]) -> np.ndarray:
    """创建违规检测可视化"""
    vis_frame = frame.copy()
    height, width = frame.shape[:2]

    # 绘制检测框和标签
    colors = {
        "Critical": (0, 0, 255),    # 红色
        "Major": (0, 140, 255),     # 橙色
        "Minor": (0, 255, 255)      # 黄色
    }

    # 绘制PPE检测结果
    ppe = detection["ppe_detection"]

    # 护目镜检测框
    if ppe["goggles"]["bbox"]:
        x1, y1, x2, y2 = ppe["goggles"]["bbox"]
        color = (0, 255, 0) if ppe["goggles"]["detected"] else (0, 0, 255)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        status = "OK" if ppe["goggles"]["detected"] else "MISSING"
        cv2.putText(vis_frame, f"Goggles {status}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 手套检测状态（在画面左上角显示）
    glove_status = "Worn" if ppe["gloves"]["detected"] else "Missing"
    glove_color = (0, 255, 0) if ppe["gloves"]["detected"] else (0, 0, 255)
    cv2.putText(vis_frame, f"Gloves: {glove_status}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, glove_color, 2)

    # 天平防风门状态
    if "analytical_balance" in detection["equipment_detection"]:
        balance = detection["equipment_detection"]["analytical_balance"]
        if balance["detected"]:
            door_status = "Closed" if balance.get("door_status") == "closed" else "Open"
            door_color = (0, 255, 0) if balance.get("door_status") == "closed" else (0, 0, 255)
            cv2.putText(vis_frame, f"Balance Door: {door_status}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, door_color, 2)

    # 绘制违规告警
    if violations:
        # 添加红色边框表示存在违规
        cv2.rectangle(vis_frame, (0, 0), (width-1, height-1), (0, 0, 255), 5)

        # 显示违规信息
        y_offset = height - 100
        for i, violation in enumerate(violations[:3]):  # 最多显示3条
            severity = violation["severity"]
            color = colors.get(severity, (255, 255, 255))

            # 违规类型和等级
            text = f"Violation: {violation['type']} [{severity}]"
            cv2.putText(vis_frame, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 添加时间戳和帧ID
    cv2.putText(vis_frame, f"Time: {detection['timestamp_sec']:.1f}s",
               (width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis_frame, f"Frame: {detection['frame_id']}",
               (width-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis_frame


def demo_weighing_sop_monitoring():
    """演示称量试剂SOP监控"""
    print("Laboratory SOP Compliance Monitoring System - Weighing Reagent Step Detection Demo")
    print("=" * 80)

    # 初始化检测引擎和分析器
    detection_engine = MockDetectionEngine()
    sop_analyzer = WeighingSOPAnalyzer()

    # 创建模拟视频帧
    demo_frame = np.zeros((400, 500, 3), dtype=np.uint8)
    demo_frame[:] = (50, 50, 50)  # 深灰色背景

    # 添加一些模拟内容
    cv2.rectangle(demo_frame, (250, 100), (400, 250), (100, 100, 100), -1)  # 天平
    cv2.rectangle(demo_frame, (100, 80), (220, 300), (80, 80, 80), -1)     # 人员

    print("\nStarting detection analysis...")

    # 模拟检测过程
    for frame_num in range(3):
        print(f"\nFrame {frame_num + 1}:")

        # 执行检测
        detection_result = detection_engine.detect_frame(demo_frame)

        # 分析违规
        violations = sop_analyzer.analyze_violations(detection_result)

        # 生成告警报告
        alert_report = sop_analyzer.generate_alert_report(violations)

        # 输出结果
        print(json.dumps(alert_report, ensure_ascii=False, indent=2))

        # 创建可视化
        vis_frame = create_demo_visualization(demo_frame, violations, detection_result)

        # 保存可视化结果
        output_path = f"outputs/demo/weighing_violation_frame_{frame_num + 1}.jpg"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, vis_frame)
        print(f"Visualization saved: {output_path}")

        print("-" * 40)

    print("\nDemo completed!")
    print("\nSystem Capability Summary:")
    print("1. PPE Detection: Can identify goggles and gloves wearing status")
    print("2. Equipment Status Detection: Can detect analytical balance door status")
    print("3. SOP Rule Matching: Can judge violation behaviors based on rules")
    print("4. Severity Level Assessment: Can classify violations (Critical/Major/Minor)")
    print("5. Real-time Alert Generation: Can generate structured violation reports")
    print("6. Visualization Annotation: Can annotate violation info on video frames")

    print("\nTechnical Implementation:")
    print("- Layer 1: YOLO26 object detection (goggles, gloves, balance)")
    print("- Layer 2: Pose estimation (operator behavior analysis)")
    print("- Layer 3: Equipment status recognition (balance door status)")
    print("- Layer 4: SOP rule engine (violation judgment and classification)")


if __name__ == "__main__":
    demo_weighing_sop_monitoring()
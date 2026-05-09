"""
VLM视觉语言大模型集成 - 称量试剂SOP违规检测
基于Qwen3-VL的零样本违规识别与结构化输出
"""
from __future__ import annotations

import json
import base64
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio

# 导入之前的模拟检测器
import sys
sys.path.append(str(Path(__file__).parent))
from weighing_sop_demo import MockDetectionEngine

class VLMWeighingAnalyzer:
    """基于VLM的称量试剂SOP分析器"""

    def __init__(self, model_name: str = "Qwen3-VL-8B"):
        self.model_name = model_name
        self.sop_context = self._build_sop_context()

    def _build_sop_context(self) -> str:
        """构建SOP上下文信息"""
        return """
        当前SOP步骤为"称量试剂"，具体要求如下：
        1. 操作人员必须佩戴护目镜和乳胶手套
        2. 使用分析天平时应关闭防风门
        3. 称量前应验证试剂标签
        4. 使用镊子或药匙取用试剂，禁止直接用手接触
        5. 称量后应及时清理天平台面
        """

    def create_analysis_prompt(self) -> str:
        """创建VLM分析提示词"""
        return f"""
分析以下实验室监控画面。{self.sop_context}

请仔细观察画面中的以下要素：
1. 操作人员是否佩戴护目镜和乳胶手套
2. 分析天平的防风门是否关闭
3. 操作人员的手部动作是否符合规范
4. 试剂瓶标签是否清晰可见
5. 天平台面是否整洁

请判断：
1. 操作人员是否存在SOP违规行为？
2. 如有违规，描述具体违规内容、对应SOP条款、严重等级(Critical/Major/Minor)
3. 给出纠正建议

请严格按照以下JSON格式输出：
{{
    "violation": true/false,
    "description": "违规描述",
    "severity": "Critical/Major/Minor/None",
    "sop_ref": "SOP条款引用",
    "recommendation": "纠正建议",
    "detailed_analysis": {{
        "ppe_status": {{
            "goggles": {{"detected": true/false, "confidence": 0.0-1.0}},
            "gloves": {{"detected": true/false, "confidence": 0.0-1.0}}
        }},
        "equipment_status": {{
            "balance_door": "open/closed/unknown",
            "balance_detected": true/false
        }},
        "operator_actions": ["action1", "action2"],
        "environment_check": {{
            "workspace_clean": true/false,
            "labels_visible": true/false
        }}
    }}
}}
"""

    def simulate_vlm_analysis(self, image_path: str) -> Dict[str, Any]:
        """模拟VLM分析过程（实际应用中会调用真实的VLM API）"""
        # 在实际应用中，这里会：
        # 1. 加载图像并编码为base64
        # 2. 调用Qwen3-VL API
        # 3. 解析返回的JSON结果

        # 模拟VLM检测结果
        mock_vlm_result = {
            "violation": True,
            "description": "操作人员未佩戴乳胶手套，且分析天平防风门未关闭",
            "severity": "Major",
            "sop_ref": "SOP-LAB-COMPLIANCE-001-PPE-002, SOP-LAB-COMPLIANCE-001-EQUIP-001",
            "recommendation": "立即佩戴乳胶手套，关闭分析天平防风门，等待读数稳定后重新称量",
            "detailed_analysis": {
                "ppe_status": {
                    "goggles": {"detected": True, "confidence": 0.92},
                    "gloves": {"detected": False, "confidence": 0.15}
                },
                "equipment_status": {
                    "balance_door": "open",
                    "balance_detected": True
                },
                "operator_actions": ["weighing_reagent", "handling_balance"],
                "environment_check": {
                    "workspace_clean": True,
                    "labels_visible": True
                }
            },
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model_name,
            "analysis_confidence": 0.87
        }

        return mock_vlm_result

    def enhance_with_traditional_cv(self, vlm_result: Dict[str, Any],
                                  cv_detections: Dict[str, Any]) -> Dict[str, Any]:
        """结合传统CV检测结果增强VLM分析"""
        enhanced_result = vlm_result.copy()

        # 融合PPE检测结果
        if "ppe_detection" in cv_detections:
            cv_ppe = cv_detections["ppe_detection"]
            vlm_ppe = vlm_result.get("detailed_analysis", {}).get("ppe_status", {})

            # 如果VLM和CV检测结果不一致，以置信度高的为准
            for item in ["goggles", "gloves"]:
                if item in cv_ppe and item in vlm_ppe:
                    cv_conf = cv_ppe[item].get("confidence", 0)
                    vlm_conf = vlm_ppe[item].get("confidence", 0)

                    if cv_conf > vlm_conf:
                        enhanced_result["detailed_analysis"]["ppe_status"][item] = {
                            "detected": cv_ppe[item]["detected"],
                            "confidence": cv_conf,
                            "source": "traditional_cv"
                        }
                    else:
                        enhanced_result["detailed_analysis"]["ppe_status"][item]["source"] = "vlm"

        # 融合设备检测结果
        if "equipment_detection" in cv_detections:
            cv_equipment = cv_detections["equipment_detection"]
            if "analytical_balance" in cv_equipment:
                balance = cv_equipment["analytical_balance"]
                enhanced_result["detailed_analysis"]["equipment_status"].update({
                    "balance_detected": balance["detected"],
                    "balance_confidence": balance["confidence"],
                    "door_status_from_cv": balance.get("door_status", "unknown")
                })

        return enhanced_result


class HybridSOPDetector:
    """混合SOP检测器 - 结合传统CV和VLM"""

    def __init__(self):
        self.vlm_analyzer = VLMWeighingAnalyzer()
        self.cv_detector = MockDetectionEngine()  # 复用之前的模拟检测器

    def analyze_frame(self, frame: np.ndarray, frame_path: Optional[str] = None) -> Dict[str, Any]:
        """分析单帧图像"""
        # 1. 传统CV检测
        cv_detections = self.cv_detector.detect_frame(frame)

        # 2. VLM分析（如果有图像路径）
        if frame_path:
            vlm_result = self.vlm_analyzer.simulate_vlm_analysis(frame_path)
        else:
            vlm_result = self.vlm_analyzer.simulate_vlm_analysis("mock_frame.jpg")

        # 3. 融合结果
        enhanced_result = self.vlm_analyzer.enhance_with_traditional_cv(
            vlm_result, cv_detections
        )

        # 4. 生成最终告警
        final_alert = self._generate_final_alert(enhanced_result)

        return {
            "cv_detections": cv_detections,
            "vlm_analysis": vlm_result,
            "enhanced_analysis": enhanced_result,
            "final_alert": final_alert
        }

    def _generate_final_alert(self, enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终告警信息"""
        # 基于增强后的分析结果生成告警
        ppe_status = enhanced_result.get("detailed_analysis", {}).get("ppe_status", {})
        equipment_status = enhanced_result.get("detailed_analysis", {}).get("equipment_status", {})

        violations = []

        # 检查PPE违规
        if not ppe_status.get("goggles", {}).get("detected", True):
            violations.append({
                "type": "missing_goggles",
                "severity": "Critical",
                "description": "操作人员未佩戴护目镜",
                "confidence": ppe_status.get("goggles", {}).get("confidence", 0)
            })

        if not ppe_status.get("gloves", {}).get("detected", True):
            violations.append({
                "type": "missing_gloves",
                "severity": "Major",
                "description": "操作人员未佩戴乳胶手套",
                "confidence": ppe_status.get("gloves", {}).get("confidence", 0)
            })

        # 检查设备违规
        if equipment_status.get("balance_door") == "open":
            violations.append({
                "type": "balance_door_open",
                "severity": "Major",
                "description": "分析天平防风门未关闭",
                "confidence": equipment_status.get("balance_confidence", 0.95)
            })

        # 生成最终告警
        if violations:
            # 按严重等级排序
            severity_order = {"Critical": 0, "Major": 1, "Minor": 2}
            violations.sort(key=lambda x: severity_order.get(x["severity"], 3))

            primary_violation = violations[0]
            return {
                "violation": True,
                "description": primary_violation["description"],
                "severity": primary_violation["severity"],
                "sop_ref": "SOP-LAB-COMPLIANCE-001",
                "recommendation": enhanced_result.get("recommendation", "请按照SOP规范操作"),
                "all_violations": violations,
                "total_violations": len(violations),
                "analysis_method": "hybrid_cv_vlm",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "violation": False,
                "description": "操作符合SOP规范，未检测到违规行为",
                "severity": "None",
                "sop_ref": "SOP-LAB-COMPLIANCE-001",
                "recommendation": "继续按照标准操作流程进行",
                "analysis_method": "hybrid_cv_vlm",
                "timestamp": datetime.now().isoformat()
            }


def demo_vlm_weighing_analysis():
    """演示VLM称量试剂分析"""
    print("VLM-based Laboratory SOP Compliance Analysis - Weighing Reagent Step")
    print("=" * 80)

    # 初始化混合检测器
    hybrid_detector = HybridSOPDetector()

    # 创建模拟图像
    demo_frame = np.zeros((400, 500, 3), dtype=np.uint8)
    demo_frame[:] = (50, 50, 50)

    # 保存模拟图像
    frame_path = "outputs/demo/mock_weighing_frame.jpg"
    Path(frame_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(frame_path, demo_frame)

    print("\nAnalyzing frame with hybrid CV+VLM approach...")

    # 执行分析
    analysis_result = hybrid_detector.analyze_frame(demo_frame, frame_path)

    # 输出详细结果
    print("\n=== Traditional CV Detection ===")
    print(json.dumps(analysis_result["cv_detections"], ensure_ascii=False, indent=2))

    print("\n=== VLM Analysis ===")
    print(json.dumps(analysis_result["vlm_analysis"], ensure_ascii=False, indent=2))

    print("\n=== Enhanced Analysis (CV+VLM Fusion) ===")
    print(json.dumps(analysis_result["enhanced_analysis"], ensure_ascii=False, indent=2))

    print("\n=== Final Alert ===")
    print(json.dumps(analysis_result["final_alert"], ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("\nSystem Capabilities Demonstrated:")
    print("1. VLM Integration: Zero-shot violation detection using natural language descriptions")
    print("2. Hybrid Approach: Combines traditional CV with VLM for robust detection")
    print("3. Structured Output: Generates standardized JSON alerts for system integration")
    print("4. Confidence Fusion: Intelligently combines detection results from multiple sources")
    print("5. SOP Compliance: Maps violations to specific SOP条款 and severity levels")

    print("\nTechnical Advantages:")
    print("- VLM enables detection of unseen violation types without retraining")
    print("- Natural language SOP descriptions can be easily updated")
    print("- Multi-modal fusion improves detection robustness")
    print("- Structured output enables seamless integration with alert systems")


if __name__ == "__main__":
    demo_vlm_weighing_analysis()
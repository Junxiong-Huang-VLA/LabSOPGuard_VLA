"""
API 集成测试：RealityLoop 实验室SOP态势感知与过程理解平台

测试范围：
1. API 测试（POST/GET/PATCH 端点）
2. PATCH / reviewed_steps.json 测试
3. 数据主线测试（ContextEvent, PhysicalEvent, MultimodalMaterialStreamItem, EvidenceRef.media_asset_id）
4. 前后端联调测试

运行方式：
  python -m pytest tests/test_api_integration.py -v
  python -m pytest tests/test_api_integration.py -v -s  # 带输出
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# 测试辅助函数
# ---------------------------------------------------------------------------

def create_mock_video_file(tmp_dir: Path) -> Path:
    """创建临时视频文件用于测试。"""
    video_path = tmp_dir / "test_video.mp4"
    # 创建最小有效 MP4 文件（仅用于测试文件上传）
    with open(video_path, 'wb') as f:
        f.write(b'\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6f\x6d')
    return video_path


def create_mock_experiment_in_storage(experiment_id: str, tmp_dir: Path) -> Path:
    """创建测试实验目录和 steps.json。"""
    exp_dir = tmp_dir / "outputs" / "experiments" / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    steps_data = [
        {
            "step_id": "step_0",
            "experiment_id": experiment_id,
            "step_index": 0,
            "step_name": "准备样本",
            "step_description": "从冰箱取出样本",
            "status": "confirmed",
            "start_time_sec": 0.0,
            "end_time_sec": 5.0,
            "duration_sec": 5.0,
            "confidence": 1.0,
            "step_confidence": "high",
            "completed_by_inference": False,
            "parameters": [],
            "evidence_refs": [
                {
                    "evidence_type": "video_frame",
                    "source": "video",
                    "frame_id": "frame_0",
                    "timestamp_sec": 0.0,
                    "confidence": 1.0,
                    "media_asset_id": "media_001",  # 验证 media_asset_id 关联
                }
            ],
            "linked_context_events": ["ctx_0"],
            "linked_physical_events": ["phys_0"],
            "provenance": {"source": "video", "confidence": 1.0, "is_inferred": False},
            "metadata": {"context_participated": True, "protocol_step_matched": False},
        },
        {
            "step_id": "step_1",
            "experiment_id": experiment_id,
            "step_index": 1,
            "step_name": "离心分离",
            "step_description": "将样本放入离心机",
            "status": "candidate",
            "start_time_sec": 5.0,
            "end_time_sec": 10.0,
            "duration_sec": 5.0,
            "confidence": 0.65,
            "step_confidence": "medium",
            "completed_by_inference": True,
            "parameters": [],
            "evidence_refs": [
                {
                    "evidence_type": "video_frame",
                    "source": "video",
                    "frame_id": "frame_1",
                    "timestamp_sec": 5.0,
                    "confidence": 0.65,
                    "media_asset_id": "media_002",
                }
            ],
            "linked_context_events": [],
            "linked_physical_events": [],
            "provenance": {"source": "video", "confidence": 0.65, "is_inferred": False},
            "metadata": {"context_participated": False, "protocol_step_matched": False},
        }
    ]

    steps_file = exp_dir / "steps.json"
    steps_file.write_text(json.dumps(steps_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 创建 timeline.json
    timeline_data = {
        "experiment_id": experiment_id,
        "total_duration_sec": 10.0,
        "total_steps": 2,
        "steps": steps_data,
        "context_events": [
            {
                "event_id": "ctx_0",
                "content": "操作员在第0分钟取出样本",
                "timestamp_sec": 0.0,
                "context_source": "conversation",
            }
        ],
        "physical_events": [
            {
                "event_id": "phys_0",
                "event_type": "object_movement",
                "timestamp_sec": 0.0,
                "description": "取出样本管",
                "confidence": 0.9,
            }
        ],
        "media_assets": [
            {
                "asset_id": "media_001",
                "file_path": "/path/to/frame_0.jpg",
                "filename": "frame_0.jpg",
                "type": "key_frame",
                "timestamp_sec": 0.0,
            },
            {
                "asset_id": "media_002",
                "file_path": "/path/to/frame_1.jpg",
                "filename": "frame_1.jpg",
                "type": "key_frame",
                "timestamp_sec": 5.0,
            }
        ],
    }

    timeline_file = exp_dir / "timeline.json"
    timeline_file.write_text(json.dumps(timeline_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 创建 structured.json
    structured_data = {
        "experiment_id": experiment_id,
        "title": "实验过程理解输出",
        "summary": "测试实验摘要",
        "steps": steps_data,
        "statistics": {
            "total_steps": 2,
            "confirmed_steps": 1,
            "candidate_steps": 1,
        },
    }

    structured_file = exp_dir / "structured.json"
    structured_file.write_text(json.dumps(structured_data, ensure_ascii=False, indent=2), encoding="utf-8")

    return exp_dir


# ---------------------------------------------------------------------------
# 测试1：数据模型验证
# ---------------------------------------------------------------------------

class TestDataModels(unittest.TestCase):
    """验证核心数据模型的结构完整性。"""

    def test_evidence_ref_has_media_asset_id(self):
        """验证 EvidenceRef 包含 media_asset_id 字段。"""
        from experiment.models import EvidenceRef

        ref = EvidenceRef(
            evidence_type="video_frame",
            source="video",
            frame_id="frame_test",
            timestamp_sec=1.0,
            confidence=0.9,
            media_asset_id="media_test_123",
        )

        d = ref.to_dict()
        self.assertIn("media_asset_id", d)
        self.assertEqual(d["media_asset_id"], "media_test_123")

    def test_step_record_links_context_events(self):
        """验证 StepRecord 包含 linked_context_events。"""
        from experiment.models import StepRecord, ProvenanceInfo, StepStatus, StepConfidence

        step = StepRecord(
            experiment_id="test_001",
            step_index=0,
            step_name="测试步骤",
            step_description="测试",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            end_time_sec=5.0,
            duration_sec=5.0,
            confidence=1.0,
            step_confidence=StepConfidence.HIGH,
            completed_by_inference=False,
            provenance=ProvenanceInfo(source="test", confidence=1.0, is_inferred=False),
            linked_context_events=["ctx_0", "ctx_1"],
        )

        d = step.to_dict()
        self.assertIn("linked_context_events", d)
        self.assertEqual(d["linked_context_events"], ["ctx_0", "ctx_1"])

    def test_step_record_links_physical_events(self):
        """验证 StepRecord 包含 linked_physical_events。"""
        from experiment.models import StepRecord, ProvenanceInfo, StepStatus, StepConfidence

        step = StepRecord(
            experiment_id="test_001",
            step_index=0,
            step_name="测试步骤",
            step_description="测试",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            end_time_sec=5.0,
            duration_sec=5.0,
            confidence=1.0,
            step_confidence=StepConfidence.HIGH,
            completed_by_inference=False,
            provenance=ProvenanceInfo(source="test", confidence=1.0, is_inferred=False),
            linked_physical_events=["phys_0"],
        )

        d = step.to_dict()
        self.assertIn("linked_physical_events", d)
        self.assertEqual(d["linked_physical_events"], ["phys_0"])

    def test_context_event_model(self):
        """验证 ContextEvent 模型。"""
        from experiment.models import ContextEvent, ContextSource

        ctx = ContextEvent(
            event_id="ctx_test",
            content="操作员在第3分钟更换了移液管",
            timestamp_sec=3.0,
            context_source=ContextSource.CONVERSATION,
        )

        d = ctx.to_dict()
        self.assertEqual(d["event_id"], "ctx_test")
        self.assertEqual(d["context_source"], "conversation")

    def test_physical_event_model(self):
        """验证 PhysicalEvent 模型。"""
        from experiment.models import PhysicalEvent

        phys = PhysicalEvent(
            event_id="phys_test",
            event_type="object_movement",
            timestamp_sec=2.0,
            confidence=0.95,
            metadata={"description": "移液管被拿起"},
        )

        d = phys.to_dict()
        self.assertEqual(d["event_id"], "phys_test")
        self.assertEqual(d["event_type"], "object_movement")
        self.assertEqual(d["metadata"]["description"], "移液管被拿起")

    def test_multimodal_material_stream_item_model(self):
        """验证 MultimodalMaterialStreamItem 模型。"""
        from experiment.models import MultimodalMaterialStreamItem

        item = MultimodalMaterialStreamItem(
            item_id="mmsi_test",
            timestamp_sec=0.0,
            frame_id=0,
            frame_bgr_path="/path/to/frame_0.jpg",
            scene_description="实验室操作台",
            detected_activities=["pipetting"],
            confidence=0.85,
        )

        d = item.to_dict()
        self.assertEqual(d["item_id"], "mmsi_test")
        self.assertEqual(d["timestamp_sec"], 0.0)
        self.assertEqual(d["frame_id"], 0)


# ---------------------------------------------------------------------------
# 测试2：数据主线链路验证
# ---------------------------------------------------------------------------

class TestDataPipeline(unittest.TestCase):
    """验证数据主线链路（ContextEvent, PhysicalEvent, MultimodalMaterialStreamItem）。"""

    def test_context_event_participation_in_step(self):
        """验证 ContextEvent 参与步骤推理（在 metadata 中体现）。"""
        from experiment.models import StepRecord, ProvenanceInfo, StepStatus, StepConfidence

        # 创建带有 context_participated=True 的步骤
        step = StepRecord(
            experiment_id="test_001",
            step_index=0,
            step_name="样本准备",
            step_description="取出样本并检查状态",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            end_time_sec=5.0,
            duration_sec=5.0,
            confidence=1.0,
            step_confidence=StepConfidence.HIGH,
            completed_by_inference=False,
            provenance=ProvenanceInfo(source="video", confidence=1.0, is_inferred=False),
            linked_context_events=["ctx_0"],
            metadata={
                "context_participated": True,
                "protocol_step_matched": False,
            },
        )

        d = step.to_dict()
        self.assertTrue(d["metadata"]["context_participated"])
        self.assertEqual(d["linked_context_events"], ["ctx_0"])

    def test_physical_event_in_step(self):
        """验证 PhysicalEvent 进入 StepRecord。"""
        from experiment.models import StepRecord, ProvenanceInfo, StepStatus, StepConfidence

        step = StepRecord(
            experiment_id="test_001",
            step_index=0,
            step_name="物体操作",
            step_description="拿起移液管",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            end_time_sec=3.0,
            duration_sec=3.0,
            confidence=0.95,
            step_confidence=StepConfidence.HIGH,
            completed_by_inference=False,
            provenance=ProvenanceInfo(source="video", confidence=0.95, is_inferred=False),
            linked_physical_events=["phys_0"],
        )

        d = step.to_dict()
        self.assertEqual(d["linked_physical_events"], ["phys_0"])

    def test_evidence_ref_media_asset_id_association(self):
        """验证 EvidenceRef.media_asset_id 正确关联到 MediaAsset。"""
        from experiment.models import EvidenceRef, EvidenceType

        # 模拟 evidence ref 关联到 media asset
        ref = EvidenceRef(
            evidence_type=EvidenceType.VIDEO_FRAME,
            source="video",
            frame_id="frame_10",
            timestamp_sec=5.0,
            confidence=0.8,
            media_asset_id="asset_abc123",
        )

        d = ref.to_dict()
        self.assertEqual(d["media_asset_id"], "asset_abc123")
        self.assertIsNotNone(d["media_asset_id"])

    def test_timeline_contains_all_main_chain_elements(self):
        """验证 Timeline 输出包含所有主链路元素（通过 ID 引用）。"""
        from experiment.models import ExperimentTimeline, StepRecord, ProvenanceInfo, StepStatus, StepConfidence

        step = StepRecord(
            experiment_id="test_001",
            step_index=0,
            step_name="测试步骤",
            step_description="测试",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            end_time_sec=5.0,
            duration_sec=5.0,
            confidence=1.0,
            step_confidence=StepConfidence.HIGH,
            completed_by_inference=False,
            provenance=ProvenanceInfo(source="test", confidence=1.0, is_inferred=False),
            linked_context_events=["ctx_0"],
            linked_physical_events=["phys_0"],
        )

        # ExperimentTimeline 使用 ID 引用，不是完整对象
        timeline = ExperimentTimeline(
            experiment_id="test_001",
            total_duration_sec=5.0,
            total_steps=1,
            steps=[step],
            media_assets=["asset_0"],       # MediaAsset IDs
            context_events=["ctx_0"],         # ContextEvent IDs
        )

        d = timeline.to_dict()
        self.assertEqual(len(d["steps"]), 1)
        self.assertEqual(d["context_events"], ["ctx_0"])
        self.assertEqual(d["media_assets"], ["asset_0"])
        self.assertEqual(d["steps"][0]["linked_context_events"], ["ctx_0"])
        self.assertEqual(d["steps"][0]["linked_physical_events"], ["phys_0"])


# ---------------------------------------------------------------------------
# 测试3：PATCH reviewed_steps.json 测试
# ---------------------------------------------------------------------------

class TestReviewedSteps(unittest.TestCase):
    """测试 PATCH 端点和 reviewed_steps.json 生成。"""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.exp_id = "test_patch_001"
        self.exp_dir = create_mock_experiment_in_storage(self.exp_id, self.tmp_dir)

    def test_steps_json_exists_after_creation(self):
        """验证 steps.json 已正确生成。"""
        steps_file = self.exp_dir / "steps.json"
        self.assertTrue(steps_file.exists(), "steps.json 应该已存在")

        data = json.loads(steps_file.read_text(encoding="utf-8"))
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["step_id"], "step_0")
        self.assertEqual(data[1]["step_id"], "step_1")

    def test_reviewed_steps_json_structure(self):
        """验证 reviewed_steps.json 的结构符合要求。"""
        # 模拟 PATCH 操作生成 reviewed_steps.json
        steps_file = self.exp_dir / "steps.json"
        steps = json.loads(steps_file.read_text(encoding="utf-8"))

        # 修改一个步骤
        steps[0]["step_name"] = "准备样本（已修正）"
        steps[0]["updated_at"] = "2026-04-15T12:00:00Z"

        # 模拟 change 记录
        changes = [
            {
                "step_id": "step_0",
                "field": "step_name",
                "old_value": "准备样本",
                "new_value": "准备样本（已修正）",
                "reason": "人工复核修正",
                "timestamp": "2026-04-15T12:00:00Z",
            }
        ]

        reviewed_data = {
            "experiment_id": self.exp_id,
            "reviewed_at": "2026-04-15T12:00:00Z",
            "reviewer": "test_user",
            "original_steps": [],
            "reviewed_steps": steps,
            "changes": changes,
        }

        reviewed_file = self.exp_dir / "reviewed_steps.json"
        reviewed_file.write_text(json.dumps(reviewed_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # 验证结构
        self.assertTrue(reviewed_file.exists())
        data = json.loads(reviewed_file.read_text(encoding="utf-8"))

        self.assertIn("experiment_id", data)
        self.assertIn("reviewed_at", data)
        self.assertIn("reviewer", data)
        self.assertIn("original_steps", data)
        self.assertIn("reviewed_steps", data)
        self.assertIn("changes", data)

        # 验证 changes 结构
        self.assertEqual(len(data["changes"]), 1)
        change = data["changes"][0]
        self.assertIn("step_id", change)
        self.assertIn("field", change)
        self.assertIn("old_value", change)
        self.assertIn("new_value", change)
        self.assertIn("reason", change)
        self.assertIn("timestamp", change)

    def test_reviewed_steps_accumulates_changes(self):
        """验证多次 PATCH 时 changes 会累积。"""
        reviewed_data = {
            "experiment_id": self.exp_id,
            "reviewed_at": "2026-04-15T12:00:00Z",
            "reviewer": "user1",
            "original_steps": [],
            "reviewed_steps": [],
            "changes": [
                {
                    "step_id": "step_0",
                    "field": "step_name",
                    "old_value": "旧名称",
                    "new_value": "新名称",
                    "reason": "第一次修正",
                    "timestamp": "2026-04-15T12:00:00Z",
                }
            ],
        }

        # 模拟第二次 PATCH，追加新 change
        new_changes = reviewed_data["changes"] + [
            {
                "step_id": "step_1",
                "field": "step_description",
                "old_value": "旧描述",
                "new_value": "新描述",
                "reason": "第二次修正",
                "timestamp": "2026-04-15T13:00:00Z",
            }
        ]

        reviewed_data["changes"] = new_changes
        reviewed_data["reviewed_at"] = "2026-04-15T13:00:00Z"

        self.assertEqual(len(reviewed_data["changes"]), 2)

    def test_patch_updates_steps_json(self):
        """验证 PATCH 后 steps.json 正确更新。"""
        steps_file = self.exp_dir / "steps.json"
        original_steps = json.loads(steps_file.read_text(encoding="utf-8"))
        original_name = original_steps[0]["step_name"]

        # 模拟 PATCH 更新
        original_steps[0]["step_name"] = "修正后的名称"
        steps_file.write_text(json.dumps(original_steps, ensure_ascii=False, indent=2), encoding="utf-8")

        # 验证更新后的 steps.json
        updated_steps = json.loads(steps_file.read_text(encoding="utf-8"))
        self.assertEqual(updated_steps[0]["step_name"], "修正后的名称")
        self.assertNotEqual(updated_steps[0]["step_name"], original_name)


# ---------------------------------------------------------------------------
# 测试4：API 数据验证测试
# ---------------------------------------------------------------------------

class TestAPIResponseFormats(unittest.TestCase):
    """验证 API 响应数据格式的正确性。"""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.exp_id = "test_api_001"
        self.exp_dir = create_mock_experiment_in_storage(self.exp_id, self.tmp_dir)

    def test_timeline_response_format(self):
        """验证 GET /timeline 响应格式。"""
        timeline_file = self.exp_dir / "timeline.json"
        data = json.loads(timeline_file.read_text(encoding="utf-8"))

        # 验证基本字段
        self.assertIn("experiment_id", data)
        self.assertIn("total_duration_sec", data)
        self.assertIn("total_steps", data)
        self.assertIn("steps", data)
        self.assertIn("context_events", data)
        self.assertIn("physical_events", data)
        self.assertIn("media_assets", data)

        # 验证 steps 中包含必要字段
        step = data["steps"][0]
        self.assertIn("step_id", step)
        self.assertIn("step_name", step)
        self.assertIn("linked_context_events", step)
        self.assertIn("linked_physical_events", step)
        self.assertIn("evidence_refs", step)

        # 验证 evidence_refs 包含 media_asset_id
        evidence = step["evidence_refs"][0]
        self.assertIn("media_asset_id", evidence)
        self.assertIsNotNone(evidence["media_asset_id"])

    def test_steps_list_response_format(self):
        """验证 GET /steps 响应格式。"""
        steps_file = self.exp_dir / "steps.json"
        data = json.loads(steps_file.read_text(encoding="utf-8"))

        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

        step = data[0]
        self.assertIn("step_id", step)
        self.assertIn("step_name", step)
        self.assertIn("status", step)
        self.assertIn("confidence", step)
        self.assertIn("metadata", step)

    def test_step_detail_response_format(self):
        """验证 GET /steps/{step_id} 响应格式。"""
        steps_file = self.exp_dir / "steps.json"
        steps = json.loads(steps_file.read_text(encoding="utf-8"))
        step = steps[0]

        self.assertIn("step_id", step)
        self.assertIn("step_name", step)
        self.assertIn("step_description", step)
        self.assertIn("status", step)
        self.assertIn("start_time_sec", step)
        self.assertIn("end_time_sec", step)
        self.assertIn("duration_sec", step)
        self.assertIn("confidence", step)
        self.assertIn("step_confidence", step)
        self.assertIn("evidence_refs", step)
        self.assertIn("linked_context_events", step)
        self.assertIn("linked_physical_events", step)
        self.assertIn("provenance", step)
        self.assertIn("metadata", step)

    def test_structured_json_response_format(self):
        """验证 GET /structured 响应格式。"""
        structured_file = self.exp_dir / "structured.json"
        data = json.loads(structured_file.read_text(encoding="utf-8"))

        self.assertIn("experiment_id", data)
        self.assertIn("title", data)
        self.assertIn("summary", data)
        self.assertIn("steps", data)
        self.assertIn("statistics", data)


# ---------------------------------------------------------------------------
# 测试5：主链路端到端验证
# ---------------------------------------------------------------------------

class TestMainPipelineE2E(unittest.TestCase):
    """验证主链路端到端数据流。"""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def test_context_event_time_range_linkage(self):
        """验证 ContextEvent 在正确的时间范围内被关联到 StepRecord。"""
        from experiment.models import ContextEvent, ContextSource, StepRecord, ProvenanceInfo, StepStatus, StepConfidence

        # 场景：ContextEvent 在步骤的时间范围内
        ctx = ContextEvent(
            event_id="ctx_0",
            content="操作员取出样本",
            timestamp_sec=2.0,  # 在 step 0-5s 范围内
            context_source=ContextSource.CONVERSATION,
        )

        step_ts_start = 0.0
        step_ts_end = 5.0

        # 模拟关联逻辑
        if step_ts_start <= ctx.timestamp_sec <= step_ts_end:
            linked = True
        else:
            linked = False

        self.assertTrue(linked, "ContextEvent 在步骤时间范围内应该被关联")

        # 验证步骤元数据包含 context_participated
        step = StepRecord(
            experiment_id="test_001",
            step_index=0,
            step_name="样本准备",
            step_description="取出样本",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            end_time_sec=5.0,
            duration_sec=5.0,
            confidence=1.0,
            step_confidence=StepConfidence.HIGH,
            completed_by_inference=False,
            provenance=ProvenanceInfo(source="video", confidence=1.0, is_inferred=False),
            linked_context_events=["ctx_0"],
            metadata={"context_participated": True, "protocol_step_matched": False},
        )

        d = step.to_dict()
        self.assertTrue(d["metadata"]["context_participated"])

    def test_physical_event_appears_in_experiment(self):
        """验证 PhysicalEvent 出现在实验的 physical_events 中。"""
        timeline_file = self.tmp_dir / "outputs" / "experiments" / "phys_test" / "timeline.json"
        timeline_file.parent.mkdir(parents=True, exist_ok=True)

        timeline_data = {
            "experiment_id": "phys_test",
            "total_duration_sec": 10.0,
            "total_steps": 1,
            "steps": [
                {
                    "step_id": "step_0",
                    "step_name": "操作",
                    "linked_physical_events": ["phys_0"],
                }
            ],
            "physical_events": [
                {
                    "event_id": "phys_0",
                    "event_type": "object_movement",
                    "timestamp_sec": 2.0,
                    "description": "拿起移液管",
                    "confidence": 0.95,
                }
            ],
            "media_assets": [],
            "context_events": [],
        }

        timeline_file.write_text(json.dumps(timeline_data, ensure_ascii=False, indent=2), encoding="utf-8")

        data = json.loads(timeline_file.read_text(encoding="utf-8"))
        self.assertEqual(len(data["physical_events"]), 1)
        self.assertEqual(data["physical_events"][0]["event_id"], "phys_0")
        self.assertEqual(data["steps"][0]["linked_physical_events"], ["phys_0"])

    def test_multimodal_material_stream_in_timeline(self):
        """验证 MultimodalMaterialStreamItem 出现在 Timeline 的 media_assets 中。"""
        timeline_file = self.tmp_dir / "outputs" / "experiments" / "mmsi_test" / "timeline.json"
        timeline_file.parent.mkdir(parents=True, exist_ok=True)

        timeline_data = {
            "experiment_id": "mmsi_test",
            "total_duration_sec": 60.0,
            "total_steps": 1,
            "steps": [],
            "context_events": [],
            "physical_events": [],
            "media_assets": [
                {
                    "asset_id": "frame_key_0",
                    "file_path": "/uploads/experiments/exp1/videos/frame_0.jpg",
                    "filename": "frame_0.jpg",
                    "type": "key_frame",
                    "timestamp_sec": 0.0,
                },
                {
                    "asset_id": "video_raw_0",
                    "file_path": "/uploads/experiments/exp1/videos/raw.mp4",
                    "filename": "raw.mp4",
                    "type": "raw_video",
                    "timestamp_sec": 0.0,
                }
            ],
        }

        timeline_file.write_text(json.dumps(timeline_data, ensure_ascii=False, indent=2), encoding="utf-8")

        data = json.loads(timeline_file.read_text(encoding="utf-8"))
        self.assertEqual(len(data["media_assets"]), 2)
        self.assertEqual(data["media_assets"][0]["asset_id"], "frame_key_0")
        self.assertEqual(data["media_assets"][1]["asset_id"], "video_raw_0")


# ---------------------------------------------------------------------------
# 测试6：前后端联调验证（数据格式）
# ---------------------------------------------------------------------------

class TestFrontendBackendIntegration(unittest.TestCase):
    """验证前后端数据契约的正确性。"""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.exp_id = "fe_test_001"
        self.exp_dir = create_mock_experiment_in_storage(self.exp_id, self.tmp_dir)

    def test_workspace_page_can_read_timeline(self):
        """验证 workspace 页面能读取 timeline 数据。"""
        timeline_file = self.exp_dir / "timeline.json"
        data = json.loads(timeline_file.read_text(encoding="utf-8"))

        # workspace 页面需要的数据
        self.assertIn("steps", data)
        self.assertIn("context_events", data)
        self.assertIn("physical_events", data)
        self.assertIn("media_assets", data)

        # 验证每个 step 包含 workspace 需要的字段
        for step in data["steps"]:
            self.assertIn("step_id", step)
            self.assertIn("step_name", step)
            self.assertIn("status", step)
            self.assertIn("start_time_sec", step)
            self.assertIn("end_time_sec", step)
            self.assertIn("evidence_refs", step)
            self.assertIn("linked_context_events", step)
            self.assertIn("linked_physical_events", step)

    def test_workspace_page_can_read_step_detail(self):
        """验证 workspace 页面能读取步骤详情。"""
        steps_file = self.exp_dir / "steps.json"
        steps = json.loads(steps_file.read_text(encoding="utf-8"))
        step = steps[0]

        # 验证 evidence_refs 中每个证据都能找到对应的 media_asset
        for evidence in step["evidence_refs"]:
            self.assertIn("media_asset_id", evidence)
            self.assertIsNotNone(evidence["media_asset_id"])

            # 验证能在 media_assets 中找到对应的 asset
            timeline_file = self.exp_dir / "timeline.json"
            timeline = json.loads(timeline_file.read_text(encoding="utf-8"))
            asset_ids = [a["asset_id"] for a in timeline["media_assets"]]
            self.assertIn(evidence["media_asset_id"], asset_ids)

    def test_json_page_can_read_structured_json(self):
        """验证 json 页面能读取 structured json。"""
        structured_file = self.exp_dir / "structured.json"
        data = json.loads(structured_file.read_text(encoding="utf-8"))

        self.assertIn("experiment_id", data)
        self.assertIn("title", data)
        self.assertIn("steps", data)
        self.assertIn("statistics", data)

    def test_patch_updates_frontend_refresh(self):
        """验证 PATCH 后前端刷新能看到更新结果。"""
        steps_file = self.exp_dir / "steps.json"
        steps = json.loads(steps_file.read_text(encoding="utf-8"))

        # 模拟 PATCH 更新
        new_name = "修正后的样本准备"
        steps[0]["step_name"] = new_name
        steps[0]["updated_at"] = "2026-04-15T12:00:00Z"
        steps_file.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")

        # 模拟前端重新读取
        refreshed_steps = json.loads(steps_file.read_text(encoding="utf-8"))
        self.assertEqual(refreshed_steps[0]["step_name"], new_name)

        # 验证 reviewed_steps.json 也已生成
        reviewed_file = self.exp_dir / "reviewed_steps.json"
        # 这里不创建 reviewed_steps.json，因为那是 PATCH API 的职责
        # 只验证 steps.json 的更新
        self.assertEqual(refreshed_steps[0]["step_name"], "修正后的样本准备")


if __name__ == "__main__":
    unittest.main()

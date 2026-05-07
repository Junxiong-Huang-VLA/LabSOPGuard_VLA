"""
端到端集成测试：实验过程理解完整链路

测试流程：
1. 创建实验
2. 上传视频/上下文/protocol
3. 处理实验（生成 StepRecords + Timeline）
4. 验证输出结构

运行方式：
  python -m pytest tests/test_experiment_e2e.py -v
  python -m pytest tests/test_experiment_e2e.py::test_full_pipeline_with_vlm -v -s  # 带 VLM
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestExperimentModels(unittest.TestCase):
    """测试核心数据模型。"""

    def test_step_record_creation_confirmed(self):
        """测试创建确认步骤。"""
        from experiment.models import (
            StepRecord, StepStatus, StepConfidence, ProvenanceInfo, EvidenceRef, EvidenceType
        )

        step = StepRecord(
            experiment_id="test_exp_001",
            step_index=0,
            step_name="准备样本",
            step_description="从冰箱取出样本",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            end_time_sec=5.0,
            confidence=1.0,
            step_confidence=StepConfidence.HIGH,
            completed_by_inference=False,
            provenance=ProvenanceInfo(
                source="video",
                confidence=1.0,
                is_inferred=False,
            ),
        )

        d = step.to_dict()
        self.assertEqual(d["experiment_id"], "test_exp_001")
        self.assertEqual(d["status"], "confirmed")
        self.assertEqual(d["completed_by_inference"], False)
        self.assertEqual(d["provenance"]["is_inferred"], False)
        self.assertEqual(d["confidence"], 1.0)

    def test_step_record_creation_inferred(self):
        """测试创建推断步骤。"""
        from experiment.models import (
            StepRecord, StepStatus, ProvenanceInfo, make_inferred_step
        )

        # 使用工厂函数
        step = make_inferred_step(
            experiment_id="test_exp_001",
            step_index=1,
            step_name="离心分离",
            start_time_sec=5.0,
            end_time_sec=10.0,
            confidence=0.65,
            inference_method="qwen_vl_temporal_reasoning",
            inference_model="qwen-vl-max",
            description="根据动作推断的离心操作",
            evidence_notes="基于移液动作和离心机声音的推断",
        )

        d = step.to_dict()
        self.assertEqual(d["status"], "inferred")  # 0.65 < 0.7 → inferred
        self.assertEqual(d["completed_by_inference"], True)
        self.assertEqual(d["provenance"]["is_inferred"], True)
        self.assertEqual(d["inference_method"], "qwen_vl_temporal_reasoning")
        self.assertEqual(d["inference_model"], "qwen-vl-max")
        self.assertEqual(d["confidence"], 0.65)

    def test_timeline_computation(self):
        """测试时间线统计计算。"""
        from experiment.models import (
            ExperimentTimeline, StepRecord, StepStatus, ProvenanceInfo, _uuid
        )

        steps = []
        for i, (name, status, conf, start, end) in enumerate([
            ("准备", "confirmed", 1.0, 0.0, 2.0),
            ("混合", "inferred", 0.6, 2.0, 5.0),
            ("离心", "candidate", 0.75, 5.0, 8.0),
            ("收集", "confirmed", 1.0, 8.0, 10.0),
        ]):
            steps.append(StepRecord(
                experiment_id="test",
                step_index=i,
                step_name=name,
                status=StepStatus(status),
                start_time_sec=start,
                end_time_sec=end,
                confidence=conf,
                completed_by_inference=(status == "inferred"),
                provenance=ProvenanceInfo(source="video" if status == "confirmed" else "system", confidence=conf, is_inferred=(status != "confirmed")),
            ))

        timeline = ExperimentTimeline(
            experiment_id="test",
            title="测试实验",
            steps=steps,
            video_duration_sec=10.0,
        )
        timeline.compute_stats()

        self.assertEqual(timeline.total_steps, 4)
        self.assertEqual(timeline.confirmed_steps, 2)
        self.assertEqual(timeline.inferred_steps, 1)
        self.assertEqual(timeline.candidate_steps, 1)
        self.assertEqual(timeline.inference_count, 1)
        self.assertEqual(timeline.avg_confidence, round((1.0 + 0.6 + 0.75 + 1.0) / 4, 4))
        self.assertEqual(timeline.total_duration_sec, 10.0)
        self.assertEqual(timeline.video_coverage_ratio, 1.0)

    def test_experiment_sync_stats(self):
        """测试实验统计同步。"""
        from experiment.models import (
            Experiment, ExperimentStatus, ExperimentTimeline, StepRecord, StepStatus, ProvenanceInfo
        )

        step = StepRecord(
            experiment_id="test",
            step_index=0,
            step_name="测试",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            confidence=0.9,
            completed_by_inference=False,
            evidence_refs=[],
            provenance=ProvenanceInfo(source="video", confidence=0.9, is_inferred=False),
        )

        timeline = ExperimentTimeline(
            experiment_id="test",
            title="测试",
            steps=[step],
        )
        timeline.compute_stats()

        exp = Experiment(
            experiment_id="test",
            title="测试实验",
            status=ExperimentStatus.COMPLETED,
            timeline=timeline,
        )
        exp.sync_stats()

        self.assertEqual(exp.total_steps, 1)
        self.assertEqual(exp.inferred_steps, 0)
        self.assertEqual(exp.avg_confidence, 0.9)
        self.assertEqual(exp.evidence_count, 0)


class TestExperimentService(unittest.TestCase):
    """测试实验服务。"""

    def setUp(self):
        """创建临时视频文件用于测试。"""
        import numpy as np
        import cv2

        self.temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(self.temp_dir, "test_video.mp4")

        # 创建一个简单的测试视频（5秒，15fps，100帧）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 15.0, (320, 240))
        for i in range(75):  # 5秒 * 15fps = 75帧
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        self.video_path = video_path

    def tearDown(self):
        """清理临时文件。"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_service_initialization(self):
        """测试服务初始化。"""
        from experiment.service import ExperimentService

        service = ExperimentService(
            vlm_api_key="fake_key",
            vlm_base_url="http://localhost:8001",
            vlm_model="qwen-vl-max",
        )

        self.assertIsNone(service._experiment)
        self.assertIsNone(service._timeline)
        self.assertEqual(service._frame_extractor.max_frames, 30)
        self.assertEqual(service._frame_extractor.sample_interval_sec, 2.0)

    def test_service_set_video(self):
        """测试设置视频。"""
        from experiment.service import ExperimentService

        service = ExperimentService()
        service.set_video(self.video_path)
        service.set_context("操作员做了样本准备")
        service.set_protocol("1. 准备 2. 离心")

        self.assertEqual(service._video_path, self.video_path)
        self.assertEqual(service._context_text, "操作员做了样本准备")
        self.assertEqual(service._protocol_text, "1. 准备 2. 离心")

    def test_service_process_fallback(self):
        """测试处理流程（无 VLM，fallback 模式）。"""
        from experiment.service import ExperimentService

        service = ExperimentService()  # 无 API key，使用 fallback
        service.set_video(self.video_path)
        service.set_context("操作员做了离心")
        service.set_protocol("1. 离心 2. 收集")

        result = service.process(
            experiment_id="test_exp_001",
            experiment_title="测试实验",
        )

        # 验证输出
        self.assertIn("experiment", result)
        self.assertIn("timeline", result)
        self.assertIn("steps", result)

        exp = result["experiment"]
        timeline = result["timeline"]
        steps = result["steps"]

        self.assertEqual(exp.experiment_id, "test_exp_001")
        self.assertEqual(exp.title, "测试实验")
        self.assertEqual(len(exp.video_assets), 1)
        self.assertEqual(exp.processing_stage.value, "output_generation")

        self.assertIsNotNone(timeline)
        self.assertEqual(timeline.experiment_id, "test_exp_001")
        self.assertEqual(len(timeline.steps), len(steps))
        self.assertGreater(timeline.total_steps, 0)

        # 验证步骤
        self.assertGreater(len(steps), 0)
        for step in steps:
            self.assertEqual(step.experiment_id, "test_exp_001")
            self.assertIsNotNone(step.step_name)
            self.assertIsNotNone(step.start_time_sec)
            # 所有 fallback 步骤都是 inferred
            self.assertTrue(step.completed_by_inference or step.confidence < 1.0)

        print(f"[Test] Processed with {len(steps)} steps, avg_conf={timeline.avg_confidence:.3f}")

    def test_service_save_outputs(self):
        """测试保存输出。"""
        from experiment.service import ExperimentService

        service = ExperimentService()
        service.set_video(self.video_path)
        service.process(experiment_id="test_exp_002", experiment_title="输出测试")

        output_dir = os.path.join(self.temp_dir, "outputs")
        paths = service.save_outputs(output_dir=output_dir)

        self.assertIn("experiment", paths)
        self.assertIn("timeline", paths)
        self.assertIn("steps", paths)

        # 验证文件存在
        for key, path in paths.items():
            self.assertTrue(os.path.exists(path), f"{key} file not found: {path}")

        # 验证 JSON 格式
        with open(paths["experiment"], "r", encoding="utf-8") as f:
            exp_data = json.load(f)
            self.assertEqual(exp_data["experiment_id"], "test_exp_002")

        with open(paths["timeline"], "r", encoding="utf-8") as f:
            timeline_data = json.load(f)
            self.assertIn("steps", timeline_data)

        print(f"[Test] Saved outputs to {output_dir}")


class TestAPIModels(unittest.TestCase):
    """测试 API 数据模型（Schema 验证）。"""

    def test_step_record_schema(self):
        """测试 StepRecord JSON Schema。"""
        from experiment.models import (
            get_step_record_schema, StepRecord, StepStatus, ProvenanceInfo
        )
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")

        schema = get_step_record_schema()

        # 创建测试数据
        step = StepRecord(
            experiment_id="test",
            step_index=0,
            step_name="测试步骤",
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            confidence=1.0,
            completed_by_inference=False,
            provenance=ProvenanceInfo(source="video", confidence=1.0, is_inferred=False),
        )

        data = step.to_dict()

        # 验证
        jsonschema.validate(data, schema)
        print("[Test] StepRecord schema validated")


class TestIntegrationDemo(unittest.TestCase):
    """演示集成测试（需要真实视频）。"""

    @unittest.skipUnless(
        os.environ.get("DASHSCOPE_API_KEY"),
        "需要 DASHSCOPE_API_KEY 环境变量"
    )
    def test_full_pipeline_with_vlm(self):
        """完整流程测试（带 VLM）。"""
        from experiment.service import ExperimentService

        # 这里需要真实的视频文件路径
        # 如果有真实视频，可以在这里测试
        print("[Test] Skipping - need real video file")

        # 示例：
        # service = ExperimentService(
        #     vlm_api_key=os.environ["DASHSCOPE_API_KEY"],
        #     vlm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # )
        # service.set_video("/path/to/real/video.mp4")
        # service.set_context("实验员进行了蛋白质纯化")
        # service.set_protocol("1. 离心 2. 上清转移 3. 加标")
        # result = service.process()
        # print(result["timeline"].step_summary())


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
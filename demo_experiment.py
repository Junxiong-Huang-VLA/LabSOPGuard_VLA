#!/usr/bin/env python3
"""
实验过程理解与步骤推理 - 完整 Demo 脚本

本脚本演示完整的主链路：
视频 + 对话上下文 + Protocol → StepRecords + ExperimentTimeline

前置要求：
  pip install openai  # DashScope 使用 OpenAI 兼容接口

环境变量：
  DASHSCOPE_API_KEY    - 阿里云 DashScope API Key
  DASHSCOPE_BASE_URL   - API Base URL（默认: https://dashscope.aliyuncs.com/compatible-mode/v1）

运行方式：
  # 方式1: 使用 DashScope Qwen-VL（需要 API Key）
  export DASHSCOPE_API_KEY=your_key
  export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  python demo_experiment.py

  # 方式2: Fallback 模式（无需 API Key）
  python demo_experiment.py --no-vlm

  # 方式3: 指定视频路径
  python demo_experiment.py --video /path/to/video.mp4

输出：
  outputs/experiments/{experiment_id}/
    ├── experiment.json    # 完整实验记录
    ├── timeline.json      # 时间线（含所有步骤）
    └── steps.json         # 步骤列表
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).resolve().parents[0]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def create_sample_video(output_path: str, duration_sec: int = 10) -> str:
    """
    创建一个简单的测试视频。
    视频内容：随机噪声 + 模拟实验室操作场景文字。
    """
    import cv2
    import numpy as np

    width, height = 640, 480
    fps = 15
    frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    for i in range(frames):
        t = i / fps
        # 创建带文字的帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 背景色（浅灰色）
        frame[:] = (30, 30, 30)

        # 模拟实验室场景文字
        if t < 2:
            label = "准备样本"
            color = (100, 200, 100)
        elif t < 5:
            label = "加入缓冲液"
            color = (200, 200, 100)
        elif t < 8:
            label = "离心分离"
            color = (100, 200, 200)
        else:
            label = "收集上清"
            color = (200, 100, 200)

        cv2.putText(frame, label, (width // 2 - 100, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        cv2.putText(frame, f"t={t:.1f}s frame={i}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        out.write(frame)

    out.release()
    print(f"[Demo] Created sample video: {output_path} ({frames} frames, {duration_sec}s)")
    return output_path


def run_demo(
    video_path: str,
    context_text: str,
    protocol_text: str,
    use_vlm: bool = True,
    max_frames: int = 30,
) -> dict:
    """
    运行完整 Demo。

    Returns:
        {"experiment": Experiment, "timeline": ExperimentTimeline}
    """
    from experiment.service import ExperimentService

    print("\n" + "=" * 60)
    print("实验过程理解 Demo")
    print("=" * 60)

    # 配置 VLM
    vlm_api_key = os.environ.get("DASHSCOPE_API_KEY") if use_vlm else None
    vlm_base_url = os.environ.get("DASHSCOPE_BASE_URL")

    print(f"\n[1] 配置信息:")
    print(f"    视频: {video_path}")
    print(f"    VLM: {'DashScope Qwen-VL' if vlm_api_key else 'Fallback (无 API Key)'}")
    print(f"    最大帧数: {max_frames}")

    print(f"\n[2] 对话上下文:")
    print(f"    {context_text[:80]}...")

    print(f"\n[3] Protocol:")
    for line in protocol_text.strip().split('\n'):
        if line.strip():
            print(f"    {line}")

    # 创建服务
    print("\n[4] 初始化实验服务...")
    service = ExperimentService(
        vlm_api_key=vlm_api_key,
        vlm_base_url=vlm_base_url,
        vlm_model="qwen-vl-max",
        frame_sample_interval=2.0,
        max_frames=max_frames,
    )

    # 设置输入
    print("[5] 设置输入数据...")
    service.set_video(video_path)
    service.set_context(context_text)
    service.set_protocol(protocol_text)

    # 处理
    print("[6] 开始处理...\n")
    start = time.time()
    result = service.process(
        experiment_id="demo_exp_001",
        experiment_title="蛋白质纯化实验 Demo",
    )
    elapsed = time.time() - start

    experiment = result["experiment"]
    timeline = result["timeline"]
    steps = result["steps"]

    # 输出统计
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  视频: {experiment.video_assets[0].filename if experiment.video_assets else 'N/A'}")
    print(f"  模型: {', '.join(experiment.models_used)}")
    print(f"\n  统计:")
    print(f"    总步骤: {timeline.total_steps}")
    print(f"    已确认: {timeline.confirmed_steps}")
    print(f"    候选:   {timeline.candidate_steps}")
    print(f"    已推断: {timeline.inferred_steps}")
    print(f"    平均置信度: {timeline.avg_confidence:.4f}")
    print(f"    视频覆盖: {timeline.video_coverage_ratio:.1%}")
    print(f"    处理阶段: {experiment.processing_stage.value}")

    # 保存输出
    print("\n[7] 保存输出...")
    output_paths = service.save_outputs(
        output_dir=str(PROJECT_ROOT / "outputs" / "experiments")
    )
    for key, path in output_paths.items():
        print(f"    {key}: {path}")

    return result


def print_step_records(timeline) -> None:
    """格式化打印步骤记录。"""
    print("\n" + "=" * 60)
    print("StepRecords 详情")
    print("=" * 60)

    for step in timeline.steps:
        flag = " [INFERRED]" if step.completed_by_inference else ""
        conf_str = "[HIGH]" if step.confidence >= 0.8 else "[MED]" if step.confidence >= 0.5 else "[LOW]"

        print(f"\n  Step {step.step_index}: {step.step_name}{flag}")
        print(f"    Status: {step.status.value}")
        print(f"    Time: {step.start_time_sec:.1f}s -> {step.end_time_sec or '?'}s")
        print(f"    Confidence: {conf_str} {step.confidence:.4f}")
        print(f"    Description: {step.step_description[:80] or '(none)'}")

        if step.completed_by_inference:
            print(f"    Inference Method: {step.inference_method}")
            print(f"    Inference Model: {step.inference_model}")

        if step.evidence_refs:
            print(f"    Evidence ({len(step.evidence_refs)} refs):")
            for ref in step.evidence_refs[:3]:
                ts = ref.timestamp_sec
                fid = ref.frame_id
                etype = ref.evidence_type.value
                print(f"      - [{etype}] frame={fid}, @={ts:.1f}s")

        prov = step.provenance
        if prov:
            print(f"    Provenance: source={prov.source}, "
                  f"inferred={prov.is_inferred}, conf={prov.confidence:.2f}")


def print_timeline_summary(timeline) -> None:
    """打印时间线摘要 JSON。"""
    print("\n" + "=" * 60)
    print("ExperimentTimeline JSON（摘要）")
    print("=" * 60)

    summary = {
        "timeline_id": timeline.timeline_id,
        "experiment_id": timeline.experiment_id,
        "title": timeline.title,
        "total_steps": timeline.total_steps,
        "confirmed_steps": timeline.confirmed_steps,
        "inferred_steps": timeline.inferred_steps,
        "avg_confidence": timeline.avg_confidence,
        "start_time_sec": timeline.start_time_sec,
        "end_time_sec": timeline.end_time_sec,
        "total_duration_sec": timeline.total_duration_sec,
        "video_coverage_ratio": timeline.video_coverage_ratio,
        "models_used": timeline.models_used,
        "processing_stage": timeline.processing_stage.value,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def print_structured_json(timeline) -> None:
    """打印完整的结构化 JSON。"""
    print("\n" + "=" * 60)
    print("完整 Timeline JSON（前50行）")
    print("=" * 60)

    full = timeline.to_dict()
    lines = json.dumps(full, ensure_ascii=False, indent=2).split('\n')
    for line in lines[:50]:
        print(line)
    if len(lines) > 50:
        print(f"  ... ({len(lines) - 50} more lines)")


def check_vlm_health() -> bool:
    """检查 VLM API 是否可用。"""
    try:
        from experiment.vlm_client import DashScopeVLClient
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            print("[VLM] 未设置 DASHSCOPE_API_KEY，跳过 VLM")
            return False
        client = DashScopeVLClient(
            api_key=api_key,
            base_url=os.environ.get("DASHSCOPE_BASE_URL"),
        )
        health = client.check_health()
        if health:
            print(f"[VLM] DashScope Qwen-VL ({client.model}) 连接正常")
        else:
            print("[VLM] DashScope Qwen-VL 连接失败（但会继续运行）")
        return health
    except Exception as e:
        print(f"[VLM] VLM 初始化失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="实验过程理解 Demo")
    parser.add_argument("--video", type=str, default=None,
                        help="视频路径（默认：创建临时测试视频）")
    parser.add_argument("--no-vlm", action="store_true",
                        help="禁用 VLM，使用 fallback 模式")
    parser.add_argument("--max-frames", type=int, default=30,
                        help="最大处理帧数（默认: 30）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认: outputs/experiments）")
    args = parser.parse_args()

    # 示例对话上下文
    SAMPLE_CONTEXT = """
操作员在上午10:00开始实验。
10:05 从4°C冰箱取出蛋白质样本溶液。
10:08 使用移液器加入1mL缓冲液（PBS, pH7.4）。
10:12 将混合液转移至离心管。
10:15 启动离心机，3000rpm，5分钟。
10:20 离心结束，小心取出上清。
10:22 记录实验数据，实验完成。
    """.strip()

    # 示例 Protocol
    SAMPLE_PROTOCOL = """
蛋白质纯化标准操作流程 (SOP):

材料:
  - 蛋白质样品溶液
  - PBS 缓冲液 (pH 7.4)
  - 离心管 (15mL)
  - 移液器和枪头

步骤:
  1. 准备阶段
     - 从冰箱取出样本，平衡至室温 (5min)
     - 标记离心管

  2. 样品处理
     - 使用移液器准确量取1mL样品
     - 加入等体积PBS缓冲液
     - 轻柔吹打混匀（≥5次）

  3. 离心分离
     - 将混合液转移至离心管
     - 离心: 3000rpm, 5min, 4°C
     - 注意：配平离心管

  4. 收集产物
     - 离心结束后，小心取出上清
     - 避免触碰沉淀
     - 将上清转移至新离心管

  5. 记录与结束
     - 记录体积和外观
     - 标记储存条件
     - 清理工作台
    """.strip()

    print("\n" + "#" * 60)
    print("#  实验过程理解与步骤推理 Demo")
    print("#  Lab Experiment Process Understanding & Step Reasoning")
    print("#" * 60)

    # 确定视频路径
    if args.video:
        video_path = args.video
        if not Path(video_path).exists():
            print(f"[ERROR] 视频文件不存在: {video_path}")
            sys.exit(1)
        print(f"[Demo] 使用指定视频: {video_path}")
    else:
        # 创建临时测试视频
        temp_video = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, prefix="demo_video_"
        )
        temp_video.close()
        video_path = create_sample_video(temp_video.name, duration_sec=10)

    # 检查 VLM
    use_vlm = not args.no_vlm
    if use_vlm:
        check_vlm_health()

    # 运行 Demo
    result = run_demo(
        video_path=video_path,
        context_text=SAMPLE_CONTEXT,
        protocol_text=SAMPLE_PROTOCOL,
        use_vlm=use_vlm,
        max_frames=args.max_frames,
    )

    timeline = result["timeline"]
    experiment = result["experiment"]

    # 打印结果
    print_step_records(timeline)
    print_timeline_summary(timeline)
    print_structured_json(timeline)

    # Web 访问信息
    print("\n" + "=" * 60)
    print("Web UI 访问")
    print("=" * 60)
    print(f"  启动后端: python backend/main.py  (http://localhost:8000)")
    print(f"  启动前端: python frontend/dashboard.py  (http://localhost:8080)")
    print(f"  实验列表: http://localhost:8080/experiments")
    print(f"  实验详情: http://localhost:8080/experiments/{experiment.experiment_id}")
    print(f"  新建上传: http://localhost:8080/upload")

    # 输出路径
    out_dir = args.output_dir or str(PROJECT_ROOT / "outputs" / "experiments")
    exp_dir = Path(out_dir) / experiment.experiment_id
    print(f"\n  输出目录: {exp_dir}")
    print(f"  Timeline JSON: {exp_dir}/timeline.json")
    print(f"  Steps JSON: {exp_dir}/steps.json")
    print(f"  Experiment JSON: {exp_dir}/experiment.json")

    print("\n" + "#" * 60)
    print("#  Demo 完成!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()

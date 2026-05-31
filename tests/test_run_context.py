from __future__ import annotations

import time

from key_action_indexer.pipeline import RunContext, PIPELINE_STAGES


def test_run_id_generated():
    ctx = RunContext()
    assert len(ctx.run_id) == 36
    assert "-" in ctx.run_id


def test_unique_run_ids():
    a = RunContext()
    b = RunContext()
    assert a.run_id != b.run_id


def test_begin_end_stage():
    ctx = RunContext()
    ctx.begin_stage("detection")
    time.sleep(0.005)
    ctx.end_stage(inputs=1, outputs=5)
    assert len(ctx.stages) == 1
    assert ctx.stages[0]["stage"] == "detection"
    assert ctx.stages[0]["inputs"] == 1
    assert ctx.stages[0]["outputs"] == 5
    assert ctx.stages[0]["duration_sec"] >= 0.004


def test_multiple_stages():
    ctx = RunContext()
    ctx.begin_stage("validation")
    ctx.end_stage(inputs=1, outputs=1)
    ctx.begin_stage("detection")
    ctx.end_stage(inputs=1, outputs=3)
    ctx.begin_stage("micro_segmentation")
    ctx.end_stage(inputs=3, outputs=8)
    ctx.begin_stage("vector_index")
    ctx.end_stage(inputs=8, outputs=8)
    stats = ctx.stage_stats()
    assert stats["stage_count"] == 4
    assert stats["run_id"] == ctx.run_id
    assert stats["total_duration_sec"] >= 0.0
    stage_names = [s["stage"] for s in stats["stages"]]
    assert stage_names == ["validation", "detection", "micro_segmentation", "vector_index"]


def test_end_without_begin_is_noop():
    ctx = RunContext()
    ctx.end_stage(inputs=1, outputs=1)
    assert len(ctx.stages) == 0


def test_double_begin_overwrites():
    ctx = RunContext()
    ctx.begin_stage("first")
    ctx.begin_stage("second")
    ctx.end_stage(inputs=0, outputs=0)
    assert len(ctx.stages) == 1
    assert ctx.stages[0]["stage"] == "second"


def test_stage_stats_format():
    ctx = RunContext()
    ctx.begin_stage("test")
    ctx.end_stage(inputs=2, outputs=4, errors=1)
    stats = ctx.stage_stats()
    assert "run_id" in stats
    assert "stage_count" in stats
    assert "total_duration_sec" in stats
    assert "stages" in stats
    stage = stats["stages"][0]
    assert "stage" in stage
    assert "duration_sec" in stage
    assert "inputs" in stage
    assert "outputs" in stage
    assert "errors" in stage
    assert stage["errors"] == 1


def test_pipeline_stages_constant():
    assert len(PIPELINE_STAGES) == 4
    assert "YOLO_PRIMARY" in PIPELINE_STAGES
    assert "MICRO_SEGMENT" in PIPELINE_STAGES

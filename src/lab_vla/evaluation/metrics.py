from __future__ import annotations

from lab_vla.core.contracts import ExecutionResult, RuntimeMetrics, SceneState


class MetricsCollector:
    def __init__(self) -> None:
        self.metrics = RuntimeMetrics()

    def on_scene(self, scene: SceneState) -> None:
        self.metrics.frames_total += 1
        if scene.target_bbox is not None:
            self.metrics.frames_with_target += 1

    def on_plan(self) -> None:
        self.metrics.plans_total += 1

    def on_safety_block(self) -> None:
        self.metrics.safety_blocks += 1

    def on_execution(self, result: ExecutionResult) -> None:
        if result.status == "ok":
            self.metrics.executions_ok += 1
        else:
            self.metrics.executions_failed += 1


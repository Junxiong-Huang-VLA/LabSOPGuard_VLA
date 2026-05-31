from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from labsopguard.input_layer import ExperimentInputBundle
from labsopguard.output_layer import StructuredOutputBuilder
from labsopguard.preprocessing import MultiModalPreprocessor
from labsopguard.reasoning import StepGraphReasoner


@dataclass
class FormalExperimentWorkflow:
    preprocessor: MultiModalPreprocessor = MultiModalPreprocessor()
    reasoner: StepGraphReasoner = StepGraphReasoner()
    output_builder: StructuredOutputBuilder = StructuredOutputBuilder()

    def build_structured_output(self, experiment_record: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        experiment = result["experiment"]
        timeline = result["timeline"]
        physical_events = result.get("physical_events", [])
        material_stream = result.get("material_stream", [])
        video_paths = experiment_record.get("video_paths", []) or []
        video_inputs = experiment_record.get("video_inputs", []) or []
        primary_video_path = video_paths[0] if video_paths else (video_inputs[0].get("video_path", "") if video_inputs else "")
        input_bundle = ExperimentInputBundle.from_experiment_record(
            experiment_id=experiment_record["experiment_id"],
            title=experiment_record.get("title", experiment_record["experiment_id"]),
            video_path=primary_video_path,
            protocol_text=experiment_record.get("protocol_text", ""),
            context_inputs=experiment_record.get("context_inputs", []),
            video_paths=video_paths,
            video_metadata=experiment_record.get("video_metadata", []),
        )
        artifact = self.preprocessor.build_artifact(
            duration_sec=getattr(timeline, "video_duration_sec", 0.0) or 0.0,
            context_text=input_bundle.context_text,
            protocol_text=input_bundle.protocol_text,
            physical_events=physical_events,
            material_stream=material_stream,
            context_records=input_bundle.user_texts,
            video_assets=getattr(experiment, "video_assets", []),
        )
        protocol_nodes = self.reasoner.parse_protocol(input_bundle.protocol_text)
        matched_steps = self.reasoner.match_timeline_steps(protocol_nodes, timeline.steps)
        for step, matched in zip(timeline.steps, matched_steps):
            step.metadata["stage_name"] = matched["stage_name"]
            step.metadata["completion_type"] = matched["completion_type"]
        return self.output_builder.build(experiment, timeline, timeline.steps, artifact)

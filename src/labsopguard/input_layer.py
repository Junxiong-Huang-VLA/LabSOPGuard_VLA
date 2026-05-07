from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimeAnchoredText:
    source_type: str
    content: str
    timestamp_sec: Optional[float] = None
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    anchor_video_index: Optional[int] = None
    anchor_video_asset_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentInputBundle:
    experiment_id: str
    title: str
    video_path: str
    video_paths: List[str] = field(default_factory=list)
    video_metadata: List[Dict[str, Any]] = field(default_factory=list)
    protocol_text: str = ""
    context_text: str = ""
    user_texts: List[TimeAnchoredText] = field(default_factory=list)
    ai_replies: List[TimeAnchoredText] = field(default_factory=list)
    transcripts: List[TimeAnchoredText] = field(default_factory=list)
    uploaded_documents: List[TimeAnchoredText] = field(default_factory=list)
    knowledge_snippets: List[TimeAnchoredText] = field(default_factory=list)
    step_priors: List[TimeAnchoredText] = field(default_factory=list)

    @classmethod
    def from_experiment_record(
        cls,
        experiment_id: str,
        title: str,
        video_path: str,
        protocol_text: str,
        context_inputs: List[Dict[str, Any]] | None,
        video_paths: Optional[List[str]] = None,
        video_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> "ExperimentInputBundle":
        context_texts: List[str] = []
        user_texts: List[TimeAnchoredText] = []
        for item in context_inputs or []:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                context_texts.append(text)
                timestamp_sec = item.get("timestamp_sec")
                if timestamp_sec is None:
                    timestamp_sec = item.get("start_time_sec")
                user_texts.append(
                    TimeAnchoredText(
                        source_type="user_text",
                        content=text,
                        timestamp_sec=timestamp_sec,
                        start_time_sec=item.get("start_time_sec"),
                        end_time_sec=item.get("end_time_sec"),
                        anchor_video_index=item.get("video_index"),
                        anchor_video_asset_id=item.get("video_asset_id"),
                        metadata={k: v for k, v in item.items() if k != "text"},
                    )
                )
            elif item:
                text = str(item).strip()
                context_texts.append(text)
                user_texts.append(TimeAnchoredText(source_type="user_text", content=text))
        return cls(
            experiment_id=experiment_id,
            title=title,
            video_path=video_path,
            video_paths=list(video_paths or ([video_path] if video_path else [])),
            video_metadata=list(video_metadata or []),
            protocol_text=protocol_text or "",
            context_text="\n".join(context_texts).strip(),
            user_texts=user_texts,
            transcripts=[t for t in user_texts if "transcript" in t.metadata.get("kind", "")],
            uploaded_documents=[t for t in user_texts if t.metadata.get("kind") == "document"],
            knowledge_snippets=[TimeAnchoredText(source_type="knowledge", content=protocol_text)] if protocol_text else [],
        )

from __future__ import annotations

from pathlib import Path

from .schemas import TranscriptSource, TranscriptUtterance
from .time_alignment import align_transcript_to_global_time


def load_aligned_transcript(transcript_source: TranscriptSource | None) -> list[TranscriptUtterance]:
    if transcript_source is None:
        return []
    path = Path(transcript_source.path)
    if not path.exists():
        return []
    return align_transcript_to_global_time(path, transcript_source)

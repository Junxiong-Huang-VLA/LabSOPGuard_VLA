from __future__ import annotations

import json
import shutil
import subprocess
import uuid
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from labsopguard.asr import TranscriptResult, TranscriptSegment, transcribe_audio_file


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ASRJob:
    job_id: str
    experiment_id: str
    source_file: str
    status: str = "queued"
    provider: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    prompt: Optional[str] = None
    text: str = ""
    segment_count: int = 0
    context_count: int = 0
    chunk_count: int = 0
    retry_count: int = 0
    error: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ASRJobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def create(self, experiment_id: str, source_file: str, *, language: Optional[str], prompt: Optional[str]) -> ASRJob:
        job = ASRJob(
            job_id=f"asr_{uuid.uuid4().hex[:12]}",
            experiment_id=experiment_id,
            source_file=source_file,
            language=language,
            prompt=prompt,
        )
        self.save(job)
        return job

    def load(self, job_id: str) -> ASRJob:
        path = self.path_for(job_id)
        if not path.exists():
            raise FileNotFoundError(job_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ASRJob(**payload)

    def save(self, job: ASRJob) -> None:
        job.updated_at = _utc_now_iso()
        self.path_for(job.job_id).write_text(json.dumps(job.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _split_wav(audio_path: Path, chunk_duration_sec: float, chunks_dir: Path) -> List[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[Path] = []
    with wave.open(str(audio_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frame_rate = source.getframerate()
        frames_per_chunk = max(1, int(frame_rate * chunk_duration_sec))
        index = 0
        while True:
            frames = source.readframes(frames_per_chunk)
            if not frames:
                break
            chunk_path = chunks_dir / f"chunk_{index:04d}.wav"
            with wave.open(str(chunk_path), "wb") as target:
                target.setnchannels(channels)
                target.setsampwidth(sample_width)
                target.setframerate(frame_rate)
                target.writeframes(frames)
            chunks.append(chunk_path)
            index += 1
    return chunks


def split_audio_for_asr(
    audio_path: str | Path,
    chunks_dir: str | Path,
    *,
    chunk_duration_sec: float = 60.0,
    force_chunk: bool = False,
    max_single_file_mb: float = 9.5,
) -> List[Path]:
    source = Path(audio_path)
    chunks = Path(chunks_dir)
    size_mb = source.stat().st_size / (1024 * 1024)
    if not force_chunk and size_mb <= max_single_file_mb:
        return [source]

    chunks.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".wav":
        return _split_wav(source, chunk_duration_sec, chunks)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to split non-WAV audio/video files for long ASR jobs")
    output_pattern = str(chunks / "chunk_%04d.wav")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-f",
        "segment",
        "-segment_time",
        str(max(1.0, float(chunk_duration_sec))),
        "-ac",
        "1",
        "-ar",
        "16000",
        output_pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed: {proc.stderr[-1000:]}")
    return sorted(chunks.glob("chunk_*.wav"))


def transcribe_audio_in_chunks(
    audio_path: str | Path,
    work_dir: str | Path,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    chunk_duration_sec: float = 60.0,
    force_chunk: bool = False,
) -> tuple[TranscriptResult, int]:
    chunks = split_audio_for_asr(
        audio_path,
        Path(work_dir) / "chunks",
        chunk_duration_sec=chunk_duration_sec,
        force_chunk=force_chunk,
    )
    merged_text: List[str] = []
    merged_segments: List[TranscriptSegment] = []
    provider = ""
    model = ""
    for index, chunk in enumerate(chunks):
        offset = index * float(chunk_duration_sec)
        result = transcribe_audio_file(chunk, language=language, prompt=prompt)
        provider = result.provider
        model = result.model
        if result.text:
            merged_text.append(result.text)
        for segment in result.segments:
            merged_segments.append(
                TranscriptSegment(
                    text=segment.text,
                    start_time_sec=(segment.start_time_sec or 0.0) + offset,
                    end_time_sec=(segment.end_time_sec + offset) if segment.end_time_sec is not None else None,
                    speaker=segment.speaker,
                    confidence=segment.confidence,
                    metadata={**segment.metadata, "chunk_index": index, "chunk_file": str(chunk)},
                )
            )
    return (
        TranscriptResult(
            text="\n".join(part for part in merged_text if part),
            segments=merged_segments,
            provider=provider or "qwen_dashscope",
            model=model,
            language=language,
        ),
        len(chunks),
    )

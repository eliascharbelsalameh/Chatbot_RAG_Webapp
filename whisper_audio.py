"""Local audio transcription via faster-whisper (replaces Deepgram)."""
from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import config


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel  # type: ignore
    return WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE,
    )


def transcribe_bytes(audio_bytes: bytes, language: str | None = None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        segments, _ = _model().transcribe(path, language=language, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
    finally:
        Path(path).unlink(missing_ok=True)

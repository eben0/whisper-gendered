"""faster-whisper transcription wrapper.

The WhisperModel is expensive to construct, so it is loaded lazily on first use
and cached as a process-wide singleton. A threading lock guards construction so
concurrent first requests don't load the model twice.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from config import settings

log = logging.getLogger("pipeline.transcribe")

_model: WhisperModel | None = None
_lock = threading.Lock()


@dataclass
class Segment:
    start: float
    end: float
    text: str


def get_model() -> WhisperModel:
    """Return the singleton WhisperModel, loading it on first call."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                log.info(
                    "Loading WhisperModel %s (device=%s, compute_type=%s)...",
                    settings.WHISPER_MODEL, settings.DEVICE, settings.COMPUTE_TYPE,
                )
                _model = WhisperModel(
                    settings.WHISPER_MODEL,
                    device=settings.DEVICE,
                    compute_type=settings.COMPUTE_TYPE,
                )
                log.info("WhisperModel loaded.")
    return _model


def model_loaded() -> bool:
    return _model is not None


def transcribe(audio_path: Path, language: str | None = "en") -> list[Segment]:
    """Transcribe an audio file into a list of timed Segments."""
    model = get_model()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )
    # The generator runs inference lazily; materialize it into a list here so
    # the work happens inside the worker thread, not later on the event loop.
    segments = [
        Segment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments_iter
    ]
    log.info(
        "Transcribed %d segments (detected language=%s, prob=%.2f)",
        len(segments), getattr(info, "language", language),
        getattr(info, "language_probability", 0.0),
    )
    return segments


def warmup(audio_path: Path) -> None:
    """Run a throwaway pass so CUDA kernels/weights are ready before traffic."""
    try:
        transcribe(audio_path)
        log.info("Whisper warm-up complete.")
    except Exception:  # pragma: no cover - warm-up must never crash startup
        log.exception("Whisper warm-up failed (continuing anyway).")

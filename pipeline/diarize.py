"""pyannote.audio speaker-diarization wrapper.

Same lazy-singleton pattern as transcribe.py. The pipeline is moved to CUDA when
available and every inference call runs under torch.inference_mode().
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

from config import settings
from pipeline.transcribe import Segment

log = logging.getLogger("pipeline.diarize")

_pipeline: Pipeline | None = None
_lock = threading.Lock()


def get_pipeline() -> Pipeline:
    """Return the singleton diarization pipeline, loading it on first call."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                token = settings.require_hf_token()
                log.info("Loading pyannote speaker-diarization-3.1 pipeline...")
                pipe = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=token,
                )
                if torch.cuda.is_available():
                    pipe.to(torch.device("cuda"))
                    log.info("Diarization pipeline moved to CUDA.")
                _pipeline = pipe
                log.info("Diarization pipeline loaded.")
    return _pipeline


def diarize(audio_path: Path) -> Annotation:
    """Run speaker diarization and return a pyannote Annotation."""
    pipe = get_pipeline()
    # Load the audio in-memory and hand pyannote a {waveform, sample_rate} dict.
    # This bypasses torchcodec/FFmpeg-DLL decoding, which is awkward to install
    # on Windows. Input is already 16 kHz mono WAV from the encode step.
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # (channel, time)
    with torch.inference_mode():
        result = pipe({"waveform": waveform, "sample_rate": sr})
    # pyannote 4.x may return a wrapped output object; older pipelines return a
    # bare Annotation. Accept either.
    annotation = getattr(result, "speaker_diarization", result)
    speakers = annotation.labels()
    log.info("Diarized %d speaker(s): %s", len(speakers), speakers)
    return annotation


def assign_speaker(segment: Segment, diarization: Annotation) -> str | None:
    """Return the speaker label active at the midpoint of ``segment``.

    Falls back to the speaker with the greatest temporal overlap with the
    segment, and finally to None if nothing overlaps.
    """
    midpoint = (segment.start + segment.end) / 2.0

    # Primary: whoever is speaking at the midpoint.
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= midpoint <= turn.end:
            return speaker

    # Fallback: speaker with the most overlap across the whole segment.
    best_speaker: str | None = None
    best_overlap = 0.0
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker

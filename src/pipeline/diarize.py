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

from pipeline.segment import Segment

log = logging.getLogger("pipeline.diarize")


class Diarizer:
    """Lazy-loading pyannote speaker-diarization singleton."""

    def __init__(self, settings) -> None:
        self._settings = settings
        self._pipeline: Pipeline | None = None
        self._lock = threading.Lock()

    def get_pipeline(self) -> Pipeline:
        """Return the singleton diarization pipeline, loading it on first call."""
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    token = self._settings.require_hf_token()
                    log.info("Loading pyannote speaker-diarization-3.1 pipeline...")
                    pipe = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=token,
                    )
                    if torch.cuda.is_available():
                        pipe.to(torch.device("cuda"))
                        log.info("Diarization pipeline moved to CUDA.")
                    self._pipeline = pipe
                    log.info("Diarization pipeline loaded.")
        return self._pipeline

    def diarize_waveform(self, waveform: np.ndarray, sr: int) -> Annotation:
        """Diarize an in-memory waveform.

        ``waveform`` is float32, shape ``(time,)`` for mono or ``(channel, time)``.
        We hand pyannote a {waveform, sample_rate} dict directly rather than a file
        path: that bypasses torchcodec/FFmpeg-DLL audio decoding, which is awkward
        to install on Windows. The dtype is coerced to float32 so callers passing
        slices that may have upcast (e.g. via numpy ops) still get correct results.
        """
        pipe = self.get_pipeline()
        waveform = waveform.astype(np.float32, copy=False)
        wav2d = waveform[np.newaxis, :] if waveform.ndim == 1 else waveform
        tensor = torch.from_numpy(np.ascontiguousarray(wav2d))
        with torch.inference_mode():
            result = pipe({"waveform": tensor, "sample_rate": sr})
        annotation = result if isinstance(result, Annotation) else result.speaker_diarization
        speakers = annotation.labels()
        log.info("Diarized %d speaker(s): %s", len(speakers), speakers)
        return annotation

    def diarize(self, audio_path: Path) -> Annotation:
        """Run speaker diarization on a WAV file (used for warm-up / whole-file)."""
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        return self.diarize_waveform(data.T, sr)  # data.T -> (channel, time)

    def assign_speaker(self, segment: Segment, diarization: Annotation) -> str | None:
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


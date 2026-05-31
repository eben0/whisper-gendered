"""faster-whisper transcription wrapper.

The WhisperModel is expensive to construct, so it is loaded lazily on first use
and cached per ``Transcriber`` instance. A threading lock guards construction so
concurrent first requests don't load the model twice.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from faster_whisper import WhisperModel

from pipeline.segment import Segment  # re-export: keeps pipeline.transcribe.Segment valid

log = logging.getLogger("pipeline.transcribe")


class Transcriber:
    """Lazy-loading faster-whisper singleton.

    Instantiate once; the model loads on first call to ``transcribe()`` or
    ``warmup()`` and is reused for every subsequent request.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    def get_model(self) -> WhisperModel:
        """Return the WhisperModel, loading it on first call."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    log.info(
                        "Loading WhisperModel %s (device=%s, compute_type=%s)...",
                        self._settings.WHISPER_MODEL, self._settings.DEVICE,
                        self._settings.COMPUTE_TYPE,
                    )
                    self._model = WhisperModel(
                        self._settings.WHISPER_MODEL,
                        device=self._settings.DEVICE,
                        compute_type=self._settings.COMPUTE_TYPE,
                    )
                    log.info("WhisperModel loaded.")
        return self._model

    def model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_path: Path, language: str | None = "en") -> list[Segment]:
        """Transcribe an audio file into a list of timed Segments.

        ``word_timestamps=True`` is enabled (Plan Task 5) so we can re-anchor
        each segment's start/end to its first/last word's timestamps. Whisper's
        segment-level timestamps drift up to several seconds over long audio
        (~4s after 4 min observed in S04E05); the word-level timestamps stay
        tightly aligned because they're produced directly from the model's
        attention pattern rather than the post-hoc segmentation. When a
        segment has no word data (rare), we fall back to its native
        timestamps rather than crash.
        """
        model = self.get_model()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )
        # Materialize the lazy generator inside the worker thread so the
        # inference work doesn't leak onto the event loop.
        segments: list[Segment] = []
        for s in segments_iter:
            start = s.start
            end = s.end
            words = getattr(s, "words", None)
            if words:
                start = words[0].start
                end = words[-1].end
            segments.append(Segment(start=start, end=end, text=s.text.strip()))
        log.info(
            "Transcribed %d segments (detected language=%s, prob=%.2f)",
            len(segments), getattr(info, "language", language),
            getattr(info, "language_probability", 0.0),
        )
        return segments

    def warmup(self, audio_path: Path) -> None:
        """Run a throwaway pass so CUDA kernels/weights are ready before traffic."""
        try:
            self.transcribe(audio_path)
            log.info("Whisper warm-up complete.")
        except Exception:  # pragma: no cover - warm-up must never crash startup
            log.exception("Whisper warm-up failed (continuing anyway).")


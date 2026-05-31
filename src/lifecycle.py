"""Startup warm-up: pre-loads ML models so the first request isn't penalised."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings
    from src.core.concurrency import ConcurrencyManager
    from src.audio import Audio
    from src.backends.factory import TranslationBackend
    from pipeline.transcribe import Transcriber
    from pipeline.diarize import Diarizer
    from pipeline.gender.ml import GenderMLClassifier

log = logging.getLogger("lifecycle")


class Lifecycle:
    """Manages application startup warm-up with all dependencies injected."""

    def __init__(
        self,
        concurrency: "ConcurrencyManager",
        audio: "Audio",
        backend: "TranslationBackend",
        settings: "Settings",
        transcriber: "Transcriber | None" = None,
        diarizer: "Diarizer | None" = None,
        gender_ml_classifier: "GenderMLClassifier | None" = None,
    ) -> None:
        self._concurrency = concurrency
        self._audio = audio
        self._backend = backend
        self._settings = settings
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._gender_ml = gender_ml_classifier

    async def warmup(self) -> None:
        """Pre-load Whisper, pyannote, gender_ml, and translation model."""
        settings = self._settings
        log.info(
            "Starting up. model=%s device=%s target_language=%s gender_aware=%s "
            "translation_backend=%s",
            settings.WHISPER_MODEL, settings.DEVICE, settings.TARGET_LANGUAGE,
            settings.is_gender_aware(), settings.TRANSLATION_BACKEND,
        )
        tmp = Path(tempfile.gettempdir()) / f"warmup_{uuid.uuid4().hex}.wav"
        try:
            self._audio.write_silent_wav(tmp)
            if self._transcriber is not None:
                await self._concurrency.run_in_thread(self._transcriber.warmup, tmp)
            else:
                from pipeline import transcribe
                await self._concurrency.run_in_thread(transcribe.warmup, tmp)

            if settings.is_gender_aware():
                try:
                    if self._diarizer is not None:
                        await self._concurrency.run_in_thread(self._diarizer.diarize, tmp)
                    else:
                        from pipeline import diarize
                        await self._concurrency.run_in_thread(diarize.diarize, tmp)
                    log.info("Pyannote warm-up complete.")
                except Exception:
                    log.exception("Pyannote warm-up failed (continuing anyway).")

            # Warm the wav2vec2 gender classifier when it's actually going to be
            # used. This amortizes the ~1.2 GB model load so the first request
            # doesn't pay for it (and so the Task 8 perf gate doesn't
            # false-positive on cold start with ml_dt = pitch + 5–30s of load).
            if (
                settings.GENDER_CLASSIFIER.strip().lower() in ("ml", "ensemble")
                and settings.is_gender_aware()
            ):
                try:
                    if self._gender_ml is not None:
                        await self._concurrency.run_in_thread(self._gender_ml.warmup)
                    else:
                        from pipeline import gender_ml
                        await self._concurrency.run_in_thread(gender_ml.warmup)
                except Exception:
                    log.exception("gender_ml warm-up failed; falling back to lazy load.")

            # Local translation model is large (NLLB-200 distilled is ~2.4 GB) and
            # would otherwise load on the first /asr request — well past Bazarr's
            # client timeout. Pre-load it here. Claude backend has nothing to warm.
            await self._backend.warmup()
        finally:
            tmp.unlink(missing_ok=True)

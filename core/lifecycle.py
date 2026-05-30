"""Startup warm-up: pre-loads ML models so the first request isn't penalised."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from config import settings
from core import audio, backends, concurrency

log = logging.getLogger("core.lifecycle")


async def warmup() -> None:
    """Pre-load Whisper, pyannote, gender_ml, and (if local) the translation model.

    Called from the FastAPI startup event. Idempotent — running twice is harmless
    (the pipeline singletons check if already loaded).
    """
    log.info(
        "Starting up. model=%s device=%s target_language=%s gender_aware=%s "
        "translation_backend=%s",
        settings.WHISPER_MODEL, settings.DEVICE, settings.TARGET_LANGUAGE,
        settings.is_gender_aware(), settings.TRANSLATION_BACKEND,
    )
    tmp = Path(tempfile.gettempdir()) / f"warmup_{uuid.uuid4().hex}.wav"
    try:
        audio._write_silent_wav(tmp)
        from pipeline import transcribe, diarize
        await concurrency.run_in_thread(transcribe.warmup, tmp)
        if settings.is_gender_aware():
            try:
                await concurrency.run_in_thread(diarize.diarize, tmp)
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
                from pipeline import gender_ml
                await concurrency.run_in_thread(gender_ml.warmup)
            except Exception:
                log.exception("gender_ml warm-up failed; falling back to lazy load.")
        # Local translation model is large (NLLB-200 distilled is ~2.4 GB) and
        # would otherwise load on the first /asr request — well past Bazarr's
        # client timeout. Pre-load it here. Claude backend has nothing to warm.
        await backends.backend.warmup()
    finally:
        tmp.unlink(missing_ok=True)

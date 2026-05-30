"""Bazarr-compatible Whisper ASR provider with gender-aware translation.

Endpoints:
  GET  /         -> liveness + model info
  GET  /status   -> queue depth + model-loaded flag
  POST /asr      -> transcribe (and optionally translate) an uploaded audio file

All ML inference (faster-whisper, pyannote, librosa) and the Claude API calls are
blocking, so every pipeline run is dispatched to a ThreadPoolExecutor and gated
by an asyncio.Semaphore sized to CONCURRENT_JOBS — the event loop never blocks.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from core import audio, backends, concurrency, cuda, lifecycle, orchestrator, side_file
cuda.bootstrap()

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from config import settings
from core.artifacts import PipelineArtifacts
from pipeline import diarize, gender, transcribe
from pipeline.format import render
from pipeline.lang import language_name
from pipeline.transcribe import Segment

from core.logging_config import configure as _configure_logging

_configure_logging()
log = logging.getLogger("server")

VERSION = "1.0.0"

app = FastAPI(title="Gender-Aware Hebrew Subtitle Server", version=VERSION)



# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

@app.on_event("startup")
async def warmup() -> None:
    await lifecycle.warmup()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
async def root():
    return JSONResponse(
        {"status": "ok", "version": VERSION, "model": settings.WHISPER_MODEL}
    )


@app.get("/status")
async def status():
    return JSONResponse(
        {
            "status": "ok",
            "queue_depth": concurrency.job_depth(),
            "model_loaded": transcribe.model_loaded(),
        }
    )


@app.post("/asr")
async def asr(
    request: Request,
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe"),
    language: str = Query("en"),
    output: str = Query("srt"),
    encode: bool = Query(True),
):
    request_id = uuid.uuid4().hex[:8]
    workdir = Path(tempfile.mkdtemp(prefix=f"asr_{request_id}_"))
    raw_path = workdir / (audio_file.filename or "input")

    concurrency.inc_jobs()
    started = time.monotonic()
    try:
        with raw_path.open("wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        log.info("[%s] received %s (task=%s lang=%s output=%s encode=%s)",
                 request_id, raw_path.name, task, language, output, encode)

        audio_path = workdir / "audio.wav"
        if encode:
            await concurrency.run_in_thread(audio.encode_to_wav, raw_path, audio_path)
        else:
            await concurrency.run_in_thread(audio.prepare_unencoded, raw_path, audio_path)

        alt_target = None  # initialized for the post-semaphore block

        async with concurrency.semaphore:
            # Reclaim cached-but-unused VRAM left by a previous request before
            # touching the GPU. The PyTorch caching allocator does not return
            # memory between requests on its own, and Whisper large-v3 + pyannote
            # + NLLB resident together leave only ~3 GB of slack on an 8 GB card.
            # Without this, a back-to-back /asr has been observed to OOM in the
            # transcribe step (CUDA error: out of memory) even though both
            # requests fit individually when the process is fresh.
            cuda.empty_cuda_cache()
            # Capture primary pipeline artifacts (raw segments + per-chunk
            # diarization annotations) only when the A/B alt pass will
            # actually run — keeps memory cost of the out-container at zero
            # for typical requests where GENDER_AB_OUTPUT is off.
            artifacts_out: list[PipelineArtifacts] | None = (
                [] if settings.GENDER_AB_OUTPUT else None
            )
            source_segments, target_segments = await orchestrator.run_pipeline_async(
                audio_path, language, _artifacts_out=artifacts_out,
            )
            # A/B alt-classifier pass: must run INSIDE the semaphore because
            # run_pipeline_alt_classifier mutates the global
            # settings.GENDER_CLASSIFIER for the duration of the call. If we
            # released the semaphore first, a concurrent request entering it
            # mid-alt would observe the flipped value and use the wrong
            # classifier on its primary pass. Render + side-file save remain
            # outside the semaphore below — they're I/O-only, no global
            # mutation. The alt pass reuses the primary's transcribe and
            # diarize results via ``artifacts`` so it doesn't OOM on
            # 8 GB cards (back-to-back Whisper inferences peaked over budget).
            if (
                settings.GENDER_AB_OUTPUT
                and target_segments is not None
                and artifacts_out
            ):
                try:
                    _, alt_target = await orchestrator.run_pipeline_alt_classifier(
                        audio_path, language, artifacts_out[0],
                    )
                except Exception:
                    log.exception(
                        "[%s] A/B alt-classifier pass failed; primary "
                        "side-file will still be written",
                        request_id,
                    )
                    alt_target = None

        # The HTTP response carries the SOURCE-language transcript so Bazarr's
        # whisperai plugin writes the response to its language-matching path
        # (e.g. ``*.en.srt``) with source-language content. The translation
        # — when present — only lands at the configured side-file path, so
        # pre-existing English subtitle files are never overwritten by Hebrew.
        body, content_type = render(source_segments, output)
        wall = time.monotonic() - started
        video_file_url = request.query_params.get("video_file") or ""

        if target_segments is not None:
            # Translated run: render the translation separately for the side-
            # file save, and build the summary from the translated segments so
            # the script-check ratio remains meaningful.
            target_body, _ = render(target_segments, output)
            backend = settings.TRANSLATION_BACKEND
            backend_model = backends.backend.model_name()
            summary = side_file._build_translation_summary(
                request_id=request_id,
                video_file_url=video_file_url,
                source_language_iso=language,
                source_language_name=language_name(language),
                target_language=settings.TARGET_LANGUAGE,
                backend=backend,
                backend_model=backend_model,
                segments=target_segments,
                wall_seconds=wall,
            )
            log.info(
                "[%s] %s",
                request_id, summary.replace("\n", "\n[" + request_id + "] "),
            )
            await concurrency.run_in_thread(
                side_file._try_save_side_file, target_body, summary, video_file_url,
            )

            # Alt side-file save (outside semaphore — just I/O, no global
            # mutation). alt_target is only non-None when GENDER_AB_OUTPUT is
            # on and the alt pass succeeded.
            if alt_target is not None:
                alt_body, _ = render(alt_target, output)
                await concurrency.run_in_thread(
                    side_file._try_save_side_file,
                    alt_body, None,    # no summary file for the alt
                    video_file_url,
                    side_file.ALT_CLASSIFIER_SRT_SUFFIX,
                )
                log.info(
                    "[%s] A/B alt-classifier side-file saved",
                    request_id,
                )

        log.info("[%s] done in %.1fs (%d segments)",
                 request_id, wall, len(source_segments))
        return PlainTextResponse(body, media_type=content_type)
    finally:
        concurrency.dec_jobs()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=settings.PORT,
        workers=1,
        log_level="info",
    )

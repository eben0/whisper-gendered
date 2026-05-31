"""Bazarr-compatible Whisper ASR server — thin HTTP wiring only.

Endpoints:
  GET  /       -> liveness + model info
  GET  /status -> queue depth + model-loaded flag
  POST /asr    -> transcribe and optionally translate uploaded audio
"""

from __future__ import annotations

# ORDERING-CRITICAL: bootstrap before pyannote/orchestrator imports
from src.core.cuda import Cuda as _CudaClass
_cuda = _CudaClass()
_cuda.bootstrap()

from src.core.logging_config import configure as _configure_logging
_configure_logging()

import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from src.config import settings
from src.core.concurrency import ConcurrencyManager
from src.backends.factory import create_backend
from src.audio import Audio
from src.side_file import SideFile, ALT_CLASSIFIER_SRT_SUFFIX
from src.lifecycle import Lifecycle
from src.artifacts import PipelineArtifacts
from src import orchestrator
from pipeline.format import render
from pipeline.lang import language_name

log = logging.getLogger("server")
VERSION = "1.0.0"

# Construct all instances at module load (startup)
_concurrency = ConcurrencyManager(settings.CONCURRENT_JOBS)
_backend = create_backend(settings)
_audio = Audio()
_side_file = SideFile(settings)
_lifecycle = Lifecycle(_concurrency, _audio, _backend, settings)

app = FastAPI(title="Gender-Aware Hebrew Subtitle Server", version=VERSION)


@app.on_event("startup")
async def warmup() -> None:
    await _lifecycle.warmup()


@app.get("/")
async def root():
    return JSONResponse(
        {"status": "ok", "version": VERSION, "model": settings.WHISPER_MODEL}
    )


@app.get("/status")
async def status():
    from pipeline import transcribe
    return JSONResponse(
        {
            "status": "ok",
            "queue_depth": _concurrency.job_depth(),
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

    _concurrency.inc_jobs()
    started = time.monotonic()
    try:
        with raw_path.open("wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        log.info("[%s] received %s (task=%s lang=%s output=%s encode=%s)",
                 request_id, raw_path.name, task, language, output, encode)

        audio_path = workdir / "audio.wav"
        if encode:
            await _concurrency.run_in_thread(_audio.encode_to_wav, raw_path, audio_path)
        else:
            await _concurrency.run_in_thread(_audio.prepare_unencoded, raw_path, audio_path)

        alt_target = None  # initialized for the post-semaphore block

        async with _concurrency.semaphore:
            # Reclaim cached-but-unused VRAM left by a previous request before
            # touching the GPU. The PyTorch caching allocator does not return
            # memory between requests on its own, and Whisper large-v3 + pyannote
            # + NLLB resident together leave only ~3 GB of slack on an 8 GB card.
            # Without this, a back-to-back /asr has been observed to OOM in the
            # transcribe step (CUDA error: out of memory) even though both
            # requests fit individually when the process is fresh.
            _cuda.empty_cache()
            # Capture primary pipeline artifacts (raw segments + per-chunk
            # diarization annotations) only when the A/B alt pass will
            # actually run — keeps memory cost of the out-container at zero
            # for typical requests where GENDER_AB_OUTPUT is off.
            artifacts_out: list[PipelineArtifacts] | None = (
                [] if settings.GENDER_AB_OUTPUT else None
            )
            source_segments, target_segments = await orchestrator.run_pipeline_async(
                audio_path, language,
                concurrency_mgr=_concurrency, audio_obj=_audio, backend=_backend,
                _artifacts_out=artifacts_out,
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
                        concurrency_mgr=_concurrency, audio_obj=_audio, backend=_backend,
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
            backend_model = _backend.model_name()
            summary = _side_file.build_summary(
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
            await _concurrency.run_in_thread(
                _side_file.try_save, target_body, summary, video_file_url,
            )

            # Alt side-file save (outside semaphore — just I/O, no global
            # mutation). alt_target is only non-None when GENDER_AB_OUTPUT is
            # on and the alt pass succeeded.
            if alt_target is not None:
                alt_body, _ = render(alt_target, output)
                await _concurrency.run_in_thread(
                    _side_file.try_save,
                    alt_body, None,    # no summary file for the alt
                    video_file_url,
                    ALT_CLASSIFIER_SRT_SUFFIX,
                )
                log.info(
                    "[%s] A/B alt-classifier side-file saved",
                    request_id,
                )

        log.info("[%s] done in %.1fs (%d segments)",
                 request_id, wall, len(source_segments))
        return PlainTextResponse(body, media_type=content_type)
    finally:
        _concurrency.dec_jobs()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=settings.PORT, workers=1)

"""ASR request handler — all /asr business logic in one class."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import UploadFile
from fastapi.responses import PlainTextResponse

from pipeline.format import render
from pipeline.lang import language_name
from src.artifacts import PipelineArtifacts
from src.config import settings
from src.side_file import ALT_CLASSIFIER_SRT_SUFFIX

if TYPE_CHECKING:
    from src.core.concurrency import ConcurrencyManager
    from src.core.cuda import Cuda
    from src.audio import Audio
    from src.side_file import SideFile
    from src.orchestrator import Orchestrator

log = logging.getLogger("asr_handler")


class AsrHandler:
    """Handles a single POST /asr request end-to-end.

    All business logic (file I/O, audio prep, pipeline execution, side-file
    saves) lives here. The FastAPI route handler is a one-liner.
    """

    def __init__(
        self,
        concurrency: "ConcurrencyManager",
        cuda: "Cuda",
        audio: "Audio",
        side_file: "SideFile",
        orchestrator: "Orchestrator",
    ) -> None:
        self._concurrency = concurrency
        self._cuda = cuda
        self._audio = audio
        self._side_file = side_file
        self._orchestrator = orchestrator

    async def handle(
        self,
        audio_file: UploadFile,
        task: str,
        language: str,
        output: str,
        encode: bool,
        video_file: str = "",
    ) -> PlainTextResponse:
        """Process one ASR request. Returns the rendered subtitle response.

        ``video_file`` is the Bazarr-provided path used for side-file saves.
        Passed explicitly so the handler has no dependency on the HTTP
        ``Request`` object and can be called from a CLI with explicit args.
        """
        request_id = uuid.uuid4().hex[:8]
        workdir = Path(tempfile.mkdtemp(prefix=f"asr_{request_id}_"))
        raw_path = workdir / (audio_file.filename or "input")

        self._concurrency.inc_jobs()
        started = time.monotonic()
        try:
            with raw_path.open("wb") as f:
                shutil.copyfileobj(audio_file.file, f)
            log.info("[%s] received %s (task=%s lang=%s output=%s encode=%s)",
                     request_id, raw_path.name, task, language, output, encode)

            audio_path = workdir / "audio.wav"
            if encode:
                await self._concurrency.run_in_thread(self._audio.encode_to_wav, raw_path, audio_path)
            else:
                await self._concurrency.run_in_thread(self._audio.prepare_unencoded, raw_path, audio_path)

            alt_target = None  # initialized for the post-semaphore block

            async with self._concurrency.semaphore:
                # Reclaim cached-but-unused VRAM left by a previous request before
                # touching the GPU. The PyTorch caching allocator does not return
                # memory between requests on its own, and Whisper large-v3 + pyannote
                # + NLLB resident together leave only ~3 GB of slack on an 8 GB card.
                # Without this, a back-to-back /asr has been observed to OOM in the
                # transcribe step (CUDA error: out of memory) even though both
                # requests fit individually when the process is fresh.
                self._cuda.empty_cache()
                # Capture primary pipeline artifacts (raw segments + per-chunk
                # diarization annotations) only when the A/B alt pass will
                # actually run — keeps memory cost of the out-container at zero
                # for typical requests where GENDER_AB_OUTPUT is off.
                artifacts_out: list[PipelineArtifacts] | None = (
                    [] if settings.GENDER_AB_OUTPUT else None
                )
                source_segments, target_segments = await self._orchestrator.run_pipeline(
                    audio_path, language,
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
                        _, alt_target = await self._orchestrator.run_pipeline_alt_classifier(
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
            video_file_url = video_file

            if target_segments is not None:
                # Translated run: render the translation separately for the side-
                # file save, and build the summary from the translated segments so
                # the script-check ratio remains meaningful.
                target_body, _ = render(target_segments, output)
                backend = settings.TRANSLATION_BACKEND
                backend_model = self._orchestrator._backend.model_name()
                summary = self._side_file.build_summary(
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
                await self._concurrency.run_in_thread(
                    self._side_file.try_save, target_body, summary, video_file_url,
                )

                # Alt side-file save (outside semaphore — just I/O, no global
                # mutation). alt_target is only non-None when GENDER_AB_OUTPUT is
                # on and the alt pass succeeded.
                if alt_target is not None:
                    alt_body, _ = render(alt_target, output)
                    await self._concurrency.run_in_thread(
                        self._side_file.try_save,
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
            self._concurrency.dec_jobs()
            shutil.rmtree(workdir, ignore_errors=True)

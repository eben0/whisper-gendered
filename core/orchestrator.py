"""Pipeline orchestration: transcribe → (diarize+gender) ⇄ translate, overlapped.

Web-agnostic. A future CLI calls ``run_pipeline_async`` directly. Blocking ML work is
dispatched via ``core.concurrency.run_in_thread`` and gated by the caller's semaphore.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from config import settings
from core import audio, backends, concurrency
from core.artifacts import PipelineArtifacts
from pipeline import diarize, gender, transcribe
from pipeline.chunk import make_chunks
from pipeline.lang import language_name
from pipeline.segment import Segment

log = logging.getLogger("core.orchestrator")


def _group_consecutive(items: list[tuple[Segment, str | None]]) -> list[tuple[str | None, list[Segment]]]:
    """Group consecutive (segment, speaker) pairs by speaker, preserving order."""
    groups: list[tuple[str | None, list[Segment]]] = []
    for seg, speaker in items:
        if groups and groups[-1][0] == speaker:
            groups[-1][1].append(seg)
        else:
            groups.append((speaker, [seg]))
    return groups


def _diarize_and_gender(audio: Any, sr: int, start: float, end: float):
    """Diarize one chunk's audio slice and detect per-speaker gender.

    Runs in a worker thread (GPU work). The returned annotation and the gender
    map are both in slice-local time (turn times relative to ``start``).
    """
    import numpy as np
    i0 = max(0, int(start * sr))
    i1 = min(len(audio), int(end * sr))
    slice_ = audio[i0:i1]
    annotation = diarize.diarize_waveform(slice_, sr)
    genders = gender.detect_genders(slice_, sr, annotation)
    return annotation, genders


async def _translate_chunk(
    idx: int,
    groups: list[tuple[str | None, list[Segment]]],
    genders: dict[str, str],
    target: str,
    sem: asyncio.Semaphore,
    prev_speaker_gender: str | None,
    source_language: str = "English",
    context_window: list[tuple[str | None, str]] | None = None,
) -> None:
    """Translate one chunk's pre-built speaker groups in place.

    ``groups`` is the output of ``_group_consecutive`` for this chunk's segments
    (built by the orchestrator so it can also derive the chunk's last-group
    gender for cross-chunk carry). ``prev_speaker_gender`` seeds the addressee
    rotation: the first group's addressee is "whoever spoke before this chunk"
    (or ``None`` for the very first chunk). ``source_language`` is the
    request's audio language (display name) used in the translation prompt.

    ``context_window`` (Plan Task 7) is a SHARED mutable rolling buffer of the
    most recent source-text lines seen by the orchestrator. Each group's
    translate call receives an immutable snapshot of the window at call time as
    ``previous_context``; after the call, this function appends the group's
    source texts and trims to ``settings.TRANSLATE_CONTEXT_LINES`` so the next
    group/chunk sees the updated window. Pass the SAME list reference to every
    chunk's task so cross-chunk flow accumulates. ``None`` disables the
    feature (no snapshot taken, no mutation).

    Concurrency caveat: with ``TRANSLATE_CONCURRENCY > 1`` chunks run in
    parallel and the order in which groups extend ``context_window`` becomes
    non-deterministic — the window then degrades to a best-effort temporal
    ordering, which is still a useful LLM context cue but not a strict
    transcript prefix. Tests pin ``TRANSLATE_CONCURRENCY=1`` to keep
    assertions deterministic.
    """
    async with sem:
        prev_group_gender = prev_speaker_gender
        try:
            for speaker, group in groups:
                spk_gender = genders.get(speaker, "male") if speaker else "male"
                # Pass the addressee hint only if the feature flag is on; when
                # off the broader "you"-form guidance in the system prompt still
                # applies, but no specific addressee gender is asserted.
                addressee = (
                    prev_group_gender
                    if settings.ADDRESSEE_GENDER_HINT_ENABLED
                    else None
                )
                source_texts = [s.text for s in group]
                ctx_snapshot = (
                    list(context_window) if context_window else []
                )
                # DIAGNOSTIC (temporary): log the exact context handed to
                # Claude alongside what's being translated. DEBUG-level so
                # subtitle text doesn't leak into default prod logs; flip
                # DEBUG=true in .env to re-enable when investigating an
                # addressee or context-window issue (e.g. the Oz S04E05
                # line-52 case that motivated this log).
                log.debug(
                    "[chunk %d] BATCH speaker=%s addressee=%s "
                    "ctx_len=%d ctx=%r src=%r",
                    idx, spk_gender, addressee, len(ctx_snapshot),
                    ctx_snapshot, source_texts,
                )
                translated = await backends.backend.translate_batch_async(
                    source_texts, spk_gender, target,
                    addressee_gender=addressee,
                    source_language=source_language,
                    previous_context=ctx_snapshot,
                )
                for seg, text in zip(group, translated):
                    seg.text = text
                if context_window is not None:
                    # Capture TARGET-language text plus this group's speaker
                    # gender. Target text is chosen on purpose: gendered
                    # languages encode the addressee in their verb forms, so
                    # later batches can reconstruct who was addressing whom —
                    # source English ``"you"`` reveals nothing about
                    # addressee gender. See ``pipeline/translate.ContextLine``.
                    context_window.extend(
                        (spk_gender, t) for t in translated
                    )
                    max_n = settings.TRANSLATE_CONTEXT_LINES
                    if max_n <= 0:
                        del context_window[:]
                    elif len(context_window) > max_n:
                        del context_window[: len(context_window) - max_n]
                prev_group_gender = spk_gender
        except Exception:
            log.exception("[chunk %d] translation failed", idx)
            raise


async def _run_gender_aware(
    audio_path: Path,
    segments: list[Segment],
    target: str,
    source_language: str = "English",
    precomputed_annotations: list[Any] | None = None,
) -> tuple[list[Segment], list[Any]]:
    """Gender-aware chunked translate. Returns ``(segments, annotations_used)``.

    When ``precomputed_annotations`` is provided (A/B alt path), reuses
    those instead of running pyannote diarization per chunk — saves the
    ~2-3 minutes of GPU time and the VRAM pressure of a second diarize.
    Gender classification still runs on every call (it's cheap, and the
    classifier may have been flipped between passes).
    """
    waveform, sr = await concurrency.run_in_thread(audio._load_wav_mono, audio_path)
    chunks = make_chunks(segments, settings.CHUNK_DURATION_SEC)
    annotations_used: list[Any] = []
    sem = asyncio.Semaphore(settings.TRANSLATE_CONCURRENCY)
    # Per-request rolling window of recent source-language lines, SHARED
    # across all chunks so chunk N's last group's source text flows into
    # chunk N+1's first call as previous_context. This is safe because the
    # window holds SOURCE text — unlike per-chunk speaker labels, source
    # text has no pyannote-renumbering problem at chunk boundaries.
    # ``None`` disables the feature entirely (TRANSLATE_CONTEXT_LINES=0).
    context_window: list[tuple[str | None, str]] | None = (
        [] if settings.TRANSLATE_CONTEXT_LINES > 0 else None
    )
    tasks: list[asyncio.Task] = []
    t1 = time.monotonic()
    try:
        for idx, chunk in enumerate(chunks):
            if precomputed_annotations is not None:
                # A/B alt path: reuse the primary pass's diarization. Only
                # re-run gender classification (cheap) against the current
                # ``settings.GENDER_CLASSIFIER`` which the alt wrapper has
                # flipped. The audio slice is regenerated locally because it's
                # cheap (a numpy view, not a copy).
                annotation = precomputed_annotations[idx]
                i0 = max(0, int(chunk.start * sr))
                i1 = min(len(waveform), int(chunk.end * sr))
                slice_ = waveform[i0:i1]
                genders = await concurrency.run_in_thread(
                    gender.detect_genders, slice_, sr, annotation,
                )
                log.info(
                    "chunk %d/%d alt-pass gender only (%d speakers; "
                    "diarization reused from primary)",
                    idx + 1, len(chunks), len(genders),
                )
            else:
                annotation, genders = await concurrency.run_in_thread(
                    _diarize_and_gender, waveform, sr, chunk.start, chunk.end
                )
                log.info(
                    "chunk %d/%d diarize+gender done (%d speakers)",
                    idx + 1, len(chunks), len(genders),
                )
            annotations_used.append(annotation)
            # Build chunk-local assignment + groups.
            assigned: list[tuple[Segment, str | None]] = []
            for seg in chunk.segments:
                local = Segment(seg.start - chunk.start, seg.end - chunk.start, seg.text)
                assigned.append((seg, diarize.assign_speaker(local, annotation)))
            groups = _group_consecutive(assigned)
            tasks.append(asyncio.create_task(
                _translate_chunk(
                    # ``prev_speaker_gender`` is intentionally hard-coded
                    # to None here (Plan Task 7). pyannote re-numbers
                    # speakers per chunk, so the previous chunk's last
                    # speaker label cannot be reliably matched against
                    # the next chunk's first speaker label — carrying
                    # the gender across the boundary was wrong as often
                    # as it was right. Within-chunk rotation in
                    # ``_translate_chunk`` is unaffected.
                    idx, groups, genders, target, sem,
                    None, source_language,
                    context_window=context_window,
                )
            ))
        await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    log.info("gender-aware chunks complete: %.1fs", time.monotonic() - t1)
    return [seg for chunk in chunks for seg in chunk.segments], annotations_used


async def _run_plain_translate(
    segments: list[Segment],
    target: str,
    source_language: str = "English",
) -> list[Segment]:
    chunks = make_chunks(segments, settings.CHUNK_DURATION_SEC)
    sem = asyncio.Semaphore(settings.TRANSLATE_CONCURRENCY)
    # Same rolling-window pattern as ``_run_gender_aware`` (Plan Task 7):
    # a single shared list flows across all chunks of the request. Best-effort
    # temporal order when TRANSLATE_CONCURRENCY > 1; deterministic at 1.
    context_window: list[tuple[str | None, str]] | None = (
        [] if settings.TRANSLATE_CONTEXT_LINES > 0 else None
    )

    async def translate_one(chunk):
        async with sem:
            source_texts = [s.text for s in chunk.segments]
            ctx_snapshot = list(context_window) if context_window else []
            # DEBUG-level diagnostic — see _run_gender_aware sibling for context.
            log.debug(
                "PLAIN BATCH ctx_len=%d ctx=%r src=%r",
                len(ctx_snapshot), ctx_snapshot, source_texts,
            )
            translated = await backends.backend.translate_batch_async(
                source_texts, None, target,
                source_language=source_language,
                previous_context=ctx_snapshot,
            )
            for seg, text in zip(chunk.segments, translated):
                seg.text = text
            if context_window is not None:
                # Plain-translate has no speaker gender → push (None, text).
                # See ``_run_gender_aware`` for the rationale on capturing
                # target-language text rather than source.
                context_window.extend((None, t) for t in translated)
                max_n = settings.TRANSLATE_CONTEXT_LINES
                if max_n <= 0:
                    del context_window[:]
                elif len(context_window) > max_n:
                    del context_window[: len(context_window) - max_n]

    tasks = [asyncio.create_task(translate_one(c)) for c in chunks]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [seg for chunk in chunks for seg in chunk.segments]


async def run_pipeline_async(
    audio_path: Path, language: str,
    _artifacts_out: list[PipelineArtifacts] | None = None,
) -> tuple[list[Segment], list[Segment] | None]:
    """Transcribe and (optionally) translate, overlapping diarize and translate.

    Returns ``(source_segments, target_segments_or_None)``. ``source_segments``
    is the raw transcript (in the request's ``language``); ``target_segments``
    is the translation when ``TARGET_LANGUAGE`` is enabled, else ``None``.

    Keeping both lists lets the HTTP response return the *source* transcript —
    matching whatever language Bazarr asked for in ``?language=en`` — while the
    translation is written only to the configured side-file. Without this,
    Bazarr's whisperai plugin stores our response body to disk under a filename
    derived from the requested language (e.g. ``*.en.srt``), and a Hebrew body
    would clobber any pre-existing English subtitle there.

    ``_artifacts_out`` is an optional out-container: when the caller passes
    an empty list, this function appends a single ``PipelineArtifacts``
    holding the raw source segments and per-chunk diarization annotations.
    Used by the /asr handler so the A/B alt-classifier pass can skip
    re-transcribing and re-diarizing (which OOM'd on 8GB cards). Tests
    leave the parameter at its default and observe no behavior change.
    """
    t0 = time.monotonic()
    segments = await concurrency.run_in_thread(transcribe.transcribe, audio_path, language)
    transcribe_elapsed = time.monotonic() - t0
    # Two log lines: the historical timing line, then an explicit-count
    # line worded for grep ("transcribed N segments"). The count line
    # pairs with the post-translate counter below so a missing-segment
    # investigation can simply diff the two numbers.
    log.info("transcribe: %d segments in %.1fs",
             len(segments), transcribe_elapsed)
    log.info("transcribed %d segments from audio (input count for translate)",
             len(segments))

    if not settings.translation_enabled() or not segments:
        return segments, None

    # Snapshot the source-language text BEFORE the pipeline mutates each
    # Segment in place. The chunk/orchestrator layers reuse the input
    # Segment instances (``make_chunks`` does not copy), so by the time the
    # translation finishes ``segments[i].text`` holds the target language.
    # Reconstructing the source list from this snapshot is cheaper and less
    # invasive than threading a "no-mutation" mode through three call sites.
    source_texts = [s.text for s in segments]

    target = settings.TARGET_LANGUAGE
    # The audio's source language: Bazarr sends an ISO 639-1 code in `language`;
    # convert to a display name for the translation prompts. Falls back to
    # "English" on empty/unknown inputs (the safe pre-feature default).
    source_language = language_name(language)
    annotations: list[Any] = []
    if settings.is_gender_aware():
        target_segments, annotations = await _run_gender_aware(
            audio_path, segments, target, source_language,
        )
    else:
        target_segments = await _run_plain_translate(
            segments, target, source_language,
        )

    # Pipeline metrics — pair with the post-transcribe count above.
    # A mismatch means something between transcribe and translate dropped
    # segments (Claude returning a shorter JSON array than asked, chunking
    # losing a segment at a boundary, etc.). Surface it at ERROR so an
    # operator notices without grepping; the per-line counts are at INFO
    # for the happy path.
    diff = len(segments) - len(target_segments)
    log.info("translated %d segments (input was %d; difference = %d)",
             len(target_segments), len(segments), diff)
    if diff != 0:
        log.error(
            "segment count mismatch — pipeline lost %d segments between "
            "transcribe and translate (input=%d, output=%d). Translation "
            "JSON length mismatch or chunk-boundary drop likely; check "
            "_translate_chunk in server.py and the alignment fallback in "
            "pipeline/translate.py.",
            diff, len(segments), len(target_segments),
        )

    # Pair the snapshotted source text with the (now-translated) instants so
    # both lists share start/end stamps but carry different ``.text``. New
    # Segment instances guarantee independence: a later mutation to one list
    # cannot leak into the other. ``zip`` truncates to the shorter list,
    # so a mismatch (already flagged above) results in a shorter source
    # list rather than a hard error.
    source_segments = [
        Segment(t.start, t.end, src)
        for t, src in zip(target_segments, source_texts)
    ]
    if _artifacts_out is not None:
        # Hand the primary pass's reusable intermediates back to the caller
        # (the /asr handler) so an A/B alt-classifier pass can skip
        # transcribe + diarize. ``raw_segments`` are fresh instances built
        # from the pre-translation source text so a later alt-pass
        # mutation can't leak into ``source_segments``.
        _artifacts_out.append(PipelineArtifacts(
            raw_segments=[
                Segment(s.start, s.end, src)
                for s, src in zip(segments, source_texts)
            ],
            annotations=annotations,
        ))
    return source_segments, target_segments


async def run_pipeline_alt_classifier(
    audio_path: Path, language: str, artifacts: PipelineArtifacts,
) -> tuple[list[Segment], list[Segment] | None]:
    """A/B alt pass: re-translate using the alternate gender classifier,
    REUSING the primary pass's transcribe and diarize results.

    Skips Whisper (~4 min on this hardware) and pyannote (~2 min). Only
    re-runs the cheap gender classification and the LLM translation per
    chunk with the flipped ``GENDER_CLASSIFIER``. Without this reuse the
    second pass triggers a back-to-back Whisper inference and CUDA OOMs
    on 8 GB cards — that's the bug this function exists to fix.

    ``artifacts`` is the value the primary ``run_pipeline_async`` call
    appended into the caller's ``_artifacts_out`` list.

    Concurrency: this function mutates the process-global
    ``settings.GENDER_CLASSIFIER`` for the duration of the call. The
    caller MUST hold the GPU semaphore (or otherwise serialize against
    other pipeline invocations) before calling, otherwise a concurrent
    request could observe the flipped value and use the wrong classifier
    on its own pipeline pass. The mutation is try/finally-restored, but
    that only protects the *post*-call state, not concurrent in-flight
    work.
    """
    primary = settings.GENDER_CLASSIFIER.strip().lower()
    alt = {
        "pitch": "ml",
        "ml": "pitch",
        "ensemble": "pitch",   # alt of ensemble is pitch-only for clarity
    }.get(primary, "ml")
    old = settings.GENDER_CLASSIFIER
    try:
        settings.GENDER_CLASSIFIER = alt

        # Build fresh source segments so this pass's in-place text mutation
        # doesn't touch the primary's outputs (or the artifacts themselves —
        # the caller may want to re-invoke alt on the same artifacts).
        fresh_segments = [
            Segment(s.start, s.end, s.text) for s in artifacts.raw_segments
        ]
        if not settings.translation_enabled() or not fresh_segments:
            return fresh_segments, None

        target = settings.TARGET_LANGUAGE
        source_language = language_name(language)

        if settings.is_gender_aware():
            # The win: pass precomputed_annotations so _run_gender_aware
            # skips the expensive per-chunk diarize call.
            target_segments, _ = await _run_gender_aware(
                audio_path, fresh_segments, target, source_language,
                precomputed_annotations=artifacts.annotations,
            )
        else:
            # Non-gender-aware target: gender label is unused by the
            # prompt, so flipping the classifier produces the same output
            # as the primary. Still translate to keep the contract (caller
            # may rely on a non-None second SRT).
            target_segments = await _run_plain_translate(
                fresh_segments, target, source_language,
            )

        # Reconstruct source from the artifact's raw_segments (which carry
        # the un-translated text — fresh_segments has been mutated above).
        source_segments = [
            Segment(t.start, t.end, raw.text)
            for t, raw in zip(target_segments, artifacts.raw_segments)
        ]
        return source_segments, target_segments
    finally:
        settings.GENDER_CLASSIFIER = old

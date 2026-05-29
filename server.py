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

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _register_cuda_dll_dirs() -> None:
    """Add the pip-installed NVIDIA cuBLAS/cuDNN bin dirs to the DLL search path.

    faster-whisper transcribes via CTranslate2, which dynamically loads
    cublas64_12.dll / cudnn_*.dll. faster-whisper is supposed to register the
    nvidia-*-cu12 wheel directories automatically, but that doesn't always fire
    on Windows, producing 'Library cublas64_12.dll is not found'. Registering
    them here (before CTranslate2 loads) makes the GPU transcription path robust
    regardless of how the server was launched.
    """
    if not sys.platform.startswith("win"):
        return
    site_nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn"):
        bin_dir = site_nvidia / sub / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


_register_cuda_dll_dirs()


def _preimport_librosa() -> None:
    """Force librosa's lazily-loaded audio core to import before pyannote does.

    librosa uses lazy_loader: the first access to `librosa.load` imports
    `librosa.core.audio`, whose module body runs `lazy.load("samplerate")` ->
    `inspect.stack()`. pyannote pulls in speechbrain, which registers a lazy
    `speechbrain.integrations.k2_fsa` module in sys.modules whose __getattr__
    eagerly imports the (Windows-unavailable) `k2` package. If that lazy
    speechbrain module is present when inspect walks sys.modules, the frame walk
    triggers `import k2` and the whole librosa.load call dies with
    'Please install k2'. Importing librosa here -- before speechbrain exists --
    runs that inspect.stack() against a clean module table once, so later
    librosa.load / librosa.pyin calls never re-trigger it.
    """
    import librosa  # noqa: F401

    _ = librosa.load
    _ = librosa.pyin


_preimport_librosa()

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from config import settings
from pipeline import diarize, gender, transcribe
# Backend factory: at import time, ``translate`` points at the module whose
# ``translate_batch_async`` matches the requested TRANSLATION_BACKEND. Both
# modules expose the same call signature, so the orchestrator's call site at
# ``translate.translate_batch_async(...)`` is unchanged either way.
# ``client`` is ignored by the local backend; the Claude backend uses it.
if settings.TRANSLATION_BACKEND.strip().lower() == "local":
    from pipeline import translate_local as translate  # type: ignore[no-redef]
else:
    from pipeline import translate  # type: ignore[no-redef]
from pipeline.chunk import make_chunks
from pipeline.format import render
from pipeline.lang import language_name, target_script_ratio
from pipeline.transcribe import Segment

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("server")

VERSION = "1.0.0"

app = FastAPI(title="Gender-Aware Hebrew Subtitle Server", version=VERSION)

# Gate concurrent GPU jobs; acquire BEFORE dispatching to the executor.
_semaphore = asyncio.Semaphore(settings.CONCURRENT_JOBS)
_executor = ThreadPoolExecutor(max_workers=max(2, settings.CONCURRENT_JOBS + 1))
_jobs_in_system = 0  # queued + running, for /status

async def run_in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


_async_anthropic_client = None


def get_async_anthropic_client():
    global _async_anthropic_client
    if _async_anthropic_client is None:
        import anthropic
        _async_anthropic_client = anthropic.AsyncAnthropic(
            api_key=settings.require_anthropic_key(),
            max_retries=settings.CLAUDE_MAX_RETRIES,
        )
    return _async_anthropic_client


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #

def encode_to_wav(src: Path, dst: Path) -> None:
    """Re-encode any input to 16 kHz mono WAV via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-15:]
        raise RuntimeError(
            "ffmpeg failed (exit %d) encoding %s:\n%s"
            % (proc.returncode, src.name, "\n".join(stderr_tail))
        )


def prepare_unencoded(src: Path, dst: Path, sr: int = 16000) -> None:
    """Turn an ``encode=false`` upload into a 16 kHz mono WAV at ``dst``.

    Per the whisper-asr-webservice contract that Bazarr follows, ``encode=false``
    means the body is *headerless* raw 16-bit little-endian PCM (mono, 16 kHz) --
    the reference server reads it straight as ``np.int16``. faster-whisper's
    container decoder rejects that (AVERROR_INVALIDDATA), so we wrap it in a real
    WAV here. A few clients instead send an actual container (WAV/FLAC); we detect
    that by trying to open it first and only fall back to raw PCM if that fails.
    """
    try:
        with sf.SoundFile(str(src)):
            pass
        shutil.copyfile(src, dst)  # real container; downstream handles sr/channels
        return
    except (RuntimeError, sf.LibsndfileError):
        pass
    pcm = np.frombuffer(src.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
    sf.write(str(dst), pcm, sr, subtype="PCM_16")


def _write_silent_wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)


def _empty_cuda_cache() -> None:
    """Best-effort release of cached-but-unused CUDA memory.

    Cheap when no CUDA is present (or torch isn't loaded yet), never raises.
    Called at the start of each /asr request to defragment the allocator
    between requests — see the call site comment for the OOM rationale.
    """
    try:
        import torch  # local import: keeps module-load time unchanged for tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - the call is purely best-effort
        log.debug("cuda.empty_cache failed (continuing)", exc_info=True)


def _compute_side_file_path(
    video_file_url: str,
    src_prefix: str,
    dst_prefix: str,
    suffix: str,
) -> Path | None:
    """Translate the Bazarr ``video_file`` URL into a local SRT path on the share.

    Returns ``None`` when the feature is disabled (either prefix empty), the URL
    is empty, or the URL does not start with the configured source prefix.
    Pure function — no filesystem I/O — so it is unit-testable without mounting
    a share. The actual write happens in ``_try_save_side_file``.
    """
    if not src_prefix or not dst_prefix or not video_file_url:
        return None
    import urllib.parse
    decoded = urllib.parse.unquote(video_file_url)
    if not decoded.startswith(src_prefix):
        return None
    rel = decoded[len(src_prefix):].replace("/", "\\")
    local = Path(dst_prefix + rel)
    stem = local.with_suffix("")
    return stem.with_name(stem.name + suffix)


def _compute_summary_path(srt_path: Path) -> Path:
    """Derive the summary file path from an SRT path, paired by name.

    ``episode.he.srt`` -> ``episode.he.summary.txt`` (preserving the language
    suffix so the pair stays visually grouped in a directory listing).
    Non-``.srt`` inputs just get ``.summary.txt`` appended.

    Pure function so the naming contract is testable without filesystem I/O.
    """
    name = srt_path.name
    if name.lower().endswith(".srt"):
        return srt_path.with_name(name[: -len(".srt")] + ".summary.txt")
    return srt_path.with_name(name + ".summary.txt")


def _try_save_side_file(
    body: str,
    summary: str | None,
    video_file_url: str,
    suffix: str | None = None,
) -> None:
    """Write the SRT body — and optionally a sibling summary file — to the
    translated path next to the source video.

    ``suffix`` overrides ``settings.SAVE_SRT_SUFFIX`` when set. Used by Plan
    Task 5 to emit a parallel alt-classifier SRT (``*.he.alt-classifier.srt``)
    for A/B comparison.

    Failures (missing share, permission denied, malformed URL) log a warning
    and do not affect the HTTP response. Atomic-ish: write to ``<target>.tmp``
    first, then ``os.replace`` onto the final name. The summary file uses the
    same atomic pattern with a ``.summary.txt`` sibling derived via
    ``_compute_summary_path`` so e.g. ``Show.S01E01.he.srt`` pairs with
    ``Show.S01E01.he.summary.txt``.
    """
    effective_suffix = suffix or settings.SAVE_SRT_SUFFIX
    try:
        target = _compute_side_file_path(
            video_file_url,
            settings.SAVE_SRT_VIDEO_PREFIX,
            settings.SAVE_SRT_LOCAL_PREFIX,
            effective_suffix,
        )
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        import os as _os
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        _os.replace(str(tmp), str(target))
        log.info("side-file saved: %s", target)
        if summary:
            summary_target = _compute_summary_path(target)
            tmp_sum = summary_target.with_name(summary_target.name + ".tmp")
            tmp_sum.write_text(summary, encoding="utf-8")
            _os.replace(str(tmp_sum), str(summary_target))
            log.info("side-file summary saved: %s", summary_target)
    except Exception:
        log.exception("side-file save failed; continuing without it")


def _build_translation_summary(
    *,
    request_id: str,
    video_file_url: str,
    source_language_iso: str,
    source_language_name: str,
    target_language: str,
    backend: str,
    backend_model: str | None,
    segments: list[Segment],
    wall_seconds: float,
) -> str:
    """Produce a multi-line human-readable summary of a completed translation.

    Designed to live next to the produced SRT and to be eyeball-checkable
    without opening the full subtitle file. The most important line is the
    target-script ratio — a ratio < ~0.5 on a known non-Latin target almost
    always indicates a backend mis-translation (we hit exactly this when
    NLLB's ``forced_bos_token_id`` plumbing broke and the model produced
    Spanish/Romanian/Tswana instead of Hebrew).
    """
    import urllib.parse

    # Joined text for ratio/length stats. ``\n`` joins keep per-line counts
    # roughly comparable to the SRT body.
    text = "\n".join(s.text for s in segments)
    total_chars = len(text)
    n_segs = len(segments)
    ratio = target_script_ratio(text, target_language)

    if ratio is None:
        script_line = (
            f"Script check:    n/a (target language has no non-Latin script signal)"
        )
    else:
        pct = ratio * 100
        flag = " ✓" if ratio >= 0.50 else " ⚠ WRONG LANGUAGE?"
        script_line = (
            f"Script check:    {pct:5.1f}% of letters are in the {target_language} "
            f"script{flag}"
        )

    # Sample lines — first, two from middle quarters, last. Useful for a quick
    # eyeball without opening the file.
    samples: list[tuple[str, Segment]] = []
    if n_segs:
        idxs = sorted({0, n_segs // 4, (3 * n_segs) // 4, n_segs - 1})
        labels = ["first", "early", "late", "last"]
        for label, idx in zip(labels, idxs):
            samples.append((label, segments[idx]))

    sample_block = ""
    if samples:
        rows: list[str] = []
        for label, seg in samples:
            ts = _fmt_timestamp(seg.start)
            # Truncate very long lines to keep the file scannable.
            shown = seg.text if len(seg.text) <= 120 else seg.text[:117] + "..."
            rows.append(f"  {label:<6} ({ts}) {shown}")
        sample_block = "Sample lines:\n" + "\n".join(rows)

    decoded_video = (
        urllib.parse.unquote(video_file_url) if video_file_url else "(none)"
    )
    backend_str = f"{backend}" + (f" ({backend_model})" if backend_model else "")

    lines = [
        "=== Translation summary ===",
        f"Request:         {request_id}",
        f"Video:           {decoded_video}",
        f"Source language: {source_language_name} (lang={source_language_iso!r})",
        f"Target language: {target_language}",
        f"Backend:         {backend_str}",
        f"Segments:        {n_segs}",
        f"Output chars:    {total_chars}",
        script_line,
        f"Wall time:       {wall_seconds:.1f}s ({int(wall_seconds)//60:02d}:{int(wall_seconds)%60:02d})",
    ]
    if sample_block:
        lines.append("")
        lines.append(sample_block)
    return "\n".join(lines) + "\n"


def _fmt_timestamp(t: float) -> str:
    """Render a float-seconds time as ``HH:MM:SS`` for the summary file."""
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --------------------------------------------------------------------------- #
# Pipeline (runs entirely inside a worker thread)
# --------------------------------------------------------------------------- #

def _group_consecutive(items: list[tuple[Segment, str | None]]) -> list[tuple[str | None, list[Segment]]]:
    """Group consecutive (segment, speaker) pairs by speaker, preserving order."""
    groups: list[tuple[str | None, list[Segment]]] = []
    for seg, speaker in items:
        if groups and groups[-1][0] == speaker:
            groups[-1][1].append(seg)
        else:
            groups.append((speaker, [seg]))
    return groups


def _load_wav_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as a mono float32 waveform + sample rate."""
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _diarize_and_gender(audio: np.ndarray, sr: int, start: float, end: float):
    """Diarize one chunk's audio slice and detect per-speaker gender.

    Runs in a worker thread (GPU work). The returned annotation and the gender
    map are both in slice-local time (turn times relative to ``start``).
    """
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
    client,
    sem: asyncio.Semaphore,
    prev_speaker_gender: str | None,
    source_language: str = "English",
    context_window: list[str] | None = None,
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
                # Capture source text BEFORE mutating seg.text below — otherwise
                # the rolling window would accumulate target-language strings.
                source_texts = [s.text for s in group]
                ctx_snapshot = (
                    list(context_window) if context_window else []
                )
                translated = await translate.translate_batch_async(
                    source_texts, spk_gender, target, client,
                    addressee_gender=addressee,
                    source_language=source_language,
                    previous_context=ctx_snapshot,
                )
                for seg, text in zip(group, translated):
                    seg.text = text
                if context_window is not None:
                    context_window.extend(source_texts)
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
    client,
    source_language: str = "English",
) -> list[Segment]:
    audio, sr = await run_in_thread(_load_wav_mono, audio_path)
    chunks = make_chunks(segments, settings.CHUNK_DURATION_SEC)
    sem = asyncio.Semaphore(settings.TRANSLATE_CONCURRENCY)
    # Per-request rolling window of recent source-language lines, SHARED
    # across all chunks so chunk N's last group's source text flows into
    # chunk N+1's first call as previous_context. This is safe because the
    # window holds SOURCE text — unlike per-chunk speaker labels, source
    # text has no pyannote-renumbering problem at chunk boundaries.
    # ``None`` disables the feature entirely (TRANSLATE_CONTEXT_LINES=0).
    context_window: list[str] | None = (
        [] if settings.TRANSLATE_CONTEXT_LINES > 0 else None
    )
    tasks: list[asyncio.Task] = []
    t1 = time.monotonic()
    try:
        for idx, chunk in enumerate(chunks):
            annotation, genders = await run_in_thread(
                _diarize_and_gender, audio, sr, chunk.start, chunk.end
            )
            log.info(
                "chunk %d/%d diarize+gender done (%d speakers)",
                idx + 1, len(chunks), len(genders),
            )
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
                    idx, groups, genders, target, client, sem,
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
    return [seg for chunk in chunks for seg in chunk.segments]


async def _run_plain_translate(
    segments: list[Segment],
    target: str,
    client,
    source_language: str = "English",
) -> list[Segment]:
    chunks = make_chunks(segments, settings.CHUNK_DURATION_SEC)
    sem = asyncio.Semaphore(settings.TRANSLATE_CONCURRENCY)
    # Same rolling-window pattern as ``_run_gender_aware`` (Plan Task 7):
    # a single shared list flows across all chunks of the request. Best-effort
    # temporal order when TRANSLATE_CONCURRENCY > 1; deterministic at 1.
    context_window: list[str] | None = (
        [] if settings.TRANSLATE_CONTEXT_LINES > 0 else None
    )

    async def translate_one(chunk):
        async with sem:
            # Capture source BEFORE mutation so the window doesn't accumulate
            # target-language text.
            source_texts = [s.text for s in chunk.segments]
            ctx_snapshot = list(context_window) if context_window else []
            translated = await translate.translate_batch_async(
                source_texts, None, target, client,
                source_language=source_language,
                previous_context=ctx_snapshot,
            )
            for seg, text in zip(chunk.segments, translated):
                seg.text = text
            if context_window is not None:
                context_window.extend(source_texts)
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
    audio_path: Path, language: str
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
    """
    t0 = time.monotonic()
    segments = await run_in_thread(transcribe.transcribe, audio_path, language)
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
    # Only the Claude backend needs an Anthropic client; the local backend
    # ignores the parameter. Skipping the call also avoids requiring
    # ANTHROPIC_API_KEY when running fully on-device.
    if settings.TRANSLATION_BACKEND.strip().lower() == "local":
        client = None
    else:
        client = get_async_anthropic_client()
    if settings.is_gender_aware():
        target_segments = await _run_gender_aware(
            audio_path, segments, target, client, source_language,
        )
    else:
        target_segments = await _run_plain_translate(
            segments, target, client, source_language,
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
    return source_segments, target_segments


async def run_pipeline_alt_classifier(
    audio_path: Path, language: str,
) -> tuple[list[Segment], list[Segment] | None]:
    """Same as ``run_pipeline_async`` but flips ``GENDER_CLASSIFIER`` to
    the alternate of whatever's currently configured, runs the pipeline,
    and restores the original value.

    Cost: one extra full pipeline pass per request. Only invoked when
    ``GENDER_AB_OUTPUT=true``; defaults are off.

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
        return await run_pipeline_async(audio_path, language)
    finally:
        settings.GENDER_CLASSIFIER = old


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

@app.on_event("startup")
async def warmup() -> None:
    log.info(
        "Starting up. model=%s device=%s target_language=%s gender_aware=%s "
        "translation_backend=%s",
        settings.WHISPER_MODEL, settings.DEVICE, settings.TARGET_LANGUAGE,
        settings.is_gender_aware(), settings.TRANSLATION_BACKEND,
    )
    tmp = Path(tempfile.gettempdir()) / f"warmup_{uuid.uuid4().hex}.wav"
    try:
        _write_silent_wav(tmp)
        await run_in_thread(transcribe.warmup, tmp)
        if settings.is_gender_aware():
            try:
                await run_in_thread(diarize.diarize, tmp)
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
                await run_in_thread(gender_ml.warmup)
            except Exception:
                log.exception("gender_ml warm-up failed; falling back to lazy load.")
        # Local translation model is large (NLLB-200 distilled is ~2.4 GB) and
        # would otherwise load on the first /asr request — well past Bazarr's
        # client timeout. Pre-load it here. Claude backend has nothing to warm.
        if settings.TRANSLATION_BACKEND.strip().lower() == "local":
            await run_in_thread(translate.warmup)
    finally:
        tmp.unlink(missing_ok=True)


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
            "queue_depth": _jobs_in_system,
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
    global _jobs_in_system
    request_id = uuid.uuid4().hex[:8]
    workdir = Path(tempfile.mkdtemp(prefix=f"asr_{request_id}_"))
    raw_path = workdir / (audio_file.filename or "input")

    _jobs_in_system += 1
    started = time.monotonic()
    try:
        with raw_path.open("wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        log.info("[%s] received %s (task=%s lang=%s output=%s encode=%s)",
                 request_id, raw_path.name, task, language, output, encode)

        audio_path = workdir / "audio.wav"
        if encode:
            await run_in_thread(encode_to_wav, raw_path, audio_path)
        else:
            await run_in_thread(prepare_unencoded, raw_path, audio_path)

        alt_target = None  # initialized for the post-semaphore block

        async with _semaphore:
            # Reclaim cached-but-unused VRAM left by a previous request before
            # touching the GPU. The PyTorch caching allocator does not return
            # memory between requests on its own, and Whisper large-v3 + pyannote
            # + NLLB resident together leave only ~3 GB of slack on an 8 GB card.
            # Without this, a back-to-back /asr has been observed to OOM in the
            # transcribe step (CUDA error: out of memory) even though both
            # requests fit individually when the process is fresh.
            _empty_cuda_cache()
            source_segments, target_segments = await run_pipeline_async(
                audio_path, language,
            )
            # A/B alt-classifier pass: must run INSIDE the semaphore because
            # run_pipeline_alt_classifier mutates the global
            # settings.GENDER_CLASSIFIER for the duration of the call. If we
            # released the semaphore first, a concurrent request entering it
            # mid-alt would observe the flipped value and use the wrong
            # classifier on its primary pass. Render + side-file save remain
            # outside the semaphore below — they're I/O-only, no global
            # mutation.
            if settings.GENDER_AB_OUTPUT and target_segments is not None:
                try:
                    _, alt_target = await run_pipeline_alt_classifier(
                        audio_path, language,
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
            backend_model = (
                settings.LOCAL_TRANSLATION_MODEL
                if backend.strip().lower() == "local"
                else settings.CLAUDE_MODEL
            )
            summary = _build_translation_summary(
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
            await run_in_thread(
                _try_save_side_file, target_body, summary, video_file_url,
            )

            # Alt side-file save (outside semaphore — just I/O, no global
            # mutation). alt_target is only non-None when GENDER_AB_OUTPUT is
            # on and the alt pass succeeded.
            if alt_target is not None:
                alt_body, _ = render(alt_target, output)
                await run_in_thread(
                    _try_save_side_file,
                    alt_body, None,    # no summary file for the alt
                    video_file_url,
                    ".he.alt-classifier.srt",
                )
                log.info(
                    "[%s] A/B alt-classifier side-file saved",
                    request_id,
                )

        log.info("[%s] done in %.1fs (%d segments)",
                 request_id, wall, len(source_segments))
        return PlainTextResponse(body, media_type=content_type)
    finally:
        _jobs_in_system -= 1
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

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
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from config import settings
from pipeline import diarize, gender, transcribe, translate
from pipeline.chunk import make_chunks
from pipeline.format import render
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
) -> None:
    """Translate one chunk's pre-built speaker groups in place.

    ``groups`` is the output of ``_group_consecutive`` for this chunk's segments
    (built by the orchestrator so it can also derive the chunk's last-group
    gender for cross-chunk carry). ``prev_speaker_gender`` seeds the addressee
    rotation: the first group's addressee is "whoever spoke before this chunk"
    (or ``None`` for the very first chunk).
    """
    async with sem:
        prev_group_gender = prev_speaker_gender
        try:
            for speaker, group in groups:
                spk_gender = genders.get(speaker, "male") if speaker else "male"
                addressee = prev_group_gender
                translated = await translate.translate_batch_async(
                    [s.text for s in group], spk_gender, target, client,
                    addressee_gender=addressee,
                )
                for seg, text in zip(group, translated):
                    seg.text = text
                prev_group_gender = spk_gender
        except Exception:
            log.exception("[chunk %d] translation failed", idx)
            raise


async def _run_gender_aware(
    audio_path: Path,
    segments: list[Segment],
    target: str,
    client,
) -> list[Segment]:
    audio, sr = await run_in_thread(_load_wav_mono, audio_path)
    chunks = make_chunks(segments, settings.CHUNK_DURATION_SEC)
    sem = asyncio.Semaphore(settings.TRANSLATE_CONCURRENCY)
    tasks: list[asyncio.Task] = []
    prev_speaker_gender: str | None = None
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
            # Build chunk-local assignment + groups here so we know the chunk's
            # last group's speaker gender (for the next chunk's addressee carry)
            # before launching the translate task.
            assigned: list[tuple[Segment, str | None]] = []
            for seg in chunk.segments:
                local = Segment(seg.start - chunk.start, seg.end - chunk.start, seg.text)
                assigned.append((seg, diarize.assign_speaker(local, annotation)))
            groups = _group_consecutive(assigned)
            tasks.append(asyncio.create_task(
                _translate_chunk(idx, groups, genders, target, client, sem, prev_speaker_gender)
            ))
            # Carry: the next chunk's first group's addressee is this chunk's
            # final group's speaker gender.
            if groups:
                last_speaker = groups[-1][0]
                prev_speaker_gender = (
                    genders.get(last_speaker, "male") if last_speaker else "male"
                )
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
) -> list[Segment]:
    chunks = make_chunks(segments, settings.CHUNK_DURATION_SEC)
    sem = asyncio.Semaphore(settings.TRANSLATE_CONCURRENCY)

    async def translate_one(chunk):
        async with sem:
            translated = await translate.translate_batch_async(
                [s.text for s in chunk.segments], None, target, client
            )
            for seg, text in zip(chunk.segments, translated):
                seg.text = text

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


async def run_pipeline_async(audio_path: Path, language: str) -> list[Segment]:
    """Transcribe and (optionally) translate, overlapping diarize and translate."""
    t0 = time.monotonic()
    segments = await run_in_thread(transcribe.transcribe, audio_path, language)
    log.info("transcribe: %d segments in %.1fs", len(segments), time.monotonic() - t0)

    if not settings.translation_enabled() or not segments:
        return segments

    target = settings.TARGET_LANGUAGE
    client = get_async_anthropic_client()
    if settings.is_gender_aware():
        return await _run_gender_aware(audio_path, segments, target, client)
    return await _run_plain_translate(segments, target, client)


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

@app.on_event("startup")
async def warmup() -> None:
    log.info(
        "Starting up. model=%s device=%s target_language=%s gender_aware=%s",
        settings.WHISPER_MODEL, settings.DEVICE, settings.TARGET_LANGUAGE,
        settings.is_gender_aware(),
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

        async with _semaphore:
            segments = await run_pipeline_async(audio_path, language)

        body, content_type = render(segments, output)
        log.info("[%s] done in %.1fs (%d segments)",
                 request_id, time.monotonic() - started, len(segments))
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

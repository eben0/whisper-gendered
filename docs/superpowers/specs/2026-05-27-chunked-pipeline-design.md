# Chunked, Overlapping Pipeline — Design

**Date:** 2026-05-27
**Status:** Approved design (pending spec review)
**Branch:** `feature/chunked-pipeline`

## Problem

A full-episode `/asr` request runs its stages strictly serially:

```
transcribe(225s) → diarize+gender(848s) → translate(867s)   = 1940s (~32 min)
                                            ↑ GPU idle the whole translate phase
```

Two costs: (1) the GPU sits idle for the ~14-min Claude translation phase, and
(2) diarizing the entire 59-min waveform at once pushes VRAM to ~96% (7.85/8 GB)
and over-segments into 44 speakers.

## Goal

**Primary: wall-clock speedup via overlap.** Pipeline the work so chunk N+1's GPU
diarization overlaps chunk N's Claude translation. Target ~35–40% wall-clock
reduction (≈1940s → ≈1150s). Secondary wins (free): lower peak VRAM (diarize a
~5-min slice, not the whole hour) and more realistic per-chunk speaker counts.

## Key insight

This pipeline does **not** need global speaker identity. Diarization exists only
to (a) group consecutive same-speaker lines and (b) assign each line a gender from
pitch. Gender is a per-utterance property — it does not matter that `SPEAKER_00`
in chunk 1 is a different person than `SPEAKER_00` in chunk 5. This is what makes
per-chunk independent diarization safe, and removes the hardest part of chunked
diarization.

## Chosen approach

**Strategy: transcribe whole, chunk after.** Transcribe the full file once (clean,
no mid-utterance cuts, ~11% of runtime), then split the resulting *segments* into
contiguous chunks. Per chunk: diarize that audio sub-range + detect gender +
translate. Diarization runs serially on the GPU; translation runs concurrently in
the background, overlapping the next chunk's diarization.

Rejected: chunking the raw audio and transcribing per chunk — it also parallelizes
transcription but cuts utterances at window boundaries, hurting accuracy for only
~11% potential extra gain.

## Architecture & data flow

```
asr()  ──►  run_pipeline_async(audio_path, language)
              │
              ├─ transcribe(whole)               [GPU thread]  → segments[0..N]
              ├─ load audio.wav into memory once  [thread]      → waveform (float32, 16k mono)
              ├─ make_chunks(segments)                          → [chunk0, chunk1, …]
              │
              └─ async pipeline:
                   for each chunk (sequential on GPU):
                       ann, genders = await run_in_thread(diarize_slice + gender, waveform[chunk_range])
                       create_task( translate_chunk(...) )   ◄─ fires; does NOT block next diarize
                   await gather(all translate tasks)
                   → concat segments in chunk order → render()
```

### Invariants

- **GPU work stays serial.** Diarization is `await`ed one chunk at a time inside
  the loop, so only one pyannote run touches the GPU at once — preserves the
  `CONCURRENT_JOBS`/VRAM guarantee. Translation tasks run concurrently in the
  background; that overlap is the speedup.
- **Waveform loaded once**, sliced per chunk by sample offset
  (`waveform[start*sr : end*sr]`). No per-chunk file re-read; pyannote only sees a
  ~5-min slice (the VRAM win).
- **Timestamps.** Segments keep absolute times from transcription. A slice's
  diarization returns slice-local turn times, so speaker assignment is done in
  chunk-local coordinates (subtract `chunk_start` from segment times before
  `assign_speaker`). Final segment list is already in absolute order →
  `format.render()` renumbers as today.
- **No cross-chunk shared mutable state.** Each chunk mutates only its own
  segments' `.text`; results are concatenated by sorted chunk index.

## Chunking

`make_chunks(segments, target_sec=CHUNK_DURATION_SEC) -> list[Chunk]`:

- Walk segments, accumulate into the current chunk until
  `current_last.end - current_first.start >= target_sec`, then start a new chunk at
  the next segment.
- A trailing chunk shorter than `0.5 * target_sec` is merged into the previous
  chunk (avoids a tiny final chunk that diarizes poorly).
- Each `Chunk` = `{ segments: list[Segment], start: float, end: float }` where
  `start = segments[0].start`, `end = segments[-1].end`.

**Default `CHUNK_DURATION_SEC = 300`** (~12 chunks for a 59-min episode): long
enough for pyannote to cluster speakers well, short enough for the VRAM and
overlap-granularity wins.

Edge cases:
- A single segment longer than `target_sec` → its own chunk.
- One chunk total (short clip) → behaves exactly like the current non-chunked path.
- Zero segments → handled upstream (returns early, no translation).

## Concurrency & translation

- Translation uses **`anthropic.AsyncAnthropic`**, bounded by
  `asyncio.Semaphore(TRANSLATE_CONCURRENCY)` (default 3). The semaphore is the
  single concurrency control — no thread-pool starvation to reason about.
  Diarization continues to use the existing thread executor via `run_in_thread`.
- `translate.py` gains `translate_batch_async(texts, gender, target, client)` that
  mirrors the current sync logic (sub-batching by `MAX_BATCH_SEGMENTS=40` /
  `MAX_BATCH_CHARS=2500`, structured-output JSON schema, length validation/padding)
  — just awaited on the async client. Within a chunk, its sub-batch calls run
  sequentially; concurrency is *across* chunks via the semaphore.
- **Retry:** the async client is configured with `max_retries=CLAUDE_MAX_RETRIES`
  (default 4). The Anthropic SDK auto-retries 429 / 529-overloaded / connection
  errors with exponential backoff. If a call still fails, the exception propagates
  → `asyncio.gather` fails → the request returns 500 (chosen failure policy), with a
  log line naming the failed chunk index.

`translate_chunk(idx, chunk, genders)`:
```
async with translate_sem:
    for speaker, group in group_consecutive(assign chunk.segments to chunk-local annotation):
        spk_gender = genders.get(speaker, "male")
        translated = await translate_batch_async([s.text for s in group], spk_gender, target, client)
        for seg, text in zip(group, translated): seg.text = text
```

### Non-gender translate path (`TARGET_LANGUAGE` set but not gender-aware)

Also chunked + bounded-concurrent translation, with `gender=None` and no
diarization. Small, free win; reuses the same semaphore and async translate.

### Plain transcription (`TARGET_LANGUAGE=none`)

Unchanged — no translation, no chunking needed.

## Failure policy

Retry transient Claude errors (via SDK `max_retries`); if a chunk still fails after
retries, fail the whole request with 500 — same contract as today, so Bazarr
retries the episode. No half-translated subtitle files. A warning logs the failing
chunk index and stage.

## Refactors

Focused, following existing patterns:

- **`pipeline/diarize.py`**: extract `diarize_waveform(waveform: np.ndarray, sr: int)
  -> Annotation` (the in-memory `{waveform, sample_rate}` path). Existing
  `diarize(path)` becomes a thin wrapper that `sf.read`s then calls it (kept for
  warmup / non-chunked callers).
- **`pipeline/gender.py`**: `detect_genders(waveform: np.ndarray, sr: int,
  diarization) -> dict[str,str]` — operate on the in-memory slice; drop the
  `librosa.load` file re-read. Pitch/threshold logic unchanged.
- **`server.py`**: add `get_async_anthropic_client()`; replace `run_pipeline` with
  async `run_pipeline_async`; `asr()` awaits it directly (no longer wrapped in
  `run_in_thread`, since the orchestrator is async and dispatches its own thread
  work). The request-level `_semaphore` (`CONCURRENT_JOBS`) is unchanged and still
  wraps the whole request — overlap is within a request; cross-request behavior is
  untouched (less risk).
- **`config.py`**: add `CHUNK_DURATION_SEC=300`, `TRANSLATE_CONCURRENCY=3`,
  `CLAUDE_MAX_RETRIES=4`. Document in `.env.example` and the README config table.

## Testing

- **Unit `make_chunks`**: boundary accumulation at `target_sec`; trailing-chunk
  merge; a single over-long segment; single-chunk input; empty input.
- **Unit slice-diarization offset**: `assign_speaker` returns the correct label when
  the annotation is in slice-local time and segments are in absolute time
  (verifies the `chunk_start` subtraction).
- **Integration**: a synthetic ≥2-chunk, 2-speaker (male+female) WAV with Claude
  **mocked** (no real API). Assert: (a) final segment count/order matches the
  transcription, (b) rendered SRT indices are contiguous and time-ordered,
  (c) per-speaker gender applied (mock echoes gender into text), (d) no deadlock;
  the run completes. Optionally assert overlap occurred (chunk 2 diarize begins
  before chunk 1 translate resolves) via instrumented fakes.
- **Equivalence**: for a single-chunk short clip, chunked output matches the
  current non-chunked output (same segments, order, numbering).

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Tiny final chunk diarizes poorly | Merge trailing `< 0.5×target` into previous chunk |
| Ordering corruption under concurrency | Per-chunk-only mutation; concat by sorted chunk index |
| Async client error semantics | Rely on SDK `max_retries`; explicit per-chunk failure logging |
| Full waveform in RAM | ~227 MB for 59 min @ 16k mono — acceptable |
| Backward compatibility | Single-chunk and `none`/non-gender paths produce identical output |

## Explicitly out of scope (YAGNI)

- Global speaker identity across chunks (gender is per-utterance — not needed).
- Progressive / streaming subtitle output.
- Parallelizing transcription (only ~11% of runtime; high seam risk).

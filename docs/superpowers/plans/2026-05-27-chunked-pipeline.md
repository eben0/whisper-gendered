# Chunked, Overlapping Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlap GPU diarization with Claude translation by transcribing the whole file once, splitting the resulting segments into time-bounded chunks, and pipelining each chunk's diarization against the previous chunk's translation.

**Architecture:** `run_pipeline` becomes an async orchestrator. Transcription runs whole-file; the waveform is loaded once and sliced per chunk. Diarization runs serially on the GPU (one chunk at a time, via the existing thread executor); each finished chunk's translation is fired as a concurrent asyncio task bounded by a semaphore, so chunk N+1 diarizes while chunk N translates. Gender is per-utterance, so per-chunk independent diarization needs no global speaker identity.

**Tech Stack:** Python 3.12, FastAPI, faster-whisper, pyannote.audio 4.x, librosa, `anthropic.AsyncAnthropic`, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-05-27-chunked-pipeline-design.md`

---

## File Structure

- **Create `pipeline/chunk.py`** — pure `Chunk` dataclass + `make_chunks()` segment-grouping logic. No model/audio deps; trivially unit-testable.
- **Modify `pipeline/diarize.py`** — extract `diarize_waveform(waveform, sr)`; `diarize(path)` becomes a thin wrapper.
- **Modify `pipeline/gender.py`** — `detect_genders(audio, sr, diarization)` operates on an in-memory slice instead of re-reading a file.
- **Modify `pipeline/translate.py`** — add async `translate_batch_async` / `_translate_one_batch_async` using `anthropic.AsyncAnthropic`.
- **Modify `config.py`** — add `CHUNK_DURATION_SEC`, `TRANSLATE_CONCURRENCY`, `CLAUDE_MAX_RETRIES`.
- **Modify `server.py`** — `get_async_anthropic_client()`, async `run_pipeline_async` + helpers (`_load_wav_mono`, `_diarize_and_gender`, `_translate_chunk`, `_run_gender_aware`, `_run_plain_translate`); wire `asr()` to await it.
- **Create test scaffolding** — root `conftest.py`, `pytest.ini`, `tests/` with one test module per unit.
- **Modify `requirements.txt`, `.env.example`, `README.md`** — dev deps + new config docs.

---

## Task 1: Test scaffolding

**Files:**
- Create: `conftest.py`
- Create: `pytest.ini`
- Modify: `requirements.txt` (append)
- Create: `tests/__init__.py`

- [ ] **Step 1: Add an empty root conftest so pytest puts the repo root on sys.path**

Create `conftest.py` (repo root):

```python
# Presence of this file makes the repo root pytest's rootdir, so tests can
# `import config`, `import server`, and `from pipeline import ...`.
```

- [ ] **Step 2: Add pytest config enabling async tests**

Create `pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 3: Create the tests package marker**

Create `tests/__init__.py`:

```python
```

(empty file)

- [ ] **Step 4: Append test dependencies**

Append these two lines to `requirements.txt`:

```
pytest
pytest-asyncio
```

- [ ] **Step 5: Install and verify pytest collects nothing yet**

Run: `.\.venv\Scripts\python.exe -m pip install pytest pytest-asyncio`
Then: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: `no tests ran` (exit code 5), no import/collection errors.

- [ ] **Step 6: Commit**

```bash
git add conftest.py pytest.ini tests/__init__.py requirements.txt
git -c commit.gpgsign=false commit -m "test: add pytest scaffolding (asyncio mode, root conftest)"
```

---

## Task 2: Config — chunking and concurrency settings

**Files:**
- Modify: `config.py:49-63` (Settings fields)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
import importlib

import config as config_module


def test_new_settings_have_defaults():
    s = config_module.Settings()
    assert s.CHUNK_DURATION_SEC == 300
    assert s.TRANSLATE_CONCURRENCY == 3
    assert s.CLAUDE_MAX_RETRIES == 4


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CHUNK_DURATION_SEC", "120")
    monkeypatch.setenv("TRANSLATE_CONCURRENCY", "2")
    monkeypatch.setenv("CLAUDE_MAX_RETRIES", "1")
    importlib.reload(config_module)
    s = config_module.Settings()
    assert s.CHUNK_DURATION_SEC == 120
    assert s.TRANSLATE_CONCURRENCY == 2
    assert s.CLAUDE_MAX_RETRIES == 1
    importlib.reload(config_module)  # restore module-level singleton
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'CHUNK_DURATION_SEC'`.

- [ ] **Step 3: Add the three fields**

In `config.py`, inside the `Settings` dataclass, immediately after the
`GENDER_THRESHOLD_HZ` line (currently line 62), add:

```python
    # Chunked-pipeline tuning.
    CHUNK_DURATION_SEC: int = _env_int("CHUNK_DURATION_SEC", 300)
    TRANSLATE_CONCURRENCY: int = _env_int("TRANSLATE_CONCURRENCY", 3)
    CLAUDE_MAX_RETRIES: int = _env_int("CLAUDE_MAX_RETRIES", 4)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git -c commit.gpgsign=false commit -m "feat(config): add CHUNK_DURATION_SEC, TRANSLATE_CONCURRENCY, CLAUDE_MAX_RETRIES"
```

---

## Task 3: `pipeline/chunk.py` — segment chunking

**Files:**
- Create: `pipeline/chunk.py`
- Test: `tests/test_chunk.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunk.py`:

```python
from pipeline.transcribe import Segment
from pipeline.chunk import Chunk, make_chunks


def _seg(start, end, text="x"):
    return Segment(start=start, end=end, text=text)


def test_empty_returns_no_chunks():
    assert make_chunks([], target_sec=300) == []


def test_single_chunk_when_under_target():
    segs = [_seg(0, 5), _seg(5, 10), _seg(10, 15)]
    chunks = make_chunks(segs, target_sec=300)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == 15
    assert chunks[0].segments == segs


def test_splits_when_current_span_reaches_target():
    # target 10s: seg c starts a new chunk because [0,10) already spans >= 10.
    a, b, c, d = _seg(0, 4), _seg(4, 10), _seg(10, 14), _seg(14, 20)
    chunks = make_chunks([a, b, c, d], target_sec=10)
    assert [ [s for s in ch.segments] for ch in chunks ] == [[a, b], [c, d]]
    assert (chunks[0].start, chunks[0].end) == (0, 10)
    assert (chunks[1].start, chunks[1].end) == (10, 20)


def test_long_single_segment_is_its_own_chunk():
    a, b = _seg(0, 50), _seg(50, 55)
    chunks = make_chunks([a, b], target_sec=10)
    assert chunks[0].segments == [a]
    assert chunks[1].segments == [b]


def test_tiny_trailing_chunk_merges_into_previous():
    # target 10, merge_ratio 0.5 -> trailing chunk shorter than 5s merges back.
    a, b, c = _seg(0, 6), _seg(6, 12), _seg(12, 13)  # c spans 1s, < 5s
    chunks = make_chunks([a, b, c], target_sec=10, merge_ratio=0.5)
    assert len(chunks) == 1
    assert chunks[0].segments == [a, b, c]
    assert chunks[0].end == 13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_chunk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.chunk'`.

- [ ] **Step 3: Implement `pipeline/chunk.py`**

Create `pipeline/chunk.py`:

```python
"""Group transcribed segments into contiguous, time-bounded chunks.

A chunk is a run of consecutive segments whose audio span approaches
``target_sec``. Chunk boundaries fall on segment gaps (natural pauses), so no
utterance is ever split. The pipeline diarizes and translates one chunk at a
time, overlapping diarization of the next chunk with translation of the current.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.transcribe import Segment


@dataclass
class Chunk:
    segments: list[Segment]
    start: float
    end: float


def make_chunks(
    segments: list[Segment],
    target_sec: float,
    merge_ratio: float = 0.5,
) -> list[Chunk]:
    """Split ``segments`` into chunks each spanning roughly ``target_sec``.

    A new chunk starts once the current chunk's span (last.end - first.start)
    reaches ``target_sec``. A trailing chunk shorter than ``merge_ratio *
    target_sec`` is merged into the previous chunk so the final chunk is never
    too short to diarize well.
    """
    if not segments:
        return []

    groups: list[list[Segment]] = []
    current: list[Segment] = []
    for seg in segments:
        if current and (current[-1].end - current[0].start) >= target_sec:
            groups.append(current)
            current = []
        current.append(seg)
    if current:
        groups.append(current)

    if len(groups) >= 2:
        last = groups[-1]
        if (last[-1].end - last[0].start) < merge_ratio * target_sec:
            groups[-2].extend(groups.pop())

    return [
        Chunk(segments=g, start=g[0].start, end=g[-1].end) for g in groups
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_chunk.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/chunk.py tests/test_chunk.py
git -c commit.gpgsign=false commit -m "feat(chunk): segment chunking with trailing-merge"
```

---

## Task 4: `diarize.diarize_waveform` — in-memory diarization

**Files:**
- Modify: `pipeline/diarize.py:48-63` (the `diarize` function)
- Test: `tests/test_diarize_waveform.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_diarize_waveform.py`:

```python
import numpy as np
from pyannote.core import Annotation, Segment as PSegment

import pipeline.diarize as diarize


class _FakePipe:
    """Records the dict it was called with; returns a fixed annotation."""

    def __init__(self):
        self.called_with = None

    def __call__(self, payload):
        self.called_with = payload
        ann = Annotation()
        ann[PSegment(0.0, 1.0)] = "SPEAKER_00"
        return ann


def test_diarize_waveform_wraps_1d_to_channel_time(monkeypatch):
    fake = _FakePipe()
    monkeypatch.setattr(diarize, "get_pipeline", lambda: fake)

    mono = np.zeros(16000, dtype=np.float32)  # 1 s @ 16k, shape (time,)
    ann = diarize.diarize_waveform(mono, 16000)

    assert ann.labels() == ["SPEAKER_00"]
    waveform = fake.called_with["waveform"]
    assert fake.called_with["sample_rate"] == 16000
    assert tuple(waveform.shape) == (1, 16000)  # (channel, time)


def test_diarize_waveform_unwraps_speaker_diarization_attr(monkeypatch):
    inner = Annotation()
    inner[PSegment(0.0, 0.5)] = "SPEAKER_01"

    class _Wrapped:
        speaker_diarization = inner

    class _WrappingPipe:
        def __call__(self, payload):
            return _Wrapped()

    monkeypatch.setattr(diarize, "get_pipeline", lambda: _WrappingPipe())
    ann = diarize.diarize_waveform(np.zeros(8000, dtype=np.float32), 16000)
    assert ann.labels() == ["SPEAKER_01"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_diarize_waveform.py -v`
Expected: FAIL — `AttributeError: module 'pipeline.diarize' has no attribute 'diarize_waveform'`.

- [ ] **Step 3: Refactor `diarize.py`**

In `pipeline/diarize.py`, replace the current `diarize` function (lines 48-63)
with these two functions:

```python
def diarize_waveform(waveform: np.ndarray, sr: int) -> Annotation:
    """Diarize an in-memory waveform.

    ``waveform`` is float32, shape ``(time,)`` for mono or ``(channel, time)``.
    Bypasses torchcodec/FFmpeg by handing pyannote a {waveform, sample_rate}
    dict directly.
    """
    pipe = get_pipeline()
    wav2d = waveform[np.newaxis, :] if waveform.ndim == 1 else waveform
    tensor = torch.from_numpy(np.ascontiguousarray(wav2d))
    with torch.inference_mode():
        result = pipe({"waveform": tensor, "sample_rate": sr})
    annotation = getattr(result, "speaker_diarization", result)
    speakers = annotation.labels()
    log.info("Diarized %d speaker(s): %s", len(speakers), speakers)
    return annotation


def diarize(audio_path: Path) -> Annotation:
    """Run speaker diarization on a WAV file (used for warm-up / whole-file)."""
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    return diarize_waveform(data.T, sr)  # data.T -> (channel, time)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_diarize_waveform.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/diarize.py tests/test_diarize_waveform.py
git -c commit.gpgsign=false commit -m "refactor(diarize): extract diarize_waveform for in-memory slices"
```

---

## Task 5: `gender.detect_genders` — in-memory waveform

**Files:**
- Modify: `pipeline/gender.py:30-32` (signature + audio load)
- Test: `tests/test_gender.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gender.py`:

```python
import numpy as np
from pyannote.core import Annotation, Segment as PSegment

from pipeline.gender import detect_genders

SR = 16000


def _tone(freq, seconds=1.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_detect_genders_classifies_low_and_high_pitch():
    # SPEAKER_LOW = 120 Hz (male), SPEAKER_HIGH = 230 Hz (female).
    low = _tone(120.0)
    high = _tone(230.0)
    audio = np.concatenate([low, high])

    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "SPEAKER_LOW"
    ann[PSegment(1.0, 2.0)] = "SPEAKER_HIGH"

    genders = detect_genders(audio, SR, ann)
    assert genders["SPEAKER_LOW"] == "male"
    assert genders["SPEAKER_HIGH"] == "female"


def test_detect_genders_defaults_to_male_when_no_audio():
    ann = Annotation()
    ann[PSegment(0.0, 0.1)] = "SPEAKER_TINY"  # below MIN_SEGMENT_SEC (0.3s)
    genders = detect_genders(np.zeros(16000, dtype=np.float32), SR, ann)
    assert genders["SPEAKER_TINY"] == "male"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gender.py -v`
Expected: FAIL — `TypeError` (detect_genders still expects `(audio_path, diarization)` and calls `librosa.load` on a numpy array).

- [ ] **Step 3: Change the signature and drop the file read**

In `pipeline/gender.py`, replace the function signature and the first two lines
of the body (currently lines 30-33):

```python
def detect_genders(audio_path: Path, diarization: Annotation) -> dict[str, str]:
    """Return {speaker_label: "male" | "female"} for every diarized speaker."""
    audio, _sr = librosa.load(str(audio_path), sr=SR, mono=True)
    total = len(audio)
```

with:

```python
def detect_genders(
    audio: np.ndarray, sr: int, diarization: Annotation
) -> dict[str, str]:
    """Return {speaker_label: "male" | "female"} for every diarized speaker.

    ``audio`` is a mono float32 waveform; ``diarization`` turn times must be in
    the same time base as ``audio`` (i.e. both relative to the same slice start).
    """
    total = len(audio)
```

Then, in the `librosa.pyin` call further down (currently line 56-58), replace
the hardcoded `sr=SR` with the passed-in `sr`:

```python
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            signal, sr=sr, fmin=FMIN_HZ, fmax=FMAX_HZ,
        )
```

Also replace the two remaining `int(turn.start * SR)` / `int(turn.end * SR)`
index computations (currently lines 45-46) with `sr`:

```python
            i0 = max(0, int(turn.start * sr))
            i1 = min(total, int(turn.end * sr))
```

Leave the `Path` import in place (still imported elsewhere is fine) — but it is
now unused here; remove `from pathlib import Path` if your linter flags it.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gender.py -v`
Expected: PASS (2 passed). (pyin on a clean tone resolves close to the tone
frequency; 120 Hz and 230 Hz sit clearly on opposite sides of the 165 Hz
threshold.)

- [ ] **Step 5: Commit**

```bash
git add pipeline/gender.py tests/test_gender.py
git -c commit.gpgsign=false commit -m "refactor(gender): detect_genders on in-memory waveform"
```

---

## Task 6: `translate.translate_batch_async` — async Claude calls

**Files:**
- Modify: `pipeline/translate.py` (add async functions; keep sync ones)
- Test: `tests/test_translate_async.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_translate_async.py`:

```python
import json

import pytest

from pipeline import translate


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = 0

    async def create(self, **kwargs):
        text = self._payloads[self.calls]
        self.calls += 1
        return _FakeResponse(text)


class _FakeAsyncClient:
    def __init__(self, payloads):
        self.messages = _FakeMessages(payloads)


@pytest.mark.asyncio
async def test_translate_batch_async_returns_translations():
    client = _FakeAsyncClient([json.dumps({"translations": ["a-he", "b-he"]})])
    out = await translate.translate_batch_async(["a", "b"], "male", "Hebrew", client)
    assert out == ["a-he", "b-he"]


@pytest.mark.asyncio
async def test_translate_batch_async_pads_on_count_mismatch():
    client = _FakeAsyncClient([json.dumps({"translations": ["only-one"]})])
    out = await translate.translate_batch_async(["a", "b"], None, "Hebrew", client)
    assert len(out) == 2
    assert out[0] == "only-one"
    assert out[1] == "b"  # padded from source text


@pytest.mark.asyncio
async def test_translate_batch_async_empty_input():
    client = _FakeAsyncClient([])
    out = await translate.translate_batch_async([], "female", "Hebrew", client)
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py -v`
Expected: FAIL — `AttributeError: module 'pipeline.translate' has no attribute 'translate_batch_async'`.

- [ ] **Step 3: Add async functions to `translate.py`**

In `pipeline/translate.py`, add these functions at the end of the file (after
`translate_batch`). They reuse the existing `_system_prompt`, `_chunks`,
`_OUTPUT_FORMAT`, `MAX_TOKENS`, and the same parse/validate logic as the sync
path:

```python
async def _translate_one_batch_async(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: "anthropic.AsyncAnthropic",
) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    response = await client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(target_language, gender),
        output_config={"format": _OUTPUT_FORMAT},
        messages=[{"role": "user", "content": numbered}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "")
    try:
        translations = json.loads(raw).get("translations", [])
    except (json.JSONDecodeError, AttributeError):
        log.warning("Could not parse translation JSON; returning source text.")
        return texts

    if len(translations) != len(texts):
        log.warning(
            "Translation count mismatch (got %d, expected %d); aligning by index.",
            len(translations), len(texts),
        )
        translations = (translations + texts[len(translations):])[: len(texts)]
    return [str(t) for t in translations]


async def translate_batch_async(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: "anthropic.AsyncAnthropic",
) -> list[str]:
    """Async counterpart of ``translate_batch`` for the chunked orchestrator.

    Sub-batches run sequentially within one call; cross-chunk concurrency is
    handled by the caller's semaphore. Output length always equals input length.
    """
    if not texts:
        return []
    out: list[str] = []
    for batch in _chunks(texts):
        out.extend(
            await _translate_one_batch_async(batch, gender, target_language, client)
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/translate.py tests/test_translate_async.py
git -c commit.gpgsign=false commit -m "feat(translate): async translate_batch_async"
```

---

## Task 7: Server orchestration — `run_pipeline_async`

**Files:**
- Modify: `server.py` — imports, async client, replace `run_pipeline`, wire `asr()`, update warm-up call site
- Test: `tests/test_orchestration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_orchestration.py`:

```python
import numpy as np
import pytest
from pyannote.core import Annotation, Segment as PSegment

import server
from pipeline.transcribe import Segment


@pytest.fixture
def two_chunk_segments():
    # Two 6s segments -> with target 5s these land in separate chunks.
    return [
        Segment(start=0.0, end=6.0, text="hello"),
        Segment(start=6.0, end=12.0, text="world"),
    ]


def _install_fakes(monkeypatch, segments, gender_aware):
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew" if gender_aware else "Japanese")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)

    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segments))
    monkeypatch.setattr(server, "_load_wav_mono", lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "SPEAKER_00"
        return ann

    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)
    monkeypatch.setattr(server.diarize, "assign_speaker", lambda seg, ann: "SPEAKER_00")
    monkeypatch.setattr(server.gender, "detect_genders", lambda audio, sr, ann: {"SPEAKER_00": "female"})

    async def fake_translate(texts, gender, target, client):
        return [f"{t}|{gender}" for t in texts]

    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)


@pytest.mark.asyncio
async def test_gender_aware_preserves_order_and_applies_gender(monkeypatch, two_chunk_segments):
    _install_fakes(monkeypatch, two_chunk_segments, gender_aware=True)
    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    assert [s.text for s in out] == ["hello|female", "world|female"]
    assert [(s.start, s.end) for s in out] == [(0.0, 6.0), (6.0, 12.0)]


@pytest.mark.asyncio
async def test_plain_translate_no_diarization(monkeypatch, two_chunk_segments):
    _install_fakes(monkeypatch, two_chunk_segments, gender_aware=False)
    # If diarization were called in the plain path this would raise.
    monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a: (_ for _ in ()).throw(AssertionError("diarize called in plain path")))
    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    assert [s.text for s in out] == ["hello|None", "world|None"]


@pytest.mark.asyncio
async def test_transcription_only_when_target_none(monkeypatch, two_chunk_segments):
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "none")
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(two_chunk_segments))
    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    assert [s.text for s in out] == ["hello", "world"]


@pytest.mark.asyncio
async def test_chunk_local_offset_maps_speakers(monkeypatch, two_chunk_segments):
    # Verifies _translate_chunk subtracts chunk.start before assign_speaker, using
    # the REAL assign_speaker (not mocked). Each chunk is one 6s segment; the
    # per-chunk annotation is slice-local with a turn over [0, 6).
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(two_chunk_segments))
    monkeypatch.setattr(server, "_load_wav_mono", lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "S"  # slice-local turn
        return ann

    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)
    monkeypatch.setattr(server.gender, "detect_genders", lambda audio, sr, ann: {"S": "female"})

    async def fake_translate(texts, gender, target, client):
        return [f"{t}|{gender}" for t in texts]

    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Chunk 2's segment is absolute 6-12. Only if chunk_start (6.0) is subtracted
    # does its local midpoint (3.0) fall inside the [0,6) turn -> speaker "S" ->
    # "female". Without the subtraction it would miss the turn -> default "male".
    assert [s.text for s in out] == ["hello|female", "world|female"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py -v`
Expected: FAIL — `AttributeError: module 'server' has no attribute 'run_pipeline_async'`.

- [ ] **Step 3: Add the async client getter**

In `server.py`, after `get_anthropic_client()` (ends ~line 112), add:

```python
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
```

- [ ] **Step 4: Add the chunk import and orchestration functions**

In `server.py`, update the pipeline import line (currently
`from pipeline import diarize, gender, transcribe, translate`) to also import the
chunk module:

```python
from pipeline import diarize, gender, transcribe, translate
from pipeline.chunk import make_chunks
```

Then replace the entire `run_pipeline` function (currently lines 174-213) with
the async orchestrator and its helpers:

```python
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


async def _translate_chunk(idx, chunk, annotation, genders, target, client, sem):
    """Translate one chunk's segments in place, grouped by speaker + gender."""
    async with sem:
        assigned: list[tuple[Segment, str | None]] = []
        for seg in chunk.segments:
            local = Segment(seg.start - chunk.start, seg.end - chunk.start, seg.text)
            assigned.append((seg, diarize.assign_speaker(local, annotation)))
        try:
            for speaker, group in _group_consecutive(assigned):
                spk_gender = genders.get(speaker, "male") if speaker else "male"
                translated = await translate.translate_batch_async(
                    [s.text for s in group], spk_gender, target, client
                )
                for seg, text in zip(group, translated):
                    seg.text = text
        except Exception:
            log.exception("[chunk %d] translation failed", idx)
            raise


async def _run_gender_aware(audio_path, segments, target, client):
    audio, sr = await run_in_thread(_load_wav_mono, audio_path)
    chunks = make_chunks(segments, settings.CHUNK_DURATION_SEC)
    sem = asyncio.Semaphore(settings.TRANSLATE_CONCURRENCY)
    tasks = []
    t1 = time.monotonic()
    for idx, chunk in enumerate(chunks):
        annotation, genders = await run_in_thread(
            _diarize_and_gender, audio, sr, chunk.start, chunk.end
        )
        log.info(
            "chunk %d/%d diarize+gender done (%d speakers)",
            idx + 1, len(chunks), len(genders),
        )
        tasks.append(asyncio.create_task(
            _translate_chunk(idx, chunk, annotation, genders, target, client, sem)
        ))
    await asyncio.gather(*tasks)
    log.info("gender-aware chunks complete: %.1fs", time.monotonic() - t1)
    return [seg for chunk in chunks for seg in chunk.segments]


async def _run_plain_translate(segments, target, client):
    chunks = make_chunks(segments, settings.CHUNK_DURATION_SEC)
    sem = asyncio.Semaphore(settings.TRANSLATE_CONCURRENCY)

    async def translate_one(chunk):
        async with sem:
            translated = await translate.translate_batch_async(
                [s.text for s in chunk.segments], None, target, client
            )
            for seg, text in zip(chunk.segments, translated):
                seg.text = text

    await asyncio.gather(*(translate_one(c) for c in chunks))
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
```

- [ ] **Step 5: Wire `asr()` to the async orchestrator**

In `server.py`, inside `asr()`, replace:

```python
        async with _semaphore:
            segments = await run_in_thread(run_pipeline, audio_path, language)
```

with:

```python
        async with _semaphore:
            segments = await run_pipeline_async(audio_path, language)
```

- [ ] **Step 6: Run the orchestration tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Run the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all tests pass (config, chunk, diarize_waveform, gender, translate_async, orchestration).

- [ ] **Step 8: Commit**

```bash
git add server.py tests/test_orchestration.py
git -c commit.gpgsign=false commit -m "feat(server): chunked async pipeline overlapping diarize and translate"
```

---

## Task 8: Docs — `.env.example` and README

**Files:**
- Modify: `.env.example` (append new vars)
- Modify: `README.md` (config table)

- [ ] **Step 1: Document the new settings in `.env.example`**

Append to `.env.example`:

```
# Chunked pipeline tuning
CHUNK_DURATION_SEC=300
TRANSLATE_CONCURRENCY=3
CLAUDE_MAX_RETRIES=4
```

- [ ] **Step 2: Add the new rows to the README config table**

In `README.md`, add these rows to the configuration table (after the
`GENDER_THRESHOLD_HZ` row):

```
| `CHUNK_DURATION_SEC` | `300`                | Target seconds of audio per pipeline chunk                            |
| `TRANSLATE_CONCURRENCY`| `3`                | Max chunk translations running concurrently (Claude)                  |
| `CLAUDE_MAX_RETRIES` | `4`                  | Anthropic SDK auto-retry attempts on 429/529/connection errors        |
```

Also add a short paragraph under "How it works" noting that for translated
output the file is transcribed whole, then split into `CHUNK_DURATION_SEC`
chunks whose diarization overlaps the previous chunk's translation.

- [ ] **Step 3: Commit**

```bash
git add .env.example README.md
git -c commit.gpgsign=false commit -m "docs: document chunked-pipeline settings"
```

---

## Task 9: End-to-end manual verification (no new code)

**Files:** none (manual run against the real server).

- [ ] **Step 1: Restart the server**

Stop any running instance on port 9000, then:
`.\.venv\Scripts\python.exe server.py *>> "$env:TEMP\wg_server.log" 2>&1` (background)
Poll `GET http://localhost:9000/` until it returns `{"status":"ok",...}`.

- [ ] **Step 2: Re-run the two-speaker smoke clip**

POST the existing short two-speaker WAV (`$env:TEMP\wg_sample.wav`) with
`output=srt`, `encode=true`. Expected: valid Hebrew SRT, two lines, gendered
forms differing per speaker — identical semantics to the pre-chunking output
(this clip is a single chunk, proving backward compatibility).

- [ ] **Step 3: Verify overlap on a long input**

Trigger a full-episode request (or a multi-minute clip spanning ≥2 chunks).
In the server log, confirm: per-chunk `diarize+gender done` lines interleave
with translation, the request returns 200, and wall-clock is meaningfully below
the previous ~1940s for an equivalent episode.

- [ ] **Step 4: Final full-suite run before handoff**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: all green.

---

## Notes for the implementer

- **Run pytest from the repo root** (`D:\git\whisper-gend`) using the venv
  interpreter `.\.venv\Scripts\python.exe -m pytest` so the CUDA DLL setup and
  installed packages resolve.
- **Do not read or edit `.env`** — it holds secrets and is gitignored. Tests use
  `monkeypatch` on `settings`; they never need real API keys.
- The gender test exercises real `librosa.pyin` on synthetic tones (CPU, a few
  seconds) — no GPU or network. All other heavy dependencies (pyannote, Whisper,
  Claude) are faked via `monkeypatch`.
- Segments are mutated in place inside their own chunk; the final flat list is
  rebuilt from `chunks` in order, so concurrency cannot reorder output.

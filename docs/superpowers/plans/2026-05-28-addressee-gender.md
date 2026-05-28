# Addressee-Aware Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Hebrew/gender-marked second-person ("you") translation by passing an `addressee_gender` hint (the previous group's speaker gender) to Claude and extending the system prompt with explicit guidance on choosing the right form.

**Architecture:** `_system_prompt` and the translate batch functions gain an optional `addressee_gender` parameter. Inside `_run_gender_aware`, groups are built before the translate task is launched, the task rotates `addressee_gender` per group ("previous group's speaker gender"), and the orchestrator threads the last group's speaker gender across chunk boundaries so the carry survives chunk seams.

**Tech Stack:** Python 3.12, asyncio, `anthropic.AsyncAnthropic`, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-05-28-addressee-gender-design.md`

---

## File Structure

- **Modify `pipeline/translate.py`** — add optional `addressee_gender` parameter to `_system_prompt`, `_translate_one_batch`, `translate_batch`, `_translate_one_batch_async`, `translate_batch_async`. Prompt body gains a paragraph about handling "you".
- **Modify `server.py`** — `_translate_chunk` now takes pre-built `groups` and a new `prev_speaker_gender` argument and rotates addressee per group. `_run_gender_aware` builds groups before launching each translate task and threads the last group's speaker gender into the next chunk's call.
- **Modify `tests/test_translate_async.py`** — add a test asserting the addressee sentence appears in the system prompt when set, and is absent when `None`.
- **Modify `tests/test_orchestration.py`** — update the existing two fake `translate_batch_async` functions to accept `addressee_gender=None`; add two new tests (within-chunk rotation, cross-chunk carry).

---

## Task 1: Prompt body — addressee paragraph

**Files:**
- Modify: `pipeline/translate.py` lines 47-65 (`_system_prompt`)
- Test: `tests/test_translate_async.py` (new test)

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_translate_async.py` (it uses the existing top-of-file imports — `pipeline.translate as translate`):

```python
def test_system_prompt_includes_addressee_sentence_when_set():
    prompt = translate._system_prompt("Hebrew", "female", addressee_gender="male")
    assert "male" in prompt
    assert "addressee" in prompt.lower()


def test_system_prompt_omits_addressee_sentence_when_unset():
    prompt = translate._system_prompt("Hebrew", "female")
    assert "addressee" not in prompt.lower()


def test_system_prompt_addresses_number_when_target_is_gender_aware():
    # Number guidance should appear for any gender-marked target language.
    prompt = translate._system_prompt("Hebrew", "male", addressee_gender=None)
    # The "you" / number guidance should be present regardless of addressee_gender.
    assert "plural" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py::test_system_prompt_includes_addressee_sentence_when_set tests/test_translate_async.py::test_system_prompt_omits_addressee_sentence_when_unset tests/test_translate_async.py::test_system_prompt_addresses_number_when_target_is_gender_aware -v`

Expected: FAIL — `_system_prompt()` got an unexpected keyword argument `'addressee_gender'` (and the "plural" / "addressee" substrings don't exist in the prompt yet).

- [ ] **Step 3: Update `_system_prompt`**

In `pipeline/translate.py`, REPLACE the entire `_system_prompt` function (currently lines 47-65) with:

```python
def _system_prompt(
    target_language: str,
    gender: str | None,
    addressee_gender: str | None = None,
) -> str:
    base = (
        f"You are an expert subtitle translator. Translate each numbered line "
        f"from English into {target_language}. Produce natural, idiomatic, "
        f"concise {target_language} suitable for on-screen subtitles. Preserve "
        f"meaning and tone; do not add notes, explanations, or transliteration."
    )
    if gender is not None:
        base += (
            f" The speaker of these lines is {gender}. Use grammatically correct "
            f"{gender} forms throughout — verb conjugation, adjective and "
            f"participle agreement, imperatives, and pronouns must all match a "
            f"{gender} speaker referring to themselves."
        )
        base += (
            f" When the speaker addresses another person (English \"you\"), "
            f"choose the {target_language} form matching the addressee's number "
            f"and gender."
        )
        if addressee_gender is not None:
            base += (
                f" The most likely addressee in this exchange is "
                f"{addressee_gender}; prefer that form for singular \"you\" "
                f"unless context clearly implies a different addressee."
            )
        base += (
            " Infer number from context — collective cues like \"you all\", "
            "\"you guys\", or plural verbs imply plural. When number is "
            "ambiguous in a multi-person scene, prefer the inclusive plural "
            "form (e.g., אתם in Hebrew). Do not mix forms within a single line."
        )
    base += (
        " Return a JSON object with a 'translations' array containing exactly "
        "one translated string per input line, in the same order."
    )
    return base
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py -v`

Expected: all tests pass (the 4 original async tests + 3 new prompt tests = 7 passed).

- [ ] **Step 5: Run full suite to confirm no regressions**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: 27 passed (24 prior + 3 new). No existing test should break — the addressee paragraph only appears when `gender is not None`, and the existing translate-async tests pass `gender="male"` / `"female"` / `None` but never inspect the prompt body.

- [ ] **Step 6: Commit**

```bash
git add pipeline/translate.py tests/test_translate_async.py
git -c commit.gpgsign=false commit -m "feat(translate): system prompt addressee paragraph"
```

---

## Task 2: Thread `addressee_gender` through translate functions

**Files:**
- Modify: `pipeline/translate.py` — `_translate_one_batch`, `translate_batch`, `_translate_one_batch_async`, `translate_batch_async` signatures + forwarding.
- Test: `tests/test_translate_async.py` (new test using a recording fake)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_translate_async.py` (reuses `_FakeBlock`, `_FakeResponse`, `_FakeMessages`, `_FakeAsyncClient` defined at the top of the file):

```python
class _RecordingMessages:
    """Captures the system prompt of every create() call."""

    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = 0
        self.systems: list[str] = []

    async def create(self, **kwargs):
        self.systems.append(kwargs.get("system", ""))
        text = self._payloads[self.calls]
        self.calls += 1
        return _FakeResponse(text)


class _RecordingAsyncClient:
    def __init__(self, payloads):
        self.messages = _RecordingMessages(payloads)


@pytest.mark.asyncio
async def test_translate_batch_async_forwards_addressee_into_prompt():
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    await translate.translate_batch_async(
        ["hello"], "female", "Hebrew", client, addressee_gender="male",
    )
    assert "male" in client.messages.systems[0]
    assert "addressee" in client.messages.systems[0].lower()


@pytest.mark.asyncio
async def test_translate_batch_async_no_addressee_when_unset():
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    await translate.translate_batch_async(["hello"], "female", "Hebrew", client)
    assert "addressee" not in client.messages.systems[0].lower()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py::test_translate_batch_async_forwards_addressee_into_prompt tests/test_translate_async.py::test_translate_batch_async_no_addressee_when_unset -v`

Expected: FAIL — `translate_batch_async()` got an unexpected keyword argument `'addressee_gender'`.

- [ ] **Step 3: Add `addressee_gender` to all four batch functions**

In `pipeline/translate.py`:

REPLACE the `_translate_one_batch` signature + body (currently lines 88-116) with:

```python
def _translate_one_batch(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: anthropic.Anthropic,
    addressee_gender: str | None = None,
) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    response = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(target_language, gender, addressee_gender),
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
```

REPLACE the `translate_batch` signature + body (currently lines 119-135) with:

```python
def translate_batch(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: anthropic.Anthropic,
    addressee_gender: str | None = None,
) -> list[str]:
    """Translate ``texts`` into ``target_language``, returning one string each.

    ``gender`` is ``"male"``/``"female"`` for gender-aware languages, or ``None``
    to request a plain translation. ``addressee_gender`` (optional) hints the
    grammatical "you" form for languages where second-person is gender-marked.
    Output length always equals input length.
    """
    if not texts:
        return []
    out: list[str] = []
    for batch in _chunks(texts):
        out.extend(
            _translate_one_batch(batch, gender, target_language, client, addressee_gender)
        )
    return out
```

REPLACE the `_translate_one_batch_async` signature + body with:

```python
async def _translate_one_batch_async(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: anthropic.AsyncAnthropic,
    addressee_gender: str | None = None,
) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    response = await client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(target_language, gender, addressee_gender),
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
```

REPLACE the `translate_batch_async` signature + body with:

```python
async def translate_batch_async(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: anthropic.AsyncAnthropic,
    addressee_gender: str | None = None,
) -> list[str]:
    """Async counterpart of ``translate_batch`` for the chunked orchestrator.

    Sub-batches run sequentially within one call; cross-chunk concurrency is
    handled by the caller's semaphore. ``addressee_gender`` (optional) hints the
    grammatical "you" form. Output length always equals input length.
    """
    if not texts:
        return []
    out: list[str] = []
    for batch in _chunks(texts):
        out.extend(
            await _translate_one_batch_async(
                batch, gender, target_language, client, addressee_gender
            )
        )
    return out
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py -v`

Expected: 9 passed (7 from Task 1 + 2 new).

- [ ] **Step 5: Run full suite — confirm no regressions**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: 29 passed. All earlier translate tests (including the parse-failure fallback test) still pass because they don't pass `addressee_gender`; the default `None` preserves their existing behavior.

- [ ] **Step 6: Commit**

```bash
git add pipeline/translate.py tests/test_translate_async.py
git -c commit.gpgsign=false commit -m "feat(translate): thread addressee_gender through batch helpers"
```

---

## Task 3: Orchestrator — addressee rotation + cross-chunk carry

**Files:**
- Modify: `server.py` — `_translate_chunk` (new signature: takes pre-built groups + `prev_speaker_gender`), `_run_gender_aware` (builds groups before launching task, threads `prev_speaker_gender`).
- Modify: `tests/test_orchestration.py` — update existing fakes' signatures (accept `addressee_gender=None`); add two new tests.

- [ ] **Step 1: Write the failing tests**

In `tests/test_orchestration.py`:

**(a) Update the two existing fakes** so they accept the new kwarg. The fake inside `_install_fakes` and the one inside `test_chunk_local_offset_maps_speakers` both currently look like:

```python
    async def fake_translate(texts, gender, target, client):
        return [f"{t}|{gender}" for t in texts]
```

Change BOTH occurrences to:

```python
    async def fake_translate(texts, gender, target, client, addressee_gender=None):
        return [f"{t}|{gender}" for t in texts]
```

**(b) Append two new tests** to `tests/test_orchestration.py` (uses existing top-of-file imports and the existing `two_chunk_segments` fixture):

```python
@pytest.mark.asyncio
async def test_addressee_rotates_within_chunk(monkeypatch):
    # Three consecutive groups in a single chunk: M then F then M.
    # Each segment is short enough (1s each) that make_chunks keeps them in one chunk
    # with target_sec=5; assign_speaker returns the speaker label we map by segment.
    segs = [
        Segment(start=0.0, end=1.0, text="a"),
        Segment(start=1.0, end=2.0, text="b"),
        Segment(start=2.0, end=3.0, text="c"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segs))
    monkeypatch.setattr(server, "_load_wav_mono", lambda path: (np.zeros(16000 * 5, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "S_M1"
    ann[PSegment(1.0, 2.0)] = "S_F"
    ann[PSegment(2.0, 3.0)] = "S_M2"
    monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a, **k: ann)
    monkeypatch.setattr(
        server.gender, "detect_genders",
        lambda audio, sr, a: {"S_M1": "male", "S_F": "female", "S_M2": "male"},
    )

    addressees: list[str | None] = []

    async def fake_translate(texts, gender, target, client, addressee_gender=None):
        addressees.append(addressee_gender)
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]

    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Three groups -> three translate calls, with addressee rotating from the
    # previous group's speaker gender. The very first group has addressee=None
    # (no prior speaker exists at all).
    assert addressees == [None, "male", "female"]
    assert [s.text for s in out] == ["a|male|None", "b|female|male", "c|male|female"]


@pytest.mark.asyncio
async def test_addressee_carries_across_chunks(monkeypatch):
    # Two chunks. Chunk 1 ends with a female speaker. Chunk 2's first group should
    # receive addressee_gender="female" (carried from chunk 1's tail).
    segs = [
        Segment(start=0.0, end=6.0, text="a"),   # chunk 1, group 0
        Segment(start=6.0, end=12.0, text="b"),  # chunk 2, group 0
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segs))
    monkeypatch.setattr(server, "_load_wav_mono", lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    # Each chunk independently diarizes to one speaker covering its full slice-local
    # range [0, 6). The genders differ by chunk.
    chunk_index = {"i": 0}
    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "S"
        return ann
    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)

    def fake_detect(audio, sr, ann):
        # Chunk 1 -> female, chunk 2 -> male, based on call order.
        result = {"S": "female" if chunk_index["i"] == 0 else "male"}
        chunk_index["i"] += 1
        return result
    monkeypatch.setattr(server.gender, "detect_genders", fake_detect)

    addressees: list[str | None] = []

    async def fake_translate(texts, gender, target, client, addressee_gender=None):
        addressees.append(addressee_gender)
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]
    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

    out = await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Chunk 1 first group: no prior speaker -> None.
    # Chunk 2 first group: prior chunk's last group was female -> "female".
    # asyncio.gather order is by chunk index, but tasks may resolve in any order;
    # sort by which chunk's segment we know was translated.
    assert sorted(addressees, key=lambda x: (x is not None, x)) == sorted([None, "female"], key=lambda x: (x is not None, x))
    # Stronger: identify each chunk's call by inspecting output text.
    text_by_seg = {s.text.split("|")[0]: s.text for s in out}
    assert text_by_seg["a"] == "a|female|None"
    assert text_by_seg["b"] == "b|male|female"
```

- [ ] **Step 2: Run the new tests to confirm they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py::test_addressee_rotates_within_chunk tests/test_orchestration.py::test_addressee_carries_across_chunks -v`

Expected: FAIL — `_translate_chunk` doesn't pass `addressee_gender` yet (the fake never receives a value other than `None`, so the rotation assertion fails); the cross-chunk carry isn't implemented either.

ALSO: the *existing* orchestration tests should still pass (or fail in a predictable way) — the fake signature update in Step 1(a) was a prerequisite. If the existing tests fail with `unexpected keyword argument 'addressee_gender'`, that's expected pre-implementation but only because the orchestrator now passes the kwarg; ensure you completed Step 1(a) for both fakes before moving on.

- [ ] **Step 3: Refactor `_translate_chunk` to take pre-built groups + `prev_speaker_gender`**

In `server.py`, REPLACE the entire `_translate_chunk` function (currently lines 199-227) with:

```python
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
```

- [ ] **Step 4: Update `_run_gender_aware` to build groups outside the task and thread `prev_speaker_gender`**

In `server.py`, REPLACE the entire `_run_gender_aware` function (currently lines 230-264) with:

```python
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
```

- [ ] **Step 5: Run the orchestration tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py -v`

Expected: 6 passed (4 prior + 2 new). The 4 prior tests pass because:
- They monkeypatch `fake_translate` with the now-`addressee_gender`-accepting signature (Step 1(a)).
- They don't inspect `addressee_gender`, so any value is fine.
- The first/only group of each chunk receives `addressee_gender=None` (no prior speaker), and the existing `f"{t}|{gender}"` outputs are unchanged.

- [ ] **Step 6: Run the full suite — confirm no regressions**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: 31 passed (29 after Task 2 + 2 new in this task).

- [ ] **Step 7: Commit**

```bash
git add server.py tests/test_orchestration.py
git -c commit.gpgsign=false commit -m "feat(server): rotate addressee per group, carry across chunks"
```

---

## Task 4: End-to-end manual verification (no new code)

**Files:** none (manual).

- [ ] **Step 1: Restart the server on `feature/addressee-gender`**

Stop any running instance on port 9000 (`Get-NetTCPConnection -LocalPort 9000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }`), then:

`.\.venv\Scripts\python.exe server.py *>> "$env:TEMP\wg_server.log" 2>&1` (background)

Poll `GET http://localhost:9000/` until it returns `{"status":"ok",...}`.

- [ ] **Step 2: Re-run the two-speaker smoke clip**

POST the existing short two-speaker WAV (`$env:TEMP\wg_sample.wav`) with `output=srt`, `encode=true`. Expected: valid Hebrew SRT, two lines, gendered correctly (the male's first line should keep "הולך" — masculine self-reference is unchanged). Optionally inspect the female's line: with the new addressee hint, the addressee "אליך" (masc. sing.) should still appear because the prior speaker was male — same as before, just now driven explicitly instead of from context.

- [ ] **Step 3 (optional): Re-trigger the Bazarr request on a real episode**

Trigger a fresh subtitle search in Bazarr for an episode with multi-speaker scenes. Compare the resulting Hebrew SRT against yesterday's output for the same kind of episode. Spot-check a few lines that contain "you" — the masculine plural `אתם` should appear in group scenes; addressee gender should follow the previous on-screen speaker in dialogue.

- [ ] **Step 4: Final full-suite run**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all 31 green.

---

## Notes for the implementer

- Run pytest from the repo root (`D:\git\whisper-gend`) using the venv interpreter `.\.venv\Scripts\python.exe -m pytest`.
- Do NOT read or edit `.env` — it holds secrets and is gitignored. All translate tests use fakes.
- `_translate_chunk`'s `groups` parameter is the return value of `_group_consecutive` — a `list[tuple[speaker_label | None, list[Segment]]]`. The orchestrator builds it; the task consumes it.
- The "addressee = previous group's speaker gender" rule applies independent of speaker label — two consecutive groups by different *male* speakers still correctly produce `addressee_gender="male"` for the second group.
- The cross-chunk carry uses the chunk's final group's speaker gender, not the gender of the chunk's most-spoken speaker — what matters is "who just finished talking" right before the chunk seam.

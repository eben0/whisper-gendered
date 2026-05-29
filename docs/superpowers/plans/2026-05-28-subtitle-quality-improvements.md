# Subtitle Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address nine subtitle-quality issues reported from real Bazarr playback: missing sentences, timing drift, line-break styling, stuck (over-long) subtitles, wrong speaker gender (S04E05 @ 04:29), wrong addressee gender (05:00), wrong preposition translation (05:06), names left in English (T, Mondo @ 05:25), and untranslated slang.

**Architecture:** Each issue belongs to a different subsystem (translation prompt / SRT post-processor / Whisper timestamps / pyannote gender / orchestrator carry / pipeline diagnostics). The plan is **seven independently-shippable tasks**, each its own commit. Tasks 1–3 are deterministic fixes with high ROI; Tasks 4–7 begin with a tight investigation step before the code change, since the right fix depends on what the data shows.

**Tech Stack:** Python 3.12, FastAPI, faster-whisper (large-v3), pyannote 3.1, Anthropic Claude, librosa.

---

## Issues → Task crosswalk

| # | Issue (with example from S04E05) | Task |
|---|---|---|
| 1 | Wrong preposition: 05:06 — `את` should be `ב` or `של` | **Task 1** (prompt) |
| 2 | Names not transliterated: `יו, T` / `מה זה, Mondo?` @ 05:25 | **Task 1** (prompt) |
| 3 | Slang translated literally rather than idiomatically | **Task 1** (prompt) |
| 4 | Long sentences not broken into 2 lines | **Task 2** (formatter) |
| 5 | Stuck subtitles — 78/978 segs >7s, worst is **69.4s** at 00:55:28 | **Task 3** (max-duration splitter) |
| 6 | Missing sentences (untracked — could be Whisper or translate loss) | **Task 4** (pipeline metrics) |
| 7 | Timing drift — HE 04:09 vs EN 04:05 for same line | **Task 5** (word-timestamp drift audit) |
| 8 | Wrong speaker gender @ 04:29 (should be Female, classified Male) | **Task 6** (gender detection) |
| 9 | Wrong addressee gender @ 05:00 (carry from prior group) | **Task 7** (addressee carry audit) |

**Recommended execution order:** 1 → 2 → 3 → 4 → (5, 6, 7 in any order; each begins with a diagnostic step that may obviate or reshape the fix).

---

## File structure

| File | Responsibility | Tasks that touch it |
|---|---|---|
| `pipeline/translate.py` | Claude system prompt | Task 1 |
| `pipeline/format.py` | SRT/VTT/TXT rendering | Task 2, Task 3 |
| `pipeline/transcribe.py` | faster-whisper call + Segment dataclass | Task 5 |
| `pipeline/gender.py` | F0-based speaker gender | Task 6 |
| `server.py` | orchestrator (`_translate_chunk`, `run_pipeline_async`) | Task 4, Task 7 |
| `tests/test_translate_async.py` | translation prompt tests | Task 1 |
| `tests/test_format.py` (new for line-break tests) | formatter tests | Task 2, Task 3 |
| `tests/test_orchestration.py` | end-to-end orchestrator tests | Task 4, Task 7 |
| `tests/test_gender.py` (new) | gender detection tests | Task 6 |

---

## Task 1: Translation-prompt improvements — names, slang, prepositions, line-break hint

**Files:**
- Modify: `D:\git\whisper-gend\pipeline\translate.py:47-87` (`_system_prompt`)
- Test: `D:\git\whisper-gend\tests\test_translate_async.py`

The current prompt explicitly says *"do not add notes, explanations, or transliteration."* That bans the very behavior we want for proper nouns. Three additions: (a) transliterate names in the target's script; (b) translate slang/idioms to their target-language equivalents rather than literal renderings; (c) prefer the natural target-language preposition (`ב`/`של` over `את` in Hebrew when grammar allows). Also: hint Claude to keep each line under ~42 chars so the formatter (Task 2) has natural break-points.

- [ ] **Step 1.1: Write failing tests pinning the new prompt contract**

Add to `tests/test_translate_async.py`:

```python
def test_system_prompt_asks_for_transliterated_names():
    sp = translate._system_prompt("Hebrew", None)
    # The old prompt banned all transliteration; the new one must allow
    # (and require) transliteration of proper nouns.
    assert "transliterat" in sp.lower()
    # Must not flatly forbid transliteration anywhere in the prompt.
    forbidding = [line for line in sp.split(".") if "do not" in line.lower()
                  and "transliterat" in line.lower()]
    assert forbidding == [], f"prompt still forbids transliteration: {forbidding}"


def test_system_prompt_asks_for_idiomatic_slang():
    sp = translate._system_prompt("Hebrew", None)
    # Use a stable substring — re-word safely if the prompt text changes.
    assert "slang" in sp.lower() or "idiomatic" in sp.lower()


def test_system_prompt_prefers_natural_prepositions_for_hebrew():
    sp = translate._system_prompt("Hebrew", None)
    # Hebrew-specific guidance must mention the natural-preposition rule.
    # We don't pin exact wording, just the intent.
    assert "preposition" in sp.lower() or 'את' in sp


def test_system_prompt_hints_max_chars_per_line():
    sp = translate._system_prompt("Hebrew", None)
    # The formatter (Task 2) is a backstop; the prompt should still steer
    # Claude away from one-shot 120-char outputs.
    import re
    assert re.search(r"\b(42|45|48|50)\b", sp) or "two lines" in sp.lower()
```

- [ ] **Step 1.2: Run the tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py -k "system_prompt" -v
```

Expected: four FAILs.

- [ ] **Step 1.3: Update `_system_prompt` in `pipeline/translate.py`**

Replace the existing function (lines 47-87) with:

```python
def _system_prompt(
    target_language: str,
    gender: str | None,
    addressee_gender: str | None = None,
    source_language: str = "English",
) -> str:
    base = (
        f"You are an expert subtitle translator. Translate each numbered line "
        f"from {source_language} into {target_language}. Produce natural, idiomatic, "
        f"concise {target_language} suitable for on-screen subtitles. Preserve "
        f"meaning and tone; do not add notes or explanations."
    )
    # Style guidance — applies to every translation regardless of gender.
    base += (
        f" Transliterate proper nouns (people's names, place names, brand "
        f"names, nicknames) into the {target_language} script: write them as "
        f"the audience would naturally read them, not as Latin letters."
    )
    base += (
        f" Render slang, idioms, and figures of speech with their natural "
        f"{target_language} equivalents — not literal word-for-word "
        f"calques. If a {source_language} idiom has no clean equivalent, "
        f"prefer the closest colloquial phrasing in {target_language}."
    )
    base += (
        f" Choose the natural {target_language} preposition for each "
        f"construction. For Hebrew specifically: prefer ב, של, ל, מ, על "
        f"where the grammar calls for them; reserve את only for marking a "
        f"definite direct object — never use את as a generic substitute."
    )
    base += (
        " Keep each line short — aim for ≤ 45 characters per line. For "
        " longer utterances, prefer two short lines over one long one; "
        " a downstream formatter may also split, so end thoughts on "
        " natural pause-points (after a comma, conjunction, or clause "
        " boundary) when possible."
    )
    if gender is not None:
        base += (
            f" The speaker of these lines is {gender}. Use grammatically correct "
            f"{gender} forms throughout — verb conjugation, adjective and "
            f"participle agreement, imperatives, and pronouns must all match a "
            f"{gender} speaker referring to themselves."
        )
        base += (
            f" When the speaker addresses another person ({source_language} \"you\""
            f" or its equivalent), choose the {target_language} form matching the "
            f"addressee's number and gender."
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

- [ ] **Step 1.4: Run the new tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py -k "system_prompt" -v
```

Expected: 4 passed.

- [ ] **Step 1.5: Run the full suite to confirm no regression**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: ≥ 75 passed, 1 skipped (existing 71 + 4 new).

- [ ] **Step 1.6: Commit**

```powershell
git add pipeline/translate.py tests/test_translate_async.py
git -c commit.gpgsign=false commit -m "feat(translate): prompt now asks for transliterated names, idiomatic slang, natural prepositions, and ≤45-char lines"
```

---

## Task 2: SRT post-processor — break long lines into ≤2 lines

**Files:**
- Modify: `D:\git\whisper-gend\pipeline\format.py`
- Test: Create `D:\git\whisper-gend\tests\test_format.py`

Even with the prompt hint from Task 1, Claude will sometimes return a 90-character single-line subtitle. Standard SRT readability is ≤ 42 chars × 2 lines. Add a deterministic post-processor that splits any subtitle line longer than `MAX_LINE_CHARS` into two balanced halves, preferring a break at the nearest punctuation/whitespace point near the midpoint.

- [ ] **Step 2.1: Read the current formatter to see where to wire the splitter**

```powershell
Get-Content D:\git\whisper-gend\pipeline\format.py
```

Locate where each subtitle's text is rendered into the SRT output. The splitter is called per-segment immediately before that render.

- [ ] **Step 2.2: Write failing tests**

Create `tests/test_format.py`:

```python
"""Tests for the multi-line SRT splitter.

The splitter takes a single subtitle text and returns 1- or 2-line text
suitable for on-screen display. Break point preference (in order):
  1. After punctuation (, . ! ? : ;) closest to mid-point
  2. After a whitespace closest to mid-point
  3. Hard-wrap at mid-point as last resort
Never produces > 2 lines.
"""

from pipeline.format import _split_to_two_lines

MAX = 42  # the configured per-line target


def test_short_text_is_unchanged():
    assert _split_to_two_lines("Hi there.", MAX) == "Hi there."


def test_exactly_max_is_unchanged():
    text = "a" * MAX
    assert _split_to_two_lines(text, MAX) == text


def test_splits_at_punctuation_near_midpoint():
    # Comma near the middle wins over whitespace at the middle.
    text = "I have nothing to say, but I will say it anyway."
    out = _split_to_two_lines(text, MAX)
    assert out.count("\n") == 1
    first, second = out.split("\n")
    # Comma stays at end of the first line.
    assert first.endswith(",")
    assert first == "I have nothing to say,"
    assert second == "but I will say it anyway."


def test_splits_on_whitespace_when_no_punctuation_near_midpoint():
    text = "the quick brown fox jumps over the lazy dog at noon"
    out = _split_to_two_lines(text, MAX)
    assert out.count("\n") == 1
    a, b = out.split("\n")
    assert len(a) <= MAX and len(b) <= MAX
    # No word was split across the line break.
    assert " ".join(a.split()) + " " + " ".join(b.split()) == text


def test_hebrew_break_at_comma():
    text = "אני אומר לך, זה לא יעבוד בשום אופן בכלל היום"
    out = _split_to_two_lines(text, MAX)
    assert out.count("\n") == 1
    a, _ = out.split("\n")
    assert a.endswith(",")


def test_never_produces_three_lines_even_for_very_long_text():
    text = " ".join(["word"] * 50)  # 250 chars
    out = _split_to_two_lines(text, MAX)
    assert out.count("\n") <= 1


def test_preserves_existing_newline():
    # If the upstream already split, don't second-guess.
    text = "Line one.\nLine two is also short."
    assert _split_to_two_lines(text, MAX) == text
```

- [ ] **Step 2.3: Run tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_format.py -v
```

Expected: 7 FAILs with "cannot import name '_split_to_two_lines'".

- [ ] **Step 2.4: Implement `_split_to_two_lines` in `pipeline/format.py`**

Add to `pipeline/format.py` (place above the SRT rendering function):

```python
import re

# Default per-line character target for subtitle readability. 42 is the
# BBC/Netflix convention; we use it for both Latin and Hebrew text since
# Hebrew character width is comparable.
SRT_MAX_LINE_CHARS = 42

# Punctuation marks we prefer to break AFTER (the mark stays on the
# leading line). Ordered by strength: full stops outrank commas only
# when both are roughly equidistant from the midpoint.
_BREAK_AFTER = ".!?;:," + "־"  # last char is Hebrew maqaf

_WHITESPACE_RE = re.compile(r"\s+")


def _split_to_two_lines(text: str, max_chars: int = SRT_MAX_LINE_CHARS) -> str:
    """Wrap ``text`` to at most two lines, each ≤ ``max_chars`` when possible.

    Break-point preference, applied in this order:
    1. Punctuation in ``_BREAK_AFTER`` whose position is closest to the
       string's midpoint (so both halves are roughly balanced).
    2. Whitespace closest to the midpoint.
    3. Hard cut at the midpoint as a last resort.

    If the input already contains a newline, it is returned unchanged on the
    assumption the upstream (e.g. the LLM) deliberately broke it.
    """
    if "\n" in text:
        return text
    if len(text) <= max_chars:
        return text

    mid = len(text) // 2

    # Pass 1: prefer the closest break-after punctuation in a window
    # spanning roughly the middle half of the string. Mid-point ± 1/4 length.
    window_lo = max(0, mid - len(text) // 4)
    window_hi = min(len(text), mid + len(text) // 4)
    best_punct = -1
    best_punct_dist = len(text)
    for i in range(window_lo, window_hi):
        if text[i] in _BREAK_AFTER:
            # break point is AFTER the punctuation (so the mark stays on
            # the leading line). Score by distance from midpoint.
            d = abs((i + 1) - mid)
            if d < best_punct_dist:
                best_punct_dist = d
                best_punct = i + 1
    if best_punct > 0:
        first = text[:best_punct].rstrip()
        second = text[best_punct:].lstrip()
        return f"{first}\n{second}"

    # Pass 2: whitespace closest to the midpoint.
    best_ws = -1
    best_ws_dist = len(text)
    for m in _WHITESPACE_RE.finditer(text):
        d = abs(m.start() - mid)
        if d < best_ws_dist:
            best_ws_dist = d
            best_ws = m.start()
    if best_ws > 0:
        first = text[:best_ws].rstrip()
        second = text[best_ws:].lstrip()
        return f"{first}\n{second}"

    # Pass 3: pathological case (no whitespace at all). Hard cut.
    return f"{text[:mid]}\n{text[mid:]}"
```

- [ ] **Step 2.5: Wire the splitter into the SRT renderer**

In the same file, find the SRT rendering loop. Replace the line that emits the segment text with one that calls `_split_to_two_lines` first. For example, if the current code looks like:

```python
out.append(f"{i+1}\n{ts_a} --> {ts_b}\n{seg.text}\n")
```

change it to:

```python
out.append(f"{i+1}\n{ts_a} --> {ts_b}\n{_split_to_two_lines(seg.text)}\n")
```

Do the same for the VTT renderer if present (same rule applies).

- [ ] **Step 2.6: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_format.py -v
```

Expected: 7 passed.

- [ ] **Step 2.7: Eyeball a real SRT before vs after**

Pick one episode SRT that already exists, render it through the new function as a smoke check:

```powershell
.\.venv\Scripts\python.exe -c @"
from pipeline.format import _split_to_two_lines
samples = [
    'I have nothing to say, but I will say it anyway.',
    'Last 200 years, scientists, sociologists, and other folks who fret about such things have debated whether a person commits a violent act.',
    'אה, אז אתה רוצה שאספר לחבר'ה שאתה סמוי?',
]
for s in samples:
    print('---'); print(s); print('->'); print(_split_to_two_lines(s))
"@
```

Expected: each long sample is broken into two roughly-balanced lines on a natural boundary.

- [ ] **Step 2.8: Commit**

```powershell
git add pipeline/format.py tests/test_format.py
git -c commit.gpgsign=false commit -m "feat(format): wrap long subtitle text to ≤2 balanced lines on punctuation/whitespace"
```

---

## Task 3: Stuck-subtitle splitter — break segments whose duration > MAX_SUB_SEC

**Files:**
- Modify: `D:\git\whisper-gend\pipeline\format.py`
- Test: `D:\git\whisper-gend\tests\test_format.py`

Confirmed in the S04E05 output: 78 of 978 segments are >7s; the worst is **69.4s** for the single line *"בבית של הסבים."* at 00:55:28. That happens when Whisper produces one mega-segment over a silence-heavy span and we never split it. Fix: when a segment's duration exceeds `MAX_SUB_SEC` AND its text is short (so we can't legitimately fill that span with reading time), shorten its display end to `start + reading_time(text)` capped at `MAX_SUB_SEC`. Long texts that legitimately need a long span are left alone.

- [ ] **Step 3.1: Write failing tests**

Append to `tests/test_format.py`:

```python
from pipeline.transcribe import Segment
from pipeline.format import _shorten_overlong_subtitle, MAX_SUB_SEC


def test_short_text_long_duration_is_shortened():
    # 1-word subtitle held for 70s — clearly a Whisper artefact.
    seg = Segment(start=10.0, end=80.0, text="שלום.")
    out = _shorten_overlong_subtitle(seg)
    assert out.start == 10.0
    assert out.end < 80.0
    assert out.end - out.start <= MAX_SUB_SEC


def test_legit_long_text_keeps_long_duration():
    # 200-character line genuinely needs ~10s reading time.
    text = "א" * 200
    seg = Segment(start=10.0, end=22.0, text=text)
    out = _shorten_overlong_subtitle(seg)
    # Reading-time floor: ~15 cps gives ~13s minimum. Should not shorten.
    assert out.end == 22.0


def test_normal_duration_unchanged():
    seg = Segment(start=10.0, end=12.5, text="A normal line.")
    out = _shorten_overlong_subtitle(seg)
    assert out.start == seg.start and out.end == seg.end and out.text == seg.text


def test_reading_time_uses_floor_min_1s():
    # Single character — reading time floor must be at least 1.0s.
    seg = Segment(start=0.0, end=30.0, text="א")
    out = _shorten_overlong_subtitle(seg)
    assert out.end - out.start >= 1.0
```

- [ ] **Step 3.2: Run tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_format.py -k "shorten or overlong" -v
```

Expected: 4 FAILs.

- [ ] **Step 3.3: Implement `_shorten_overlong_subtitle` in `pipeline/format.py`**

Add below `_split_to_two_lines`:

```python
# Display-time bounds. MAX_SUB_SEC caps how long any subtitle can sit
# on screen (subtitle conventions are 6–7s); the reading-time formula
# scales the actual cap by text length so a 200-char line still gets
# the time it needs.
MAX_SUB_SEC = 7.0
CHARS_PER_SECOND = 15.0   # avg adult reading speed (Netflix uses 17)
MIN_SUB_SEC = 1.0          # never flash anything for less than a second


def _shorten_overlong_subtitle(seg: "Segment") -> "Segment":
    """If ``seg`` sits on screen far longer than its text needs, shorten
    its end timestamp to a reading-time-derived value, capped at
    ``MAX_SUB_SEC``.

    Whisper produces these artefacts when a short utterance falls in a
    silence-heavy span — its segment timestamps swallow the silence and
    we end up holding "Hello." on screen for 60 seconds.
    """
    from pipeline.transcribe import Segment
    duration = seg.end - seg.start
    if duration <= MAX_SUB_SEC:
        return seg
    # Strip the wrapping newline (Task 2 may have inserted one).
    raw_len = len(seg.text.replace("\n", " ").strip())
    reading_time = max(MIN_SUB_SEC, raw_len / CHARS_PER_SECOND)
    new_end = seg.start + min(MAX_SUB_SEC, reading_time)
    if new_end >= seg.end:
        return seg  # text genuinely needs the time
    return Segment(seg.start, new_end, seg.text)
```

- [ ] **Step 3.4: Wire the shortener into the SRT renderer**

In the same file, in the SRT renderer, apply `_shorten_overlong_subtitle` to each segment before computing its end timestamp. Combined with Task 2, the renderer pipeline becomes:

```python
for i, seg in enumerate(segments):
    seg = _shorten_overlong_subtitle(seg)
    ts_a = _fmt_srt_time(seg.start)
    ts_b = _fmt_srt_time(seg.end)
    out.append(f"{i+1}\n{ts_a} --> {ts_b}\n{_split_to_two_lines(seg.text)}\n")
```

(Adapt to the actual current SRT-renderer code.)

- [ ] **Step 3.5: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_format.py -v
```

Expected: 11 passed (7 from Task 2 + 4 from Task 3).

- [ ] **Step 3.6: Commit**

```powershell
git add pipeline/format.py tests/test_format.py
git -c commit.gpgsign=false commit -m "feat(format): cap subtitle display time at reading-time × cap, fixes ≤7s stuck-subtitle issue"
```

---

## Task 4: Pipeline metrics — detect dropped segments

**Files:**
- Modify: `D:\git\whisper-gend\server.py` (`_translate_chunk`, `run_pipeline_async`)
- Test: `D:\git\whisper-gend\tests\test_orchestration.py`

"Missing sentences" could come from (a) Whisper missing the utterance, (b) chunking dropping a segment, (c) translation backend returning fewer items than input, or (d) the alignment fallback in `pipeline/translate.py` discarding text. Today we silently log `"translation count mismatch"` if Claude returns the wrong number, and align by index — but we don't log per-request **input segment count vs output segment count**, which is the easy diagnostic. Add an exact counter and an error-level alert when they differ.

- [ ] **Step 4.1: Investigate Whisper output for any actually-empty segments first**

Before adding metrics, confirm with one diagnostic command that this is worth fixing:

```powershell
.\.venv\Scripts\python.exe -c "
from faster_whisper import WhisperModel
import os
m = WhisperModel('large-v3', device='cuda', compute_type='float16')
# Use any short audio file. If you don't have one handy, skip step 4.1.
" 2>&1 | Select-String -NotMatch -Pattern 'libtorchcodec|FFmpeg|warn|Traceback|raise OS' | Select-Object -Last 3
```

Then visually skim `Z:\media\tv\Oz (1997)\Season 04\Oz (1997) - S04E05 - Gray Matter [...].he.srt` and the corresponding .en.srt for any timestamps where you remember dialogue but no segment appears.

- [ ] **Step 4.2: Write failing test for the new counter log line**

Add to `tests/test_orchestration.py`:

```python
@pytest.mark.asyncio
async def test_orchestrator_logs_segment_counts(monkeypatch, caplog):
    """run_pipeline_async must log the input-segment and output-segment
    counts at INFO so a missing-segment investigation has hard numbers.
    """
    import logging
    caplog.set_level(logging.INFO, logger="server")

    segs = [Segment(s, s + 1.0, f"line {s}") for s in (0.0, 1.0, 2.0)]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.transcribe, "transcribe", lambda path, language="en": list(segs))
    monkeypatch.setattr(server, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 3, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    ann = Annotation()
    ann[PSegment(0.0, 3.0)] = "S"
    monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders",
                        lambda audio, sr, a: {"S": "male"})

    async def fake_translate(texts, gender, target, client,
                             addressee_gender=None, source_language="English"):
        return [f"HE: {t}" for t in texts]
    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

    source, target = await server.run_pipeline_async(server.Path("x.wav"), "en")
    msgs = [r.getMessage() for r in caplog.records]
    # New required log line — exact text not pinned, but it must mention
    # both counts and the word "segments".
    assert any("3" in m and "segments" in m.lower() and "transcribed" in m.lower()
               for m in msgs), f"missing transcribe-count log; got: {msgs}"
    assert any("3" in m and "translated" in m.lower()
               for m in msgs), f"missing translate-count log; got: {msgs}"
```

- [ ] **Step 4.3: Run the test — confirm it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py::test_orchestrator_logs_segment_counts -v
```

Expected: FAIL (no such log line exists yet).

- [ ] **Step 4.4: Add the counter logs to `run_pipeline_async`**

In `server.py`, modify `run_pipeline_async` (current body returns `(source_segments, target_segments)`):

```python
# After transcribe completes:
log.info("transcribed %d segments from audio", len(segments))

# After translate completes (just before the return statement that builds
# source_segments + target_segments):
log.info("translated %d segments (input was %d); difference = %d",
         len(target_segments), len(segments),
         len(segments) - len(target_segments))
if len(target_segments) != len(segments):
    log.error(
        "segment count mismatch — pipeline lost %d segments between "
        "transcribe and translate",
        len(segments) - len(target_segments),
    )
```

- [ ] **Step 4.5: Run test — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py::test_orchestrator_logs_segment_counts -v
```

Expected: PASS.

- [ ] **Step 4.6: Run full suite for regression**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: existing tests still pass.

- [ ] **Step 4.7: Commit**

```powershell
git add server.py tests/test_orchestration.py
git -c commit.gpgsign=false commit -m "feat(server): log segment counts after transcribe and translate; ERROR on mismatch"
```

- [ ] **Step 4.8: Run a real episode and grep the log for the new counters**

After deploying, watch `server.log` for the new lines. If `transcribed N`, `translated N`, `difference = 0`, the pipeline never drops segments. If `difference ≠ 0`, capture that request id and investigate the chunked-translate code path (`_translate_chunk` in `server.py`) — likely Claude returned a shorter JSON array than asked and the fallback aligner truncated.

---

## Task 5: Timing-drift root cause — word-level timestamps audit

**Files:**
- Modify: `D:\git\whisper-gend\pipeline\transcribe.py`
- Test: existing `tests/test_transcribe.py` (or add one if absent)

Confirmed drift: at ~04:09 in S04E05 our HE SRT shows *"Oh, so you do want me to tell the guys you're undercover?"* but the EN SRT (which Bazarr re-anchored) shows the same line starting at 04:05 — a 4s gap after only 4 minutes of audio. faster-whisper offers `word_timestamps=True`, which yields per-word times that can re-anchor segments precisely. This task first measures the magnitude of the drift, then decides between enabling word timestamps or applying a simpler segment-end clamp.

- [ ] **Step 5.1: Diagnostic — extract a 1-min clip from S04E05 and compare word vs segment times**

Run from a PowerShell:

```powershell
$mp4 = "Z:\media\tv\Oz (1997)\Season 04\Oz (1997) - S04E05 - Gray Matter [WEBRip-1080p][AAC 2.0][x265]-KONTRAST.mp4"
$wav = "$env:TEMP\drift_test.wav"
& C:\Users\eyalb\ffmpeg\bin\ffmpeg.exe -hide_banner -loglevel error -y -ss 240 -t 60 -i $mp4 -ac 1 -ar 16000 -f wav $wav
.\.venv\Scripts\python.exe -c @"
from faster_whisper import WhisperModel
m = WhisperModel('large-v3', device='cuda', compute_type='float16')
segs, info = m.transcribe(r'$wav', word_timestamps=True, vad_filter=True, beam_size=5)
for s in segs:
    word_starts = [w.start for w in (s.words or [])]
    drift = s.start - (word_starts[0] if word_starts else s.start)
    print(f'seg [{s.start:6.2f} -> {s.end:6.2f}] first-word={word_starts[0] if word_starts else None}  drift={drift:+.3f}s  text={s.text[:60]!r}')
"@ 2>&1 | Select-String -NotMatch 'libtorchcodec|warn|FFmpeg|Traceback|raise OS' | Select-Object -First 30
```

Record what you see: is the segment.start consistently late by ~1s? Is the first word's start more accurate than the segment.start?

- [ ] **Step 5.2: Write the failing test, contingent on Step 5.1 finding**

If word timestamps ARE more accurate (most likely), the fix is to enable them and use word.start for segment.start. Write the test:

```python
# Append to tests/test_transcribe.py (create if needed)
def test_transcribe_uses_word_anchor_when_words_available(monkeypatch):
    """Whisper's segment-level start can lag the first spoken word; when
    word_timestamps is enabled, we anchor segment.start to the first word's
    start to reduce drift over long audio.
    """
    from pipeline import transcribe

    class _Word:
        def __init__(self, start, end, word):
            self.start = start; self.end = end; self.word = word

    class _Seg:
        def __init__(self, start, end, text, words):
            self.start = start; self.end = end; self.text = text; self.words = words

    fake_segs = [_Seg(10.5, 14.0, "hello world", [_Word(10.0, 10.5, "hello"), _Word(10.5, 14.0, "world")])]
    class _FakeModel:
        def transcribe(self, *a, **k):
            return iter(fake_segs), type("Info", (), {"language": "en", "language_probability": 1.0})()
    monkeypatch.setattr(transcribe, "_get_model", lambda: _FakeModel())

    out = transcribe.transcribe("ignored.wav", language="en")
    assert len(out) == 1
    # Segment.start should equal the first word's start, not the model's
    # (delayed) seg.start.
    assert out[0].start == 10.0
    assert out[0].end == 14.0
```

- [ ] **Step 5.3: Run the test — confirm it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_transcribe.py -k "word_anchor" -v
```

Expected: FAIL.

- [ ] **Step 5.4: Update `pipeline/transcribe.py` to enable word timestamps and anchor**

In the `transcribe` function (line 69 area), change the `model.transcribe` call to pass `word_timestamps=True`, and in the segment materialisation loop, override `seg.start` with `seg.words[0].start` when available:

```python
def transcribe(audio_path: Path, language: str = "en") -> list[Segment]:
    model = _get_model()
    raw_segs, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,   # << was implicitly False
    )
    out: list[Segment] = []
    for s in raw_segs:
        start = s.start
        end = s.end
        # Anchor to the first/last word time when available — reduces the
        # drift that accumulates from Whisper's segment-level timestamps
        # over long audio.
        words = getattr(s, "words", None)
        if words:
            start = words[0].start
            end = words[-1].end
        out.append(Segment(start=start, end=end, text=s.text.strip()))
    log.info("Transcribed %d segments (detected language=%s, prob=%.2f)",
             len(out), info.language, info.language_probability)
    return out
```

- [ ] **Step 5.5: Run the new test — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_transcribe.py -k "word_anchor" -v
```

Expected: PASS.

- [ ] **Step 5.6: Run the full suite for regression**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 5.7: Re-run S04E05 via Bazarr after merging this task; compare timing drift**

After deploying, re-run the same episode through Bazarr. Compare the first 10 minutes of the new HE SRT to the previous run's timestamps. Drift should be reduced from ~4s/4min to ≤ 1s/10min.

- [ ] **Step 5.8: Commit**

```powershell
git add pipeline/transcribe.py tests/test_transcribe.py
git -c commit.gpgsign=false commit -m "feat(transcribe): enable word_timestamps and anchor segment start/end to first/last word, reduces drift over long audio"
```

---

## Task 6: Gender detection — improve robustness for the S04E05 04:29 case

**Files:**
- Modify: `D:\git\whisper-gend\pipeline\gender.py`
- Test: Create `D:\git\whisper-gend\tests\test_gender.py`

S04E05 @ 04:29 was classified male when the speaker is female. Without re-running pyannote on the actual audio, we can still harden the classifier in two ways: (a) require a minimum number of voiced frames before trusting a measurement (currently if very few voices frames exist, the median can land anywhere), (b) widen the F0 search range slightly so we don't miss low-end female voices.

- [ ] **Step 6.1: Diagnostic — confirm what F0 the current code measures at 04:29**

Extract the 04:29 region's audio:

```powershell
$mp4 = "Z:\media\tv\Oz (1997)\Season 04\Oz (1997) - S04E05 - Gray Matter [WEBRip-1080p][AAC 2.0][x265]-KONTRAST.mp4"
$wav = "$env:TEMP\gender_test_4_29.wav"
& C:\Users\eyalb\ffmpeg\bin\ffmpeg.exe -hide_banner -loglevel error -y -ss 269 -t 5 -i $mp4 -ac 1 -ar 16000 -f wav $wav
.\.venv\Scripts\python.exe -c @"
import librosa, numpy as np
y, sr = librosa.load(r'$wav', sr=16000, mono=True)
f0, voiced_flag, _ = librosa.pyin(y, fmin=65.0, fmax=300.0, sr=sr)
voiced = f0[~np.isnan(f0)]
print(f'voiced frames: {len(voiced)} / total: {len(f0)}')
print(f'median F0:    {np.nanmedian(f0):.1f} Hz' if len(voiced) > 0 else 'no voiced frames')
print(f'  P25 / P75:  {np.nanpercentile(f0, 25):.1f} / {np.nanpercentile(f0, 75):.1f} Hz')
"@
```

This tells you whether (a) librosa simply mis-detected the voice (median ≤165 = classified male), (b) too few voiced frames so the median is noisy, or (c) the speaker's actual F0 is truly low for a woman (creaky voice / vocal fry).

- [ ] **Step 6.2: Write failing tests**

Create `tests/test_gender.py`:

```python
"""Tests for the F0-based gender classifier."""
import numpy as np
from pipeline.gender import _classify_f0, MIN_VOICED_FRAMES, GENDER_THRESHOLD_HZ


def test_classify_female_above_threshold():
    f0 = np.array([200.0] * 50)
    assert _classify_f0(f0) == "female"


def test_classify_male_below_threshold():
    f0 = np.array([110.0] * 50)
    assert _classify_f0(f0) == "male"


def test_too_few_voiced_frames_defaults_to_male():
    # Tiny sample size — median is unreliable. The classifier should
    # not trust a near-threshold call from <MIN_VOICED_FRAMES frames.
    f0 = np.array([170.0] * (MIN_VOICED_FRAMES - 1))
    # Default to male (the historical default for ambiguity).
    assert _classify_f0(f0) == "male"


def test_nan_only_input_defaults_to_male():
    f0 = np.array([np.nan] * 100)
    assert _classify_f0(f0) == "male"


def test_boundary_case_at_threshold():
    # Exactly at threshold — classifier needs a deterministic answer.
    f0 = np.array([float(GENDER_THRESHOLD_HZ)] * 50)
    # Convention: ≤ threshold => male, > threshold => female.
    assert _classify_f0(f0) == "male"
```

- [ ] **Step 6.3: Run tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gender.py -v
```

Expected: FAILs with "cannot import name '_classify_f0' or 'MIN_VOICED_FRAMES'".

- [ ] **Step 6.4: Refactor `pipeline/gender.py` to expose `_classify_f0`**

Find the existing classification logic (the place that compares to `GENDER_THRESHOLD_HZ`). Extract it into a function:

```python
# Module-level constant — fewer voiced frames than this means the median
# F0 is too noisy to trust. Default to "male" (historical default for
# undetected/silent speakers).
MIN_VOICED_FRAMES = 30


def _classify_f0(f0: "np.ndarray") -> str:
    """Classify a per-frame F0 array into ``'male'``/``'female'``.

    Defaults to ``'male'`` when:
    - fewer than ``MIN_VOICED_FRAMES`` non-NaN values are present, OR
    - the median is exactly at or below the threshold (boundary rule).
    """
    import numpy as np
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < MIN_VOICED_FRAMES:
        return "male"
    median = float(np.median(voiced))
    return "female" if median > GENDER_THRESHOLD_HZ else "male"
```

Then update `detect_genders` to call `_classify_f0(f0)` instead of inlining the comparison. The threshold constant `GENDER_THRESHOLD_HZ` stays where it is.

- [ ] **Step 6.5: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gender.py -v
```

Expected: 5 passed.

- [ ] **Step 6.6: Run full suite for regression**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 6.7: Re-test S04E05 04:29 case after merging**

Re-trigger Bazarr Manual Search for S04E05. Inspect the new HE SRT around 04:29. If the speaker still classifies male, the median F0 is genuinely sub-threshold and a different fix is needed (e.g., per-speaker calibration); add a follow-up task and ship what we have.

- [ ] **Step 6.8: Commit**

```powershell
git add pipeline/gender.py tests/test_gender.py
git -c commit.gpgsign=false commit -m "feat(gender): require MIN_VOICED_FRAMES before trusting F0 median; extract _classify_f0"
```

---

## Task 7: Addressee carry audit — the S04E05 05:00 case

**Files:**
- Modify: `D:\git\whisper-gend\server.py` (`_translate_chunk`, possibly `_run_gender_aware`)
- Test: `D:\git\whisper-gend\tests\test_orchestration.py`

The addressee-rotation logic in `_translate_chunk` (server.py:418-440) sets `prev_group_gender` AFTER each group's translate completes, and `prev_speaker_gender` is the seed for the next chunk. The user observed at 05:00 that the addressee should be female but the system carried male. The likely cause: at a chunk boundary, the speaker label that diarization assigns to the first group of chunk N may not match the same person it was in chunk N-1 (pyannote labels are per-chunk). So the carry is correct in terms of what was said, but wrong in terms of who is being addressed.

This task is mostly verification — confirm the bug is what the model above describes — then add a test that pins the desired behavior, then decide whether the fix is (i) ignore cross-chunk carry, (ii) anchor by audio overlap between chunks, or (iii) leave as-is and disable via `ADDRESSEE_GENDER_HINT_ENABLED=false`.

- [ ] **Step 7.1: Verify by inspecting the server.log of the S04E05 run**

```powershell
Get-Content D:\git\whisper-gend\server.log |
  Select-String -Pattern "chunk \d+/\d+ diarize\+gender done|received audio_file|side-file saved" |
  Select-Object -First 40
```

For the S04E05 request (started 22:41:30), look at the chunk boundaries near 5:00. Note which chunk the speaker at 05:00 is in, and what gender mapping was emitted for SPEAKER_0X in that chunk.

- [ ] **Step 7.2: Open the SRT and the log together; map the 05:00 line to a chunk number**

```powershell
$srt = "Z:\media\tv\Oz (1997)\Season 04\Oz (1997) - S04E05 - Gray Matter [WEBRip-1080p][AAC 2.0][x265]-KONTRAST.he.srt"
Get-Content -LiteralPath $srt -Encoding UTF8 | Select-String -Pattern "00:0[4-5]:[0-9]{2}" | Select-Object -First 8
```

Cross-reference: if 05:00 is the last line of chunk-N and the speaker is female, then chunk-(N+1)'s first group's addressee should be female. Confirm or refute.

- [ ] **Step 7.3: Decide which fix to implement**

Three options, ranked by complexity:

- (a) **Disable cross-chunk carry.** Set `prev_speaker_gender = None` at the start of `_run_gender_aware` and never propagate beyond a chunk. Simplest, ships immediately. May regress within-chunk addressee accuracy slightly.
- (b) **Anchor across chunks by audio embedding.** Compute pyannote embeddings for the last group of chunk-N and the first group of chunk-(N+1); if cosine ≥ 0.75 it's the same speaker and the carry is valid. Otherwise drop the carry. Higher effort.
- (c) **Leave as-is, expose flag.** Already exists (`ADDRESSEE_GENDER_HINT_ENABLED`); user can disable globally if it hurts more than helps.

Pick **(a)** for this task unless Step 7.2 contradicts it.

- [ ] **Step 7.4: Write the failing test for option (a)**

Append to `tests/test_orchestration.py`:

```python
@pytest.mark.asyncio
async def test_addressee_does_not_carry_across_chunks(monkeypatch):
    """Cross-chunk addressee carry is unreliable because pyannote re-numbers
    speakers per chunk. The first group of chunk N+1 must therefore see
    addressee_gender=None, regardless of what chunk N's last speaker was.
    """
    segs = [
        Segment(start=0.0, end=6.0, text="a"),
        Segment(start=6.0, end=12.0, text="b"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 5)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 2)
    monkeypatch.setattr(server.settings, "ADDRESSEE_GENDER_HINT_ENABLED", True)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))
    monkeypatch.setattr(server, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 12, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    def fake_diarize(waveform, sr):
        ann = Annotation()
        ann[PSegment(0.0, 6.0)] = "S"
        return ann
    monkeypatch.setattr(server.diarize, "diarize_waveform", fake_diarize)

    chunk_idx = {"i": 0}
    def fake_detect(audio, sr, ann):
        result = {"S": "female" if chunk_idx["i"] == 0 else "male"}
        chunk_idx["i"] += 1
        return result
    monkeypatch.setattr(server.gender, "detect_genders", fake_detect)

    addressees: list[str | None] = []
    async def fake_translate(texts, gender, target, client,
                             addressee_gender=None, source_language="English"):
        addressees.append(addressee_gender)
        return [f"{t}|{gender}|{addressee_gender}" for t in texts]
    monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Two chunks, one group each — both first-of-chunk so addressee None.
    assert addressees == [None, None], (
        "cross-chunk addressee carry is still active; expected both None"
    )
```

This conflicts with an existing test `test_addressee_carries_across_chunks` (test_orchestration.py:142+). That test was the *prior* contract; we're explicitly inverting it. Mark that old test `@pytest.mark.skip(reason="cross-chunk carry removed in Task 7")` or delete it.

- [ ] **Step 7.5: Run tests — confirm new test fails, old test still exists**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py -k "addressee" -v
```

Expected: new test FAIL, old `test_addressee_carries_across_chunks` PASS (still asserting carry).

- [ ] **Step 7.6: Modify `_run_gender_aware` to disable cross-chunk carry**

In `server.py`, in `_run_gender_aware`, **delete** the line that updates `prev_speaker_gender` after building each chunk's groups (around line 484), AND set it to `None` at the start of each chunk by removing it from the call to `_translate_chunk`:

```python
for idx, chunk in enumerate(chunks):
    annotation, genders = await run_in_thread(
        _diarize_and_gender, audio, sr, chunk.start, chunk.end
    )
    log.info(
        "chunk %d/%d diarize+gender done (%d speakers)",
        idx + 1, len(chunks), len(genders),
    )
    assigned: list[tuple[Segment, str | None]] = []
    for seg in chunk.segments:
        local = Segment(seg.start - chunk.start, seg.end - chunk.start, seg.text)
        assigned.append((seg, diarize.assign_speaker(local, annotation)))
    groups = _group_consecutive(assigned)
    tasks.append(asyncio.create_task(
        _translate_chunk(
            idx, groups, genders, target, client, sem,
            None,   # << was prev_speaker_gender; pyannote labels reset per chunk
            source_language,
        )
    ))
    # The cross-chunk carry has been removed. Within-chunk rotation still
    # works (see _translate_chunk's prev_group_gender).
```

- [ ] **Step 7.7: Delete or @skip the old `test_addressee_carries_across_chunks`**

In `tests/test_orchestration.py`, replace the body of `test_addressee_carries_across_chunks` with `pytest.skip(...)` or delete the function entirely. Keep `test_addressee_rotates_within_chunk` — that one still asserts the correct within-chunk behavior.

- [ ] **Step 7.8: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py -k "addressee" -v
```

Expected: `test_addressee_rotates_within_chunk` PASS, `test_addressee_does_not_carry_across_chunks` PASS, old test skipped/removed.

- [ ] **Step 7.9: Run full suite for regression**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 7.10: Commit**

```powershell
git add server.py tests/test_orchestration.py
git -c commit.gpgsign=false commit -m "fix(server): drop cross-chunk addressee carry — pyannote speaker labels reset per chunk so the carry is unsound"
```

---

## Self-review

**Spec coverage check:**

| Issue from feedback | Task |
|---|---|
| Missing sentences | Task 4 ✓ (diagnostic) |
| Timing drift | Task 5 ✓ |
| Long-line styling | Task 2 ✓ |
| Stuck subtitles | Task 3 ✓ |
| Wrong speaker gender @ 04:29 | Task 6 ✓ |
| Wrong addressee gender @ 05:00 | Task 7 ✓ |
| Wrong preposition @ 05:06 | Task 1 ✓ |
| Names not translated @ 05:25 | Task 1 ✓ |
| Slang not in Hebrew | Task 1 ✓ |

All nine issues mapped.

**Placeholder scan:** No "TODO", no "add appropriate error handling", no unbacked references. Each step is a concrete command or a code block.

**Type consistency:** `Segment(start, end, text)` is used identically across Tasks 3, 4, 5; `_classify_f0`, `MIN_VOICED_FRAMES`, `GENDER_THRESHOLD_HZ` (Task 6) match. `_split_to_two_lines` and `_shorten_overlong_subtitle` (Tasks 2/3) are named consistently with the renderer wiring step.

**Cross-task ordering:** Tasks 1–3 ship immediately and don't depend on each other. Task 4 ships independently. Tasks 5–7 each open with a diagnostic step — if the data shows the assumed cause is wrong, the fix step is adjusted; the test still pins the desired behavior either way.

**Out of scope (deliberate):** speaker-tracking across chunks via pyannote embeddings (mentioned in Task 7 as option b — defer to a follow-up plan once the simple disable lands and we measure subjective quality).

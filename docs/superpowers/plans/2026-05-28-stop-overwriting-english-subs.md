# Stop Overwriting English Subtitles Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a Bazarr request, leave existing English subtitle files on the share untouched. Only the translated (Hebrew) SRT and its summary should appear under `$SAVE_SRT_LOCAL_PREFIX`.

**Architecture:** Today the `/asr` response body IS the translated Hebrew SRT, and Bazarr's whisperai plugin stores that body via its own subtitle-storage logic — under a filename Bazarr chooses, which can collide with existing English subtitle files (`*.srt`, `*.en.srt`). Decouple the two: return the **English transcript** to Bazarr in the HTTP response (matches the language Bazarr submitted with `language=en`), and write the **target-language translation** only to the configured side-file path. Existing English files are then either unchanged or refreshed with English content; the Hebrew lands only at `*.he.srt`.

**Tech Stack:** Python 3.12, FastAPI, faster-whisper, pyannote, NLLB (or Claude); existing `_try_save_side_file` mechanism.

---

## Task 1: Confirm root cause (diagnostic only — no code changes)

**Files:** none

- [ ] **Step 1.1: Read user's current `SAVE_SRT_SUFFIX`**

  Ask the user to paste the line from their `.env` (or confirm "default `.he.srt`"). If the value is `.srt` or `.en.srt`, the entire bug is a config issue and Tasks 2–4 are unnecessary — they just need to change the env var.

- [ ] **Step 1.2: Capture Z: directory before next Bazarr run**

  ```powershell
  $dir = "Z:\media\tv\Oz (1997)\Season 04"
  Get-ChildItem -LiteralPath $dir -Filter "*S04E15*" |
    Select-Object Name, Length, LastWriteTime,
                  @{n='head';e={(Get-Content -LiteralPath $_.FullName -Encoding UTF8 -TotalCount 4) -join '`n'}} |
    Format-List
  ```

  Save this output. Expected: 3+ files (`*.srt`, `*.en.srt`, `*.he.srt`).

- [ ] **Step 1.3: Trigger Bazarr Manual Search for that episode, then re-capture**

  Run the same snapshot command. Diff against step 1.2.

  - If `*.he.srt` mtime is new and others are unchanged → **no bug**, the user may have been confused.
  - If `*.srt` or `*.en.srt` mtime changed → bug confirmed. Note which file changed and whether its content is now Hebrew.

- [ ] **Step 1.4: Decide path**

  - Config fix only (step 1.1 result was wrong suffix) → done, no further work.
  - Real overwrite confirmed → continue to Task 2.

---

## Task 2: Preserve the English transcript through the pipeline

**Files:**
- Modify: `D:\git\whisper-gend\server.py:280-302` (`_translate_chunk`)
- Modify: `D:\git\whisper-gend\server.py:305+` (`_run_gender_aware`, `_run_plain_translate`)
- Modify: `D:\git\whisper-gend\server.py` (`run_pipeline_async` return shape)
- Test: `D:\git\whisper-gend\tests\test_orchestration.py`

Today `_translate_chunk` mutates `Segment.text` in place after translation. The original English text is lost. We need to keep both.

- [ ] **Step 2.1: Write the failing test**

  Append to `tests/test_orchestration.py`:

  ```python
  @pytest.mark.asyncio
  async def test_pipeline_returns_both_english_and_target(monkeypatch):
      # Pipeline must preserve the English transcript alongside the translation.
      seg = Segment(start=0.0, end=1.0, text="hello world")
      monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
      monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
      monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 1)
      monkeypatch.setattr(server.transcribe, "transcribe",
                          lambda path, language="en": [seg])
      monkeypatch.setattr(server, "_load_wav_mono",
                          lambda path: (np.zeros(16000, dtype=np.float32), 16000))
      monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

      ann = Annotation()
      ann[PSegment(0.0, 1.0)] = "S"
      monkeypatch.setattr(server.diarize, "diarize_waveform", lambda *a, **k: ann)
      monkeypatch.setattr(server.gender, "detect_genders",
                          lambda audio, sr, a: {"S": "male"})

      async def fake_translate(texts, gender, target, client,
                               addressee_gender=None, source_language="English"):
          return ["שלום עולם"]
      monkeypatch.setattr(server.translate, "translate_batch_async", fake_translate)

      english, hebrew = await server.run_pipeline_async(
          server.Path("ignored.wav"), "en"
      )
      assert [s.text for s in english] == ["hello world"]
      assert [s.text for s in hebrew] == ["שלום עולם"]
      # Same instants, different texts:
      assert english[0].start == hebrew[0].start
      assert english[0].end == hebrew[0].end
      # Must be independent objects so future mutation can't leak:
      assert english[0] is not hebrew[0]
  ```

- [ ] **Step 2.2: Run the test to verify it fails**

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py::test_pipeline_returns_both_english_and_target -v
  ```

  Expected: FAIL — current `run_pipeline_async` returns a single list, and segments are mutated in place.

- [ ] **Step 2.3: Refactor `_translate_chunk` to produce parallel translated segments instead of mutating**

  Change the inner loop to build a parallel list keyed by chunk index. Replace:

  ```python
  for seg, text in zip(group, translated):
      seg.text = text
  ```

  with:

  ```python
  from dataclasses import replace
  for seg, text in zip(group, translated):
      translated_segs.append(replace(seg, text=text))
  ```

  Collect `translated_segs` into a chunk-keyed structure the orchestrator can re-assemble in order.

- [ ] **Step 2.4: Make `_run_gender_aware` and `_run_plain_translate` return `(english, target)`**

  Both already iterate the original segment list to build chunks. They now also gather the translated segments per chunk and stitch them back in the same order, returning the tuple.

- [ ] **Step 2.5: Update `run_pipeline_async` signature**

  ```python
  async def run_pipeline_async(audio_path: Path, language: str) -> tuple[list[Segment], list[Segment]]:
  ```

  Update its callers accordingly (only the `/asr` handler).

- [ ] **Step 2.6: Re-run the test to verify it passes**

  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py::test_pipeline_returns_both_english_and_target -v
  ```

- [ ] **Step 2.7: Update the 5 existing orchestration tests**

  Existing tests assert on a single returned segment list and the mutated `.text`. Change them to unpack `(english, target)` and assert against the appropriate list. The text they currently check (e.g., `"a|male|None"`) is the translated text, which now lives on `target`.

- [ ] **Step 2.8: Run the full suite**

  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q
  ```

  Expected: all green.

- [ ] **Step 2.9: Commit**

  ```bash
  git add server.py tests/test_orchestration.py
  git commit -m "refactor(server): pipeline returns (english, target) segment lists"
  ```

---

## Task 3: Wire the `/asr` handler to return English and side-file Hebrew

**Files:**
- Modify: `D:\git\whisper-gend\server.py` (the `/asr` endpoint body)

- [ ] **Step 3.1: Write the failing test**

  Append to `tests/test_orchestration.py` (or a new endpoint test if cleaner):

  ```python
  @pytest.mark.asyncio
  async def test_asr_response_is_english_side_file_is_target(monkeypatch, tmp_path):
      # Pipe a stub pipeline result; verify the HTTP response body contains
      # the English transcript and _try_save_side_file is called with the
      # target-language SRT body.
      english = [Segment(start=0.0, end=1.0, text="hello world")]
      target  = [Segment(start=0.0, end=1.0, text="שלום עולם")]

      async def fake_pipeline(audio_path, language):
          return english, target
      monkeypatch.setattr(server, "run_pipeline_async", fake_pipeline)

      captured: dict[str, str] = {}
      def fake_save(body, summary, video_file_url):
          captured["body"] = body
          captured["summary"] = summary
      monkeypatch.setattr(server, "_try_save_side_file", fake_save)

      # ... call the endpoint via fastapi.testclient with a small audio_file ...
      from fastapi.testclient import TestClient
      client = TestClient(server.app)
      response = client.post(
          "/asr?task=transcribe&language=en&output=srt&encode=false"
          "&video_file=/media/tv/x.mp4",
          files={"audio_file": ("x.wav", b"RIFF....", "audio/wav")},
      )
      assert response.status_code == 200
      assert "hello world" in response.text
      assert "שלום עולם" not in response.text
      assert "שלום עולם" in captured["body"]
      assert "hello world" not in captured["body"]
      # Summary stats reflect the target language (Hebrew), not English.
      assert "Hebrew" in captured["summary"]
  ```

- [ ] **Step 3.2: Run test to verify it fails**

  Expected: FAIL — current code uses `target` segments for both the response and the side-file.

- [ ] **Step 3.3: Update the `/asr` handler**

  In `server.py`, after `english, target = await run_pipeline_async(...)`:

  ```python
  body, content_type = render(english, output)
  target_body, _ = render(target, output)
  ...
  summary = _build_translation_summary(
      ...
      segments=target,                # script-ratio check on the TARGET
      ...
  )
  await run_in_thread(_try_save_side_file, target_body, summary, video_file_url)
  return PlainTextResponse(body, media_type=content_type)
  ```

- [ ] **Step 3.4: Run the new test + full suite**

  Expected: all green.

- [ ] **Step 3.5: Commit**

  ```bash
  git add server.py tests/test_orchestration.py
  git commit -m "fix(server): return English transcript to client, save translation as side-file"
  ```

---

## Task 4: Verify end-to-end against Bazarr

**Files:** none (manual verification)

- [ ] **Step 4.1: Stop server, switch to the bugfix branch, restart**

  ```powershell
  # In Claude Code via TaskStop / Bash run_in_background restart sequence
  ```

- [ ] **Step 4.2: Snapshot Z: directory**

  Same command as Task 1, step 1.2.

- [ ] **Step 4.3: Trigger Bazarr Manual Search for a *fresh* episode (not S04E15 — we've poked at it too many times)**

  Pick e.g. S04E05 (currently missing a Hebrew SRT per the directory listing we saw).

- [ ] **Step 4.4: Snapshot Z: again. Compare**

  Expected:
  - `*.he.srt` is new (was absent for S04E05)
  - `*.he.summary.txt` is new
  - `*.srt` (no-language) and `*.en.srt` files: **mtime UNCHANGED**, content UNCHANGED
  - Side-file summary `Script check: ≥ 95% Hebrew`

- [ ] **Step 4.5: Spot-check Bazarr's own subtitle storage**

  Whatever filename Bazarr now writes (the English response body): confirm its content is English text matching the transcribe output, not Hebrew. If Bazarr writes nothing (because it's smart enough to detect "you returned the source language"), even better.

- [ ] **Step 4.6: If all four checks pass, merge to main**

  ```bash
  git checkout main
  git merge --no-ff <bugfix-branch>
  git push origin main
  ```

---

## Risks & rollback

- **Risk:** Some Bazarr UI quirks may show the episode as "no Hebrew subtitle found" if Bazarr only inspects the HTTP response and now sees English. Mitigation: the side-file save still lands Hebrew on disk; Bazarr's next disk scan picks it up. If this matters for Bazarr's UI, an alternative is to have the response BE Hebrew and configure Bazarr's whisperai plugin to save under a language-specific filename — but that requires Bazarr config we don't directly control.
- **Rollback:** All changes are in `server.py` plus tests. The fix is a single merge commit; revert it cleanly with `git revert -m 1 <merge-sha>`.

## Open questions for the user

1. **What does your `.env` `SAVE_SRT_SUFFIX` actually say?** If it's already `.he.srt`, the Bazarr-side overwrite hypothesis stands. If it's `.srt`, no code change needed — just fix the env.
2. **Which file is being overwritten?** `KONTRAST.srt` (no language) or `KONTRAST.en.srt`? Snapshots before/after a Bazarr search will tell us — see Task 1 step 1.3.

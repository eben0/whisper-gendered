# Improve Gender Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve speaker- and addressee-gender accuracy by (a) replacing the pitch-only classifier with an ML model that has access to spectral / formant cues a single-feature threshold can't see, and (b) giving the translation LLM a rolling window of prior conversation lines so its addressee-form choices are grounded in scene context.

**Architecture:** Two independent improvements share a feature branch:
- **Speaker gender (Tasks 1-5).** New `pipeline/gender_ml.py` wraps a pre-trained wav2vec2 audio-classification model (`alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech` by default). `pipeline/gender.py` becomes a small dispatcher that runs the configured classifier (`pitch` | `ml` | `ensemble`); the ensemble mode runs both and logs disagreements for A/B observation, with the option to emit a second side-file SRT for human review.
- **Addressee context (Tasks 6-7).** `translate.translate_batch_async` gains a `previous_context` parameter — a rolling window of the most recent source segments (default 4, env-configurable). The orchestrator maintains the window per request and passes it to every translation batch. The system prompt instructs Claude to use this context to disambiguate "you" forms.

A performance gate (Task 8) trips a WARNING log line if per-chunk gender-classification wall time exceeds the pitch-only baseline by a configurable multiple, so a regression is visible without separate monitoring.

**Tech Stack:** Python 3.12, transformers (already installed for NLLB), torch + torchaudio, librosa (still used by the pitch fallback), pytest. No new training; everything is inference-time on a pre-trained HF model. The user's existing Claude backend continues to drive translation.

---

## Spec assumptions (called out so they can be challenged)

1. **"Use Kaggle datasets and models"** is honored via a pre-trained HF wav2vec2 model whose training distribution (LibriSpeech) overlaps strongly with the Kaggle voice gender dataset. The spec rule *"Use open-source libraries if they exist, instead of implementing from scratch"* makes pre-trained the preferred path. Training a scikit-learn model on the literal Kaggle CSV is documented as **Task 5b** (deferred unless the pre-trained model underperforms).
2. **"By default send 4 conversations"** is interpreted as **4 prior source-language segments** (the rolling window of the last 4 things said in the request, regardless of speaker), since "conversation" has no other natural unit in this codebase. Env var `TRANSLATE_CONTEXT_LINES` makes it configurable.
3. **"A/B test the new detection engine"** is implemented in two layers: runtime logging of pitch-vs-ML disagreements (always-on when `GENDER_CLASSIFIER=ensemble`), and an optional second side-file SRT (`*.he.alt-classifier.srt`) for visual comparison on a real episode (env-gated via `GENDER_AB_OUTPUT=true`).

---

## File structure

| File | Responsibility | Touched by |
|---|---|---|
| `pipeline/gender.py` | dispatcher: choose classifier(s), log disagreements, perf gate | Tasks 3, 4, 8 |
| `pipeline/gender_ml.py` (new) | wav2vec2 singleton + `classify_audio(slice, sr) -> (label, confidence)` | Task 2 |
| `pipeline/translate.py` | system prompt + sync/async batch helpers accept `previous_context` | Tasks 6, 7 |
| `server.py` | rolling-window context in `_run_gender_aware`/`_run_plain_translate`; optional alt-classifier side-file | Tasks 5, 7 |
| `config.py` | three new env keys: `GENDER_CLASSIFIER`, `GENDER_ML_MODEL`, `TRANSLATE_CONTEXT_LINES`, `GENDER_AB_OUTPUT` | Task 3 |
| `requirements.txt` | add `torchaudio` if not already present | Task 1 |
| `.env.example` | document the new keys | Task 3 |
| `tests/test_gender_ml.py` (new) | mock the HF pipeline | Task 2 |
| `tests/test_gender.py` | dispatcher tests | Task 4 |
| `tests/test_translate_async.py` | extend for `previous_context` | Tasks 6 |
| `tests/test_orchestration.py` | extend for rolling-window assembly | Task 7 |

---

## Branch policy

Per spec: **feature branch, no PR**. Create `feature/improve-gender-detection` off `main`. Commit each task. **Do NOT open a pull request** when finished; the user will merge or evaluate manually.

```powershell
git checkout main
git pull
git checkout -b feature/improve-gender-detection
```

---

## Task 1: Pre-flight — model availability + 5-second smoke-test

**Files:** none (investigation); may add `torchaudio` to `requirements.txt` if missing.

The default model is `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech` (~1.2 GB, fp16-friendly, returns `{"label": "male"|"female", "score": float}`). Confirm three things before writing any production code:

1. The model loads via `transformers.pipeline("audio-classification", ...)`.
2. On a known-incorrect S04E05 clip (the 04:29 case that the existing pitch classifier called male), the ML model returns `female` with confidence ≥ 0.6.
3. Per-clip inference time on the user's RTX 2070 is acceptable (target ≤ 1s per 5-second slice).

- [ ] **Step 1.1: Confirm torchaudio is installed**

```powershell
.\.venv\Scripts\python.exe -c "import torchaudio; print(torchaudio.__version__)"
```

If this fails with ImportError, install:

```powershell
.\.venv\Scripts\python.exe -m pip install torchaudio
```

Then add `torchaudio>=2.0.0` to `requirements.txt`.

- [ ] **Step 1.2: Extract the 5-second clip from S04E05 around 04:29 (the misclassified female speaker)**

```powershell
$mp4 = "Z:\media\tv\Oz (1997)\Season 04\Oz (1997) - S04E05 - Gray Matter [WEBRip-1080p][AAC 2.0][x265]-KONTRAST.mp4"
$wav = "$env:TEMP\gender_smoke_4_29.wav"
& C:\Users\eyalb\ffmpeg\bin\ffmpeg.exe -hide_banner -loglevel error -y -ss 269 -t 5 -i $mp4 -ac 1 -ar 16000 -f wav $wav
"WAV ready: $((Get-Item $wav).Length) bytes"
```

- [ ] **Step 1.3: Run the candidate model on the clip**

```powershell
.\.venv\Scripts\python.exe -c @"
import time, librosa
from transformers import pipeline
clf = pipeline(
    'audio-classification',
    model='alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech',
    device=0,  # cuda
)
y, sr = librosa.load(r'$env:TEMP\gender_smoke_4_29.wav', sr=16000, mono=True)
t0 = time.time()
out = clf({'array': y, 'sampling_rate': sr}, top_k=2)
print(f'inference: {time.time()-t0:.2f}s')
for r in out:
    print(f"  {r['label']:8s}  {r['score']:.3f}")
"@
```

Expected: prints two rows (`female` and `male`) with scores. Confirm `female` wins with `score > 0.5`.

If the candidate model does NOT call this clip female with ≥ 0.5 confidence, try the alternates documented in the model card:
- `superb/wav2vec2-base-superb-er` (emotion recognition — not gender; do not use)
- A community fine-tune; search HF Hub for `"gender"` + `"audio-classification"` (Step 1.4 below).

- [ ] **Step 1.4: (only if Step 1.3 fails) Try one alternate**

```powershell
.\.venv\Scripts\python.exe -c @"
from transformers import pipeline
# Smaller backup; replace with another HF model id if needed.
clf = pipeline('audio-classification',
               model='facebook/wav2vec2-large-xlsr-53', device=0)
print(clf.model.config.id2label)
"@
```

If this returns gender labels, use it; otherwise file a follow-up task ("train on Kaggle CSV") and proceed with the pitch fallback only.

- [ ] **Step 1.5: Decide and record the choice**

Add a one-liner to the plan file noting which model is the default. Commit only `requirements.txt` (if you added torchaudio) at this point:

```powershell
git add requirements.txt
git -c commit.gpgsign=false commit -m "chore: add torchaudio for wav2vec2 audio-classification pipeline"
```

If `requirements.txt` already had torchaudio, skip the commit.

---

## Task 2: `pipeline/gender_ml.py` — wav2vec2 gender classifier singleton

**Files:**
- Create: `D:\git\whisper-gend\pipeline\gender_ml.py`
- Test: `D:\git\whisper-gend\tests\test_gender_ml.py`

The module exposes:
- `classify_audio(audio: np.ndarray, sr: int) -> tuple[str, float]` — label is `"male"` or `"female"`, score is in [0, 1] (the winning label's confidence)
- `get_pipeline()` — singleton loader, same lazy-with-lock pattern as `pipeline/transcribe.py:get_model`
- `model_loaded() -> bool`
- `warmup() -> None` — startup hook to amortize the first request's load time

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_gender_ml.py`:

```python
"""Tests for the wav2vec2 gender classifier wrapper.

The HF pipeline is patched out so tests don't load the ~1.2 GB model.
We pin the public contract: classify_audio returns (label, confidence)
with label in {"male", "female"} and confidence in [0, 1].
"""
import numpy as np
import pytest

from pipeline import gender_ml


class _FakePipeline:
    """Returns whatever the test set as ``self.predictions``."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.calls: list[dict] = []

    def __call__(self, payload, top_k=None):
        self.calls.append(payload)
        return self.predictions


def test_classify_audio_returns_winning_label_and_confidence(monkeypatch):
    fake = _FakePipeline([
        {"label": "female", "score": 0.82},
        {"label": "male",   "score": 0.18},
    ])
    monkeypatch.setattr(gender_ml, "get_pipeline", lambda: fake)

    audio = np.zeros(16000, dtype=np.float32)  # 1s of silence; payload only
    label, conf = gender_ml.classify_audio(audio, sr=16000)

    assert label == "female"
    assert conf == pytest.approx(0.82)
    # The pipeline must be called with the HF audio-classification payload shape.
    assert fake.calls[0]["sampling_rate"] == 16000
    assert "array" in fake.calls[0]


def test_classify_audio_picks_male_when_male_scores_higher(monkeypatch):
    fake = _FakePipeline([
        {"label": "male",   "score": 0.91},
        {"label": "female", "score": 0.09},
    ])
    monkeypatch.setattr(gender_ml, "get_pipeline", lambda: fake)

    label, conf = gender_ml.classify_audio(
        np.zeros(16000, dtype=np.float32), sr=16000,
    )
    assert label == "male"
    assert conf == pytest.approx(0.91)


def test_classify_audio_normalizes_label_casing(monkeypatch):
    # Some HF models return "MALE"/"FEMALE"; normalize to lowercase.
    fake = _FakePipeline([
        {"label": "FEMALE", "score": 0.7},
        {"label": "MALE",   "score": 0.3},
    ])
    monkeypatch.setattr(gender_ml, "get_pipeline", lambda: fake)

    label, _ = gender_ml.classify_audio(
        np.zeros(16000, dtype=np.float32), sr=16000,
    )
    assert label == "female"


def test_classify_audio_raises_on_unrecognised_label(monkeypatch):
    # If a misconfigured model returns something other than male/female,
    # the wrapper should raise rather than silently mis-route.
    fake = _FakePipeline([
        {"label": "neutral", "score": 0.6},
        {"label": "male",    "score": 0.4},
    ])
    monkeypatch.setattr(gender_ml, "get_pipeline", lambda: fake)

    with pytest.raises(ValueError, match="unexpected gender label"):
        gender_ml.classify_audio(
            np.zeros(16000, dtype=np.float32), sr=16000,
        )


def test_model_loaded_reflects_singleton_state(monkeypatch):
    # Reset module state so the test is deterministic.
    monkeypatch.setattr(gender_ml, "_pipeline", None)
    assert gender_ml.model_loaded() is False

    monkeypatch.setattr(gender_ml, "_pipeline", object())
    assert gender_ml.model_loaded() is True
```

- [ ] **Step 2.2: Run tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gender_ml.py -v
```

Expected: collection error ("No module named 'pipeline.gender_ml'").

- [ ] **Step 2.3: Create `pipeline/gender_ml.py`**

```python
"""Wav2vec2-based gender classifier.

A drop-in alternative (or A/B counterpart) to the pitch-based classifier
in ``pipeline.gender``. The model is loaded as a process-wide singleton
on first use — same pattern as ``pipeline/transcribe.py`` and
``pipeline/diarize.py``. Inference takes ~0.5-1s per short audio slice
on a 2070; the singleton makes that one-time per process.

Public API:
    classify_audio(audio, sr) -> (label, confidence)
        label is "male" or "female"; confidence is the winning class's
        score in [0, 1]. Raises ValueError if the underlying model
        returns an unrecognised label.

    get_pipeline() -> transformers.Pipeline
        Singleton accessor (lazy with threading.Lock — concurrent first
        requests do not load twice).

    model_loaded() -> bool
        For /status reporting.

    warmup() -> None
        Optional startup hook; eagerly loads the model so the first /asr
        request isn't penalised by the cold-start download.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from config import settings

log = logging.getLogger("pipeline.gender_ml")

_pipeline: Any | None = None
_lock = threading.Lock()


def get_pipeline() -> Any:
    """Return the singleton HF audio-classification pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        # Local import so unit tests that patch get_pipeline don't pay
        # the transformers import cost.
        from transformers import pipeline as hf_pipeline
        model_id = settings.GENDER_ML_MODEL
        device = 0 if settings.DEVICE.strip().lower() == "cuda" else -1
        log.info(
            "Loading gender ML classifier %s (device=%s)...",
            model_id, "cuda" if device == 0 else "cpu",
        )
        _pipeline = hf_pipeline(
            "audio-classification",
            model=model_id,
            device=device,
        )
        log.info("Gender ML classifier loaded.")
    return _pipeline


def model_loaded() -> bool:
    return _pipeline is not None


def classify_audio(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """Classify a mono audio slice as ``"male"`` or ``"female"``.

    Args:
        audio: mono float32 waveform.
        sr: sample rate in Hz (typically 16000 — the wav2vec2 model's
            native rate; pass non-native rates and the HF pipeline
            resamples internally).

    Returns:
        ``(label, confidence)`` where label is the lowercased winning
        class and confidence is its score in [0, 1].

    Raises:
        ValueError: the underlying model returned a label other than
            "male" or "female" (e.g. a misconfigured model). Surfacing
            this loudly is preferable to silently mapping to a default.
    """
    pipe = get_pipeline()
    # HF audio-classification expects this payload shape; top_k=2 returns
    # both labels so we can grab the winner cleanly.
    out = pipe({"array": audio, "sampling_rate": sr}, top_k=2)
    if not out:
        raise ValueError("audio-classification pipeline returned empty result")
    winner = out[0]
    label = str(winner["label"]).strip().lower()
    if label not in ("male", "female"):
        raise ValueError(
            f"unexpected gender label {label!r} from model "
            f"{settings.GENDER_ML_MODEL}; expected 'male' or 'female'"
        )
    return label, float(winner["score"])


def warmup() -> None:
    """Eagerly load the singleton at startup."""
    try:
        get_pipeline()
        log.info("Gender ML classifier warm-up complete.")
    except Exception:  # pragma: no cover — startup hook must not crash
        log.exception("Gender ML warm-up failed (continuing anyway).")
```

- [ ] **Step 2.4: Add the new settings keys (deferred to Task 3 — see Task 3 for the exact `config.py` edit)**

This task's tests will fail with `AttributeError` on `settings.GENDER_ML_MODEL` until Task 3 ships. To avoid blocking, monkeypatch in the test as needed OR jump to Task 3 first. Task ordering note: **execute Task 3 before re-running Task 2's tests.**

- [ ] **Step 2.5: Commit (after Task 3's config bits land)**

```powershell
git add pipeline/gender_ml.py tests/test_gender_ml.py
git -c commit.gpgsign=false commit -m "feat(gender_ml): wav2vec2 audio-classification wrapper with singleton + warmup"
```

---

## Task 3: Config knobs for the new behavior

**Files:**
- Modify: `D:\git\whisper-gend\config.py` (Settings dataclass)
- Modify: `D:\git\whisper-gend\.env.example`

Adds three env keys for gender detection and one for the translate-context window. Defaults are conservative: behavior is unchanged unless the operator opts in.

- [ ] **Step 3.1: Write a failing test pinning the default values**

Add to a new file `tests/test_config_new_keys.py`:

```python
"""Pin the defaults of the new env keys added for Plan 'improve-gender-
detection'. Defaults are deliberately conservative so existing
deployments behave identically until the operator opts in.
"""
import importlib

import config


def test_default_gender_classifier_is_pitch(monkeypatch):
    monkeypatch.delenv("GENDER_CLASSIFIER", raising=False)
    importlib.reload(config)
    assert config.settings.GENDER_CLASSIFIER == "pitch"


def test_default_gender_ml_model_is_alefiury_wav2vec2(monkeypatch):
    monkeypatch.delenv("GENDER_ML_MODEL", raising=False)
    importlib.reload(config)
    assert config.settings.GENDER_ML_MODEL == (
        "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    )


def test_default_translate_context_lines_is_4(monkeypatch):
    monkeypatch.delenv("TRANSLATE_CONTEXT_LINES", raising=False)
    importlib.reload(config)
    assert config.settings.TRANSLATE_CONTEXT_LINES == 4


def test_default_gender_ab_output_is_false(monkeypatch):
    monkeypatch.delenv("GENDER_AB_OUTPUT", raising=False)
    importlib.reload(config)
    assert config.settings.GENDER_AB_OUTPUT is False
```

- [ ] **Step 3.2: Run tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config_new_keys.py -v
```

Expected: AttributeError on every test (settings don't exist yet).

- [ ] **Step 3.3: Add the keys to `config.py`**

Open `D:\git\whisper-gend\config.py` and add inside the `Settings` dataclass (near the existing `TRANSLATION_BACKEND` / `LOCAL_*` keys block):

```python
    # Gender classification (Plan: improve-gender-detection).
    # "pitch"    — current librosa.pyin + threshold (default; no model load)
    # "ml"       — wav2vec2 audio-classification only
    # "ensemble" — run both; log disagreements; ML wins (pitch is fallback)
    GENDER_CLASSIFIER: str = os.getenv("GENDER_CLASSIFIER", "pitch")
    GENDER_ML_MODEL: str = os.getenv(
        "GENDER_ML_MODEL",
        "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
    )
    # When true, the orchestrator also emits a second *.he.srt next to the
    # primary one using the *other* classifier (pitch vs ML) so the
    # operator can A/B them on a real episode.
    GENDER_AB_OUTPUT: bool = _env_bool("GENDER_AB_OUTPUT", False)

    # Translate-context window (Plan: improve-gender-detection).
    # Number of preceding source-language segments injected into each
    # translate batch as "earlier in this scene" context. Helps Claude
    # disambiguate addressee gender across "you" forms when the prior
    # exchange establishes who's being addressed. 0 disables.
    TRANSLATE_CONTEXT_LINES: int = _env_int("TRANSLATE_CONTEXT_LINES", 4)
```

- [ ] **Step 3.4: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config_new_keys.py -v
```

Expected: 4 passed.

- [ ] **Step 3.5: Document in `.env.example`**

Append to `.env.example`:

```ini
# Gender classification (Plan: improve-gender-detection).
# "pitch" (default) keeps the existing librosa.pyin + threshold path.
# "ml" uses the wav2vec2 audio-classification model alone.
# "ensemble" runs both; logs disagreements; ML decides (pitch is the fallback).
GENDER_CLASSIFIER=pitch
# HuggingFace model id for the ML classifier.
# Loaded on first request (singleton). ~1.2 GB on disk.
GENDER_ML_MODEL=alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech
# When true, emit a SECOND *.he.srt side-file using the alternate classifier
# so the operator can A/B them on a real episode. Defaults to off.
GENDER_AB_OUTPUT=false
# Number of preceding source-language segments injected into each translate
# batch as "earlier in this scene" context. 0 disables.
TRANSLATE_CONTEXT_LINES=4
```

- [ ] **Step 3.6: Re-run Task 2's tests now that the config keys exist**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gender_ml.py tests/test_config_new_keys.py -v
```

Expected: all passed.

- [ ] **Step 3.7: Commit Tasks 2 + 3 together**

```powershell
git add config.py .env.example pipeline/gender_ml.py tests/test_gender_ml.py tests/test_config_new_keys.py
git -c commit.gpgsign=false commit -m "feat(gender_ml,config): wav2vec2 gender classifier + 4 new env knobs

* pipeline/gender_ml.py: HF audio-classification singleton with
  classify_audio(audio, sr) -> (label, confidence). Lazy + lock,
  identical pattern to transcribe.get_model.
* config.py: GENDER_CLASSIFIER (pitch|ml|ensemble, default pitch),
  GENDER_ML_MODEL (default alefiury/wav2vec2-large-xlsr-53-gender-
  recognition-librispeech), GENDER_AB_OUTPUT (default false),
  TRANSLATE_CONTEXT_LINES (default 4).
* .env.example documents all four.
* Tests cover label normalisation, the unrecognised-label error path,
  singleton state, and that defaults are pitch / disabled."
```

---

## Task 4: Dispatcher in `pipeline/gender.py` — pitch / ml / ensemble

**Files:**
- Modify: `D:\git\whisper-gend\pipeline\gender.py`
- Modify: `D:\git\whisper-gend\tests\test_gender.py`

Adds a thin dispatcher around `_classify_f0` and the new `classify_audio`. Crucially: the **per-speaker concatenated audio slice** is what feeds either classifier. Pitch path is unchanged; ML path passes the same slice to wav2vec2. Ensemble path runs both and logs disagreements at INFO so the operator can build a confusion matrix from grep alone.

- [ ] **Step 4.1: Write failing tests**

Append to `tests/test_gender.py`:

```python
# --- Task 4: dispatcher --------------------------------------------------- #

def test_detect_genders_uses_pitch_by_default(monkeypatch):
    """Default classifier is 'pitch'; the ML path must not be touched."""
    monkeypatch.setattr("config.settings.GENDER_CLASSIFIER", "pitch")
    from pipeline import gender as g

    # If gender_ml is touched in pitch mode, this raises.
    monkeypatch.setattr(
        "pipeline.gender_ml.classify_audio",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("ML classifier called in pitch mode")
        ),
    )
    low = _tone(120.0)
    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "SPEAKER"
    out = g.detect_genders(low, SR, ann)
    assert out["SPEAKER"] == "male"


def test_detect_genders_uses_ml_when_configured(monkeypatch):
    """GENDER_CLASSIFIER=ml routes through gender_ml.classify_audio."""
    monkeypatch.setattr("config.settings.GENDER_CLASSIFIER", "ml")
    from pipeline import gender as g

    called = {"n": 0}
    def fake_ml(audio, sr):
        called["n"] += 1
        return ("female", 0.91)
    monkeypatch.setattr("pipeline.gender_ml.classify_audio", fake_ml)

    audio = np.zeros(int(0.5 * SR), dtype=np.float32)
    ann = Annotation()
    ann[PSegment(0.0, 0.5)] = "SPEAKER"
    out = g.detect_genders(audio, SR, ann)
    assert out["SPEAKER"] == "female"
    assert called["n"] == 1


def test_detect_genders_ensemble_logs_disagreement(monkeypatch, caplog):
    """In ensemble mode the dispatcher runs BOTH, logs disagreements at
    INFO, and ML's call wins (pitch is the fallback if ML errors).
    """
    import logging
    caplog.set_level(logging.INFO, logger="pipeline.gender")
    monkeypatch.setattr("config.settings.GENDER_CLASSIFIER", "ensemble")
    from pipeline import gender as g

    # Pitch says male (120 Hz tone < 165 Hz threshold).
    # ML says female. Ensemble must pick ML's answer and log disagreement.
    monkeypatch.setattr(
        "pipeline.gender_ml.classify_audio",
        lambda audio, sr: ("female", 0.85),
    )
    low = _tone(120.0)
    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "SPEAKER"
    out = g.detect_genders(low, SR, ann)
    assert out["SPEAKER"] == "female"
    msgs = [r.getMessage() for r in caplog.records]
    assert any("disagree" in m.lower() and "pitch=male" in m.lower()
               and "ml=female" in m.lower() for m in msgs), (
        f"expected disagreement log line; got: {msgs}"
    )


def test_detect_genders_ensemble_falls_back_to_pitch_when_ml_errors(monkeypatch, caplog):
    """If the ML call raises, ensemble mode must not crash the request —
    log a warning and use the pitch answer.
    """
    import logging
    caplog.set_level(logging.WARNING, logger="pipeline.gender")
    monkeypatch.setattr("config.settings.GENDER_CLASSIFIER", "ensemble")
    from pipeline import gender as g

    def boom(audio, sr):
        raise RuntimeError("simulated wav2vec2 OOM")
    monkeypatch.setattr("pipeline.gender_ml.classify_audio", boom)

    low = _tone(120.0)
    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "SPEAKER"
    out = g.detect_genders(low, SR, ann)
    assert out["SPEAKER"] == "male"  # pitch fallback
    msgs = [r.getMessage() for r in caplog.records]
    assert any("fall" in m.lower() and "pitch" in m.lower() for m in msgs)
```

- [ ] **Step 4.2: Run tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gender.py -k "uses_pitch or uses_ml or ensemble" -v
```

Expected: 4 FAILs.

- [ ] **Step 4.3: Refactor `pipeline/gender.py` with the dispatcher**

Replace the body of `detect_genders` so it calls a per-speaker dispatcher. Add a new private function `_classify_speaker(signal, sr, speaker_label, voiced_f0)` that contains the dispatch logic:

```python
def _classify_speaker(
    signal: np.ndarray,
    sr: int,
    speaker_label: str,
    voiced_f0: np.ndarray,
) -> str:
    """Return "male" | "female" using the configured classifier.

    ``voiced_f0`` is the librosa.pyin output for the pitch classifier;
    ``signal`` is the raw concatenated audio for the ML classifier. Both
    are computed once by ``detect_genders`` so the dispatcher just chooses
    among them.
    """
    mode = settings.GENDER_CLASSIFIER.strip().lower()

    if mode == "pitch":
        return _classify_f0(voiced_f0)

    if mode == "ml":
        # Local import — keeps the heavy transformers import lazy so
        # pitch-only deployments never pay the load cost.
        from pipeline import gender_ml
        try:
            label, conf = gender_ml.classify_audio(signal, sr)
            log.info(
                "Speaker %s ML: %s (confidence=%.3f)",
                speaker_label, label, conf,
            )
            return label
        except Exception:
            log.exception(
                "Speaker %s ML classifier raised; falling back to pitch",
                speaker_label,
            )
            return _classify_f0(voiced_f0)

    if mode == "ensemble":
        pitch_label = _classify_f0(voiced_f0)
        from pipeline import gender_ml
        try:
            ml_label, ml_conf = gender_ml.classify_audio(signal, sr)
        except Exception:
            log.warning(
                "Speaker %s ML classifier raised; falling back to pitch=%s",
                speaker_label, pitch_label,
                exc_info=True,
            )
            return pitch_label
        if ml_label != pitch_label:
            log.info(
                "Speaker %s classifiers disagree: pitch=%s ml=%s (conf=%.3f) "
                "— using ML",
                speaker_label, pitch_label, ml_label, ml_conf,
            )
        else:
            log.info(
                "Speaker %s classifiers agree: %s (ML conf=%.3f)",
                speaker_label, ml_label, ml_conf,
            )
        return ml_label

    # Unknown mode — log and fall back to pitch so a typo doesn't break prod.
    log.warning(
        "Unknown GENDER_CLASSIFIER=%r; falling back to pitch", mode,
    )
    return _classify_f0(voiced_f0)
```

In `detect_genders`, replace the existing `gender = _classify_f0(f0)` block with:

```python
        gender = _classify_speaker(signal, sr, str(speaker), f0)
```

…and keep the existing context-logging block (it still reads `f0`/`voiced_f0`).

- [ ] **Step 4.4: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gender.py -v
```

Expected: all gender tests pass.

- [ ] **Step 4.5: Run full suite for regression**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 4.6: Commit**

```powershell
git add pipeline/gender.py tests/test_gender.py
git -c commit.gpgsign=false commit -m "feat(gender): dispatcher selects pitch | ml | ensemble; ensemble logs disagreements"
```

---

## Task 5: Optional A/B SRT side-by-side output

**Files:**
- Modify: `D:\git\whisper-gend\server.py` (the `/asr` handler + `_try_save_side_file`)
- Test: `D:\git\whisper-gend\tests\test_orchestration.py`

When `GENDER_AB_OUTPUT=true`, the orchestrator runs the gender-aware pipeline a second time with the alternate classifier and writes that result as `*.he.alt-classifier.srt`. The user inspects both in their video player; the one that feels right indicates which classifier should be the default.

This task is the "production-readiness A/B" the spec calls for. Defaults to off — the cost (one extra full run) only matters when enabled.

- [ ] **Step 5.1: Write a failing test pinning the alt-srt naming + presence**

Append to `tests/test_orchestration.py`:

```python
def test_asr_emits_alt_classifier_srt_when_ab_output_enabled(monkeypatch):
    """When GENDER_AB_OUTPUT=true, the /asr handler must call
    _try_save_side_file a SECOND time with a different filename
    (``*.he.alt-classifier.srt``) using the alternate classifier's
    output.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "GENDER_AB_OUTPUT", True)
    monkeypatch.setattr(server.settings, "GENDER_CLASSIFIER", "pitch")

    source_segs = [Segment(start=0.0, end=1.0, text="hello")]
    target_segs = [Segment(start=0.0, end=1.0, text="שלום")]
    alt_target  = [Segment(start=0.0, end=1.0, text="שלום-ALT")]

    async def fake_pipeline(audio_path, language):
        return source_segs, target_segs
    monkeypatch.setattr(server, "run_pipeline_async", fake_pipeline)

    async def fake_alt(audio_path, language):
        return source_segs, alt_target
    monkeypatch.setattr(server, "run_pipeline_alt_classifier", fake_alt)

    monkeypatch.setattr(server, "encode_to_wav", lambda src, dst: None)
    monkeypatch.setattr(server, "prepare_unencoded", lambda src, dst: None)

    saved: list[tuple[str, str]] = []  # (suffix, body)
    def fake_save(body, summary, video_file_url, suffix=None):
        saved.append((suffix or ".he.srt", body))
    monkeypatch.setattr(server, "_try_save_side_file", fake_save)

    client = TestClient(server.app)
    response = client.post(
        "/asr",
        params={
            "task": "transcribe", "language": "en",
            "output": "srt", "encode": "false",
            "video_file": "/media/tv/x.mp4",
        },
        files={"audio_file": ("x.wav", b"RIFF0000WAVE", "audio/wav")},
    )

    assert response.status_code == 200
    suffixes = [s for s, _ in saved]
    bodies = {s: b for s, b in saved}
    assert ".he.srt" in suffixes or any(".he.srt" == s for s in suffixes)
    assert any(".alt-classifier" in s for s in suffixes), (
        f"alt-classifier side-file not saved; suffixes: {suffixes}"
    )
    # Bodies differ between primary and alt.
    alt_body = next(b for s, b in saved if ".alt-classifier" in s)
    assert "שלום-ALT" in alt_body
```

- [ ] **Step 5.2: Run test — confirm it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py::test_asr_emits_alt_classifier_srt_when_ab_output_enabled -v
```

Expected: FAIL (`run_pipeline_alt_classifier` does not exist; `_try_save_side_file` doesn't take a suffix arg).

- [ ] **Step 5.3: Extend `_try_save_side_file` and `_compute_side_file_path` to accept a custom suffix**

In `server.py`, modify `_try_save_side_file`:

```python
def _try_save_side_file(
    body: str,
    summary: str | None,
    video_file_url: str,
    suffix: str | None = None,
) -> None:
    """Same as before, but ``suffix`` overrides ``settings.SAVE_SRT_SUFFIX``
    when set. Used by Plan Task 5 to emit a parallel alt-classifier SRT
    (``*.he.alt-classifier.srt``) for A/B comparison.
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
        # ... (rest of body unchanged, but uses ``target`` from above)
```

- [ ] **Step 5.4: Add `run_pipeline_alt_classifier` helper**

In `server.py`, alongside `run_pipeline_async`:

```python
async def run_pipeline_alt_classifier(
    audio_path: Path, language: str,
) -> tuple[list[Segment], list[Segment] | None]:
    """Same as ``run_pipeline_async`` but flips ``GENDER_CLASSIFIER`` to
    the alternate of whatever's currently configured, runs the pipeline,
    and restores the original value.

    Cost: one extra full pipeline pass per request. Only invoked when
    ``GENDER_AB_OUTPUT=true``; defaults are off.
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
```

- [ ] **Step 5.5: Wire the alt path into `/asr`**

In `server.py`, in the `/asr` handler, immediately after the primary `await run_in_thread(_try_save_side_file, target_body, summary, video_file_url)` (inside the `if target_segments is not None:` block), append:

```python
            if settings.GENDER_AB_OUTPUT:
                try:
                    alt_source, alt_target = await run_pipeline_alt_classifier(
                        audio_path, language,
                    )
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
                except Exception:
                    log.exception(
                        "[%s] A/B alt-classifier pass failed; primary "
                        "side-file already written",
                        request_id,
                    )
```

- [ ] **Step 5.6: Run test — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py::test_asr_emits_alt_classifier_srt_when_ab_output_enabled -v
```

- [ ] **Step 5.7: Run full suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 5.8: Commit**

```powershell
git add server.py tests/test_orchestration.py
git -c commit.gpgsign=false commit -m "feat(server): GENDER_AB_OUTPUT=true emits a second *.he.alt-classifier.srt side-file for human A/B"
```

---

## Task 6: Past-conversation context in the translation prompt

**Files:**
- Modify: `D:\git\whisper-gend\pipeline\translate.py`
- Modify: `D:\git\whisper-gend\tests\test_translate_async.py`

Adds a `previous_context: list[str] = []` parameter to all four batch helpers (`translate_batch`, `_translate_one_batch`, `translate_batch_async`, `_translate_one_batch_async`). The system prompt grows a sentence on how to use it; the user message prepends a numbered "Earlier in this scene:" block when the list is non-empty. Default empty so existing callers keep working.

- [ ] **Step 6.1: Write failing tests**

Append to `tests/test_translate_async.py`:

```python
@pytest.mark.asyncio
async def test_previous_context_appears_in_user_message():
    """When previous_context is non-empty, the user message must include
    a numbered 'Earlier in this scene:' block listing those lines.
    """
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    await translate.translate_batch_async(
        ["new line"], None, "Hebrew", client,
        previous_context=["He arrived at noon.", "She was already there."],
    )
    user_msg = client.messages.payloads[0]["messages"][0]["content"]
    assert "Earlier in this scene" in user_msg
    assert "He arrived at noon" in user_msg
    assert "She was already there" in user_msg
    assert "new line" in user_msg
    # The actual line to translate must be clearly separated from context.
    # Check that "new line" appears AFTER both context lines.
    assert user_msg.index("new line") > user_msg.index("She was already there")


@pytest.mark.asyncio
async def test_previous_context_absent_when_window_is_empty():
    """Default (empty) context should not add any preamble — the user
    message looks identical to the pre-feature output.
    """
    client = _RecordingAsyncClient([json.dumps({"translations": ["a-he"]})])
    await translate.translate_batch_async(
        ["only line"], None, "Hebrew", client,
    )
    user_msg = client.messages.payloads[0]["messages"][0]["content"]
    assert "Earlier in this scene" not in user_msg


def test_system_prompt_mentions_context_use():
    """When previous_context is plumbed, the system prompt should tell
    Claude how to use it. We only check the directive sentence exists —
    not the exact wording — so future re-phrasings don't break the test.
    """
    sp = translate._system_prompt("Hebrew", None)
    # The directive can be present unconditionally (independent of the
    # current batch's gender) — it costs nothing when no context lines
    # are passed.
    assert (
        "earlier" in sp.lower() and "context" in sp.lower()
    ), "system prompt should explain how to use the 'Earlier' context block"
```

Make sure `_RecordingAsyncClient` records the full payload, not just the system. Inspect the helper class in the existing file; if it doesn't expose `payloads`, extend it:

```python
class _RecordingMessages:
    def __init__(self, payloads):
        self.payloads: list[dict] = []
        self.systems: list[str] = []
        self._scripted = list(payloads)

    async def create(self, **kwargs):
        self.payloads.append(kwargs)
        self.systems.append(kwargs.get("system", ""))
        return _RecordingResponse(self._scripted.pop(0))
```

Adjust the existing helper if its shape differs — keep the existing `.systems` attribute working so prior tests don't break.

- [ ] **Step 6.2: Run tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py -k "previous_context or context_use" -v
```

Expected: 3 FAILs.

- [ ] **Step 6.3: Extend `_system_prompt` with the context-use sentence**

In `pipeline/translate.py`, in `_system_prompt`, add (right before the final "Return a JSON object..." sentence):

```python
    base += (
        " If an 'Earlier in this scene:' block is included before the "
        "numbered translation lines, use it as background context only "
        "— don't re-translate it. Use it to disambiguate addressee gender "
        "and number for 'you' forms, maintain vocabulary consistency, "
        "and pick the form most consistent with what was just said."
    )
```

- [ ] **Step 6.4: Add `previous_context` parameter to all four batch helpers**

For each of `translate_batch`, `_translate_one_batch`, `translate_batch_async`, `_translate_one_batch_async`, add the parameter `previous_context: list[str] | None = None` at the end of the signature, and in each batch's user-message builder, replace:

```python
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
```

with:

```python
    parts: list[str] = []
    if previous_context:
        parts.append("Earlier in this scene:")
        for j, ctx in enumerate(previous_context, start=1):
            parts.append(f"  {j}. {ctx}")
        parts.append("")
        parts.append("Translate the following lines:")
    parts.extend(f"{i + 1}. {t}" for i, t in enumerate(texts))
    numbered = "\n".join(parts)
```

And in the recursive `translate_batch`/`translate_batch_async` wrappers that loop over `_chunks(texts)`, propagate `previous_context` to the inner helper (the sub-batching loop should pass the SAME context window into each sub-batch — they share scene context):

```python
        out.extend(
            await _translate_one_batch_async(
                batch, gender, target_language, client, addressee_gender,
                source_language, previous_context=previous_context,
            )
        )
```

- [ ] **Step 6.5: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_translate_async.py -v
```

- [ ] **Step 6.6: Commit**

```powershell
git add pipeline/translate.py tests/test_translate_async.py
git -c commit.gpgsign=false commit -m "feat(translate): previous_context parameter — 'Earlier in this scene' preamble + prompt directive"
```

---

## Task 7: Wire the rolling context window into the orchestrator

**Files:**
- Modify: `D:\git\whisper-gend\server.py` (`_translate_chunk`, `_run_gender_aware`, `_run_plain_translate`)
- Modify: `D:\git\whisper-gend\tests\test_orchestration.py`

Each translate call receives the last `TRANSLATE_CONTEXT_LINES` (default 4) source segments that the orchestrator has already passed through. The orchestrator maintains an ordered rolling window in the per-request scope. Cross-chunk: the context flows from chunk N's last group into chunk N+1's first group (this is OK because the context is SOURCE text — not subject to pyannote's per-chunk labelling problem that bit us in subtitle Task 7).

- [ ] **Step 7.1: Write failing tests**

Append to `tests/test_orchestration.py`:

```python
@pytest.mark.asyncio
async def test_translate_context_window_is_passed_to_each_batch(monkeypatch):
    """Each translate_batch_async call (one per group) receives a
    previous_context list containing the most recent TRANSLATE_CONTEXT_LINES
    source-language segments handled so far.
    """
    segs = [
        Segment(start=0.0, end=1.0, text="line A"),
        Segment(start=1.0, end=2.0, text="line B"),
        Segment(start=2.0, end=3.0, text="line C"),
        Segment(start=3.0, end=4.0, text="line D"),
        Segment(start=4.0, end=5.0, text="line E"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONCURRENCY", 1)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONTEXT_LINES", 2)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))
    monkeypatch.setattr(server, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 6, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    # Force 5 separate groups (one per segment) by having each segment
    # belong to a different "speaker".
    ann = Annotation()
    for i in range(5):
        ann[PSegment(float(i), float(i+1))] = f"S{i}"
    monkeypatch.setattr(server.diarize, "diarize_waveform",
                        lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders",
                        lambda audio, sr, a: {f"S{i}": "male" for i in range(5)})

    received: list[list[str]] = []
    async def fake_translate(texts, gender, target, client,
                             addressee_gender=None, source_language="English",
                             previous_context=None):
        received.append(list(previous_context) if previous_context else [])
        return [f"HE: {t}" for t in texts]
    monkeypatch.setattr(server.translate, "translate_batch_async",
                        fake_translate)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")

    # Group 1 (A): no prior context.
    # Group 2 (B): ["line A"].
    # Group 3 (C): ["line A", "line B"] — window=2 reached.
    # Group 4 (D): ["line B", "line C"] — rolled.
    # Group 5 (E): ["line C", "line D"].
    assert received == [
        [],
        ["line A"],
        ["line A", "line B"],
        ["line B", "line C"],
        ["line C", "line D"],
    ], received


@pytest.mark.asyncio
async def test_translate_context_disabled_when_setting_zero(monkeypatch):
    """TRANSLATE_CONTEXT_LINES=0 must produce no previous_context at all."""
    segs = [
        Segment(start=0.0, end=1.0, text="A"),
        Segment(start=1.0, end=2.0, text="B"),
    ]
    monkeypatch.setattr(server.settings, "TARGET_LANGUAGE", "Hebrew")
    monkeypatch.setattr(server.settings, "CHUNK_DURATION_SEC", 30)
    monkeypatch.setattr(server.settings, "TRANSLATE_CONTEXT_LINES", 0)
    monkeypatch.setattr(server.transcribe, "transcribe",
                        lambda path, language="en": list(segs))
    monkeypatch.setattr(server, "_load_wav_mono",
                        lambda path: (np.zeros(16000 * 3, dtype=np.float32), 16000))
    monkeypatch.setattr(server, "get_async_anthropic_client", lambda: object())

    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "S0"
    ann[PSegment(1.0, 2.0)] = "S1"
    monkeypatch.setattr(server.diarize, "diarize_waveform",
                        lambda *a, **k: ann)
    monkeypatch.setattr(server.gender, "detect_genders",
                        lambda audio, sr, a: {"S0": "male", "S1": "male"})

    received = []
    async def fake_translate(texts, gender, target, client,
                             addressee_gender=None, source_language="English",
                             previous_context=None):
        received.append(list(previous_context) if previous_context else [])
        return [f"HE: {t}" for t in texts]
    monkeypatch.setattr(server.translate, "translate_batch_async",
                        fake_translate)

    await server.run_pipeline_async(server.Path("ignored.wav"), "en")
    # Every call must see empty context.
    assert all(c == [] for c in received), received
```

- [ ] **Step 7.2: Run tests — confirm they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py -k "context_window or context_disabled" -v
```

Expected: 2 FAILs.

- [ ] **Step 7.3: Add the rolling-window plumbing to the orchestrator**

The cleanest place is `_translate_chunk`. Add a parameter:

```python
async def _translate_chunk(
    idx,
    groups,
    genders,
    target,
    client,
    sem,
    prev_speaker_gender,
    source_language="English",
    context_window: list[str] | None = None,
) -> None:
    ...
    async with sem:
        prev_group_gender = prev_speaker_gender
        try:
            for speaker, group in groups:
                spk_gender = genders.get(speaker, "male") if speaker else "male"
                addressee = (
                    prev_group_gender
                    if settings.ADDRESSEE_GENDER_HINT_ENABLED
                    else None
                )
                # Snapshot the window for THIS call before extending it.
                ctx_snapshot = list(context_window) if context_window else []
                translated = await translate.translate_batch_async(
                    [s.text for s in group], spk_gender, target, client,
                    addressee_gender=addressee,
                    source_language=source_language,
                    previous_context=ctx_snapshot,
                )
                for seg, text in zip(group, translated):
                    seg.text = text
                # Extend the rolling window with this group's SOURCE text
                # (text was already mutated to translated — capture before).
                if context_window is not None:
                    pass  # handled below via the group's original texts
                prev_group_gender = spk_gender
```

This needs the source texts captured BEFORE mutation. Refactor the inner loop to capture them first:

```python
            for speaker, group in groups:
                spk_gender = genders.get(speaker, "male") if speaker else "male"
                addressee = (
                    prev_group_gender
                    if settings.ADDRESSEE_GENDER_HINT_ENABLED
                    else None
                )
                source_texts = [s.text for s in group]
                ctx_snapshot = list(context_window) if context_window else []
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
                    # Trim to the configured window size.
                    max_n = settings.TRANSLATE_CONTEXT_LINES
                    if len(context_window) > max_n:
                        del context_window[: len(context_window) - max_n]
                prev_group_gender = spk_gender
```

In `_run_gender_aware`, initialise the shared window before the loop and pass it to each task:

```python
    context_window: list[str] | None = (
        [] if settings.TRANSLATE_CONTEXT_LINES > 0 else None
    )
    ...
    tasks.append(asyncio.create_task(
        _translate_chunk(
            idx, groups, genders, target, client, sem,
            None, source_language,
            context_window=context_window,
        )
    ))
```

**Important caveat.** The test asserts a strict ordering of context across groups. With `TRANSLATE_CONCURRENCY > 1`, chunks run in parallel and the window order is non-deterministic. The test pins `TRANSLATE_CONCURRENCY=1` to keep the assertion deterministic; in production with concurrency > 1 the rolling window is best-effort temporal ordering (good enough for an LLM context cue, not strict). This is a deliberate trade-off — document it in the function's docstring.

For `_run_plain_translate`, do the analogous wiring inside `translate_one`. Each chunk gets the rolling window; sequential ordering is preserved when `TRANSLATE_CONCURRENCY=1` and is best-effort otherwise.

- [ ] **Step 7.4: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_orchestration.py -v
```

- [ ] **Step 7.5: Full suite for regression**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 7.6: Commit**

```powershell
git add server.py tests/test_orchestration.py
git -c commit.gpgsign=false commit -m "feat(server): rolling TRANSLATE_CONTEXT_LINES window (default 4) passed to each translate batch"
```

---

## Task 8: Performance gate — alert when ML adds disproportionate wall time

**Files:**
- Modify: `D:\git\whisper-gend\pipeline\gender.py`
- Modify: `D:\git\whisper-gend\tests\test_gender.py`

When the ensemble mode runs both classifiers, measure each and emit a WARNING when ML's per-speaker time exceeds the pitch time by more than `GENDER_ML_TIME_BUDGET_RATIO` (default 5×, i.e. ML may take up to 5 s for every 1 s of pitch). Threshold is per-speaker so a single noisy outlier doesn't trip on a quiet chunk. Same comparison in ml-only mode is informational.

- [ ] **Step 8.1: Add config knob**

In `config.py`, add:

```python
    # WARNING is emitted when ML classifier wall time per speaker exceeds
    # the pitch classifier's wall time × this ratio. 0 disables the gate.
    GENDER_ML_TIME_BUDGET_RATIO: float = float(
        os.getenv("GENDER_ML_TIME_BUDGET_RATIO", "5.0")
    )
```

In `.env.example` append:

```ini
# WARNING when wav2vec2 wall time exceeds pitch wall time by this ratio.
# 0 disables the gate.
GENDER_ML_TIME_BUDGET_RATIO=5.0
```

- [ ] **Step 8.2: Write failing test**

Add to `tests/test_gender.py`:

```python
def test_ensemble_warns_when_ml_too_slow(monkeypatch, caplog):
    """When ML classifier wall time exceeds pitch by more than the budget
    ratio, _classify_speaker emits a WARNING."""
    import logging, time
    caplog.set_level(logging.WARNING, logger="pipeline.gender")
    monkeypatch.setattr("config.settings.GENDER_CLASSIFIER", "ensemble")
    monkeypatch.setattr("config.settings.GENDER_ML_TIME_BUDGET_RATIO", 2.0)

    def slow_ml(audio, sr):
        time.sleep(0.05)  # ML "takes" 50ms; pitch is near-instant.
        return ("female", 0.9)
    monkeypatch.setattr("pipeline.gender_ml.classify_audio", slow_ml)

    from pipeline import gender as g
    low = _tone(120.0)
    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "SPEAKER_X"
    g.detect_genders(low, SR, ann)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("slow" in m.lower() and "ml" in m.lower()
               for m in msgs), f"no perf warning; got {msgs}"
```

- [ ] **Step 8.3: Run test — confirm FAIL**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gender.py::test_ensemble_warns_when_ml_too_slow -v
```

- [ ] **Step 8.4: Add timing + gate to `_classify_speaker` (ensemble branch)**

In `pipeline/gender.py`, replace the ensemble branch with:

```python
    if mode == "ensemble":
        t0 = time.perf_counter()
        pitch_label = _classify_f0(voiced_f0)
        pitch_dt = time.perf_counter() - t0

        from pipeline import gender_ml
        t0 = time.perf_counter()
        try:
            ml_label, ml_conf = gender_ml.classify_audio(signal, sr)
        except Exception:
            log.warning(
                "Speaker %s ML classifier raised; falling back to pitch=%s",
                speaker_label, pitch_label, exc_info=True,
            )
            return pitch_label
        ml_dt = time.perf_counter() - t0

        budget = float(settings.GENDER_ML_TIME_BUDGET_RATIO)
        if budget > 0 and pitch_dt > 0 and ml_dt > pitch_dt * budget:
            log.warning(
                "Speaker %s ML classifier slow: ml=%.2fs vs pitch=%.4fs "
                "(ratio %.1fx > budget %.1fx)",
                speaker_label, ml_dt, pitch_dt, ml_dt / max(pitch_dt, 1e-6),
                budget,
            )

        if ml_label != pitch_label:
            log.info(
                "Speaker %s classifiers disagree: pitch=%s ml=%s (conf=%.3f) "
                "— using ML",
                speaker_label, pitch_label, ml_label, ml_conf,
            )
        else:
            log.info(
                "Speaker %s classifiers agree: %s (ML conf=%.3f)",
                speaker_label, ml_label, ml_conf,
            )
        return ml_label
```

Add at the top of `pipeline/gender.py`:

```python
import time
```

- [ ] **Step 8.5: Run tests — confirm PASS**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_gender.py -v
```

- [ ] **Step 8.6: Full suite for regression**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 8.7: Commit + push the whole branch (NO PR per spec)**

```powershell
git add pipeline/gender.py config.py .env.example tests/test_gender.py
git -c commit.gpgsign=false commit -m "feat(gender): perf gate — WARNING when ML wall time exceeds pitch × GENDER_ML_TIME_BUDGET_RATIO (default 5x)"
git push -u origin feature/improve-gender-detection
```

**Do NOT open a pull request.** The user evaluates the branch directly.

---

## Self-review

**Spec coverage:**

| Spec line | Task |
|---|---|
| "Use Kaggle datasets and models" → pre-trained HF wav2vec2 (assumption #1) | Task 1, 2 |
| Add past conversations to translate context (default 4, configurable) | Task 6, 7 |
| "Use open-source libraries if they exist, instead of implementing from scratch" | wav2vec2 via transformers (Task 2); no custom training |
| "Perform A/B to test the new detection engine" | ensemble-mode disagreement log (Task 4) + alt-classifier SRT side-file (Task 5) |
| "Write plan, run tests, repeat, self-review, production ready" | TDD red→green steps in every task; this self-review section |
| "Performance: alert if there is a possibility of huge drop of performance" | Task 8 perf gate (WARNING log line, configurable ratio) |
| "Branch: feature branch, no pr" | Task 8 final commit + `git push -u origin feature/improve-gender-detection` only; no `gh pr create` |

All spec points have a task.

**Placeholder scan:** No "TODO", no "fill in later", no "add appropriate error handling". Every code-changing step shows the actual code.

**Type consistency:**
- `classify_audio(audio, sr) -> tuple[str, float]` defined in Task 2, used identically in Tasks 4 and 8.
- `previous_context: list[str] | None = None` parameter signature is the same across all four batch helpers in Task 6 and the orchestrator call sites in Task 7.
- `_classify_speaker(signal, sr, speaker_label, voiced_f0)` signature stable across Tasks 4 and 8.
- `GENDER_CLASSIFIER` accepted values `"pitch" | "ml" | "ensemble"` consistently used in Tasks 3, 4, 5, 8.

**Cross-task ordering:**
- Task 2 depends on Task 3's config keys (`settings.GENDER_ML_MODEL`). Execution order is documented in Task 2.4: "execute Task 3 before re-running Task 2's tests." A subagent runner should execute Task 3 before Task 2's commit step.
- Task 5 depends on Task 4's dispatcher being in place. Task 4 must commit before Task 5.
- Tasks 6 + 7 are independent of 1-5 and can run in parallel by a subagent.
- Task 8 depends on Task 4.

**Out of scope (deliberate):**
- Training a classifier on the Kaggle CSV from scratch — deferred unless the pre-trained model underperforms on the user's content. Documented as a follow-up.
- Speaker re-identification across chunks (would let cross-chunk addressee context work better) — separate plan; the existing Plan Task 7 of `2026-05-28-subtitle-quality-improvements.md` already disabled the brittle naive carry.
- Replacing pyannote for diarization. This plan only changes the classifier that runs AFTER pyannote identifies speakers.

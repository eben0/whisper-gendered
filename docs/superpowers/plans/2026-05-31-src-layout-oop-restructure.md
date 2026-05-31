# `src/` Layout + Full OOP Restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Run on Sonnet 4.6 (`claude-sonnet-4-6`)**.

---

## Context

PR #2 review by eben0 (project owner) on the thin-server-core-package refactor. Comments are requirements, not optional feedback. Three structural mandates:

1. **Full OOP** — every module's logic lives in a class; class in its own dedicated file; instantiation happens in the caller; singletons use double-checked locking.
2. **`src/` layout** — all Python source moves under `src/`; tests, scripts, root config stay at repo root.
3. **`core/` = infrastructure only** — configs, loggers, concurrency; domain-specific things (backends, audio, side-file, etc.) go in dedicated subfolders or at `src/` root.

Inline comments add specific requirements:
- Merge `backends.py` + `backend_base.py` → `backend_factory.py`; move backend files to `src/backends/`
- `Audio` class in `src/audio.py`
- `ConcurrencyManager` — dedicated file, no module-level extras; init in caller
- `Cuda` class in `src/core/cuda.py`
- `Lifecycle` class in `src/lifecycle.py`
- `SideFile` class in `src/side_file.py`

---

**Goal:** Restructure the codebase into a `src/` layout with consistent OOP design — every logical unit is a class in its own file, instantiated by its caller, with proper singleton discipline.

**Architecture:** All importable Python source lives under `src/`. `src/core/` holds pure infrastructure (cuda, logging, concurrency). Domain packages (`src/backends/`) and domain modules (`src/audio.py`, `src/side_file.py`, `src/lifecycle.py`, `src/orchestrator.py`, `src/artifacts.py`) live at `src/` root level. `src/pipeline/` is unchanged. `server.py` creates all instances at startup and injects dependencies.

**Tech Stack:** Python 3.11+, FastAPI, pytest. No new dependencies.

---

## Target directory structure

```
src/
  server.py                  # thin HTTP: creates instances, wires routes
  config.py                  # Settings dataclass (unchanged)
  core/                      # infrastructure only
    __init__.py
    cuda.py                  # Cuda class
    logging_config.py        # configure() (stays procedural — stateless one-shot)
    concurrency.py           # ConcurrencyManager class (clean, no extras)
  backends/                  # translation backend package
    __init__.py
    factory.py               # TranslationBackend ABC + LOCAL/CLAUDE + is_() + create_backend()
                             #   (merged from: backend_base.py + backends.py + backend_factory.py)
    claude.py                # ClaudeBackend (renamed from backend_claude.py)
    local.py                 # LocalBackend (renamed from backend_local.py)
  audio.py                   # Audio class
  side_file.py               # SideFile class
  lifecycle.py               # Lifecycle class
  orchestrator.py            # orchestration functions (no class required — not in PR comments)
  artifacts.py               # PipelineArtifacts dataclass (unchanged)
  pipeline/
    segment.py, transcribe.py, diarize.py, gender.py, gender_ml.py,
    translate.py, translate_local.py, chunk.py, format.py, lang.py
tests/                       # unchanged (stays at root)
conftest.py                  # update sys.path to include src/
pytest.ini                   # update testpaths
run.ps1                      # update python invocation
```

---

## Dependency injection design

`server.py` owns construction of all stateful singletons:

```python
# server.py startup sequence
cuda_obj = Cuda()
cuda_obj.bootstrap()
configure_logging()
concurrency_mgr = ConcurrencyManager(settings.CONCURRENT_JOBS)
backend = create_backend(settings)         # no module-level singleton in factory
audio_obj = Audio()
side_file_obj = SideFile(settings)
lifecycle_obj = Lifecycle(concurrency_mgr, audio_obj, backend, settings)
# orchestrator functions receive concurrency_mgr, backend, audio_obj as args
```

Singletons that need lazy init (e.g. `ClaudeBackend._get_client`) use double-checked locking:
```python
import threading
_lock = threading.Lock()

def _get_client(self):
    if self._client is None:
        with _lock:
            if self._client is None:   # second check inside lock
                self._client = anthropic.AsyncAnthropic(...)
    return self._client
```

---

## Spec assumptions

1. `logging_config.py` stays as `configure()` function (stateless one-shot, no class needed).
2. `orchestrator.py` stays as module-level functions (not called out in PR comments for class conversion).
3. `pipeline/` package stays untouched (pure ML stages — not in PR comments).
4. `core/__init__.py` exists as a package marker with no auto-imports.
5. `backends/__init__.py` re-exports `TranslationBackend`, `LOCAL`, `CLAUDE`, `create_backend` for clean external imports.
6. Module-level `backend = create_backend()` singleton **removed** from `backend_factory.py`; caller (server.py) calls `create_backend(settings)` at startup.
7. `ConcurrencyManager` module-level `manager` instance and re-exports (`semaphore = manager.semaphore` etc.) **removed**; caller holds the instance.
8. Tests that currently import from `core.*` will be updated to import from `src.*` paths or via conftest sys.path.

---

## File structure table

| File | Responsibility | Task |
|---|---|---|
| `conftest.py`, `pytest.ini` | add `src/` to sys.path; update testpaths | 0 |
| `src/` (directory) | package root for all source | 0 |
| `src/config.py` | move from root `config.py` | 1 |
| `src/core/__init__.py` | package marker | 1 |
| `src/core/cuda.py` | `Cuda` class: `bootstrap()`, `empty_cache()`; replaces `core/cuda.py` | 1 |
| `src/core/logging_config.py` | `configure()` function; replaces `core/logging_config.py` | 1 |
| `src/core/concurrency.py` | `ConcurrencyManager` class only, no singleton, no re-exports | 1 |
| `src/backends/__init__.py` | re-exports | 2 |
| `src/backends/factory.py` | `TranslationBackend` ABC + `LOCAL`/`CLAUDE` + `is_()` + `create_backend(settings)` | 2 |
| `src/backends/claude.py` | `ClaudeBackend` with double-lock `_get_client()` | 2 |
| `src/backends/local.py` | `LocalBackend` | 2 |
| `src/audio.py` | `Audio` class: `encode_to_wav`, `prepare_unencoded`, `_write_silent_wav`, `_load_wav_mono` | 3 |
| `src/side_file.py` | `SideFile` class: path helpers + atomic writes + summary builder | 4 |
| `src/lifecycle.py` | `Lifecycle` class: `__init__(concurrency_mgr, audio, backend, settings)`, `async warmup()` | 5 |
| `src/artifacts.py` | `PipelineArtifacts` dataclass — move from `core/artifacts.py` | 6 |
| `src/orchestrator.py` | module-level orchestration functions — move from `core/orchestrator.py` | 6 |
| `src/pipeline/` | unchanged — move directory from root `pipeline/` | 6 |
| `src/server.py` | thin HTTP: constructs all instances at startup, wires routes | 7 |
| `tests/` | update imports to `src.*`; update monkeypatch targets | 8 |
| `run.ps1` | update python invocation to `src/server.py` | 9 |
| **DELETE** | root `config.py`, `server.py`, `core/`, `pipeline/` | 9 |

---

## Task 0: Bootstrap — `src/` directory + import infrastructure

**Files:**
- Create: `src/__init__.py` (empty)
- Modify: `conftest.py` (add `src/` to `sys.path`)
- Modify: `pytest.ini` (update `testpaths`)

- [ ] **Step 0.1: Create `src/` directory with empty `__init__.py`**

```bash
mkdir src
touch src/__init__.py
```

- [ ] **Step 0.2: Update `conftest.py`** — add `src/` to sys.path so tests can import from `src`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
```

Read current `conftest.py` first — it may already set up `sys.path` for the root.

- [ ] **Step 0.3: Update `pytest.ini`**

Change `testpaths = tests` to stay as-is (tests stay at root). Add `pythonpath = src` if using pytest ≥ 7 (`pythonpath` ini option replaces `sys.path` manipulation in conftest):

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
pythonpath = src
```

If `pythonpath` is not supported by the installed pytest version, rely on the conftest approach from Step 0.2.

- [ ] **Step 0.4: Verify pytest discovers tests without import errors**

```powershell
& "D:\git\whisper-gend\.venv\Scripts\python.exe" -m pytest --collect-only -q 2>&1 | Select-Object -Last 10
```

Expected: tests collected (even though they'll fail to import source until Tasks 1-8 complete). No `ModuleNotFoundError` for `conftest` itself.

- [ ] **Step 0.5: Commit**

```bash
git add src/__init__.py conftest.py pytest.ini
git -c commit.gpgsign=false commit -m "chore: create src/ layout; update conftest and pytest.ini"
```

---

## Task 1: `src/core/` — Cuda class, ConcurrencyManager (clean), logging_config, config

**Files:**
- Create: `src/core/__init__.py`
- Create: `src/core/cuda.py` — `Cuda` class
- Create: `src/core/logging_config.py` — `configure()` function
- Create: `src/core/concurrency.py` — `ConcurrencyManager` class, NO module instance, NO re-exports
- Create: `src/config.py` — copy of root `config.py`

### Step 1.1: `src/core/__init__.py` (package marker)

```python
"""Core infrastructure: CUDA bootstrap, logging, and concurrency."""
```

### Step 1.2: `src/core/cuda.py` — `Cuda` class

```python
"""CUDA and librosa bootstrap — must run before pyannote/CTranslate2 load."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("core.cuda")


class Cuda:
    """Encapsulates the import-ordering-critical platform bootstrap.

    Call ``bootstrap()`` first in any entrypoint — before importing
    ``orchestrator`` or ``lifecycle`` (which pull in pyannote/speechbrain).
    """

    def bootstrap(self) -> None:
        """Register NVIDIA DLL dirs and force-import librosa. Idempotent."""
        self._register_cuda_dll_dirs()
        self._preimport_librosa()

    def empty_cache(self) -> None:
        """Best-effort release of cached-but-unused CUDA memory. Never raises."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            log.debug("cuda.empty_cache failed (continuing)", exc_info=True)

    def _register_cuda_dll_dirs(self) -> None:
        # verbatim body from current core/cuda.py register_cuda_dll_dirs()
        ...

    def _preimport_librosa(self) -> None:
        # verbatim body from current core/cuda.py preimport_librosa()
        ...
```

Copy verbatim function bodies from current `core/cuda.py`.

### Step 1.3: `src/core/logging_config.py`

```python
"""Process-wide logging configuration."""

from __future__ import annotations

import logging
from config import settings


def configure() -> None:
    """Apply logging config. Call once at entrypoint start."""
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
```

### Step 1.4: `src/core/concurrency.py` — `ConcurrencyManager` class ONLY

```python
"""ConcurrencyManager: bounded executor, semaphore, and job counter."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


class ConcurrencyManager:
    """Encapsulates GPU-job concurrency. Instantiate once; inject into callers.

    No module-level singleton — the caller (server.py startup) creates the
    instance and passes it to orchestrator functions and lifecycle.
    """

    def __init__(self, concurrent_jobs: int) -> None:
        self.semaphore = asyncio.Semaphore(concurrent_jobs)
        self._executor = ThreadPoolExecutor(max_workers=max(2, concurrent_jobs + 1))
        self._jobs = 0

    async def run_in_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def inc_jobs(self) -> None:
        self._jobs += 1

    def dec_jobs(self) -> None:
        self._jobs -= 1

    def job_depth(self) -> int:
        return self._jobs
```

**No** `manager = ConcurrencyManager(...)` at module level. **No** re-exports.

### Step 1.5: `src/config.py`

Copy `config.py` from repo root verbatim.

### Step 1.6: Run collection check

```powershell
& "D:\git\whisper-gend\.venv\Scripts\python.exe" -m pytest --collect-only -q 2>&1 | Select-Object -Last 5
```

### Step 1.7: Commit

```bash
git add src/
git -c commit.gpgsign=false commit -m "feat(src/core): Cuda class, ConcurrencyManager (clean), logging_config, config"
```

---

## Task 2: `src/backends/` — merged factory, ClaudeBackend, LocalBackend

**Files:**
- Create: `src/backends/__init__.py`
- Create: `src/backends/factory.py` — merged from `backend_base.py` + `backends.py` + `backend_factory.py`
- Create: `src/backends/claude.py` — `ClaudeBackend` with double-lock
- Create: `src/backends/local.py` — `LocalBackend`

### Step 2.1: `src/backends/__init__.py`

```python
"""Translation backend package."""

from src.backends.factory import (
    TranslationBackend,
    LOCAL,
    CLAUDE,
    create_backend,
)

__all__ = ["TranslationBackend", "LOCAL", "CLAUDE", "create_backend"]
```

### Step 2.2: `src/backends/factory.py` — merge of three files

```python
"""Translation backend factory and abstract base class.

Merged from: core/backend_base.py + core/backends.py + core/backend_factory.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings

LOCAL = "local"
CLAUDE = "claude"


class TranslationBackend(ABC):
    """Common interface for all translation backends.

    Use ``backend.is_(LOCAL)`` or ``backend.is_(CLAUDE)`` for identity checks.
    """

    def is_(self, backend_type: str) -> bool:
        """Return True if this instance matches ``backend_type``."""
        return getattr(self, "_backend_type", None) == backend_type

    @abstractmethod
    async def translate_batch_async(
        self, texts: list[str], gender: str | None, target: str, **kwargs: Any
    ) -> list[str]: ...

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    def model_name(self) -> str: ...


def create_backend(settings: "Settings") -> TranslationBackend:
    """Instantiate the backend named by ``settings.TRANSLATION_BACKEND``.

    Raises ``ValueError`` for unrecognised backend names.
    No module-level singleton — caller creates and holds the instance.
    """
    from src.backends.claude import ClaudeBackend
    from src.backends.local import LocalBackend

    match settings.TRANSLATION_BACKEND.strip().lower():
        case "claude":
            return ClaudeBackend(settings)
        case "local":
            return LocalBackend()
        case other:
            raise ValueError(
                f"Unknown TRANSLATION_BACKEND {other!r}. Valid: 'claude', 'local'."
            )
```

### Step 2.3: `src/backends/claude.py` — double-lock singleton

```python
"""Claude API translation backend."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from src.backends.factory import TranslationBackend, CLAUDE

if TYPE_CHECKING:
    from src.config import Settings

log = logging.getLogger("backends.claude")

_client_lock = threading.Lock()


class ClaudeBackend(TranslationBackend):
    """Calls the Anthropic API via pipeline.translate."""

    _backend_type = CLAUDE

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._client = None

    def _get_client(self):
        """Lazy singleton with double-checked locking."""
        if self._client is None:
            with _client_lock:
                if self._client is None:
                    import anthropic
                    self._client = anthropic.AsyncAnthropic(
                        api_key=self._settings.require_anthropic_key(),
                        max_retries=self._settings.CLAUDE_MAX_RETRIES,
                    )
        return self._client

    async def translate_batch_async(self, texts, gender, target, **kwargs):
        from pipeline.translate import translate_batch_async
        return await translate_batch_async(
            texts, gender, target, self._get_client(), **kwargs
        )

    async def warmup(self) -> None:
        pass  # API-based; no model to preload

    def model_name(self) -> str:
        return self._settings.CLAUDE_MODEL
```

### Step 2.4: `src/backends/local.py`

```python
"""Local HuggingFace seq2seq translation backend."""

from __future__ import annotations

import logging

from src.backends.factory import TranslationBackend, LOCAL

log = logging.getLogger("backends.local")


class LocalBackend(TranslationBackend):
    """On-device HuggingFace seq2seq translation."""

    _backend_type = LOCAL

    async def translate_batch_async(self, texts, gender, target, **kwargs):
        from pipeline.translate_local import translate_batch_async
        return await translate_batch_async(texts, gender, target, None, **kwargs)

    async def warmup(self) -> None:
        from pipeline import translate_local
        translate_local.warmup()

    def model_name(self) -> str:
        from src.config import settings
        return settings.LOCAL_TRANSLATION_MODEL
```

### Step 2.5: Commit

```bash
git add src/backends/
git -c commit.gpgsign=false commit -m "feat(src/backends): merged factory + TranslationBackend ABC + ClaudeBackend (double-lock) + LocalBackend"
```

---

## Task 3: `src/audio.py` — `Audio` class

**Files:**
- Create: `src/audio.py`

```python
"""Audio I/O — encoding, format conversion, and waveform loading."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("audio")


class Audio:
    """Stateless audio I/O operations.

    Instantiate once; the instance holds no state — all methods are pure I/O.
    """

    def encode_to_wav(self, src: Path, dst: Path) -> None:
        """Re-encode any input to 16 kHz mono WAV via ffmpeg."""
        # verbatim body from current core/audio.py encode_to_wav

    def prepare_unencoded(self, src: Path, dst: Path, sr: int = 16000) -> None:
        """Convert a headerless PCM upload to a real WAV."""
        # verbatim body from current core/audio.py prepare_unencoded

    def write_silent_wav(self, path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
        """Write a silent WAV (for warm-up use)."""
        sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)

    def load_wav_mono(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Load a WAV as mono float32 waveform + sample rate."""
        # verbatim body from current core/audio.py _load_wav_mono
```

Note: method names drop leading underscores (these are now instance methods, not module privates).

- [ ] **Run tests after:** `& "D:\git\whisper-gend\.venv\Scripts\python.exe" -m pytest -q 2>&1 | tail -3` (tests still run against old paths — this is expected until Task 8)
- [ ] **Commit:** `git -c commit.gpgsign=false commit -m "feat(src): Audio class"`

---

## Task 4: `src/side_file.py` — `SideFile` class

**Files:**
- Create: `src/side_file.py`

```python
"""Side-file I/O: path resolution, atomic writes, and translation summaries."""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings
    from pipeline.segment import Segment

log = logging.getLogger("side_file")

TMP_SUFFIX = ".tmp"
ENCODING = "utf-8"
SUMMARY_SUFFIX = ".summary.txt"
ALT_CLASSIFIER_SRT_SUFFIX = ".he.alt-classifier.srt"


class SideFile:
    """Handles side-file path resolution, atomic writes, and summary generation.

    Instantiate with ``settings``; the instance caches prefix/suffix config.
    """

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings

    def compute_path(self, video_file_url: str, suffix: str | None = None) -> Path | None:
        """Translate a Bazarr video URL to a local SRT path. Pure — no filesystem I/O."""
        # verbatim logic from _compute_side_file_path
        effective_suffix = suffix or self._settings.SAVE_SRT_SUFFIX
        # ... (move body verbatim, using self._settings for prefix access)

    def compute_summary_path(self, srt_path: Path) -> Path:
        """Derive the summary path from an SRT path. Pure."""
        # verbatim body from _compute_summary_path

    def try_save(self, body: str, summary: str | None, video_file_url: str,
                 suffix: str | None = None) -> None:
        """Atomically write the SRT and (optionally) a summary file."""
        # verbatim body from _try_save_side_file, using self.compute_path, self._settings

    def build_summary(self, *, request_id, video_file_url, source_language_iso,
                      source_language_name, target_language, backend, backend_model,
                      segments, wall_seconds) -> str:
        """Build a human-readable translation summary string."""
        # verbatim body from _build_translation_summary

    @staticmethod
    def fmt_timestamp(t: float) -> str:
        """Render float seconds as HH:MM:SS."""
        # verbatim body from _fmt_timestamp
```

- [ ] **Commit:** `git -c commit.gpgsign=false commit -m "feat(src): SideFile class"`

---

## Task 5: `src/lifecycle.py` — `Lifecycle` class

**Files:**
- Create: `src/lifecycle.py`

```python
"""Startup warm-up: pre-loads ML models so the first request isn't penalised."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings
    from src.core.concurrency import ConcurrencyManager
    from src.audio import Audio
    from src.backends.factory import TranslationBackend

log = logging.getLogger("lifecycle")


class Lifecycle:
    """Manages application startup warm-up.

    Instantiated with all dependencies injected — no global imports needed.
    """

    def __init__(
        self,
        concurrency: "ConcurrencyManager",
        audio: "Audio",
        backend: "TranslationBackend",
        settings: "Settings",
    ) -> None:
        self._concurrency = concurrency
        self._audio = audio
        self._backend = backend
        self._settings = settings

    async def warmup(self) -> None:
        """Pre-load Whisper, pyannote, gender_ml, and (if local) the translation model."""
        settings = self._settings
        log.info(
            "Starting up. model=%s device=%s target_language=%s gender_aware=%s "
            "translation_backend=%s",
            settings.WHISPER_MODEL, settings.DEVICE, settings.TARGET_LANGUAGE,
            settings.is_gender_aware(), settings.TRANSLATION_BACKEND,
        )
        tmp = Path(tempfile.gettempdir()) / f"warmup_{uuid.uuid4().hex}.wav"
        try:
            self._audio.write_silent_wav(tmp)
            # verbatim warmup body from core/lifecycle.py, using self._concurrency,
            # self._audio, self._backend, self._settings
            ...
        finally:
            tmp.unlink(missing_ok=True)
```

- [ ] **Commit:** `git -c commit.gpgsign=false commit -m "feat(src): Lifecycle class with injected dependencies"`

---

## Task 6: Move remaining modules to `src/`

**Files:**
- Create: `src/artifacts.py` — copy `core/artifacts.py` (update import: `from pipeline.segment import Segment`)
- Create: `src/orchestrator.py` — copy `core/orchestrator.py` (update all `core.*` → `src.*` imports; update `audio._load_wav_mono` → `audio_obj.load_wav_mono` etc. — see dependency injection note below)
- Move: `pipeline/` → `src/pipeline/` (entire directory)

**Orchestrator dependency injection:** The orchestrator functions currently use module-level `concurrency.run_in_thread`, `audio._load_wav_mono`, `backends.backend.translate_batch_async`. After this task, they must receive these as parameters. Options:

**Option A (recommended for minimal disruption):** Keep orchestrator functions with added keyword params:
```python
async def run_pipeline_async(
    audio_path, language, settings, concurrency_mgr, audio_obj, backend,
    _artifacts_out=None
):
```

**Option B:** Create `Orchestrator` class (not required by PR comments, adds complexity).

Use **Option A** — the PR comments only mandated classes for Audio, Cuda, Lifecycle, SideFile.

- [ ] **Commit:** `git -c commit.gpgsign=false commit -m "feat(src): move artifacts, orchestrator, pipeline to src/"`

---

## Task 7: `src/server.py` — thin HTTP layer, constructs all instances

**Files:**
- Create: `src/server.py`

`server.py` is responsible for construction:

```python
"""Bazarr-compatible Whisper ASR server — thin HTTP wiring only.

Endpoints: GET /, GET /status, POST /asr
"""

from __future__ import annotations

# ORDERING-CRITICAL: cuda bootstrap before pyannote/orchestrator imports
from src.core.cuda import Cuda
_cuda = Cuda()
_cuda.bootstrap()

from src.core.logging_config import configure
configure()

# --- stdlib, FastAPI, config ---
import shutil, tempfile, time, uuid
from pathlib import Path
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from src.config import settings

# --- core infrastructure ---
from src.core.concurrency import ConcurrencyManager

# --- domain ---
from src.backends.factory import create_backend
from src.audio import Audio
from src.side_file import SideFile, ALT_CLASSIFIER_SRT_SUFFIX
from src.lifecycle import Lifecycle
from src.artifacts import PipelineArtifacts
from src import orchestrator
from pipeline.format import render
from pipeline.lang import language_name

VERSION = "1.0.0"

# --- construct singletons at module load (startup) ---
_concurrency = ConcurrencyManager(settings.CONCURRENT_JOBS)
_backend = create_backend(settings)
_audio = Audio()
_side_file = SideFile(settings)
_lifecycle = Lifecycle(_concurrency, _audio, _backend, settings)

app = FastAPI(title="Gender-Aware Hebrew Subtitle Server", version=VERSION)


@app.on_event("startup")
async def warmup() -> None:
    await _lifecycle.warmup()


@app.get("/")
async def root():
    return JSONResponse({"status": "ok", "version": VERSION, "model": settings.WHISPER_MODEL})


@app.get("/status")
async def status():
    from pipeline import transcribe
    return JSONResponse({
        "status": "ok",
        "queue_depth": _concurrency.job_depth(),
        "model_loaded": transcribe.model_loaded(),
    })


@app.post("/asr")
async def asr(
    request: Request,
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe"),
    language: str = Query("en"),
    output: str = Query("srt"),
    encode: bool = Query(True),
):
    # verbatim /asr body from current server.py, using:
    # _concurrency, _backend, _audio, _side_file, _cuda, orchestrator.*
    ...


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=settings.PORT, workers=1)
```

- [ ] **Commit:** `git -c commit.gpgsign=false commit -m "feat(src): thin server.py with dependency-injected instances"`

---

## Task 8: Update tests to import from `src.*`

**Files:**
- Modify: `tests/test_side_file.py` — `from src.side_file import SideFile` (and update call style to class methods)
- Modify: `tests/test_orchestration.py` — update all imports and monkeypatch targets to `src.*`
- Modify: `tests/test_translate_local.py` — update to `src.backends.factory`
- Modify: `tests/test_config.py`, `tests/test_config_new_keys.py` — `from src.config import Settings`
- Modify: `tests/test_chunk.py`, `tests/test_diarize_waveform.py`, `tests/test_format.py`, `tests/test_gender.py`, `tests/test_gender_ml.py`, `tests/test_lang.py`, `tests/test_transcribe.py`, `tests/test_translate_async.py`, `tests/test_translate_local.py` — update `from pipeline.*` → `from pipeline.*` (unchanged — pipeline stays importable from `src/pipeline/` via `sys.path`)

Run the full suite:
```powershell
& "D:\git\whisper-gend\.venv\Scripts\python.exe" -m pytest -q
```
Expected: **133 passed, 2 skipped**

- [ ] **Commit:** `git -c commit.gpgsign=false commit -m "test: update all imports and monkeypatch targets to src.* layout"`

---

## Task 9: Remove old root-level source files; update run.ps1

**Files:**
- Delete: root `server.py`, `config.py`
- Delete: root `core/` directory (all of it — replaced by `src/core/`)
- Delete: root `pipeline/` directory (moved to `src/pipeline/`)
- Modify: `run.ps1` — update invocation

In `run.ps1`, update:
```powershell
# OLD
python server.py
# or: uvicorn server:app ...

# NEW
python src/server.py
# or: uvicorn src.server:app ...
```

Final full test run:
```powershell
& "D:\git\whisper-gend\.venv\Scripts\python.exe" -m pytest -q
```
Expected: **133 passed, 2 skipped**

Smoke import:
```powershell
& "D:\git\whisper-gend\.venv\Scripts\python.exe" -c "from src.server import app, VERSION; print('ok', VERSION)"
```

- [ ] **Commit:** `git -c commit.gpgsign=false commit -m "refactor: remove root-level source; all code lives under src/"`

---

## Invariants (must survive)

- HTTP contract: `GET /`, `GET /status`, `POST /asr` — unchanged
- Import-ordering: `Cuda().bootstrap()` first in `src/server.py`, before orchestrator imports
- All ML singletons, semaphore-gated dispatch, atomic UTF-8 writes — unchanged behavior
- 133 tests green (2 pre-existing hardware skips)
- `from src.server import app` works for uvicorn

---

## Execution path

1. `/model claude-sonnet-4-6` — run on Sonnet 4.6
2. **superpowers:subagent-driven-development** — one subagent per task, two-stage review after each
3. **superpowers:requesting-code-review** + **security-review** before merge
4. **superpowers:finishing-a-development-branch** — push updated PR branch

Work in the existing worktree: `D:\git\whisper-gend\.claude\worktrees\feature+thin-server-core-package`

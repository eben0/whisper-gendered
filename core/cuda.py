"""Runtime bootstrap that MUST run before pyannote/CTranslate2 load.

``bootstrap()`` registers the NVIDIA cuBLAS/cuDNN DLL dirs and force-imports librosa's
audio core against a clean module table (the speechbrain ``k2`` lazy-import trap). It is
import-ordering critical: call it FIRST in any entrypoint, before importing
``core.orchestrator`` / ``core.lifecycle`` (which transitively import pyannote).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("core.cuda")


def register_cuda_dll_dirs() -> None:
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


def preimport_librosa() -> None:
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


def empty_cuda_cache() -> None:
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


def bootstrap() -> None:
    """Run the import-ordering-critical setup. Idempotent enough for one call/process."""
    register_cuda_dll_dirs()
    preimport_librosa()

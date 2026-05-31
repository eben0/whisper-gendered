"""Backward-compat shim — the real implementation is in pipeline.gender.ml.

Tests and callers that do ``from pipeline import gender_ml`` and then access
module-level ``get_pipeline``, ``classify_audio``, ``model_loaded``, ``warmup``,
and ``_pipeline`` continue to work through this module.

IMPORTANT: ``classify_audio`` calls the module-level ``get_pipeline()`` so that
tests which do ``monkeypatch.setattr(gender_ml, "get_pipeline", lambda: fake)``
intercept the call correctly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.gender.ml import GenderMLClassifier, _Wav2Vec2GenderPipeline  # noqa: F401

import logging

log = logging.getLogger("pipeline.gender_ml")

from config import settings as _settings  # noqa: E402

_instance = GenderMLClassifier(_settings)

# Module-level _pipeline so tests can do:
#   monkeypatch.setattr(gender_ml, "_pipeline", None)
#   monkeypatch.setattr(gender_ml, "_pipeline", object())
# and model_loaded() / classify_audio() observe it.
_pipeline: Any | None = None
_lock = _instance._lock


def get_pipeline() -> Any:
    global _pipeline
    result = _instance.get_pipeline()
    _pipeline = result
    return result


def model_loaded() -> bool:
    # Read the module-level _pipeline so that monkeypatch.setattr on
    # ``gender_ml._pipeline`` is observed here.
    return _pipeline is not None


def classify_audio(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """Classify a mono waveform — calls module-level get_pipeline() so tests can monkeypatch it."""
    pipe = get_pipeline()
    rows = pipe({"array": audio, "sampling_rate": sr})
    winner = rows[0]
    label = str(winner["label"]).lower()
    if label not in {"male", "female"}:
        raise ValueError(
            f"unexpected gender label from classifier: {winner['label']!r} "
            f"(full row={winner!r})"
        )
    return label, float(winner["score"])


def warmup() -> None:
    try:
        get_pipeline()
        log.info("Gender ML warm-up complete.")
    except Exception:  # pragma: no cover - warm-up must never crash startup
        log.exception("Gender ML warm-up failed (continuing anyway).")

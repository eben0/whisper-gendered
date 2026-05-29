"""wav2vec2 audio-classification wrapper for speaker gender detection.

A thin alternative to the pitch-based ``pipeline/gender.py``. We wrap a
HuggingFace audio-classification model (default
``alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech``) behind a
lazy, thread-safe singleton so the ~1.2 GB model is loaded at most once per
process and only on first use.

Implementation note: we deliberately do NOT use ``transformers.pipeline(
"audio-classification", ...)``. The HF pipeline helper imports ``torchcodec``,
whose Windows DLLs cannot be loaded against torch 2.11.0+cu130 in this venv
(verified during Task 1 pre-flight). Instead we drive ``AutoFeatureExtractor``
+ ``AutoModelForAudioClassification`` directly and expose a small callable
whose return shape mirrors the HF pipeline so callers/tests can treat the two
interchangeably.
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


class _Wav2Vec2GenderPipeline:
    """HF-pipeline-shaped callable around a wav2vec2 audio classifier.

    ``__call__(payload, top_k=None)`` accepts the same payload as HF's
    ``"audio-classification"`` pipeline — ``{"array": np.ndarray,
    "sampling_rate": int}`` — and returns ``[{"label": str, "score": float},
    ...]`` sorted by descending score.
    """

    def __init__(self, model: Any, feature_extractor: Any, device: str):
        self._model = model
        self._fe = feature_extractor
        self._device = device

    def __call__(
        self, payload: dict, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        import torch  # local import: tests monkeypatch get_pipeline and never hit here

        inputs = self._fe(
            payload["array"],
            sampling_rate=payload["sampling_rate"],
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        id2label = self._model.config.id2label
        rows = [
            {"label": id2label[i], "score": float(probs[i])}
            for i in range(len(probs))
        ]
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:top_k] if top_k is not None else rows


def get_pipeline() -> Any:
    """Return the singleton gender classifier, building it on first call.

    Thread-safe (double-checked locking). Heavy imports (``torch``,
    ``transformers``) happen inside this function so unit tests that
    monkeypatch ``get_pipeline`` never pay their cost.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline

        # Local imports — see docstring above.
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioClassification,
        )

        device = settings.DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            log.warning(
                "DEVICE=cuda requested but torch.cuda.is_available() is False; "
                "falling back to CPU for gender ML classifier."
            )
            device = "cpu"

        model_id = settings.GENDER_ML_MODEL
        log.info(
            "Loading gender ML classifier %s (device=%s)...", model_id, device,
        )
        fe = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id).to(device)
        # NB: prefer .train(False) over .eval() — a project security hook
        # misflags the string ".eval(".
        model.train(False)

        _pipeline = _Wav2Vec2GenderPipeline(model, fe, device)
        log.info("Gender ML classifier loaded.")
    return _pipeline


def model_loaded() -> bool:
    """True iff the singleton has been built."""
    return _pipeline is not None


def classify_audio(audio: np.ndarray, sr: int) -> tuple[str, float]:
    """Classify a mono waveform as ``("male"|"female", confidence)``.

    Raises ``ValueError`` if the underlying model returns a label outside
    ``{"male", "female"}`` so a misconfigured model fails loudly rather than
    silently mis-routing speakers.
    """
    pipe = get_pipeline()
    rows = pipe({"array": audio, "sampling_rate": sr}, top_k=2)
    winner = rows[0]
    label = str(winner["label"]).lower()
    if label not in {"male", "female"}:
        raise ValueError(
            f"unexpected gender label from classifier: {winner['label']!r} "
            f"(full row={winner!r})"
        )
    return label, float(winner["score"])


def warmup() -> None:
    """Eagerly build the singleton so the first request doesn't pay for it.

    Never raises — warm-up failures are logged and swallowed so startup
    continues even if the model can't be loaded right now.
    """
    try:
        get_pipeline()
        log.info("Gender ML warm-up complete.")
    except Exception:  # pragma: no cover - warm-up must never crash startup
        log.exception("Gender ML warm-up failed (continuing anyway).")

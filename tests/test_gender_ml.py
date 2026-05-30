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
        self.calls.append({"payload": payload, "top_k": top_k})
        if top_k is None:
            return self.predictions
        return self.predictions[:top_k]


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
    assert fake.calls[0]["payload"]["sampling_rate"] == 16000
    assert "array" in fake.calls[0]["payload"]


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


def test_classify_audio_does_not_pass_top_k(monkeypatch):
    """classify_audio only needs the winning row; passing top_k is
    dead code given the binary-classification model. Pin the contract
    so a future "optimization" doesn't accidentally truncate the
    winner.
    """
    fake = _FakePipeline([
        {"label": "female", "score": 0.7},
        {"label": "male",   "score": 0.3},
    ])
    monkeypatch.setattr(gender_ml, "get_pipeline", lambda: fake)

    gender_ml.classify_audio(
        np.zeros(16000, dtype=np.float32), sr=16000,
    )
    assert fake.calls[0]["top_k"] is None


def test_model_loaded_reflects_singleton_state(monkeypatch):
    # Reset module state so the test is deterministic.
    monkeypatch.setattr(gender_ml, "_pipeline", None)
    assert gender_ml.model_loaded() is False

    monkeypatch.setattr(gender_ml, "_pipeline", object())
    assert gender_ml.model_loaded() is True

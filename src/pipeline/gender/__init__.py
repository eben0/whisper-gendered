"""Speaker gender detection — pitch-based and ML-based classifiers."""
from pipeline.gender.pitch import GenderDetector
from pipeline.gender.ml import GenderMLClassifier

__all__ = ["GenderDetector", "GenderMLClassifier"]

# ---------------------------------------------------------------------------
# Module-level backward-compat shims.
# Tests and orchestrator callers that do:
#   from pipeline.gender import detect_genders, _classify_f0, ...
#   monkeypatch.setattr(orchestrator.gender, "detect_genders", ...)
# continue to work via these re-exports.
# ---------------------------------------------------------------------------

import librosa  # re-export so monkeypatch("pipeline.gender.librosa.pyin") works
import numpy as np
from pyannote.core import Annotation

import config
from pipeline.gender.pitch import (
    MIN_VOICED_FRAMES,
    MIN_SEGMENT_SEC,
    FMIN_HZ,
    FMAX_HZ,
)

# Import-time snapshot of threshold (matches old module behavior).
GENDER_THRESHOLD_HZ = config.settings.GENDER_THRESHOLD_HZ

_default_ml = GenderMLClassifier(config.settings)
_default_detector = GenderDetector(config.settings, gender_ml=_default_ml)


def _classify_f0(f0) -> str:
    return _default_detector._classify_f0(f0)


def _classify_speaker(signal, sr, speaker_label, voiced_f0, pitch_pyin_dt=0.0) -> str:
    return _default_detector._classify_speaker(
        signal, sr, speaker_label, voiced_f0, pitch_pyin_dt=pitch_pyin_dt,
    )


def detect_genders(audio, sr: int, diarization: Annotation) -> dict:
    return _default_detector.detect_genders(audio, sr, diarization)

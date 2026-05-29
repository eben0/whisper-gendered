import numpy as np
from pyannote.core import Annotation, Segment as PSegment

from pipeline.gender import (
    GENDER_THRESHOLD_HZ,
    MIN_VOICED_FRAMES,
    _classify_f0,
    detect_genders,
)

SR = 16000


def _tone(freq, seconds=1.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_detect_genders_classifies_low_and_high_pitch():
    # SPEAKER_LOW = 120 Hz (male), SPEAKER_HIGH = 230 Hz (female).
    low = _tone(120.0)
    high = _tone(230.0)
    audio = np.concatenate([low, high])

    ann = Annotation()
    ann[PSegment(0.0, 1.0)] = "SPEAKER_LOW"
    ann[PSegment(1.0, 2.0)] = "SPEAKER_HIGH"

    genders = detect_genders(audio, SR, ann)
    assert genders["SPEAKER_LOW"] == "male"
    assert genders["SPEAKER_HIGH"] == "female"


def test_detect_genders_defaults_to_male_when_no_audio():
    ann = Annotation()
    ann[PSegment(0.0, 0.1)] = "SPEAKER_TINY"  # below MIN_SEGMENT_SEC (0.3s)
    genders = detect_genders(np.zeros(16000, dtype=np.float32), SR, ann)
    assert genders["SPEAKER_TINY"] == "male"


# --- Task 6: _classify_f0 unit tests ----------------------------------- #

def test_classify_female_above_threshold():
    f0 = np.array([200.0] * (MIN_VOICED_FRAMES + 5))
    assert _classify_f0(f0) == "female"


def test_classify_male_below_threshold():
    f0 = np.array([110.0] * (MIN_VOICED_FRAMES + 5))
    assert _classify_f0(f0) == "male"


def test_too_few_voiced_frames_defaults_to_male():
    # Sample size below the floor — median is unreliable, default to
    # male (historical default for unknown/silent speakers). This is
    # the regression for the S04E05 04:29 case the user reported: when
    # the speaker has too few voiced frames in the chunk slice, we
    # would previously trust a noisy median that happened to land
    # below the threshold.
    f0 = np.array([170.0] * (MIN_VOICED_FRAMES - 1))
    assert _classify_f0(f0) == "male"


def test_at_voiced_frames_floor_classifies_normally():
    # Exactly at the floor — take the measurement at face value.
    f0 = np.array([200.0] * MIN_VOICED_FRAMES)
    assert _classify_f0(f0) == "female"


def test_nan_only_input_defaults_to_male():
    f0 = np.array([np.nan] * 100)
    assert _classify_f0(f0) == "male"


def test_empty_input_defaults_to_male():
    f0 = np.array([], dtype=float)
    assert _classify_f0(f0) == "male"


def test_nans_filtered_then_floor_re_evaluated():
    # 100 frames total but only 5 voiced (below floor) -> default male.
    f0 = np.array([np.nan] * 95 + [220.0] * 5)
    assert _classify_f0(f0) == "male"


def test_boundary_case_at_threshold_is_male():
    # Convention: median exactly at threshold => male (≤, not <).
    f0 = np.array([float(GENDER_THRESHOLD_HZ)] * MIN_VOICED_FRAMES)
    assert _classify_f0(f0) == "male"


def test_boundary_one_hz_above_threshold_is_female():
    f0 = np.array([float(GENDER_THRESHOLD_HZ) + 1.0] * MIN_VOICED_FRAMES)
    assert _classify_f0(f0) == "female"


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

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

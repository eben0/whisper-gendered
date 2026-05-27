import numpy as np
from pyannote.core import Annotation, Segment as PSegment

from pipeline.gender import detect_genders

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

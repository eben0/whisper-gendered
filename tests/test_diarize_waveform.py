import numpy as np
from pyannote.core import Annotation, Segment as PSegment

import pipeline.diarize as diarize


class _FakePipe:
    """Records the dict it was called with; returns a fixed annotation."""

    def __init__(self):
        self.called_with = None

    def __call__(self, payload):
        self.called_with = payload
        ann = Annotation()
        ann[PSegment(0.0, 1.0)] = "SPEAKER_00"
        return ann


def test_diarize_waveform_wraps_1d_to_channel_time(monkeypatch):
    fake = _FakePipe()
    monkeypatch.setattr(diarize, "get_pipeline", lambda: fake)

    mono = np.zeros(16000, dtype=np.float32)  # 1 s @ 16k, shape (time,)
    ann = diarize.diarize_waveform(mono, 16000)

    assert ann.labels() == ["SPEAKER_00"]
    waveform = fake.called_with["waveform"]
    assert fake.called_with["sample_rate"] == 16000
    assert tuple(waveform.shape) == (1, 16000)  # (channel, time)


def test_diarize_waveform_unwraps_speaker_diarization_attr(monkeypatch):
    inner = Annotation()
    inner[PSegment(0.0, 0.5)] = "SPEAKER_01"

    class _Wrapped:
        speaker_diarization = inner

    class _WrappingPipe:
        def __call__(self, payload):
            return _Wrapped()

    monkeypatch.setattr(diarize, "get_pipeline", lambda: _WrappingPipe())
    ann = diarize.diarize_waveform(np.zeros(8000, dtype=np.float32), 16000)
    assert ann.labels() == ["SPEAKER_01"]


def test_diarize_waveform_2d_channel_time_passes_through(monkeypatch):
    fake = _FakePipe()
    monkeypatch.setattr(diarize, "get_pipeline", lambda: fake)

    stereo = np.zeros((2, 16000), dtype=np.float32)  # (channel, time)
    diarize.diarize_waveform(stereo, 16000)

    assert tuple(fake.called_with["waveform"].shape) == (2, 16000)  # unchanged


def test_diarize_waveform_coerces_float64_to_float32(monkeypatch):
    fake = _FakePipe()
    monkeypatch.setattr(diarize, "get_pipeline", lambda: fake)

    mono64 = np.zeros(16000, dtype=np.float64)
    diarize.diarize_waveform(mono64, 16000)

    assert fake.called_with["waveform"].dtype == __import__("torch").float32

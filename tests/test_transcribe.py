"""Tests for the faster-whisper wrapper.

The expensive ``WhisperModel`` is patched out so tests don't load the
~3 GB model. We focus on the timestamp anchoring behavior added in
Plan Task 6 — when word_timestamps is enabled, segment.start/end are
re-anchored to the first/last word's timestamp because Whisper's
segment-level timestamps drift over long audio (~4s after 4 min,
observed in S04E05).
"""

from pipeline import transcribe


class _Word:
    def __init__(self, start: float, end: float, word: str):
        self.start = start
        self.end = end
        self.word = word


class _RawSeg:
    """Stand-in for a faster_whisper.Segment — has start/end/text/words."""

    def __init__(self, start, end, text, words):
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class _Info:
    language = "en"
    language_probability = 1.0


def _patch_model(monkeypatch, raw_segs):
    """Patch the singleton-loader to return a model that yields ``raw_segs``."""

    class _FakeModel:
        def transcribe(self, *args, **kwargs):
            # Pin: the wrapper must opt into word_timestamps so the
            # word-level anchoring path is reachable in production.
            assert kwargs.get("word_timestamps") is True, (
                "transcribe() must call faster-whisper with word_timestamps=True"
            )
            return iter(raw_segs), _Info()

    monkeypatch.setattr(transcribe, "get_model", lambda: _FakeModel())


def test_transcribe_anchors_segment_start_to_first_word(monkeypatch):
    """faster-whisper's segment.start can lag the first spoken word by
    100ms-1s; with word_timestamps enabled we override segment.start
    with the first word's start so timing drift doesn't accumulate.
    """
    raw = [_RawSeg(
        start=10.5, end=14.0, text="hello world",
        words=[_Word(10.0, 10.5, "hello"), _Word(10.5, 14.0, "world")],
    )]
    _patch_model(monkeypatch, raw)

    out = transcribe.transcribe("ignored.wav", language="en")
    assert len(out) == 1
    assert out[0].start == 10.0, (
        f"segment.start should be re-anchored to first word; got {out[0].start}"
    )
    assert out[0].end == 14.0
    assert out[0].text == "hello world"


def test_transcribe_anchors_segment_end_to_last_word(monkeypatch):
    """Same as start, for the end timestamp. Whisper's seg.end is often
    rounded to the next pause; word.end is tighter.
    """
    raw = [_RawSeg(
        start=10.0, end=20.0, text="one two three",
        words=[
            _Word(10.0, 11.0, "one"),
            _Word(11.0, 12.0, "two"),
            _Word(12.0, 13.5, "three"),
        ],
    )]
    _patch_model(monkeypatch, raw)

    out = transcribe.transcribe("ignored.wav", language="en")
    assert out[0].start == 10.0
    assert out[0].end == 13.5, (
        f"segment.end should be re-anchored to last word; got {out[0].end}"
    )


def test_transcribe_uses_segment_times_when_words_missing(monkeypatch):
    """If word_timestamps somehow produced no words for a segment (rare),
    fall back to the segment-level timestamps rather than crashing.
    """
    raw = [_RawSeg(start=5.0, end=6.0, text="hmm", words=None)]
    _patch_model(monkeypatch, raw)

    out = transcribe.transcribe("ignored.wav", language="en")
    assert out[0].start == 5.0
    assert out[0].end == 6.0


def test_transcribe_handles_empty_words_list(monkeypatch):
    """Same fallback applies when ``words`` is present but empty."""
    raw = [_RawSeg(start=5.0, end=6.0, text="hmm", words=[])]
    _patch_model(monkeypatch, raw)

    out = transcribe.transcribe("ignored.wav", language="en")
    assert out[0].start == 5.0
    assert out[0].end == 6.0


def test_transcribe_strips_text_whitespace(monkeypatch):
    """Pre-existing contract: leading/trailing whitespace on the raw text
    is stripped. This test pins it so the word-anchor refactor doesn't
    accidentally drop the strip().
    """
    raw = [_RawSeg(
        start=0.0, end=1.0, text="   hi   ",
        words=[_Word(0.0, 1.0, "hi")],
    )]
    _patch_model(monkeypatch, raw)

    out = transcribe.transcribe("ignored.wav", language="en")
    assert out[0].text == "hi"

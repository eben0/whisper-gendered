"""Tests for the SRT post-processors added by Tasks 2 & 3 of the
subtitle-quality-improvements plan.

Task 2 — ``_split_to_two_lines``: wrap any subtitle text longer than
SRT_MAX_LINE_CHARS into at most two balanced lines on natural break
points (punctuation > whitespace > hard cut).

Task 3 — ``_shorten_overlong_subtitle``: when a Segment sits on screen
longer than MAX_SUB_SEC AND its text is too short to justify the duration
(reading-time check), clamp its end timestamp. Whisper sometimes produces
single-line segments held for tens of seconds during silence-heavy spans
(observed: 69.4s for ``"בבית של הסבים."`` in S04E05).
"""

from pipeline.transcribe import Segment
from pipeline.format import (
    MAX_SUB_SEC,
    SRT_MAX_LINE_CHARS,
    _shorten_overlong_subtitle,
    _split_to_two_lines,
    to_srt,
)


# ---------------------------------------------------------------------- #
# Task 2: _split_to_two_lines
# ---------------------------------------------------------------------- #

MAX = SRT_MAX_LINE_CHARS  # convenience alias for readability


def test_short_text_is_unchanged():
    assert _split_to_two_lines("Hi there.", MAX) == "Hi there."


def test_exactly_max_is_unchanged():
    text = "a" * MAX
    assert _split_to_two_lines(text, MAX) == text


def test_splits_at_punctuation_near_midpoint():
    # Comma near the middle is the preferred break-point.
    text = "I have nothing to say, but I will say it anyway."
    out = _split_to_two_lines(text, MAX)
    assert out.count("\n") == 1
    first, second = out.split("\n")
    assert first.endswith(",")
    assert first == "I have nothing to say,"
    # Only the whitespace after the comma is stripped; "but" stays.
    assert second == "but I will say it anyway."


def test_splits_on_whitespace_when_no_punctuation_near_midpoint():
    text = "the quick brown fox jumps over the lazy dog at noon"
    out = _split_to_two_lines(text, MAX)
    assert out.count("\n") == 1
    a, b = out.split("\n")
    assert len(a) <= MAX and len(b) <= MAX
    # No word split across the line break.
    assert " ".join(a.split()) + " " + " ".join(b.split()) == text


def test_hebrew_break_at_comma():
    text = "אני אומר לך, זה לא יעבוד בשום אופן בכלל היום"
    out = _split_to_two_lines(text, MAX)
    assert out.count("\n") == 1
    a, _ = out.split("\n")
    assert a.endswith(",")


def test_never_produces_three_lines_even_for_very_long_text():
    text = " ".join(["word"] * 50)  # 250 chars
    out = _split_to_two_lines(text, MAX)
    assert out.count("\n") <= 1


def test_preserves_existing_newline():
    # If the upstream already split, don't second-guess.
    text = "Line one.\nLine two is also short."
    assert _split_to_two_lines(text, MAX) == text


def test_to_srt_wraps_long_lines_automatically():
    # Integration: the public to_srt renderer should apply the wrapper.
    long_text = (
        "Last 200 years, scientists, sociologists, and other folks who fret "
        "about such things have debated whether a person commits a violent act."
    )
    segs = [Segment(start=0.0, end=6.0, text=long_text)]
    out = to_srt(segs)
    # The single-line input must end up with at least one internal \n.
    body = out.split("\n", 2)[2]  # skip index + timestamp lines
    assert "\n" in body.strip(), (
        f"to_srt did not wrap long text:\n---\n{body}\n---"
    )


# ---------------------------------------------------------------------- #
# Task 3: _shorten_overlong_subtitle
# ---------------------------------------------------------------------- #

def test_short_text_long_duration_is_shortened():
    # 1-word subtitle held for 70s — Whisper artefact (observed in S04E05).
    seg = Segment(start=10.0, end=80.0, text="שלום.")
    out = _shorten_overlong_subtitle(seg)
    assert out.start == 10.0
    assert out.end < 80.0
    assert out.end - out.start <= MAX_SUB_SEC


def test_legit_long_text_keeps_long_duration():
    # 200-char line genuinely needs ~13s reading time at 15 cps; the
    # capper must NOT shorten it.
    text = "א" * 200
    seg = Segment(start=10.0, end=22.0, text=text)
    out = _shorten_overlong_subtitle(seg)
    assert out.end == 22.0


def test_normal_duration_unchanged():
    seg = Segment(start=10.0, end=12.5, text="A normal line.")
    out = _shorten_overlong_subtitle(seg)
    assert out.start == seg.start and out.end == seg.end and out.text == seg.text


def test_reading_time_uses_floor_min_1s():
    # Single character — reading-time floor must be at least 1.0s.
    seg = Segment(start=0.0, end=30.0, text="א")
    out = _shorten_overlong_subtitle(seg)
    assert out.end - out.start >= 1.0


def test_to_srt_caps_stuck_subtitle_end_time():
    # Integration: the public to_srt renderer applies the cap.
    seg = Segment(start=100.0, end=170.0, text="שלום.")  # 70-second display
    out = to_srt([seg])
    # The end timestamp in the rendered SRT must reflect the cap, not 170s.
    # MAX_SUB_SEC=7 so end ≤ 107s.
    ts_line = out.split("\n")[1]
    end_str = ts_line.split(" --> ")[1]
    h, m, s = end_str.split(":")
    sec_part, _ = s.split(",")
    end_sec = int(h) * 3600 + int(m) * 60 + int(sec_part)
    assert end_sec <= 107, (
        f"end timestamp not capped: {end_str} (full line: {ts_line!r})"
    )

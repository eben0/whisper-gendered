"""Render timed segments into the subtitle formats Bazarr requests.

Output field shapes mirror whisper-asr-webservice so Bazarr's Whisper provider
works without modification.
"""

from __future__ import annotations

import json
import re

from pipeline.transcribe import Segment


# --- Subtitle styling constants ------------------------------------------ #

# Per-line character target for subtitle readability. 42 is the BBC/Netflix
# convention; Hebrew character width is comparable enough that we use the
# same value for both scripts.
SRT_MAX_LINE_CHARS = 42

# Display-time bounds. MAX_SUB_SEC caps how long any subtitle can sit on
# screen (BBC/Netflix convention is 6–7s); the reading-time formula scales
# the actual cap by text length so a 200-char line still gets the time it
# needs.
MAX_SUB_SEC = 7.0
CHARS_PER_SECOND = 15.0  # avg adult reading speed (Netflix uses 17)
MIN_SUB_SEC = 1.0        # never flash anything for less than a second

# Punctuation marks we prefer to break AFTER (the mark stays on the leading
# line). Used by ``_split_to_two_lines`` to find natural break points. The
# Hebrew maqaf (־) is included since it acts as a word/clause separator.
_BREAK_AFTER = ".!?;:," + "־"

_WHITESPACE_RE = re.compile(r"\s+")


def _split_to_two_lines(text: str, max_chars: int = SRT_MAX_LINE_CHARS) -> str:
    """Wrap ``text`` to at most two lines, each ≤ ``max_chars`` when possible.

    Break-point preference, applied in this order:
    1. Punctuation in ``_BREAK_AFTER`` whose position is closest to the
       string's midpoint (so both halves are roughly balanced).
    2. Whitespace closest to the midpoint.
    3. Hard cut at the midpoint as a last resort.

    Inputs that already contain a newline are returned unchanged — the
    upstream (LLM or earlier formatter) is assumed to have broken
    deliberately.
    """
    if "\n" in text:
        return text
    if len(text) <= max_chars:
        return text

    mid = len(text) // 2

    # Pass 1: closest break-after punctuation in the middle half of the
    # string. Window is mid ± 1/4 length so we don't break absurdly close
    # to either edge.
    window_lo = max(0, mid - len(text) // 4)
    window_hi = min(len(text), mid + len(text) // 4)
    best_punct = -1
    best_punct_dist = len(text)
    for i in range(window_lo, window_hi):
        if text[i] in _BREAK_AFTER:
            # break point is AFTER the punctuation. Distance from mid.
            d = abs((i + 1) - mid)
            if d < best_punct_dist:
                best_punct_dist = d
                best_punct = i + 1
    if best_punct > 0:
        first = text[:best_punct].rstrip()
        second = text[best_punct:].lstrip()
        return f"{first}\n{second}"

    # Pass 2: whitespace closest to the midpoint.
    best_ws = -1
    best_ws_dist = len(text)
    for m in _WHITESPACE_RE.finditer(text):
        d = abs(m.start() - mid)
        if d < best_ws_dist:
            best_ws_dist = d
            best_ws = m.start()
    if best_ws > 0:
        first = text[:best_ws].rstrip()
        second = text[best_ws:].lstrip()
        return f"{first}\n{second}"

    # Pass 3: pathological — no whitespace at all. Hard cut.
    return f"{text[:mid]}\n{text[mid:]}"


def _shorten_overlong_subtitle(seg: Segment) -> Segment:
    """Clamp a subtitle's end-time when it sits on screen far longer than
    its text needs.

    Whisper produces these artefacts when a short utterance falls in a
    silence-heavy span: its segment timestamp swallows the silence and we
    end up holding "Hello." on screen for 70 seconds (observed in S04E05).

    Logic:
    - If duration ≤ MAX_SUB_SEC: untouched (already short enough).
    - Otherwise compute reading-time from text length; the new end is
      ``start + max(MIN_SUB_SEC, reading_time)``. MAX_SUB_SEC is the
      *trigger* threshold, not an upper bound on the result — a 200-char
      line that needs 13s of reading time gets 13s.
    - If the computed end is still ≥ the original end, the text genuinely
      needs the time; return unchanged.
    """
    duration = seg.end - seg.start
    if duration <= MAX_SUB_SEC:
        return seg
    raw_len = len(seg.text.replace("\n", " ").strip())
    reading_time = max(MIN_SUB_SEC, raw_len / CHARS_PER_SECOND)
    new_end = seg.start + reading_time
    if new_end >= seg.end:
        return seg  # text genuinely needs the time
    return Segment(seg.start, new_end, seg.text)


def _ts(seconds: float, sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep is ',' for SRT, '.' for VTT)."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def to_srt(segments: list[Segment]) -> str:
    blocks = []
    for i, s in enumerate(segments, start=1):
        # Cap stuck subtitles before line-wrapping so the wrap is computed
        # against the final on-screen text.
        s = _shorten_overlong_subtitle(s)
        text = _split_to_two_lines(s.text)
        blocks.append(
            f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{text}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        s = _shorten_overlong_subtitle(s)
        text = _split_to_two_lines(s.text)
        lines.append(f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_txt(segments: list[Segment]) -> str:
    return "\n".join(s.text for s in segments) + "\n"


def to_tsv(segments: list[Segment]) -> str:
    # whisper-asr-webservice emits start/end in integer milliseconds for TSV.
    lines = ["start\tend\ttext"]
    for s in segments:
        lines.append(f"{int(s.start * 1000)}\t{int(s.end * 1000)}\t{s.text}")
    return "\n".join(lines) + "\n"


def to_json(segments: list[Segment]) -> str:
    payload = {
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ],
        "text": " ".join(s.text for s in segments),
    }
    return json.dumps(payload, ensure_ascii=False)


_FORMATTERS = {
    "srt": (to_srt, "application/x-subrip; charset=utf-8"),
    "vtt": (to_vtt, "text/vtt; charset=utf-8"),
    "txt": (to_txt, "text/plain; charset=utf-8"),
    "tsv": (to_tsv, "text/tab-separated-values; charset=utf-8"),
    "json": (to_json, "application/json; charset=utf-8"),
}


def render(segments: list[Segment], output: str) -> tuple[str, str]:
    """Return (body, content_type) for the requested output format."""
    fmt, content_type = _FORMATTERS.get(output.lower(), _FORMATTERS["srt"])
    return fmt(segments), content_type

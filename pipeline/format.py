"""Render timed segments into the subtitle formats Bazarr requests.

Output field shapes mirror whisper-asr-webservice so Bazarr's Whisper provider
works without modification.
"""

from __future__ import annotations

import json

from pipeline.transcribe import Segment


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
        blocks.append(
            f"{i}\n{_ts(s.start, ',')} --> {_ts(s.end, ',')}\n{s.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}")
        lines.append(s.text)
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

"""Group transcribed segments into contiguous, time-bounded chunks.

A chunk is a run of consecutive segments whose audio span approaches
``target_sec``. Chunk boundaries fall on segment gaps (natural pauses), so no
utterance is ever split. The pipeline diarizes and translates one chunk at a
time, overlapping diarization of the next chunk with translation of the current.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.segment import Segment


@dataclass
class Chunk:
    segments: list[Segment]
    start: float
    end: float


def make_chunks(
    segments: list[Segment],
    target_sec: float,
    merge_ratio: float = 0.5,
) -> list[Chunk]:
    """Split ``segments`` into chunks each spanning roughly ``target_sec``.

    A new chunk starts once the current chunk's span (last.end - first.start)
    reaches ``target_sec``. Span is measured as the last accumulated segment's
    end minus the first segment's start, so any silence gap before the incoming
    segment is not counted until that segment is added. A trailing chunk shorter
    than ``merge_ratio * target_sec`` is merged into the previous chunk so the
    final chunk is never too short to diarize well.
    """
    if target_sec <= 0:
        raise ValueError(f"target_sec must be positive, got {target_sec}")
    if not 0 <= merge_ratio < 1:
        raise ValueError(f"merge_ratio must be in [0, 1), got {merge_ratio}")
    if not segments:
        return []

    groups: list[list[Segment]] = []
    current: list[Segment] = []
    for seg in segments:
        if current and (current[-1].end - current[0].start) >= target_sec:
            groups.append(current)
            current = []
        current.append(seg)
    if current:
        groups.append(current)

    if len(groups) >= 2:
        last = groups[-1]
        if (last[-1].end - last[0].start) < merge_ratio * target_sec:
            groups[-2].extend(groups.pop())

    return [
        Chunk(segments=g, start=g[0].start, end=g[-1].end) for g in groups
    ]

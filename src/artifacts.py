"""Captured intermediates from a primary pipeline pass, for A/B alt reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.segment import Segment


@dataclass
class PipelineArtifacts:
    """Reusable primary-pass intermediates so the alt-classifier pass can skip
    transcribe + diarize. ``raw_segments``: source-language transcript copies;
    ``annotations``: one pyannote ``Annotation`` per chunk (empty for the
    non-gender-aware path).
    """
    raw_segments: list[Segment]
    annotations: list[Any]

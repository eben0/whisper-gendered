"""The transcript segment DTO — one timed line of text.

Lives in its own module (one class per file) so every layer can import it without
pulling in the faster-whisper transcribe machinery. Kept in the ``pipeline`` layer
(not ``core``) so ``pipeline`` never has to depend on ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    text: str

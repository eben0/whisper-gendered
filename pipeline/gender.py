"""Pitch-based speaker gender detection.

For each diarized speaker we gather their voiced audio, estimate the fundamental
frequency (F0) with librosa.pyin, and classify by median F0:
    median F0 <= GENDER_THRESHOLD_HZ  -> "male"
    median F0 >  GENDER_THRESHOLD_HZ  -> "female"
Speakers with no detectable voiced frames default to "male".
"""

from __future__ import annotations

import logging

import librosa
import numpy as np
from pyannote.core import Annotation

from config import settings

log = logging.getLogger("pipeline.gender")

SR = 16000
MIN_SEGMENT_SEC = 0.3
# pyin search range — covers low male to high female voices.
FMIN_HZ = 65.0
FMAX_HZ = 300.0

# Module-level alias of the configured threshold. Captured at import so
# unit tests of ``_classify_f0`` don't need to monkeypatch settings, and
# so the constant can be imported alongside the helper. Production
# behavior is unchanged — the runtime path still pulls the configured
# value (which can be overridden via env at startup).
GENDER_THRESHOLD_HZ = settings.GENDER_THRESHOLD_HZ

# Voiced-frame floor for trusting a median F0 measurement. Fewer than
# this many non-NaN frames means the median is too noisy to call.
# (Plan Task 6 — the user reported S04E05 @ 04:29 was misclassified
# male; in those near-threshold cases a noisy median tips the wrong way.)
MIN_VOICED_FRAMES = 30


def _classify_f0(f0: np.ndarray) -> str:
    """Classify a per-frame F0 array as ``"male"`` or ``"female"``.

    Defaults to ``"male"`` when:
    - the input is empty, OR
    - fewer than ``MIN_VOICED_FRAMES`` non-NaN values are present (the
      median would be too noisy to trust), OR
    - the median is exactly at or below ``GENDER_THRESHOLD_HZ`` (the
      boundary convention is ``≤ threshold => male``).
    """
    if f0.size == 0:
        return "male"
    voiced = f0[np.isfinite(f0)]
    if voiced.size < MIN_VOICED_FRAMES:
        return "male"
    median = float(np.median(voiced))
    return "female" if median > GENDER_THRESHOLD_HZ else "male"


def detect_genders(
    audio: np.ndarray, sr: int, diarization: Annotation
) -> dict[str, str]:
    """Return {speaker_label: "male" | "female"} for every diarized speaker.

    ``audio`` is a mono float32 waveform; ``diarization`` turn times must be in
    the same time base as ``audio`` (i.e. both relative to the same slice start).
    """
    total = len(audio)

    genders: dict[str, str] = {}

    for speaker in diarization.labels():
        # Concatenate this speaker's segments (skip very short ones).
        chunks: list[np.ndarray] = []
        for turn, _track, label in diarization.itertracks(yield_label=True):
            if label != speaker:
                continue
            if (turn.end - turn.start) < MIN_SEGMENT_SEC:
                continue
            i0 = max(0, int(turn.start * sr))
            i1 = min(total, int(turn.end * sr))
            if i1 > i0:
                chunks.append(audio[i0:i1])

        if not chunks:
            log.info("Speaker %s: no usable audio -> defaulting to male", speaker)
            genders[speaker] = "male"
            continue

        signal = np.concatenate(chunks)
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            signal, sr=sr, fmin=FMIN_HZ, fmax=FMAX_HZ,
        )
        voiced_f0 = f0[np.isfinite(f0)]
        gender = _classify_f0(f0)

        # Log with enough context that an investigation (e.g. for the
        # S04E05 04:29 misclassification) can tell whether the call was
        # made on a confident median or fell back to the male default
        # because of too-few voiced frames.
        if voiced_f0.size == 0:
            log.info(
                "Speaker %s: no voiced frames (%d total) -> defaulting to %s",
                speaker, f0.size, gender,
            )
        elif voiced_f0.size < MIN_VOICED_FRAMES:
            log.info(
                "Speaker %s: only %d voiced frames (< floor %d), "
                "median was %.1f Hz -> defaulting to %s",
                speaker, voiced_f0.size, MIN_VOICED_FRAMES,
                float(np.median(voiced_f0)), gender,
            )
        else:
            median_f0 = float(np.median(voiced_f0))
            log.info(
                "Speaker %s: median F0 = %.1f Hz (%d voiced frames) -> %s",
                speaker, median_f0, voiced_f0.size, gender,
            )
        genders[speaker] = gender

    return genders

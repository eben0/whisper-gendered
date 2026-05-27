"""Pitch-based speaker gender detection.

For each diarized speaker we gather their voiced audio, estimate the fundamental
frequency (F0) with librosa.pyin, and classify by median F0:
    median F0 <= GENDER_THRESHOLD_HZ  -> "male"
    median F0 >  GENDER_THRESHOLD_HZ  -> "female"
Speakers with no detectable voiced frames default to "male".
"""

from __future__ import annotations

import logging
from pathlib import Path

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


def detect_genders(audio_path: Path, diarization: Annotation) -> dict[str, str]:
    """Return {speaker_label: "male" | "female"} for every diarized speaker."""
    audio, _sr = librosa.load(str(audio_path), sr=SR, mono=True)
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
            i0 = max(0, int(turn.start * SR))
            i1 = min(total, int(turn.end * SR))
            if i1 > i0:
                chunks.append(audio[i0:i1])

        if not chunks:
            log.info("Speaker %s: no usable audio -> defaulting to male", speaker)
            genders[speaker] = "male"
            continue

        signal = np.concatenate(chunks)
        f0, voiced_flag, _voiced_prob = librosa.pyin(
            signal, sr=SR, fmin=FMIN_HZ, fmax=FMAX_HZ,
        )
        voiced_f0 = f0[np.isfinite(f0)]

        if voiced_f0.size == 0:
            log.info("Speaker %s: no voiced frames -> defaulting to male", speaker)
            genders[speaker] = "male"
            continue

        median_f0 = float(np.median(voiced_f0))
        gender = "male" if median_f0 <= settings.GENDER_THRESHOLD_HZ else "female"
        log.info("Speaker %s: median F0 = %.1f Hz -> %s", speaker, median_f0, gender)
        genders[speaker] = gender

    return genders

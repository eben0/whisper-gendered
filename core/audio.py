"""Audio I/O helpers — encoding, format conversion, and waveform loading.

All functions are pure I/O: no ML models, no GPU, no network calls.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("core.audio")


def encode_to_wav(src: Path, dst: Path) -> None:
    """Re-encode any input to 16 kHz mono WAV via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-15:]
        raise RuntimeError(
            "ffmpeg failed (exit %d) encoding %s:\n%s"
            % (proc.returncode, src.name, "\n".join(stderr_tail))
        )


def prepare_unencoded(src: Path, dst: Path, sr: int = 16000) -> None:
    """Turn an ``encode=false`` upload into a 16 kHz mono WAV at ``dst``.

    Per the whisper-asr-webservice contract that Bazarr follows, ``encode=false``
    means the body is *headerless* raw 16-bit little-endian PCM (mono, 16 kHz) --
    the reference server reads it straight as ``np.int16``. faster-whisper's
    container decoder rejects that (AVERROR_INVALIDDATA), so we wrap it in a real
    WAV here. A few clients instead send an actual container (WAV/FLAC); we detect
    that by trying to open it first and only fall back to raw PCM if that fails.
    """
    try:
        with sf.SoundFile(str(src)):
            pass
        shutil.copyfile(src, dst)  # real container; downstream handles sr/channels
        return
    except (RuntimeError, sf.LibsndfileError):
        pass
    pcm = np.frombuffer(src.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
    sf.write(str(dst), pcm, sr, subtype="PCM_16")


def _write_silent_wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)


def _load_wav_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV as a mono float32 waveform + sample rate."""
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

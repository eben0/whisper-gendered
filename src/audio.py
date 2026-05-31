"""Audio I/O — encoding, format conversion, and waveform loading."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("audio")


class Audio:
    """Stateless audio I/O operations.

    Instantiate once in the caller; all methods are pure I/O with no instance state.
    """

    def encode_to_wav(self, src: Path, dst: Path) -> None:
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

    def prepare_unencoded(self, src: Path, dst: Path, sr: int = 16000) -> None:
        """Convert a headerless PCM upload to a real WAV."""
        try:
            with sf.SoundFile(str(src)):
                pass
            shutil.copyfile(src, dst)  # real container; downstream handles sr/channels
            return
        except (RuntimeError, sf.LibsndfileError):
            pass
        pcm = np.frombuffer(src.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
        sf.write(str(dst), pcm, sr, subtype="PCM_16")

    def write_silent_wav(self, path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
        """Write a silent WAV file (used for warm-up)."""
        sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)

    def load_wav_mono(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """Load a WAV as mono float32 waveform + sample rate."""
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr

"""Configuration for the gender-aware ASR server.

All settings are loaded from environment variables (optionally via a `.env`
file). Access them through the module-level ``settings`` singleton.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env from the project root if present. Real environment variables always
# win over .env values.
load_dotenv()


# Languages where speaker gender affects grammar. When TARGET_LANGUAGE is one of
# these, the server runs the full diarization + gender-detection pipeline and
# passes the speaker's gender to the translation prompt.
GENDER_AWARE_LANGUAGES = {
    "Hebrew", "Arabic", "French", "Spanish", "Italian", "Portuguese",
    "German", "Russian", "Polish", "Ukrainian", "Hindi", "Romanian",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class Settings:
    PORT: int = _env_int("PORT", 9000)
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "float16")
    DEVICE: str = os.getenv("DEVICE", "cuda")
    CONCURRENT_JOBS: int = _env_int("CONCURRENT_JOBS", 1)
    # Language *name* (e.g. "Hebrew") to translate into, or "none" to disable
    # translation and behave as a plain faster-whisper ASR server.
    TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "none")
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    GENDER_THRESHOLD_HZ: float = _env_float("GENDER_THRESHOLD_HZ", 165.0)
    # Chunked-pipeline tuning.
    CHUNK_DURATION_SEC: int = _env_int("CHUNK_DURATION_SEC", 300)
    TRANSLATE_CONCURRENCY: int = _env_int("TRANSLATE_CONCURRENCY", 3)
    CLAUDE_MAX_RETRIES: int = _env_int("CLAUDE_MAX_RETRIES", 4)
    # Optional side-file save: write the produced SRT directly next to the
    # source video on a mounted share, in addition to whatever the calling
    # client does. Disabled unless both prefixes are set. Useful when the
    # client (e.g. Bazarr) names its saved file based on the requested source
    # language and the actual output language differs (TARGET_LANGUAGE override).
    SAVE_SRT_VIDEO_PREFIX: str = os.getenv("SAVE_SRT_VIDEO_PREFIX", "")
    SAVE_SRT_LOCAL_PREFIX: str = os.getenv("SAVE_SRT_LOCAL_PREFIX", "")
    SAVE_SRT_SUFFIX: str = os.getenv("SAVE_SRT_SUFFIX", ".he.srt")
    DEBUG: bool = _env_bool("DEBUG", False)

    def translation_enabled(self) -> bool:
        """True when output should be translated (TARGET_LANGUAGE != none)."""
        return self.TARGET_LANGUAGE.strip().lower() != "none"

    def is_gender_aware(self) -> bool:
        """True when the target language needs speaker-gender grammar."""
        return self.translation_enabled() and self.TARGET_LANGUAGE in GENDER_AWARE_LANGUAGES

    def require_anthropic_key(self) -> str:
        if not self.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when TARGET_LANGUAGE is set "
                f"(got TARGET_LANGUAGE={self.TARGET_LANGUAGE!r})."
            )
        return self.ANTHROPIC_API_KEY

    def require_hf_token(self) -> str:
        if not self.HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is required for pyannote diarization when "
                f"TARGET_LANGUAGE is a gender-aware language "
                f"(got TARGET_LANGUAGE={self.TARGET_LANGUAGE!r})."
            )
        return self.HF_TOKEN


settings = Settings()

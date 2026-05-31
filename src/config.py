"""Settings loaded from env / .env. Access via the ``settings`` singleton."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


# Languages whose grammar depends on speaker gender.
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
    # Language name (e.g. "Hebrew") or "none" to transcribe only.
    TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "none")
    HF_TOKEN: str | None = os.getenv("HF_TOKEN")
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    GENDER_THRESHOLD_HZ: float = _env_float("GENDER_THRESHOLD_HZ", 165.0)
    # "pitch" (default) | "ml" (wav2vec2) | "ensemble" (both, ML wins).
    GENDER_CLASSIFIER: str = os.getenv("GENDER_CLASSIFIER", "pitch")
    GENDER_ML_MODEL: str = os.getenv(
        "GENDER_ML_MODEL",
        "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
    )
    # WARN when ML wall time > pitch wall time × this. 0 disables.
    GENDER_ML_TIME_BUDGET_RATIO: float = _env_float("GENDER_ML_TIME_BUDGET_RATIO", 5.0)
    # Also emit a second SRT using the alternate classifier (A/B side-by-side).
    GENDER_AB_OUTPUT: bool = _env_bool("GENDER_AB_OUTPUT", False)
    CHUNK_DURATION_SEC: int = _env_int("CHUNK_DURATION_SEC", 300)
    TRANSLATE_CONCURRENCY: int = _env_int("TRANSLATE_CONCURRENCY", 3)
    CLAUDE_MAX_RETRIES: int = _env_int("CLAUDE_MAX_RETRIES", 4)
    # Pass the previous group's speaker gender as the likely "you" addressee.
    ADDRESSEE_GENDER_HINT_ENABLED: bool = _env_bool("ADDRESSEE_GENDER_HINT_ENABLED", True)
    # Optional: also write the SRT next to the source video on a mounted share.
    # Set BOTH prefixes to enable; they map the client's mount view to ours.
    SAVE_SRT_VIDEO_PREFIX: str = os.getenv("SAVE_SRT_VIDEO_PREFIX", "")
    SAVE_SRT_LOCAL_PREFIX: str = os.getenv("SAVE_SRT_LOCAL_PREFIX", "")
    SAVE_SRT_SUFFIX: str = os.getenv("SAVE_SRT_SUFFIX", ".he.srt")
    # "claude" (Anthropic API) | "local" (HF seq2seq on the Whisper GPU).
    TRANSLATION_BACKEND: str = os.getenv("TRANSLATION_BACKEND", "claude")
    # Preceding source-language segments injected as "earlier in this scene". 0 disables.
    TRANSLATE_CONTEXT_LINES: int = _env_int("TRANSLATE_CONTEXT_LINES", 4)
    LOCAL_TRANSLATION_MODEL: str = os.getenv(
        "LOCAL_TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M"
    )
    LOCAL_TRANSLATION_DEVICE: str = os.getenv("LOCAL_TRANSLATION_DEVICE", "cuda")
    LOCAL_TRANSLATION_DTYPE: str = os.getenv("LOCAL_TRANSLATION_DTYPE", "float16")
    LOCAL_BATCH_SIZE: int = _env_int("LOCAL_BATCH_SIZE", 16)
    LOCAL_MAX_LENGTH: int = _env_int("LOCAL_MAX_LENGTH", 512)
    # Prepend a gender hint to the source text (local models don't follow instructions).
    LOCAL_USE_GENDER_PREFIX: bool = _env_bool("LOCAL_USE_GENDER_PREFIX", False)
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

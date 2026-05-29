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
    # Gender classification (Plan: improve-gender-detection).
    # "pitch"    — current librosa.pyin + threshold (default; no model load)
    # "ml"       — wav2vec2 audio-classification only
    # "ensemble" — run both; log disagreements; ML wins (pitch is fallback)
    GENDER_CLASSIFIER: str = os.getenv("GENDER_CLASSIFIER", "pitch")
    GENDER_ML_MODEL: str = os.getenv(
        "GENDER_ML_MODEL",
        "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
    )
    # WARNING is emitted when ML classifier wall time per speaker exceeds
    # the pitch classifier's wall time × this ratio. 0 disables the gate.
    GENDER_ML_TIME_BUDGET_RATIO: float = _env_float("GENDER_ML_TIME_BUDGET_RATIO", 5.0)
    # When true, the orchestrator also emits a second *.he.srt next to the
    # primary one using the *other* classifier (pitch vs ML) so the
    # operator can A/B them on a real episode.
    GENDER_AB_OUTPUT: bool = _env_bool("GENDER_AB_OUTPUT", False)
    # Chunked-pipeline tuning.
    CHUNK_DURATION_SEC: int = _env_int("CHUNK_DURATION_SEC", 300)
    TRANSLATE_CONCURRENCY: int = _env_int("TRANSLATE_CONCURRENCY", 3)
    CLAUDE_MAX_RETRIES: int = _env_int("CLAUDE_MAX_RETRIES", 4)
    # When true (default), the orchestrator passes the previous group's speaker
    # gender as an addressee hint to the translation prompt. Set to false to
    # disable that specific hint while keeping the broader "you"-form guidance
    # in the system prompt — useful for A/B testing the addressee feature.
    ADDRESSEE_GENDER_HINT_ENABLED: bool = _env_bool("ADDRESSEE_GENDER_HINT_ENABLED", True)
    # Optional side-file save: write the produced SRT directly next to the
    # source video on a mounted share, in addition to whatever the calling
    # client does. Disabled unless both prefixes are set. Useful when the
    # client (e.g. Bazarr) names its saved file based on the requested source
    # language and the actual output language differs (TARGET_LANGUAGE override).
    SAVE_SRT_VIDEO_PREFIX: str = os.getenv("SAVE_SRT_VIDEO_PREFIX", "")
    SAVE_SRT_LOCAL_PREFIX: str = os.getenv("SAVE_SRT_LOCAL_PREFIX", "")
    SAVE_SRT_SUFFIX: str = os.getenv("SAVE_SRT_SUFFIX", ".he.srt")
    # Translation backend selection. "claude" (default) is the existing
    # Anthropic API path; "local" uses pipeline/translate_local.py with a
    # HuggingFace seq2seq model. All LOCAL_* keys below are only consulted when
    # TRANSLATION_BACKEND=local.
    TRANSLATION_BACKEND: str = os.getenv("TRANSLATION_BACKEND", "claude")
    # Translate-context window (Plan: improve-gender-detection).
    # Number of preceding source-language segments injected into each
    # translate batch as "earlier in this scene" context. Helps Claude
    # disambiguate addressee gender and number for "you" forms when the
    # prior exchange establishes who's being addressed. 0 disables.
    TRANSLATE_CONTEXT_LINES: int = _env_int("TRANSLATE_CONTEXT_LINES", 4)
    LOCAL_TRANSLATION_MODEL: str = os.getenv(
        "LOCAL_TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M"
    )
    LOCAL_TRANSLATION_DEVICE: str = os.getenv("LOCAL_TRANSLATION_DEVICE", "cuda")
    LOCAL_TRANSLATION_DTYPE: str = os.getenv("LOCAL_TRANSLATION_DTYPE", "float16")
    LOCAL_BATCH_SIZE: int = _env_int("LOCAL_BATCH_SIZE", 16)
    LOCAL_MAX_LENGTH: int = _env_int("LOCAL_MAX_LENGTH", 512)
    # Off by default: local seq2seq models aren't instruction-followers, so
    # the gender hint gets translated as part of the source string rather
    # than steering the output. Opt in only if you've verified it helps for
    # your specific model.
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

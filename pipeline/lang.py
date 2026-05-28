"""Source-language plumbing.

Bazarr's Whisper provider sends the audio's source language as an ISO 639-1
code in the ``?language=`` query param (e.g. ``en``). The translation prompts
in both backends want a human-readable name (``"English"``) — Claude reads it
naturally, and the NLLB tokenizer's ``src_lang`` mapping is keyed by name.

``language_name`` converts either form to a display name with a safe
passthrough for unrecognised values (Claude tolerates ``"en"`` directly; NLLB
falls back to its default source language).
"""

from __future__ import annotations

import re

# ISO 639-1 → display name. Covers the languages Bazarr would realistically
# send, plus a few extras. Keep names matching what NLLB_LANGUAGE_CODES and
# GENDER_AWARE_LANGUAGES use elsewhere in this codebase.
ISO_TO_NAME: dict[str, str] = {
    "en": "English",
    "he": "Hebrew",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "de": "German",
    "ru": "Russian",
    "pl": "Polish",
    "uk": "Ukrainian",
    "hi": "Hindi",
    "ro": "Romanian",
    "ja": "Japanese",
    "tr": "Turkish",
    "zh": "Chinese",
    "ko": "Korean",
    "nl": "Dutch",
    "sv": "Swedish",
    "fi": "Finnish",
    "no": "Norwegian",
    "da": "Danish",
    "cs": "Czech",
    "el": "Greek",
}


def language_name(value: str | None, default: str = "English") -> str:
    """Map an ISO 639-1 code to its display name, or pass a name through.

    - ``"en"`` -> ``"English"``
    - ``"English"`` -> ``"English"`` (unchanged passthrough)
    - ``"en-US"`` / ``"en_us"`` -> ``"English"`` (strips region tag)
    - ``""`` / ``None`` -> ``default`` (``"English"``)
    - Any unrecognised value -> passthrough (so Claude/NLLB still get *something*).
    """
    if not value:
        return default
    raw = value.strip()
    if not raw:
        return default
    # Strip a region/script subtag if present: "en-US", "zh_Hans" -> "en", "zh".
    primary = raw.replace("_", "-").split("-", 1)[0]
    # Direct ISO match (lower-cased).
    name = ISO_TO_NAME.get(primary.lower())
    if name is not None:
        return name
    # Already a recognised display name (case-insensitive)?
    for canonical in ISO_TO_NAME.values():
        if raw.lower() == canonical.lower():
            return canonical
    # Unknown — pass through verbatim. Both backends tolerate this.
    return raw


# Primary Unicode script ranges per non-Latin target language. Used by
# ``target_script_ratio`` to sanity-check that a translated text is actually
# in the requested target language — caught a real bug where NLLB free-
# generated Spanish/Romanian/Tswana when targeting Hebrew because the
# ``forced_bos_token_id`` plumbing had silently broken on transformers 5.x.
#
# Latin-script languages (French, Spanish, German, Italian, Portuguese,
# Romanian, Polish, Dutch, etc.) are intentionally absent — there's no
# usable Unicode signal distinguishing one Latin-script language from
# another, so a ratio there would be misleading.
LANGUAGE_SCRIPTS: dict[str, re.Pattern[str]] = {
    "Hebrew": re.compile(r"[֐-׿]"),
    "Arabic": re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]"),
    "Russian": re.compile(r"[Ѐ-ӿ]"),
    "Ukrainian": re.compile(r"[Ѐ-ӿ]"),
    "Greek": re.compile(r"[Ͱ-Ͽ]"),
    "Hindi": re.compile(r"[ऀ-ॿ]"),
    "Japanese": re.compile(r"[぀-ゟ゠-ヿ一-鿿]"),
    "Korean": re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]"),
    "Chinese": re.compile(r"[一-鿿㐀-䶿]"),
}


def target_script_ratio(text: str, target_language: str) -> float | None:
    """Fraction of letters in ``text`` matching the target language's script.

    Returns a float in [0.0, 1.0] for languages in ``LANGUAGE_SCRIPTS``;
    returns ``None`` for languages with no usable script signal (Latin-script
    languages, where the ratio would not distinguish e.g. French from Spanish).
    A ratio < ~0.5 on a known non-Latin target almost always means the
    translation backend wrote the wrong language.
    """
    pat = LANGUAGE_SCRIPTS.get(target_language)
    if pat is None:
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if pat.match(c)) / len(letters)

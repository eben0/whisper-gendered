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

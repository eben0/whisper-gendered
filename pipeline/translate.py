"""Claude-API translation of subtitle segments into the target language.

Segments are translated in *batches* (one API call per group of consecutive
same-speaker segments) rather than one call per segment. This preserves
cross-line context, cuts a feature film from ~1500 calls to a few dozen, and
still lets us state the speaker's gender for grammatically correct output in
gender-aware languages.

The model is asked to return a JSON object whose ``translations`` array has one
entry per input segment, in order, via structured outputs — so parsing is
deterministic.
"""

from __future__ import annotations

import json
import logging

import anthropic

from config import settings

log = logging.getLogger("pipeline.translate")

# Cap each batch so the request/response comfortably fit in one call.
MAX_BATCH_SEGMENTS = 40
MAX_BATCH_CHARS = 2500
MAX_TOKENS = 4096

# Structured-outputs schema: an object with one string per input segment.
_OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    },
}


def _system_prompt(
    target_language: str,
    gender: str | None,
    addressee_gender: str | None = None,
    source_language: str = "English",
) -> str:
    base = (
        f"You are an expert subtitle translator. Translate each numbered line "
        f"from {source_language} into {target_language}. Produce natural, idiomatic, "
        f"concise {target_language} suitable for on-screen subtitles. Preserve "
        f"meaning and tone; do not add notes or explanations."
    )
    # Style guidance — applies to every translation regardless of gender.
    base += (
        f" Transliterate proper nouns (people's names, place names, brand "
        f"names, nicknames) into the {target_language} script: write them as "
        f"the audience would naturally read them aloud, not as Latin letters. "
        f"Example: 'Mondo' -> 'מונדו', 'T' -> 'טי'."
    )
    base += (
        f" Render slang, idioms, and figures of speech with their natural "
        f"{target_language} equivalents — not literal word-for-word calques. "
        f"If a {source_language} idiom has no clean equivalent, prefer the "
        f"closest colloquial phrasing in {target_language}."
    )
    base += (
        f" Choose the natural {target_language} preposition for each "
        f"construction. For Hebrew specifically: prefer ב, של, ל, מ, על "
        f"where the grammar calls for them; reserve את only for marking a "
        f"definite direct object — never use את as a generic substitute."
    )
    base += (
        " Keep each line short — aim for at most 42 characters per visible "
        "line. For longer utterances, prefer two short lines over one long "
        "one; end thoughts on natural pause-points (after a comma, "
        "conjunction, or clause boundary) when possible, so a downstream "
        "formatter can break cleanly."
    )
    if gender is not None:
        base += (
            f" The speaker of these lines is {gender}. Use grammatically correct "
            f"{gender} forms throughout — verb conjugation, adjective and "
            f"participle agreement, imperatives, and pronouns must all match a "
            f"{gender} speaker referring to themselves."
        )
        base += (
            f" When the speaker addresses another person ({source_language} \"you\""
            f" or its equivalent), choose the {target_language} form matching the "
            f"addressee's number and gender."
        )
        if addressee_gender is not None:
            base += (
                f" The most likely addressee in this exchange is "
                f"{addressee_gender}; prefer that form for singular \"you\" "
                f"unless context clearly implies a different addressee."
            )
        base += (
            " Infer number from context — collective cues like \"you all\", "
            "\"you guys\", or plural verbs imply plural. When number is "
            "ambiguous in a multi-person scene, prefer the inclusive plural "
            "form (e.g., אתם in Hebrew). Do not mix forms within a single line."
        )
    base += (
        " Return a JSON object with a 'translations' array containing exactly "
        "one translated string per input line, in the same order."
    )
    return base


def _chunks(texts: list[str]) -> list[list[str]]:
    """Split texts into batches bounded by segment count and total chars."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for t in texts:
        if current and (
            len(current) >= MAX_BATCH_SEGMENTS
            or current_chars + len(t) > MAX_BATCH_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(t)
        current_chars += len(t)
    if current:
        batches.append(current)
    return batches


def _translate_one_batch(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: anthropic.Anthropic,
    addressee_gender: str | None = None,
    source_language: str = "English",
) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    response = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(target_language, gender, addressee_gender, source_language),
        output_config={"format": _OUTPUT_FORMAT},
        messages=[{"role": "user", "content": numbered}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "")
    try:
        translations = json.loads(raw).get("translations", [])
    except (json.JSONDecodeError, AttributeError):
        log.warning("Could not parse translation JSON; returning source text.")
        return texts

    if len(translations) != len(texts):
        log.warning(
            "Translation count mismatch (got %d, expected %d); aligning by index.",
            len(translations), len(texts),
        )
        translations = (translations + texts[len(translations):])[: len(texts)]
    return [str(t) for t in translations]


def translate_batch(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: anthropic.Anthropic,
    addressee_gender: str | None = None,
    source_language: str = "English",
) -> list[str]:
    """Translate ``texts`` into ``target_language``, returning one string each.

    ``gender`` is ``"male"``/``"female"`` for gender-aware languages, or ``None``
    to request a plain translation. ``addressee_gender`` (optional) hints the
    grammatical "you" form for languages where second-person is gender-marked.
    ``source_language`` defaults to English so existing callers stay valid;
    the orchestrator passes the value derived from the request's ``language``
    query param.
    Output length always equals input length.
    """
    if not texts:
        return []
    out: list[str] = []
    for batch in _chunks(texts):
        out.extend(
            _translate_one_batch(
                batch, gender, target_language, client, addressee_gender, source_language,
            )
        )
    return out


async def _translate_one_batch_async(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: anthropic.AsyncAnthropic,
    addressee_gender: str | None = None,
    source_language: str = "English",
) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    response = await client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=_system_prompt(target_language, gender, addressee_gender, source_language),
        output_config={"format": _OUTPUT_FORMAT},
        messages=[{"role": "user", "content": numbered}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "")
    try:
        translations = json.loads(raw).get("translations", [])
    except (json.JSONDecodeError, AttributeError):
        log.warning("Could not parse translation JSON; returning source text.")
        return texts

    if len(translations) != len(texts):
        log.warning(
            "Translation count mismatch (got %d, expected %d); aligning by index.",
            len(translations), len(texts),
        )
        translations = (translations + texts[len(translations):])[: len(texts)]
    return [str(t) for t in translations]


async def translate_batch_async(
    texts: list[str],
    gender: str | None,
    target_language: str,
    client: anthropic.AsyncAnthropic,
    addressee_gender: str | None = None,
    source_language: str = "English",
) -> list[str]:
    """Async counterpart of ``translate_batch`` for the chunked orchestrator.

    Sub-batches run sequentially within one call; cross-chunk concurrency is
    handled by the caller's semaphore. ``addressee_gender`` (optional) hints the
    grammatical "you" form. ``source_language`` defaults to English; the
    orchestrator passes the value derived from the request's ``language``
    query param. Output length always equals input length.
    """
    if not texts:
        return []
    out: list[str] = []
    for batch in _chunks(texts):
        out.extend(
            await _translate_one_batch_async(
                batch, gender, target_language, client, addressee_gender, source_language,
            )
        )
    return out

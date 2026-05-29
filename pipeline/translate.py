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

import prompt
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
    """Assemble the system prompt from templates in ``prompt/translate/``."""
    parts: list[str] = [
        prompt.load("translate/base",
                    source_language=source_language, target_language=target_language),
        prompt.load("translate/style_transliteration", target_language=target_language),
        prompt.load("translate/style_slang",
                    source_language=source_language, target_language=target_language),
        prompt.load("translate/style_prepositions", target_language=target_language),
        prompt.load("translate/style_length"),
    ]
    if gender is not None:
        parts.append(prompt.load("translate/gender_speaker", gender=gender))
        parts.append(prompt.load("translate/gender_you_form",
                                 source_language=source_language,
                                 target_language=target_language))
        if addressee_gender is not None:
            parts.append(prompt.load("translate/gender_addressee",
                                     addressee_gender=addressee_gender))
        parts.append(prompt.load("translate/gender_number"))
    parts.append(prompt.load("translate/scene_context"))
    parts.append(prompt.load("translate/output_format"))
    return " ".join(parts)


def _build_user_message(
    texts: list[str], previous_context: list[str] | None
) -> str:
    """Compose the user message body, optionally prefixed by a scene-context block.

    When ``previous_context`` is non-empty, prepends a numbered
    'Earlier in this scene:' preamble followed by a 'Translate the
    following lines:' marker, then the numbered ``texts``. When empty or
    None, returns the bare numbered list — byte-identical to the
    pre-context behaviour, so existing callers see no change.
    """
    parts: list[str] = []
    if previous_context:
        parts.append("Earlier in this scene:")
        for j, ctx in enumerate(previous_context, start=1):
            parts.append(f"  {j}. {ctx}")
        parts.append("")
        parts.append("Translate the following lines:")
    parts.extend(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return "\n".join(parts)


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
    previous_context: list[str] | None = None,
) -> list[str]:
    numbered = _build_user_message(texts, previous_context)
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
    previous_context: list[str] | None = None,
) -> list[str]:
    """Translate ``texts`` into ``target_language``, returning one string each.

    ``gender`` is ``"male"``/``"female"`` for gender-aware languages, or ``None``
    to request a plain translation. ``addressee_gender`` (optional) hints the
    grammatical "you" form for languages where second-person is gender-marked.
    ``source_language`` defaults to English so existing callers stay valid;
    the orchestrator passes the value derived from the request's ``language``
    query param. ``previous_context`` is an optional list of recent prior
    translation lines that share scene context — they are prepended to the
    user message as a background block (not re-translated) so Claude can
    disambiguate addressee gender/number and keep vocabulary consistent.
    Output length always equals input length.
    """
    if not texts:
        return []
    out: list[str] = []
    for batch in _chunks(texts):
        out.extend(
            _translate_one_batch(
                batch, gender, target_language, client, addressee_gender,
                source_language, previous_context=previous_context,
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
    previous_context: list[str] | None = None,
) -> list[str]:
    numbered = _build_user_message(texts, previous_context)
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
    previous_context: list[str] | None = None,
) -> list[str]:
    """Async counterpart of ``translate_batch`` for the chunked orchestrator.

    Sub-batches run sequentially within one call; cross-chunk concurrency is
    handled by the caller's semaphore. ``addressee_gender`` (optional) hints the
    grammatical "you" form. ``source_language`` defaults to English; the
    orchestrator passes the value derived from the request's ``language``
    query param. ``previous_context`` (optional) is shared across all
    sub-batches of this call — the same scene-context window is prepended
    to each sub-batch's user message. Output length always equals input length.
    """
    if not texts:
        return []
    out: list[str] = []
    for batch in _chunks(texts):
        out.extend(
            await _translate_one_batch_async(
                batch, gender, target_language, client, addressee_gender,
                source_language, previous_context=previous_context,
            )
        )
    return out

"""Claude API translation backend."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, TYPE_CHECKING

import prompt
from src.backends.factory import TranslationBackend, CLAUDE
from src.config import settings

if TYPE_CHECKING:
    from src.config import Settings

log = logging.getLogger("backends.claude")

_client_lock = threading.Lock()

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


ContextLine = tuple[str | None, str]
"""One line of prior-scene context: ``(speaker_gender, target_text)``.

``speaker_gender`` is ``"male"`` / ``"female"`` for the gender-aware path,
or ``None`` when the source pipeline doesn't carry gender (plain translate).

``target_text`` is the line in the *target language* (already translated by
an earlier batch in the same request) — not the source. Target text is
chosen on purpose: gendered languages (Hebrew, Spanish, Russian, …) encode
the addressee's gender directly in verb conjugations and pronouns, giving
Claude the signal it needs to reconstruct turn-taking. Source English
``"you"`` is gender-neutral and reveals nothing about who was addressing
whom. The cost is that a mistranslated earlier line propagates its error
to later lines — but for addressee inference the trade-off pays off.
"""




def _build_user_message(
    texts: list[str],
    previous_context: list[ContextLine] | None,
    speaker_gender: str | None = None,
    addressee_gender: str | None = None,
) -> str:
    """Compose the user message body, optionally prefixed by scene context
    and a per-batch ``(speaker, addressee)`` role hint.

    When ``previous_context`` is non-empty, prepends a numbered
    'Earlier in this scene:' preamble with each context line rendered as
    ``[gender]: target_text`` — already-translated target-language text
    plus the speaker's gender label — so Claude can reconstruct turn-taking
    and infer the current speaker's addressee across chunk boundaries.

    When ``speaker_gender`` and/or ``addressee_gender`` are provided,
    emits a ``(speaker: X, addressee: Y)`` line just before the actual
    translation targets. Putting this hint in the user message — not just
    the system prompt — overrides narrative priors that a system-prompt
    directive alone can lose to (e.g. "my wife" → assume male addressee).

    When both ``previous_context`` and the role hints are absent, returns
    the bare numbered list — byte-identical to the pre-context behaviour.
    """
    parts: list[str] = []
    if previous_context:
        parts.append("Earlier in this scene:")
        for j, (gender, ctx) in enumerate(previous_context, start=1):
            prefix = f"[{gender}]: " if gender else ""
            parts.append(f"  {j}. {prefix}{ctx}")
        parts.append("")
    role_bits: list[str] = []
    if speaker_gender:
        role_bits.append(f"speaker: {speaker_gender}")
    if addressee_gender:
        role_bits.append(f"addressee: {addressee_gender}")
    if role_bits:
        parts.append(f"({', '.join(role_bits)})")
        parts.append("")
    if previous_context or role_bits:
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




def _log_usage(response, batch_size: int) -> None:
    """Emit one grep-friendly INFO line per Claude call with token counts.

    Logs raw counts only — pricing varies by model and changes over time, so
    cost math stays out of code. Grep ``TRANSLATE_USAGE`` to sum a request's
    totals; ``cache_read`` / ``cache_creation`` are 0 today (we don't pass
    ``cache_control`` yet) but populated automatically if/when we enable
    prompt caching, so the line format won't change later.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    log.info(
        "TRANSLATE_USAGE model=%s batch=%d input=%d output=%d cache_read=%d cache_creation=%d",
        settings.CLAUDE_MODEL,
        batch_size,
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


class ClaudeBackend(TranslationBackend):
    """Calls the Anthropic API directly — all translation logic lives here."""

    _backend_type = CLAUDE

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._target_language = settings.TARGET_LANGUAGE
        self._client = None

    def _system_prompt(
        self,
        gender: str | None,
        addressee_gender: str | None = None,
        source_language: str = "English",
    ) -> str:
        """Assemble the system prompt from templates in ``prompt/translate/``.

        Language-specific sections are only included when relevant:
        - transliteration guidance is skipped for Latin-script targets
          (French, Spanish, German, etc.) where proper nouns stay as-is.
        - Hebrew-specific preposition guidance is only included when the
          target is Hebrew.
        """
        from pipeline.lang import uses_non_latin_script
        target_language = self._target_language
        parts: list[str] = [
            prompt.load("translate/base",
                        source_language=source_language, target_language=target_language),
        ]
        if uses_non_latin_script(target_language):
            parts.append(prompt.load("translate/style_transliteration",
                                     target_language=target_language))
        parts.append(prompt.load("translate/style_slang",
                                 source_language=source_language,
                                 target_language=target_language))
        parts.append(prompt.load("translate/style_prepositions",
                                 source_language=source_language,
                                 target_language=target_language))
        if target_language == "Hebrew":
            parts.append(prompt.load("translate/style_prepositions_hebrew"))
        parts.append(prompt.load("translate/style_length"))
        if gender is not None:
            parts.append(prompt.load("translate/gender_speaker", gender=gender))
            parts.append(prompt.load("translate/gender_you_form",
                                     source_language=source_language,
                                     target_language=target_language))
            if addressee_gender is not None:
                parts.append(prompt.load("translate/gender_addressee",
                                         addressee_gender=addressee_gender))
            parts.append(prompt.load("translate/gender_number"))
        parts.append(prompt.load("translate/scene_context",
                                 target_language=target_language))
        parts.append(prompt.load("translate/output_format"))
        return " ".join(parts)

    def _system_blocks(
        self,
        gender: str | None,
        addressee_gender: str | None,
        source_language: str,
    ) -> list[dict]:
        """Wrap the system prompt as a single cacheable content block.

        Setting ``cache_control`` on the system block opts this request into
        Anthropic prompt caching: subsequent calls within the cache TTL that
        share the same system prompt skip re-encoding it on the input side
        (~90% input-token discount on the cached portion). All batches inside
        one ``/asr`` request use the same system prompt — same target language,
        same speaker gender, same addressee gender — so the first batch primes
        the cache and the rest hit it.

        Caveat: caching requires the cached prefix to exceed a model-specific
        minimum (currently 1024 tokens for Sonnet). Our gender-aware Hebrew
        system prompt is ~550 tokens, so the API may silently fall back to
        non-cached behaviour today. Leaving ``cache_control`` in place is
        forward-compatible: if/when prompts grow past the threshold or
        Anthropic lowers it, caching kicks in automatically.
        """
        return [{
            "type": "text",
            "text": self._system_prompt(gender, addressee_gender, source_language),
            "cache_control": {"type": "ephemeral"},
        }]

    def _get_client(self):
        """Lazy Anthropic client with double-checked locking (thread-safe)."""
        if self._client is None:
            with _client_lock:
                if self._client is None:
                    import anthropic
                    self._client = anthropic.AsyncAnthropic(
                        api_key=self._settings.require_anthropic_key(),
                        max_retries=self._settings.CLAUDE_MAX_RETRIES,
                    )
        return self._client

    def _translate_one_batch(
        self,
        texts: list[str],
        gender: str | None,
        target_language: str,
        client: Any,
        addressee_gender: str | None = None,
        source_language: str = "English",
        previous_context: list[ContextLine] | None = None,
    ) -> list[str]:
        numbered = _build_user_message(
            texts, previous_context,
            speaker_gender=gender, addressee_gender=addressee_gender,
        )
        response = client.messages.create(
            model=self._settings.CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=self._system_blocks(gender, addressee_gender, source_language),
            output_config={"format": _OUTPUT_FORMAT},
            messages=[{"role": "user", "content": numbered}],
        )
        _log_usage(response, len(texts))
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
        self,
        texts: list[str],
        gender: str | None,
        target_language: str,
        addressee_gender: str | None = None,
        source_language: str = "English",
        previous_context: list[ContextLine] | None = None,
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
        client = self._get_client()
        out: list[str] = []
        for batch in _chunks(texts):
            out.extend(
                self._translate_one_batch(
                    batch, gender, target_language, client, addressee_gender,
                    source_language, previous_context=previous_context,
                )
            )
        return out

    async def _translate_one_batch_async(
        self,
        texts: list[str],
        gender: str | None,
        target_language: str,
        client: Any,
        addressee_gender: str | None = None,
        source_language: str = "English",
        previous_context: list[ContextLine] | None = None,
    ) -> list[str]:
        numbered = _build_user_message(
            texts, previous_context,
            speaker_gender=gender, addressee_gender=addressee_gender,
        )
        response = await client.messages.create(
            model=self._settings.CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=self._system_blocks(gender, addressee_gender, source_language),
            output_config={"format": _OUTPUT_FORMAT},
            messages=[{"role": "user", "content": numbered}],
        )
        _log_usage(response, len(texts))
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
        self, texts: list[str], gender: str | None, target: str, **kwargs: Any
    ) -> list[str]:
        """Translate ``texts`` into ``target``, returning one string each.

        ``gender`` is ``"male"``/``"female"`` for gender-aware languages, or ``None``
        to request a plain translation. Extra kwargs forwarded to sub-calls:
        ``addressee_gender``, ``source_language``, ``previous_context``.

        Sub-batches run sequentially within one call; cross-chunk concurrency is
        handled by the caller's semaphore. Output length always equals input length.
        """
        if not texts:
            return []
        addressee_gender: str | None = kwargs.get("addressee_gender")
        source_language: str = kwargs.get("source_language", "English")
        previous_context: list[ContextLine] | None = kwargs.get("previous_context")
        client = self._get_client()
        out: list[str] = []
        for batch in _chunks(texts):
            out.extend(
                await self._translate_one_batch_async(
                    batch, gender, target, client, addressee_gender,
                    source_language, previous_context=previous_context,
                )
            )
        return out

    async def warmup(self) -> None:
        pass  # API-based; no model to preload

    def model_name(self) -> str:
        return self._settings.CLAUDE_MODEL

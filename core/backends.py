"""Translation-backend selection — de-magics the ``"local"`` literal.

At import time ``translate`` is bound to the module whose ``translate_batch_async``
matches ``settings.TRANSLATION_BACKEND``; both expose the same signature, so callers
use ``backends.translate.translate_batch_async(...)`` unchanged. ``client`` is ignored
by the local backend; the Claude backend uses the AsyncAnthropic client below.
"""

from __future__ import annotations

from config import settings

# Named constants replace the inline "local"/"claude" string literals.
LOCAL = "local"
CLAUDE = "claude"


def is_local() -> bool:
    """True when the configured translation backend is the on-device model."""
    return settings.TRANSLATION_BACKEND.strip().lower() == LOCAL


# Import-time backend factory (preserves prior behavior; see test_translate_local).
if is_local():
    from pipeline import translate_local as translate  # type: ignore[no-redef]
else:
    from pipeline import translate  # type: ignore[no-redef]


_async_anthropic_client = None


def get_async_anthropic_client():
    global _async_anthropic_client
    if _async_anthropic_client is None:
        import anthropic
        _async_anthropic_client = anthropic.AsyncAnthropic(
            api_key=settings.require_anthropic_key(),
            max_retries=settings.CLAUDE_MAX_RETRIES,
        )
    return _async_anthropic_client

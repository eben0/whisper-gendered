"""Claude API translation backend — wraps pipeline.translate with a managed Anthropic client."""

from __future__ import annotations

import logging

from config import settings

log = logging.getLogger("core.backend_claude")


class ClaudeBackend:
    """Translation backend that calls the Anthropic API via pipeline.translate."""

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.require_anthropic_key(),
                max_retries=settings.CLAUDE_MAX_RETRIES,
            )
        return self._client

    async def translate_batch_async(self, texts, gender, target, **kwargs):
        from pipeline import translate
        return await translate.translate_batch_async(
            texts, gender, target, self._get_client(), **kwargs
        )

    async def warmup(self) -> None:
        pass  # Claude is API-based; no model to preload

    def model_name(self) -> str:
        return settings.CLAUDE_MODEL

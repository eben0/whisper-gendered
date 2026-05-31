"""Claude API translation backend."""

from __future__ import annotations

import logging
import threading
from typing import Any, TYPE_CHECKING

from src.backends.factory import TranslationBackend, CLAUDE

if TYPE_CHECKING:
    from src.config import Settings

log = logging.getLogger("backends.claude")

_client_lock = threading.Lock()


class ClaudeBackend(TranslationBackend):
    """Calls the Anthropic API via pipeline.translate."""

    _backend_type = CLAUDE

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._client = None

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

    async def translate_batch_async(
        self, texts: list[str], gender: str | None, target: str, **kwargs: Any
    ) -> list[str]:
        from pipeline.translate import translate_batch_async
        return await translate_batch_async(
            texts, gender, target, self._get_client(), **kwargs
        )

    async def warmup(self) -> None:
        pass  # API-based; no model to preload

    def model_name(self) -> str:
        return self._settings.CLAUDE_MODEL

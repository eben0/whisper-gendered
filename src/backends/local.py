"""Local HuggingFace seq2seq translation backend."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from src.backends.factory import TranslationBackend, LOCAL

if TYPE_CHECKING:
    from pipeline.translate.local import LocalTranslator

log = logging.getLogger("backends.local")


class LocalBackend(TranslationBackend):
    """On-device HuggingFace seq2seq translation."""

    _backend_type = LOCAL

    def __init__(self, translator: "LocalTranslator") -> None:
        self._translator = translator

    async def translate_batch_async(
        self, texts: list[str], gender: str | None, target: str, **kwargs: Any
    ) -> list[str]:
        return await self._translator.translate_batch_async(texts, gender, target, None, **kwargs)

    async def warmup(self) -> None:
        await asyncio.to_thread(self._translator.warmup)

    def model_name(self) -> str:
        from src.config import settings
        return settings.LOCAL_TRANSLATION_MODEL

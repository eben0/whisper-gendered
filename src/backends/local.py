"""Local HuggingFace seq2seq translation backend."""

from __future__ import annotations

import logging
from typing import Any

from src.backends.factory import TranslationBackend, LOCAL

log = logging.getLogger("backends.local")


class LocalBackend(TranslationBackend):
    """On-device HuggingFace seq2seq translation."""

    _backend_type = LOCAL

    async def translate_batch_async(
        self, texts: list[str], gender: str | None, target: str, **kwargs: Any
    ) -> list[str]:
        from pipeline.translate_local import translate_batch_async
        return await translate_batch_async(texts, gender, target, None, **kwargs)

    async def warmup(self) -> None:
        from pipeline import translate_local
        translate_local.warmup()

    def model_name(self) -> str:
        from src.config import settings
        return settings.LOCAL_TRANSLATION_MODEL

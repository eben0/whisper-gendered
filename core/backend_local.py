"""Local HuggingFace seq2seq translation backend — wraps pipeline.translate_local."""

from __future__ import annotations

import logging

log = logging.getLogger("core.backend_local")


class LocalBackend:
    """Translation backend using a local HuggingFace model on the same GPU."""

    async def translate_batch_async(self, texts, gender, target, **kwargs):
        from pipeline import translate_local
        # local backend does not use an Anthropic client; pass None
        return await translate_local.translate_batch_async(
            texts, gender, target, None, **kwargs
        )

    async def warmup(self) -> None:
        from pipeline import translate_local
        from core import concurrency
        await concurrency.run_in_thread(translate_local.warmup)

    def model_name(self) -> str:
        from config import settings
        return settings.LOCAL_TRANSLATION_MODEL

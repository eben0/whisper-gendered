"""Backward-compat shim — the real implementation is in pipeline.translate.local.

Tests and callers that do ``from pipeline import translate_local`` and then access
module-level ``get_model_and_tokenizer``, ``translate_batch_async``, ``warmup``,
``_model``, ``_tokenizer``, ``_translate_sync``, ``_is_nllb_tokenizer``, etc.
continue to work through this module.

IMPORTANT: ``_translate_sync`` and ``translate_batch_async`` call the module-level
``get_model_and_tokenizer()`` so that tests which do
  monkeypatch.setattr(translate_local, "get_model_and_tokenizer", ...)
intercept the call correctly.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import torch

from pipeline.translate.local import (  # noqa: F401
    LocalTranslator,
    _is_nllb_tokenizer,
    _check_vram_fits,
    NLLB_LANGUAGE_CODES,
)

from config import settings as _settings  # noqa: E402

_instance = LocalTranslator(_settings)

# Module-level state so tests can monkeypatch _model/_tokenizer directly.
_model: Any | None = None
_tokenizer: Any | None = None
_lock = _instance._lock
_inference_lock = _instance._inference_lock


def get_model_and_tokenizer() -> tuple[Any, Any]:
    global _model, _tokenizer
    result = _instance.get_model_and_tokenizer()
    _model, _tokenizer = result
    return result


def model_loaded() -> bool:
    return _model is not None and _tokenizer is not None


def _translate_sync(
    texts: list[str],
    gender: Any,
    target_language: str,
    source_language: str = "English",
) -> list[str]:
    """CPU/GPU-bound work — calls module-level get_model_and_tokenizer() so tests can monkeypatch it."""
    if not texts:
        return []
    model, tokenizer = get_model_and_tokenizer()
    device = next(model.parameters()).device

    out: list[str] = []
    bsz = max(1, _settings.LOCAL_BATCH_SIZE)
    max_len = max(1, _settings.LOCAL_MAX_LENGTH)

    forced_bos: int | None = None
    with _inference_lock, torch.inference_mode():
        if _is_nllb_tokenizer(tokenizer):
            src_code = NLLB_LANGUAGE_CODES.get(source_language)
            if src_code is None:
                import logging as _logging
                _logging.getLogger("pipeline.translate_local").warning(
                    "No NLLB language code mapped for source=%r; falling back to eng_Latn.",
                    source_language,
                )
                src_code = "eng_Latn"
            tokenizer.src_lang = src_code
            target_code = NLLB_LANGUAGE_CODES.get(target_language)
            if target_code is None:
                import logging as _logging
                _logging.getLogger("pipeline.translate_local").warning(
                    "No NLLB language code mapped for target=%r; generation will fall back.",
                    target_language,
                )
            else:
                forced_bos = tokenizer.convert_tokens_to_ids(target_code)

        for i in range(0, len(texts), bsz):
            batch = [
                _instance._format_with_gender_hint(t, gender, target_language, source_language)
                for t in texts[i : i + bsz]
            ]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_len,
                "num_beams": 1,
            }
            if forced_bos is not None:
                generate_kwargs["forced_bos_token_id"] = forced_bos
            output_ids = model.generate(**inputs, **generate_kwargs)
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            out.extend(decoded)
    return out


async def translate_batch_async(
    texts: list[str],
    gender: Any = None,
    target_language: str = "Hebrew",
    client: Any = None,
    addressee_gender: Any = None,
    source_language: str = "English",
    previous_context: Any = None,
) -> list[str]:
    if not texts:
        return []
    return await asyncio.to_thread(
        _translate_sync, texts, gender, target_language, source_language,
    )


def warmup() -> None:
    try:
        get_model_and_tokenizer()
        import logging as _logging
        _logging.getLogger("pipeline.translate_local").info("Local translation warm-up complete.")
    except Exception:  # pragma: no cover - warm-up must never crash startup
        import logging as _logging
        _logging.getLogger("pipeline.translate_local").exception(
            "Local translation warm-up failed (continuing anyway)."
        )

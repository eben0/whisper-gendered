"""Translation backends — Claude API and local HuggingFace."""
from pipeline.translate.claude import translate_batch_async, translate_batch
from pipeline.translate.local import LocalTranslator

__all__ = ["translate_batch_async", "translate_batch", "LocalTranslator"]

# ---------------------------------------------------------------------------
# Module-level backward-compat shims.
# Tests that do ``from pipeline import translate`` and then access
# ``translate._system_prompt``, ``translate.ContextLine``, etc. continue
# to work via these re-exports.
# ---------------------------------------------------------------------------

from pipeline.translate.claude import (  # noqa: F401
    _system_prompt,
    _build_user_message,
    _chunks,
    _system_blocks,
    _log_usage,
    _translate_one_batch,
    _translate_one_batch_async,
    ContextLine,
    MAX_BATCH_SEGMENTS,
    MAX_BATCH_CHARS,
    MAX_TOKENS,
    _OUTPUT_FORMAT,
)

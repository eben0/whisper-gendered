"""Tests for the local HuggingFace translation backend.

Three layers, matching the style of ``tests/test_translate_async.py``:
1. Unit — mock get_model_and_tokenizer so we exercise translate_batch_async
   shape (output length, dtype, async wrapping) with zero model load.
2. Smoke — real ``Helsinki-NLP/opus-mt-en-he`` download + translate; auto-
   skipped if torch.cuda is unavailable or transformers can't reach the hub.
3. Regression — confirm that with TRANSLATION_BACKEND=claude (default),
   ``server.translate`` resolves to ``pipeline.translate`` and not to
   ``pipeline.translate_local``.
"""

from __future__ import annotations

import os
import re

import pytest

from pipeline import translate_local


# ---------------------------------------------------------------------------- #
# 1. Unit — mocked model, no HF download, no GPU
# ---------------------------------------------------------------------------- #

class _FakeInputs(dict):
    """Dict-subclass with a ``.to(device)`` no-op.

    Real HuggingFace tokenizer output is a ``BatchEncoding`` which behaves both
    like a dict (so ``**inputs`` unpacks it into ``model.generate``) and has a
    ``.to(device)`` method. Subclassing ``dict`` gives us both for free.
    """

    def to(self, device):
        return self


class _FakeTokenizer:
    """Minimal stand-in for a MarianMT tokenizer (no lang_code_to_id attr)."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts, **kwargs):
        self.calls.append(list(texts))
        # Return something dict-like (so ``**inputs`` works) carrying just
        # enough info for the fake model.generate to size its output.
        return _FakeInputs(input_ids=list(range(len(texts))))

    def batch_decode(self, ids, **kwargs):
        # Pretend each segment produced a Hebrew-tagged translation.
        return [f"HE:{i}" for i in range(len(ids))]


class _FakeModel:
    """Minimal stand-in: parameters() yields a tensor whose device is CPU."""

    class _Param:
        def __init__(self):
            import torch
            self._t = torch.zeros(1)

        @property
        def device(self):
            return self._t.device

    def parameters(self):
        return iter([self._Param()])

    def generate(self, **kwargs):
        # Return a sentinel sized to whatever the tokenizer last consumed.
        return [object()] * len(kwargs.get("forced_bos_token_id", []) or [1, 2])

    def train(self, mode):  # never called in unit tests; here for symmetry
        return self


@pytest.mark.asyncio
async def test_translate_batch_async_preserves_count_and_returns_strings(monkeypatch):
    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel()

    # generate() needs to return as many outputs as inputs for batch_decode.
    def fake_generate(**kwargs):
        # Reconstruct batch size from the tokenizer's last recorded call.
        return [None] * len(fake_tok.calls[-1])

    fake_model.generate = fake_generate

    monkeypatch.setattr(
        translate_local, "get_model_and_tokenizer",
        lambda: (fake_model, fake_tok),
    )

    out = await translate_local.translate_batch_async(
        ["a", "b", "c"], gender="male", target_language="Hebrew",
    )
    assert len(out) == 3
    assert all(isinstance(s, str) and s for s in out)
    # Confirm batching honoured the input order (single batch of 3 here).
    assert fake_tok.calls == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_translate_batch_async_empty_input():
    # Empty input must short-circuit before any model load.
    out = await translate_local.translate_batch_async(
        [], gender="female", target_language="Hebrew",
    )
    assert out == []


@pytest.mark.asyncio
async def test_translate_batch_async_accepts_and_ignores_client_and_addressee(monkeypatch):
    # Signature parity with the Claude backend: extra kwargs are accepted.
    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel()
    fake_model.generate = lambda **kw: [None]

    monkeypatch.setattr(
        translate_local, "get_model_and_tokenizer",
        lambda: (fake_model, fake_tok),
    )

    out = await translate_local.translate_batch_async(
        ["x"],
        gender="male",
        target_language="Hebrew",
        client=object(),         # ignored
        addressee_gender="female",  # ignored
    )
    assert out == ["HE:0"]


# ---------------------------------------------------------------------------- #
# 2. Smoke — real model, real translation; skips on no-GPU or offline runners
# ---------------------------------------------------------------------------- #

HEBREW_CHAR_RE = re.compile(r"[֐-׿]")


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _gpu_available(),
    reason="Real-model smoke test needs a CUDA GPU; skipping on CPU-only runners.",
)
@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_TRANSLATE_SMOKE", "").lower() not in ("1", "true", "yes"),
    reason="Set RUN_LOCAL_TRANSLATE_SMOKE=1 to run this (downloads ~300 MB of weights).",
)
async def test_smoke_opus_mt_en_he_produces_hebrew(monkeypatch):
    # Override the configured model to the lightweight fallback so the test
    # runs in <1 minute on first invocation. Subsequent runs are seconds
    # (HF cache hit).
    monkeypatch.setattr(translate_local.settings, "LOCAL_TRANSLATION_MODEL",
                        "Helsinki-NLP/opus-mt-en-he")
    monkeypatch.setattr(translate_local.settings, "LOCAL_BATCH_SIZE", 2)
    # Force a fresh load so a previously-loaded model from a different test
    # doesn't shadow our override.
    translate_local._model = None
    translate_local._tokenizer = None
    try:
        out = await translate_local.translate_batch_async(
            ["Hello, how are you?", "I am going to the market this afternoon."],
            gender=None,
            target_language="Hebrew",
        )
    finally:
        translate_local._model = None
        translate_local._tokenizer = None

    assert len(out) == 2
    for line in out:
        assert HEBREW_CHAR_RE.search(line), f"no Hebrew chars in: {line!r}"


# ---------------------------------------------------------------------------- #
# 3. Regression — TRANSLATION_BACKEND=claude (default) keeps the Claude path
# ---------------------------------------------------------------------------- #

def test_default_backend_resolves_to_claude_module():
    # Importing server should bind ``server.translate`` to pipeline.translate
    # because TRANSLATION_BACKEND defaults to "claude". This guards against a
    # future refactor accidentally flipping the default.
    import server
    assert server.translate.__name__ == "pipeline.translate", (
        f"Default backend should be pipeline.translate; got {server.translate.__name__}. "
        "Check TRANSLATION_BACKEND default in config.py."
    )

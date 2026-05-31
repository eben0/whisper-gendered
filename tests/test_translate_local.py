"""Tests for the local HuggingFace translation backend.

Three layers, matching the style of ``tests/test_translate_async.py``:
1. Unit — mock get_model_and_tokenizer so we exercise translate_batch_async
   shape (output length, dtype, async wrapping) with zero model load.
2. Smoke — real ``Helsinki-NLP/opus-mt-en-he`` download + translate; auto-
   skipped if torch.cuda is unavailable or transformers can't reach the hub.
3. Regression — confirm that with TRANSLATION_BACKEND=claude (default),
   ``create_backend`` returns a ClaudeBackend and not a LocalBackend.
"""

from __future__ import annotations

import os
import re

import pytest

from src.backends.local import LocalBackend, _is_nllb_tokenizer
from src.config import settings as _settings


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
    backend = LocalBackend(_settings)
    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel()

    # generate() needs to return as many outputs as inputs for batch_decode.
    def fake_generate(**kwargs):
        # Reconstruct batch size from the tokenizer's last recorded call.
        return [None] * len(fake_tok.calls[-1])

    fake_model.generate = fake_generate

    monkeypatch.setattr(
        backend, "get_model_and_tokenizer",
        lambda: (fake_model, fake_tok),
    )

    out = await backend.translate_batch_async(
        ["a", "b", "c"], gender="male", target="Hebrew",
    )
    assert len(out) == 3
    assert all(isinstance(s, str) and s for s in out)
    # Confirm batching honoured the input order (single batch of 3 here).
    assert fake_tok.calls == [["a", "b", "c"]]


@pytest.mark.asyncio
async def test_translate_batch_async_empty_input():
    backend = LocalBackend(_settings)
    # Empty input must short-circuit before any model load.
    out = await backend.translate_batch_async(
        [], gender="female", target="Hebrew",
    )
    assert out == []


@pytest.mark.asyncio
async def test_translate_batch_async_serialises_concurrent_calls(monkeypatch):
    # Regression: two chunks calling _translate_sync from different worker
    # threads must not enter the tokenizer concurrently. Real HF FastTokenizer
    # raises ``RuntimeError: Already borrowed`` from the Rust layer if they do
    # — we simulate the panic with a fake tokenizer that explodes when entered
    # while another thread is still inside it. With the lock in place, both
    # calls complete; without it, this test would raise.
    import asyncio
    import threading

    entered = 0
    max_concurrent = 0
    lock_witness = threading.Lock()

    class _Race:
        @staticmethod
        def _enter(label: str):
            nonlocal entered, max_concurrent
            with lock_witness:
                entered += 1
                max_concurrent = max(max_concurrent, entered)
                if entered > 1:
                    raise RuntimeError(
                        f"Already borrowed (concurrent {label})"
                    )

        @staticmethod
        def _leave():
            nonlocal entered
            with lock_witness:
                entered -= 1

    class _RaceTokenizer:
        __name__ = "NllbTokenizer"

        def __init__(self):
            self._src_lang = "eng_Latn"

        @property
        def src_lang(self):
            return self._src_lang

        @src_lang.setter
        def src_lang(self, value):
            _Race._enter("src_lang setter")
            try:
                import time
                time.sleep(0.01)
                self._src_lang = value
            finally:
                _Race._leave()

        def convert_tokens_to_ids(self, token):
            _Race._enter("convert_tokens_to_ids")
            try:
                import time
                time.sleep(0.01)
                return 256067  # heb_Hebr's real id, arbitrary here
            finally:
                _Race._leave()

        def __call__(self, texts, **kwargs):
            _Race._enter("__call__")
            try:
                import time
                time.sleep(0.05)
                return _FakeInputs(input_ids=list(range(len(texts))))
            finally:
                _Race._leave()

        def batch_decode(self, ids, **kwargs):
            return [f"HE:{i}" for i in range(len(ids))]

    _RaceTokenizer.__name__ = "NllbTokenizer"
    fake_tok = _RaceTokenizer()
    fake_model = _FakeModel()
    fake_model.generate = lambda **kw: [None]

    backend = LocalBackend(_settings)
    monkeypatch.setattr(
        backend, "get_model_and_tokenizer",
        lambda: (fake_model, fake_tok),
    )

    results = await asyncio.gather(
        backend.translate_batch_async(["a"], None, "Hebrew"),
        backend.translate_batch_async(["b"], None, "Hebrew"),
        backend.translate_batch_async(["c"], None, "Hebrew"),
    )
    assert all(r == ["HE:0"] for r in results)
    assert max_concurrent == 1, (
        f"_inference_lock failed to serialise tokenizer access "
        f"(saw {max_concurrent} concurrent entries)"
    )


@pytest.mark.asyncio
async def test_translate_batch_async_accepts_and_ignores_client_and_addressee(monkeypatch):
    # Signature parity with the Claude backend: extra kwargs are accepted.
    backend = LocalBackend(_settings)
    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel()
    fake_model.generate = lambda **kw: [None]

    monkeypatch.setattr(
        backend, "get_model_and_tokenizer",
        lambda: (fake_model, fake_tok),
    )

    out = await backend.translate_batch_async(
        ["x"],
        gender="male",
        target="Hebrew",
        addressee_gender="female",  # ignored
    )
    assert out == ["HE:0"]


@pytest.mark.asyncio
async def test_translate_batch_async_accepts_previous_context_kwarg(monkeypatch):
    # The local backend doesn't use previous_context (no instruction-following),
    # but must accept it for signature parity with the Claude backend so the
    # orchestrator can pass it uniformly regardless of TRANSLATION_BACKEND.
    captured: dict = {}

    backend = LocalBackend(_settings)

    def fake_sync(texts, gender, tgt, src):
        captured["args"] = (texts, gender, tgt, src)
        return ["X" for _ in texts]

    monkeypatch.setattr(backend, "_translate_sync", fake_sync)

    out = await backend.translate_batch_async(
        ["hello"], None, "Hebrew",
        previous_context=["earlier line 1", "earlier line 2"],
    )
    assert out == ["X"]
    # The context must NOT leak into the model call — local backend ignores it.
    assert captured["args"] == (["hello"], None, "Hebrew", "English")


def test_is_nllb_tokenizer_keys_on_class_name():
    # Regression: this used to check ``hasattr(tokenizer, "lang_code_to_id")``,
    # which silently returned False on transformers 5.x (the attribute was
    # removed). The NLLB branch was then skipped — no src_lang, no forced
    # target token — and the model generated random multilingual output. We
    # now key on the class name, which is stable across versions.
    class NllbTokenizer: pass
    class NllbTokenizerFast: pass
    class MarianTokenizer: pass
    class MarianTokenizerFast: pass

    assert _is_nllb_tokenizer(NllbTokenizer()) is True
    assert _is_nllb_tokenizer(NllbTokenizerFast()) is True
    assert _is_nllb_tokenizer(MarianTokenizer()) is False
    assert _is_nllb_tokenizer(MarianTokenizerFast()) is False

    # A historical NLLB tokenizer that still has lang_code_to_id but a
    # different (hypothetical) class name should still be detected by name.
    class NllbV2Tokenizer:
        lang_code_to_id = {"eng_Latn": 256047}
    assert _is_nllb_tokenizer(NllbV2Tokenizer()) is True

    # An unrelated tokenizer that happens to expose ``lang_code_to_id`` must
    # NOT be classified as NLLB — that was the old bug in reverse.
    class WeirdTokenizer:
        lang_code_to_id = {"foo": 1}
    assert _is_nllb_tokenizer(WeirdTokenizer()) is False


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
    backend = LocalBackend(_settings)
    monkeypatch.setattr(backend._settings, "LOCAL_TRANSLATION_MODEL",
                        "Helsinki-NLP/opus-mt-en-he")
    monkeypatch.setattr(backend._settings, "LOCAL_BATCH_SIZE", 2)
    try:
        out = await backend.translate_batch_async(
            ["Hello, how are you?", "I am going to the market this afternoon."],
            gender=None,
            target="Hebrew",
        )
    finally:
        backend._model = None
        backend._tokenizer = None

    assert len(out) == 2
    for line in out:
        assert HEBREW_CHAR_RE.search(line), f"no Hebrew chars in: {line!r}"


# ---------------------------------------------------------------------------- #
# 3. Regression — TRANSLATION_BACKEND=claude (default) keeps the Claude path
# ---------------------------------------------------------------------------- #

def test_default_backend_resolves_to_claude_module(monkeypatch):
    # When TRANSLATION_BACKEND=claude, create_backend should return a
    # ClaudeBackend instance.
    import importlib
    import src.config as config_module
    import src.backends.claude as claude_module
    import src.backends.factory as factory_module
    monkeypatch.setenv("TRANSLATION_BACKEND", "claude")
    importlib.reload(config_module)
    assert config_module.settings.TRANSLATION_BACKEND == "claude"
    importlib.reload(claude_module)
    importlib.reload(factory_module)
    from src.backends.claude import ClaudeBackend
    backend_instance = factory_module.create_backend(config_module.settings)
    assert isinstance(backend_instance, ClaudeBackend), (
        f"Claude backend should be a ClaudeBackend instance; got "
        f"{type(backend_instance)!r}."
    )


def test_config_default_translation_backend_is_claude():
    # Independent of the dev's local .env: the *source-level* default in
    # src/config.py must be "claude". Parse the file rather than relying on
    # the runtime settings (which load_dotenv would override).
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "config.py"
    text = src.read_text(encoding="utf-8")
    assert 'os.getenv("TRANSLATION_BACKEND", "claude")' in text, (
        "src/config.py TRANSLATION_BACKEND default must be 'claude' so existing "
        "deployments keep using the Claude API path without an env change."
    )

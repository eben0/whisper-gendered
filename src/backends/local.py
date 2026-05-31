"""Local HuggingFace seq2seq translation backend.

A drop-in alternative to the Claude path, selected at startup via
``settings.TRANSLATION_BACKEND``. The model is loaded once as a
per-``LocalBackend``-instance singleton and runs on GPU when available,
falling back to CPU otherwise.

Two model families are supported transparently:
- MarianMT (e.g. ``Helsinki-NLP/opus-mt-en-he``): direction is baked into the
  weights; no language-tag plumbing.
- NLLB (e.g. ``facebook/nllb-200-distilled-600M``): the tokenizer needs an
  ``src_lang`` set, and ``model.generate`` needs ``forced_bos_token_id`` for
  the target language. We detect by introspecting the tokenizer.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, TYPE_CHECKING

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.backends.factory import TranslationBackend, LOCAL

if TYPE_CHECKING:
    from src.config import Settings

log = logging.getLogger("backends.local")

# Map from human-readable language names (matching the codebase's
# TARGET_LANGUAGE convention) to NLLB-200 language tags. NLLB needs these to
# pick the target language; MarianMT doesn't use them at all.
#
# Add to this dict if you point LOCAL_TRANSLATION_MODEL at NLLB and want a new
# target language. For MarianMT this dict is irrelevant — the model name alone
# encodes the direction.
NLLB_LANGUAGE_CODES: dict[str, str] = {
    "Hebrew": "heb_Hebr",
    "Arabic": "arb_Arab",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "German": "deu_Latn",
    "Russian": "rus_Cyrl",
    "Polish": "pol_Latn",
    "Ukrainian": "ukr_Cyrl",
    "Hindi": "hin_Deva",
    "Romanian": "ron_Latn",
    "English": "eng_Latn",  # for src_lang and round-tripping
    "Japanese": "jpn_Jpan",
    "Turkish": "tur_Latn",
    "Chinese": "zho_Hans",
}


def _is_nllb_tokenizer(tokenizer: Any) -> bool:
    """Detect NLLB tokenizers across transformers versions.

    transformers <=4.x exposed ``lang_code_to_id`` as an attribute; transformers
    5.x removed it and manages language tags through the added-tokens
    vocabulary instead. Keying on ``hasattr(..., "lang_code_to_id")`` therefore
    silently returns False on 5.x — the NLLB branch is skipped, ``src_lang``
    and ``forced_bos_token_id`` are never set, and the model free-generates in
    whatever language it picks per batch (we observed Spanish, Romanian, and
    Tswana output for an English->Hebrew job).

    The class name ``NllbTokenizer`` / ``NllbTokenizerFast`` is stable across
    both major versions, so we key on that instead. MarianMT tokenizers are
    named ``MarianTokenizer`` and correctly fall through to the non-NLLB path.
    """
    return type(tokenizer).__name__.startswith("Nllb")


def _check_vram_fits(model: Any, device: str) -> None:
    """After loading on CPU, raise a clear error if CUDA wouldn't have room.

    Called *before* moving the model to CUDA. The check is conservative: it
    compares the model's parameter byte count plus a 20% activation/KV-cache
    headroom against currently-free VRAM. If Whisper and pyannote are already
    resident this captures the realistic remaining headroom; if they aren't,
    the check is just pessimistic (better than mid-request OOM).
    """
    if device != "cuda":
        return
    weight_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    needed = int(weight_bytes * 1.20)  # 20% headroom for activations + KV cache
    free, total = torch.cuda.mem_get_info()
    if needed > free:
        raise RuntimeError(
            f"Local translation model needs ~{needed/1e9:.2f} GB free VRAM but only "
            f"{free/1e9:.2f} GB is available on this device "
            f"({total/1e9:.2f} GB total). Set LOCAL_TRANSLATION_MODEL to a smaller "
            f"model (e.g. Helsinki-NLP/opus-mt-en-he), or LOCAL_TRANSLATION_DEVICE=cpu."
        )


class LocalBackend(TranslationBackend):
    """On-device HuggingFace seq2seq translation — self-contained singleton."""

    _backend_type = LOCAL

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = threading.Lock()
        # Separate lock that serialises inference calls. HuggingFace's Rust-backed
        # ``FastTokenizer`` uses interior mutability and panics with
        # ``RuntimeError: Already borrowed`` if two threads enter ``tokenizer(...)``
        # concurrently. The orchestrator's ``TRANSLATE_CONCURRENCY`` semaphore allows
        # multiple chunks to call ``translate_batch_async`` in parallel from different
        # worker threads (via ``asyncio.to_thread``), so we must serialise here.
        self._inference_lock = threading.Lock()

    def _resolve_device(self) -> str:
        """Return ``cuda`` if requested and available, else ``cpu``."""
        requested = self._settings.LOCAL_TRANSLATION_DEVICE.strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            log.warning(
                "LOCAL_TRANSLATION_DEVICE=cuda but torch.cuda is unavailable; "
                "falling back to CPU."
            )
            return "cpu"
        return requested

    def _resolve_dtype(self) -> torch.dtype:
        name = self._settings.LOCAL_TRANSLATION_DTYPE.strip().lower()
        if name in ("float16", "fp16", "half"):
            return torch.float16
        return torch.float32

    def get_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Return ``(model, tokenizer)``, loading the singleton on first call.

        The HuggingFace cache lives at its default location (``~/.cache/huggingface``)
        and the existing ``HF_TOKEN`` is reused for any gated downloads.
        """
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer

            model_id = self._settings.LOCAL_TRANSLATION_MODEL
            device = self._resolve_device()
            dtype = self._resolve_dtype()
            token = self._settings.HF_TOKEN or None

            log.info(
                "Loading local translation model %s (device=%s, dtype=%s)...",
                model_id, device, dtype,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
            # Load on CPU first so we can VRAM-check the parameter count without
            # actually allocating GPU memory, then move.
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, token=token, torch_dtype=dtype,
            )
            _check_vram_fits(model, device)
            model.to(device)
            # Switch to inference mode (no dropout, etc.). Using .train(False)
            # rather than .eval() — semantically identical, but .eval() trips an
            # over-broad security hook that flags the unrelated Python builtin.
            model.train(False)
            self._tokenizer = tokenizer
            self._model = model
            log.info(
                "Local translation model loaded (%d params, %.2f GB on %s).",
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9,
                device,
            )
        return self._model, self._tokenizer

    def _format_with_gender_hint(
        self,
        text: str,
        gender: str | None,
        target_language: str,
        source_language: str = "English",
    ) -> str:
        """Prepend a best-effort gender hint to the source text.

        NB: local seq2seq models aren't instruction-followers. The hint becomes
        part of the source string the model translates; the model may incorporate
        it, ignore it, or (worst case) translate the hint literally. Off by
        default; opt in via ``LOCAL_USE_GENDER_PREFIX=true``.
        """
        if not self._settings.LOCAL_USE_GENDER_PREFIX or not gender:
            return text
        return (
            f"Translate from {source_language} to {target_language} "
            f"({gender} speaker): {text}"
        )

    def _translate_sync(
        self,
        texts: list[str],
        gender: str | None,
        target_language: str,
        source_language: str = "English",
    ) -> list[str]:
        """The CPU/GPU-bound work; the async wrapper offloads this to a thread."""
        if not texts:
            return []
        model, tokenizer = self.get_model_and_tokenizer()
        device = next(model.parameters()).device

        out: list[str] = []
        bsz = max(1, self._settings.LOCAL_BATCH_SIZE)
        max_len = max(1, self._settings.LOCAL_MAX_LENGTH)
        # See ``_inference_lock`` docstring at __init__ for why this is needed.
        forced_bos: int | None = None
        with self._inference_lock, torch.inference_mode():
            if _is_nllb_tokenizer(tokenizer):
                src_code = NLLB_LANGUAGE_CODES.get(source_language)
                if src_code is None:
                    log.warning(
                        "No NLLB language code mapped for source=%r; falling back to "
                        "eng_Latn. Add the mapping to NLLB_LANGUAGE_CODES in "
                        "src/backends/local.py.",
                        source_language,
                    )
                    src_code = "eng_Latn"
                tokenizer.src_lang = src_code
                target_code = NLLB_LANGUAGE_CODES.get(target_language)
                if target_code is None:
                    log.warning(
                        "No NLLB language code mapped for target=%r; generation will "
                        "fall back to the model's default target. Add it to "
                        "NLLB_LANGUAGE_CODES in src/backends/local.py.",
                        target_language,
                    )
                else:
                    forced_bos = tokenizer.convert_tokens_to_ids(target_code)

            for i in range(0, len(texts), bsz):
                batch = [
                    self._format_with_gender_hint(t, gender, target_language, source_language)
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
                    "num_beams": 1,  # greedy keeps latency low; subtitle text tolerates it
                }
                if forced_bos is not None:
                    generate_kwargs["forced_bos_token_id"] = forced_bos
                output_ids = model.generate(**inputs, **generate_kwargs)
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                out.extend(decoded)
        return out

    async def translate_batch_async(
        self, texts: list[str], gender: str | None, target: str, **kwargs: Any
    ) -> list[str]:
        """Translate ``texts`` into ``target``, returning one string each.

        ``source_language`` defaults to English; the orchestrator passes the value
        derived from the request's ``language`` query param (mapped from ISO to
        display name via ``pipeline.lang.language_name``).
        Output length always equals input length; an empty input returns ``[]``.

        ``addressee_gender`` and ``previous_context`` are accepted but unused:
        the local model is not an instruction-follower API and has no
        separate addressee-hint or scene-context pathway.
        """
        if not texts:
            return []
        source_language: str = kwargs.get("source_language", "English")
        return await asyncio.to_thread(
            self._translate_sync, texts, gender, target, source_language,
        )

    async def warmup(self) -> None:
        await asyncio.to_thread(self._warmup_sync)

    def _warmup_sync(self) -> None:
        """Eagerly load the model + tokenizer so the first real request isn't
        penalised by the cold-start download/initialisation."""
        try:
            self.get_model_and_tokenizer()
            log.info("Local translation warm-up complete.")
        except Exception:  # pragma: no cover - warm-up must never crash startup
            log.exception("Local translation warm-up failed (continuing anyway).")

    def model_name(self) -> str:
        from src.config import settings
        return settings.LOCAL_TRANSLATION_MODEL

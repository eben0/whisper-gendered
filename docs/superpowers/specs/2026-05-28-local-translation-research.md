# Phase 1 — Local Translation Backend: Model Research

**Date:** 2026-05-28
**Goal:** Pick a HuggingFace en→he translation model that fits in ~2 GB VRAM alongside
faster-whisper large-v3 (~4.5 GB) + pyannote (~1.5 GB) on an RTX 2070 Super (8 GB).
**Source-of-truth direction:** English → Hebrew. Other languages may follow later.

## Candidates compared

| Model | Params | On-disk | fp16 VRAM (est.) | HF downloads/mo | Tokenizer | Notes |
|---|---|---|---|---|---|---|
| `Helsinki-NLP/opus-mt-en-he` | ~74M | ~300 MB | ~0.3 GB | 3,261 | SentencePiece | MarianMT seq2seq; direct en→he; simple tokenizer call, no language-tag plumbing |
| `facebook/nllb-200-distilled-600M` | 600M | 2.46 GB (fp32) | ~1.2 GB | 1,128,761 | SentencePiece | 200-language multilingual; needs `src_lang="eng_Latn"` + `tgt_lang="heb_Hebr"` |
| `facebook/nllb-200-distilled-1.3B` | 1.3B | ~5 GB (fp32) | ~2.6 GB | 204,000 | SentencePiece | Over the 2 GB target |
| `facebook/nllb-200-3.3B` | 3.3B | ~13 GB | ~6.6 GB | 223,000 | SentencePiece | Far over budget |
| `google/madlad400-3b-mt` | 3B | ~12 GB | ~6 GB | 289,000 | SentencePiece | Far over budget |
| `Mungert/Hunyuan-MT-7B-GGUF` | 7B | varies | n/a (GGUF) | 178,000 | n/a | Different runtime (llama.cpp), out of scope |

VRAM estimates are weights only; KV-cache during generation adds ~10–20% on top
for short subtitle segments. Net headroom alongside Whisper + pyannote, at fp16:
- `opus-mt-en-he`: ~6 GB free → very comfortable.
- `nllb-200-distilled-600M`: ~0.8 GB free → tight but workable.
- Anything ≥ 1.3B params: OOM risk.

## Quality

### Helsinki-NLP/opus-mt-en-he

The model card reports **BLEU 40.1 / chrF 0.609 on Tatoeba.en.he**. Tatoeba is
generally easier than FLORES devtest (shorter sentences, common phrasings), so
the equivalent on FLORES-200 would likely be lower (rough heuristic: −5 to −10
BLEU). For dialogue subtitles, Tatoeba is actually a reasonable proxy.

I could not find a published FLORES en→he number on the model card. The reverse
direction (he→en) has a "tc-big" successor (`Helsinki-NLP/opus-mt-tc-big-he-en`);
there is no equivalent `tc-big-en-he` for the en→he direction.

### facebook/nllb-200-distilled-600M

The NLLB paper reports chrF++ scores across all 40,602 directions on FLORES-200
devtest. The specific en→heb number was not visible in any of the model-card
excerpts I retrieved; the published metrics live at the
`https://tinyurl.com/nllb200densedst600mmetrics` link on the model card. The
**general expectation** from the NLLB paper is that the 600M distilled variant
loses ~2–4 chrF++ versus the 3.3B teacher; for a moderately-resourced language
like Hebrew this still typically beats the per-pair Helsinki-NLP MarianMT for
domain-general text, particularly on longer / lower-frequency sentences.

I'm deliberately marking the BLEU comparison as **non-rigorous**: without
running both models on the same FLORES devtest from this codebase, the
relative ranking is informed but not measured. Hebrew is morphologically rich
and the multilingual data NLLB was trained on includes substantially more
Hebrew than what Helsinki-NLP's OPUS-MT en-he saw.

## Inference speed (RTX 2070 Super, fp16)

I do not have a measured benchmark on this exact card for either model. The
rough operating expectations from the model architectures:

- `opus-mt-en-he` (encoder-decoder, ~74M): single short sentence ≈ 30–80 ms,
  batch of 16 short sentences ≈ 150–400 ms.
- `nllb-200-distilled-600M` (encoder-decoder, 600M): single short sentence ≈
  200–500 ms, batch of 16 ≈ 1.5–3 s.

For a feature-length episode (~900 segments), the local backend's wall-clock
will be dominated by the model's decoder forward passes, not the encoder. With
the chunked-pipeline architecture (`TRANSLATE_CONCURRENCY=3`) the local backend
will actually want **lower** concurrency than the Claude backend (the GPU
serializes work anyway and bigger batches per call are more efficient than
overlapping small calls). We'll surface this as a `LOCAL_BATCH_SIZE` config and
let the user tune.

Both models support batched inputs through the standard transformers API
(`model.generate(input_ids=tokenizer(batch, ...).input_ids)`).

## Recommendation

**Primary:** `facebook/nllb-200-distilled-600M`
- Best quality/size trade-off for en→he with our VRAM headroom.
- Massive community adoption (1.13M downloads/month) — the well-trodden path.
- Multilingual: if the user later switches `TARGET_LANGUAGE` to Arabic, French,
  Russian, etc., the same loaded model serves them; the per-pair Helsinki
  models would each need a separate download.
- Costs ~0.8 GB VRAM headroom; still safe with Whisper + pyannote resident.

**Fallback:** `Helsinki-NLP/opus-mt-en-he`
- Drops to ~6 GB VRAM headroom — useful if the user hits OOM on a long episode
  or wants to run on a smaller GPU.
- ~10× faster per sentence, ~7× smaller weights.
- Quality is lower for general-domain text but fine for plain dialogue
  subtitles, especially when latency matters more than nuance.
- One concrete deprecation note: `transformers v5` removed the
  `pipeline("translation")` shortcut; we'll use the explicit
  `AutoTokenizer` + `AutoModelForSeq2SeqLM` API which works on both v4 and v5.

## Defaults for the implementation

- `LOCAL_TRANSLATION_MODEL = facebook/nllb-200-distilled-600M`
- `LOCAL_TRANSLATION_DTYPE = float16` (cuts the 600M weights from ~2.4 GB to ~1.2 GB)
- `LOCAL_BATCH_SIZE = 16` — a good default for short subtitle segments
- `LOCAL_MAX_LENGTH = 512` — way above any subtitle line's actual length
- `LOCAL_TRANSLATION_DEVICE = cuda` (auto-fallback to `cpu` if CUDA unavailable)

Per the spec's hard constraint, the loader must check available VRAM before
moving the model to the device and **raise a clear error at startup** if the
weights wouldn't fit. We can read `torch.cuda.mem_get_info()` after Whisper +
pyannote warm-up to know what we have left.

## Sources

- [`Helsinki-NLP/opus-mt-en-he`](https://huggingface.co/Helsinki-NLP/opus-mt-en-he)
- [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [HF translation models filtered to `he`, sorted by downloads](https://huggingface.co/models?language=he&pipeline_tag=translation&sort=downloads)
- [NLLB-200 paper (multilingual chrF++ on FLORES-200)](https://huggingface.co/docs/transformers/en/model_doc/nllb)
- [Helsinki-NLP/OPUS-MT-leaderboard](https://github.com/Helsinki-NLP/OPUS-MT-leaderboard)

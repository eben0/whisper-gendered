# Gender-Aware Subtitle Server

A standalone, native-Windows HTTP server that acts as a **Bazarr-compatible Whisper ASR
provider**. It transcribes audio with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and — optionally — translates the result into another language via the Claude API.

For languages where speaker gender affects grammar (Hebrew, Arabic, French, …), it runs a full
pipeline that detects **who is speaking** and **their gender**, then translates each line with the
correct grammatical gender (verb conjugation, adjective agreement, imperatives, pronouns).

Drop it in anywhere you'd use `whisper-asr-webservice` or `subgen` — Bazarr's Whisper provider
works against it unmodified.

## How it works

```
POST /asr  (audio)
   │
   ├─ transcribe  (faster-whisper)                    → English segments + timestamps
   │
   ├─ TARGET_LANGUAGE=none ───────────────────────────→ return transcription as-is
   │
   ├─ gender-aware language (Hebrew, Arabic, …):
   │     diarize (pyannote) → detect gender (librosa pitch)
   │     → group consecutive same-speaker lines
   │     → translate each group with that speaker's gender (Claude)
   │
   └─ other language (Japanese, Turkish, …):
         translate (Claude, no gender) — skips diarization to save VRAM
```

Gender is detected per speaker from median fundamental frequency (F0) via `librosa.pyin`:
≤ `GENDER_THRESHOLD_HZ` (default 165 Hz) → male, otherwise female.

Translation is **batched**: consecutive lines from the same speaker go in one Claude call, which
preserves cross-line context and turns a ~1500-call feature film into a few dozen calls.

**Chunked overlap (translated output):** when translating, the file is transcribed
whole, then the resulting segments are split into ~`CHUNK_DURATION_SEC` chunks.
Per chunk, diarization runs on that slice while the previous chunk's translation
is still calling the Claude API — so GPU work overlaps the network-bound translate
phase. Per-chunk diarization also lowers peak VRAM (a few-minute slice instead of
the full file). Up to `TRANSLATE_CONCURRENCY` chunk translations run at once; the
Anthropic SDK retries transient errors (429/529/connection) up to
`CLAUDE_MAX_RETRIES` times.

## Requirements

- Windows with an NVIDIA GPU (developed for an RTX 2070, 8 GB VRAM). CPU works but is slow.
- Python 3.11+
- `ffmpeg` on `PATH` — only needed for requests with `encode=true`. Without it, send `encode=false`
  (the pipeline resamples internally). Bazarr typically sends `encode=true`, so install ffmpeg for
  production use.
- A [HuggingFace token](https://huggingface.co/settings/tokens) **and** acceptance of the licenses
  for **all three** gated models the diarization pipeline uses (only for gender-aware languages).
  A missing acceptance on any of them shows up as a `403 GatedRepoError` on the first gender-aware
  request:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)  ← easy to miss;
    diarization-3.1 depends on it.
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
    ← pyannote.audio 4.x loads this model's PLDA inside `speaker-diarization-3.1`; also easy to miss.
- An [Anthropic API key](https://console.anthropic.com/) (only needed when translating)

With `large-v3` + `float16`, Whisper uses ~4.5 GB VRAM and pyannote adds ~1–1.5 GB — ~6 GB total,
fits in 8 GB. Keep `CONCURRENT_JOBS=1`.

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the GPU is visible to torch (should print True)
python -c "import torch; print(torch.cuda.is_available())"

# 4. Create your config
Copy-Item .env.example .env   # then edit .env (see below)
```

### GPU (CUDA) torch — important

Plain `pip install torch` on Windows installs a **CPU-only** build, so step 3 prints `False`.
You must install a CUDA build that matches your driver from PyTorch's CUDA index. For a recent
NVIDIA driver (CUDA 13.x), the matching wheels are `+cu130`:

```powershell
# Replace the CPU build with the CUDA 13.0 build (torch + matching torchaudio)
pip install --force-reinstall --no-deps `
  --index-url https://download.pytorch.org/whl/cu130 `
  torch==2.11.0+cu130 torchaudio==2.11.0+cu130

# Re-verify — should now print True and your GPU name
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Notes:
- `torch` and `torchaudio` versions **must match** (e.g. both `2.11.0+cu130`), or imports fail.
- For other CUDA versions, swap the index URL (e.g. `.../whl/cu128`) and pick a version that exists
  on that index — check with
  `pip index versions torch --index-url https://download.pytorch.org/whl/cu130`.
- If `torch.cuda.is_available()` stays `False`, your `pip`/`python` is probably pointing at a
  different environment than your `.venv` — confirm with `python -c "import sys; print(sys.executable)"`.

### cuBLAS + cuDNN for faster-whisper (CTranslate2)

faster-whisper transcribes through **CTranslate2**, not torch, and CTranslate2 needs the **CUDA 12**
cuBLAS + cuDNN runtime libraries (independent of the torch CUDA build above — CTranslate2 targets
CUDA 12, and a modern driver runs them fine). Without them, the first `/asr` request fails with an
error like `Library cublas64_12.dll is not found` or `Library cudnn_ops not found`. Install:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

faster-whisper adds these pip-installed library directories to the DLL search path automatically.
If the DLL is still not found, prepend them to `PATH` before launching (`.venv` site-packages):
`$env:PATH = "$PWD\.venv\Lib\site-packages\nvidia\cublas\bin;$PWD\.venv\Lib\site-packages\nvidia\cudnn\bin;$env:PATH"`.

### Note on torchcodec / pyannote audio decoding

pyannote 4.x decodes audio via `torchcodec`, which needs FFmpeg "full-shared" DLLs on Windows. To
avoid that dependency, this server loads the audio itself (via `soundfile`) and hands pyannote an
in-memory `{waveform, sample_rate}` tensor (see `pipeline/diarize.py`). The `torchcodec is not
installed correctly` warning at startup is therefore harmless — that decode path is never used.

(pyannote diarization uses torch, so the gender-aware path is covered by the CUDA torch install
above — this step only affects transcription.)

## Configuration

All settings come from environment variables (or a `.env` file). See `.env.example`.

| Env var              | Default              | Description                                                            |
|----------------------|----------------------|------------------------------------------------------------------------|
| `PORT`               | `9000`               | HTTP port                                                              |
| `WHISPER_MODEL`      | `large-v3`           | faster-whisper model name                                              |
| `COMPUTE_TYPE`       | `float16`            | CTranslate2 quantisation                                               |
| `DEVICE`             | `cuda`               | `cuda` or `cpu`                                                        |
| `CONCURRENT_JOBS`    | `1`                  | Max parallel GPU jobs                                                  |
| `TARGET_LANGUAGE`    | `none`               | Language name to translate into (`Hebrew`, `French`, …), or `none`     |
| `HF_TOKEN`           | —                    | HuggingFace token for pyannote (required for gender-aware languages)   |
| `ANTHROPIC_API_KEY`  | —                    | Anthropic API key (required when translating)                          |
| `CLAUDE_MODEL`       | `claude-sonnet-4-6`  | Claude model used for translation                                      |
| `GENDER_THRESHOLD_HZ`| `165`                | F0 (Hz) boundary between male and female                               |
| `CHUNK_DURATION_SEC` | `300`                | Target seconds of audio per pipeline chunk                            |
| `TRANSLATE_CONCURRENCY`| `3`                | Max chunk translations running concurrently (Claude)                  |
| `CLAUDE_MAX_RETRIES` | `4`                  | Anthropic SDK auto-retry attempts on 429/529/connection errors        |
| `ADDRESSEE_GENDER_HINT_ENABLED` | `true`    | Pass the prior group's speaker gender as an addressee hint to Claude  |
| `SAVE_SRT_VIDEO_PREFIX`| —                  | Client's view of the share root (e.g. `/media`). Empty = disabled     |
| `SAVE_SRT_LOCAL_PREFIX`| —                  | Server's view of the same root (e.g. `Z:\media`). Empty = disabled    |
| `SAVE_SRT_SUFFIX`    | `.he.srt`            | Replaces the video extension on the written side-file                 |
| `DEBUG`              | `false`              | Verbose logging                                                        |

**Gender-aware languages** (full diarization + gender pipeline): Hebrew, Arabic, French, Spanish,
Italian, Portuguese, German, Russian, Polish, Ukrainian, Hindi, Romanian. Any other
`TARGET_LANGUAGE` translates without diarization. `none` disables translation entirely (plain
Whisper ASR server).

**Side-file save (optional).** Bazarr names its saved subtitle by the source-audio language
(e.g. `…en.srt`), regardless of what the server actually returned. When `TARGET_LANGUAGE`
overrides the output language (Hebrew here), the filename mislabels its contents. Set
`SAVE_SRT_VIDEO_PREFIX` and `SAVE_SRT_LOCAL_PREFIX` to enable a parallel save: the server
takes Bazarr's `video_file` query parameter, translates the path from the client's view of
the share to the server's view, and writes a copy of the SRT with `SAVE_SRT_SUFFIX` next to
the source video. Failures are logged but never break the HTTP response.

## Running

```powershell
pwsh ./run.ps1
```

`run.ps1` prints your LAN IP and the server URL, then starts the server. The first request is
warmed up at startup so it isn't penalized by CUDA/model load time.

To run directly: `python server.py`.

## Endpoints

### `GET /`
Liveness check.
```json
{ "status": "ok", "version": "1.0.0", "model": "large-v3" }
```

### `GET /status`
```json
{ "status": "ok", "queue_depth": 0, "model_loaded": true }
```

### `GET /docs`
Auto-generated FastAPI docs.

### `POST /asr`
`multipart/form-data` — the Bazarr Whisper contract.

| Field        | Type   | Default        | Notes                                          |
|--------------|--------|----------------|------------------------------------------------|
| `audio_file` | file   | —              | The audio/video bytes                          |
| `task`       | string | `transcribe`   | `transcribe` or `translate`                    |
| `language`   | string | `en`           | Source language (ISO 639-1)                    |
| `output`     | string | `srt`          | `srt` \| `vtt` \| `txt` \| `tsv` \| `json`     |
| `encode`     | bool   | `true`         | Re-encode to 16 kHz mono WAV with ffmpeg first |

Returns the raw subtitle content (UTF-8) in the requested format.

```powershell
curl.exe -F "audio_file=@clip.mp4" -F "output=srt" http://localhost:9000/asr
```

## Connecting Bazarr

In Bazarr, configure the **Whisper** provider and point its endpoint at
`http://<LAN_IP>:9000`. No code changes are needed — the `/asr` contract matches
`whisper-asr-webservice`. Set `TARGET_LANGUAGE` server-side to control the output language.

## Project layout

```
config.py            Settings + GENDER_AWARE_LANGUAGES
server.py            FastAPI app, /asr orchestration, warm-up, concurrency
pipeline/
  transcribe.py      faster-whisper wrapper (lazy singleton)
  diarize.py         pyannote speaker diarization
  gender.py          pitch-based gender detection
  translate.py       batched Claude translation (structured outputs)
  format.py          SRT/VTT/TXT/TSV/JSON renderers
run.ps1              PowerShell launcher (prints LAN IP)
```

## Notes

- Models load lazily on first request and are reused across requests (never reloaded per call).
- All ML inference and Claude calls run in a thread pool so the async event loop never blocks; a
  semaphore (`CONCURRENT_JOBS`) gates GPU work.
- Temp audio files are always cleaned up, even on error.
- No Docker — native Windows Python.

"""Bazarr-compatible Whisper ASR server — HTTP wiring only.

Endpoints: GET /, GET /status, POST /asr
"""

from __future__ import annotations

# ORDERING-CRITICAL: bootstrap before pyannote/orchestrator imports
from src.core.cuda import Cuda as _CudaClass
_cuda = _CudaClass()
_cuda.bootstrap()

from src.core.logging_config import configure as _configure_logging
_configure_logging()

import logging
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from src.config import settings
from src.core.concurrency import ConcurrencyManager
from src.backends.factory import create_backend
from src.audio import Audio
from src.side_file import SideFile
from src.lifecycle import Lifecycle
from src.orchestrator import Orchestrator
from src.asr_handler import AsrHandler
from pipeline.transcribe import Transcriber
from pipeline.diarize import Diarizer
from pipeline.gender import GenderDetector, GenderMLClassifier

log = logging.getLogger("server")
VERSION = "1.0.0"

# Construct all singletons at startup
_concurrency = ConcurrencyManager(settings.CONCURRENT_JOBS)
_backend = create_backend(settings)
_audio = Audio()
_side_file = SideFile(settings)
_transcriber = Transcriber(settings)
_diarizer = Diarizer(settings)
_gender_ml_classifier = GenderMLClassifier(settings)
_gender_detector = GenderDetector(settings, gender_ml=_gender_ml_classifier)
_orchestrator = Orchestrator(_concurrency, _audio, _backend, _transcriber, _diarizer, _gender_detector)
_lifecycle = Lifecycle(_concurrency, _audio, _backend, settings, _transcriber, _diarizer, _gender_ml_classifier)
_asr_handler = AsrHandler(_concurrency, _cuda, _audio, _side_file, _orchestrator)

app = FastAPI(title="Gender-Aware Hebrew Subtitle Server", version=VERSION)


@app.on_event("startup")
async def warmup() -> None:
    await _lifecycle.warmup()


@app.get("/")
async def root():
    return JSONResponse(
        {"status": "ok", "version": VERSION, "model": settings.WHISPER_MODEL}
    )


@app.get("/status")
async def status():
    return JSONResponse({
        "status": "ok",
        "queue_depth": _concurrency.job_depth(),
        "model_loaded": _transcriber.model_loaded(),
    })


@app.post("/asr")
async def asr(
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe"),
    language: str = Query("en"),
    output: str = Query("srt"),
    encode: bool = Query(True),
    video_file: str = Query(""),
):
    return await _asr_handler.handle(audio_file, task, language, output, encode, video_file=video_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=settings.PORT, workers=1)

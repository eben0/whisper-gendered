"""Bazarr-compatible Whisper ASR server — HTTP wiring only."""

from __future__ import annotations

# ORDERING-CRITICAL: bootstrap before any pyannote/pipeline imports
from src.core.cuda import Cuda as _CudaClass
_cuda = _CudaClass()
_cuda.bootstrap()

from src.core.logging_config import configure as _configure_logging
_configure_logging()

import logging
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from src.config import settings
from src.asr_handler import AsrHandler

log = logging.getLogger("server")
VERSION = "1.0.0"

_handler = AsrHandler(settings)

app = FastAPI(title="Gender-Aware Hebrew Subtitle Server", version=VERSION)


@app.on_event("startup")
async def warmup() -> None:
    await _handler.warmup()


@app.get("/")
async def root():
    return JSONResponse(
        {"status": "ok", "version": VERSION, "model": settings.WHISPER_MODEL}
    )


@app.get("/status")
async def status():
    return JSONResponse({
        "status": "ok",
        "queue_depth": _handler.job_depth(),
        "model_loaded": _handler.model_loaded(),
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
    return await _handler.handle(audio_file, task, language, output, encode, video_file=video_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=settings.PORT, workers=1)

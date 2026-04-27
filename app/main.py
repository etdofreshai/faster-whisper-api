import io
import os
import tempfile
from typing import Optional, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Depends
from fastapi.responses import PlainTextResponse, JSONResponse, Response
from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DOWNLOAD_ROOT = os.getenv("WHISPER_DOWNLOAD_ROOT", "/models")
API_KEY = os.getenv("API_KEY")

app = FastAPI(title="faster-whisper OpenAI-compatible API")

_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=DOWNLOAD_ROOT,
        )
    return _model


def verify_api_key(authorization: Optional[str] = Header(None)):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _format_timestamp(seconds: float, vtt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _to_srt(segments) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(str(i))
        out.append(f"{_format_timestamp(seg['start'])} --> {_format_timestamp(seg['end'])}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def _to_vtt(segments) -> str:
    out = ["WEBVTT", ""]
    for seg in segments:
        out.append(f"{_format_timestamp(seg['start'], vtt=True)} --> {_format_timestamp(seg['end'], vtt=True)}")
        out.append(seg["text"].strip())
        out.append("")
    return "\n".join(out)


def _build_response(
    segments_iter,
    info,
    response_format: str,
    timestamp_granularities: List[str],
    translate: bool,
):
    segments = []
    full_text_parts = []
    for s in segments_iter:
        seg = {
            "id": s.id,
            "seek": s.seek,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "tokens": s.tokens,
            "temperature": s.temperature,
            "avg_logprob": s.avg_logprob,
            "compression_ratio": s.compression_ratio,
            "no_speech_prob": s.no_speech_prob,
        }
        if s.words:
            seg["words"] = [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in s.words
            ]
        segments.append(seg)
        full_text_parts.append(s.text)

    text = "".join(full_text_parts).strip()

    if response_format == "text":
        return PlainTextResponse(text + "\n")
    if response_format == "srt":
        return PlainTextResponse(_to_srt(segments), media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(_to_vtt(segments), media_type="text/vtt")
    if response_format == "verbose_json":
        body = {
            "task": "translate" if translate else "transcribe",
            "language": info.language,
            "duration": info.duration,
            "text": text,
            "segments": segments,
        }
        if "word" in timestamp_granularities:
            words = []
            for seg in segments:
                for w in seg.get("words", []):
                    words.append(w)
            body["words"] = words
        return JSONResponse(body)
    return JSONResponse({"text": text})


def _run(
    file_bytes: bytes,
    filename: str,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    temperature: float,
    timestamp_granularities: List[str],
    translate: bool,
):
    valid_formats = {"json", "text", "srt", "verbose_json", "vtt"}
    if response_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Invalid response_format: {response_format}")

    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        whisper = get_model()
        word_timestamps = "word" in timestamp_granularities or response_format in {"srt", "vtt"}
        segments_iter, info = whisper.transcribe(
            tmp_path,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            task="translate" if translate else "transcribe",
            word_timestamps=word_timestamps,
            vad_filter=os.getenv("WHISPER_VAD", "false").lower() == "true",
        )
        return _build_response(segments_iter, info, response_format, timestamp_granularities, translate)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE, "device": DEVICE, "compute_type": COMPUTE_TYPE}


@app.get("/v1/models")
def list_models(_: None = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {"id": "whisper-1", "object": "model", "owned_by": "openai-compat"},
            {"id": MODEL_SIZE, "object": "model", "owned_by": "faster-whisper"},
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[List[str]] = Form(None, alias="timestamp_granularities[]"),
    _: None = Depends(verify_api_key),
):
    data = await file.read()
    return _run(
        data, file.filename or "audio",
        model, language, prompt, response_format, temperature,
        timestamp_granularities or ["segment"], translate=False,
    )


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    _: None = Depends(verify_api_key),
):
    data = await file.read()
    return _run(
        data, file.filename or "audio",
        model, None, prompt, response_format, temperature,
        ["segment"], translate=True,
    )

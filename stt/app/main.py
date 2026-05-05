import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import struct
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, cast

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field


app = FastAPI(title="OpenAI-compatible Qwen3-TTS + Faster Whisper Server")

_stt_model: WhisperModel | None = None
_stt_lock = threading.Lock()

logger = logging.getLogger("tts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_tts_warmup_started = False


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


@app.on_event("startup")
async def startup_event() -> None:
    global _tts_warmup_started
    if _tts_warmup_started:
        return
    _tts_warmup_started = True
    await _warmup_tts_upstream()


def get_stt_model() -> WhisperModel:
    global _stt_model
    if _stt_model is None:
        with _stt_lock:
            if _stt_model is None:
                _stt_model = WhisperModel(
                    _env("STT_MODEL_SIZE", "base"),
                    device=_env("STT_DEVICE", "cpu"),
                    compute_type=_env("STT_COMPUTE_TYPE", "int8"),
                    download_root=_env("STT_DOWNLOAD_ROOT", "/models/faster-whisper"),
                )
    return _stt_model


def _fish_base_url() -> str:
    return _env("COSY_TTS_BASE_URL", _env("FISH_TTS_BASE_URL", "http://cosyvoice:50000")).rstrip("/")


def _fish_model_name() -> str:
    return _env("COSY_TTS_MODEL", _env("FISH_TTS_MODEL", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"))


def _fish_timeout() -> float:
    return float(_env("COSY_TTS_TIMEOUT", _env("FISH_TTS_TIMEOUT", "600")))


def _fish_chunk_bytes() -> int:
    return int(_env("COSY_TTS_STREAM_CHUNK_BYTES", _env("FISH_TTS_STREAM_CHUNK_BYTES", "8192")))


def _tts_upstream_base_url() -> str:
    return _env("TTS_UPSTREAM_BASE_URL", _env("QWEN_TTS_BASE_URL", "http://qwen3-tts:8091")).rstrip("/")


def _tts_upstream_model_name() -> str:
    return _env("TTS_UPSTREAM_MODEL", _env("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))


def _tts_upstream_timeout() -> float:
    return float(_env("TTS_UPSTREAM_TIMEOUT", _env("QWEN_TTS_TIMEOUT", _env("COSY_TTS_TIMEOUT", "600"))))


def _tts_upstream_api_key() -> str:
    return _env("TTS_UPSTREAM_API_KEY", _env("QWEN_TTS_API_KEY", "EMPTY"))


def _tts_upstream_headers() -> dict[str, str]:
    api_key = _tts_upstream_api_key()
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _tts_warmup_enabled() -> bool:
    return _env("TTS_WARMUP_ENABLED", "1") != "0"


def _tts_warmup_text() -> str:
    return _env("TTS_WARMUP_TEXT", "こんにちは。")


def _default_task_type() -> Literal["CustomVoice", "VoiceDesign", "Base"]:
    task_type = _env("TTS_DEFAULT_TASK_TYPE", _env("QWEN_TTS_DEFAULT_TASK_TYPE", "CustomVoice"))
    if task_type not in {"CustomVoice", "VoiceDesign", "Base"}:
        task_type = "CustomVoice"
    return cast(Literal["CustomVoice", "VoiceDesign", "Base"], task_type)


def _to_pcm16(audio: np.ndarray) -> bytes:
    return np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    channels = 1
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _complete_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    raw = _to_pcm16(audio)
    return _wav_header(sample_rate, len(raw)) + raw


def _mp3_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    from pydub import AudioSegment

    segment = AudioSegment(_to_pcm16(audio), frame_rate=sample_rate, sample_width=2, channels=1)
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


def _read_audio_file(audio_path: str) -> bytes:
    path = Path(audio_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"参照音声が見つかりません: {audio_path}")
    return path.read_bytes()


def _audio_reference_to_data_url(audio_ref: str) -> str:
    if audio_ref.startswith(("http://", "https://", "data:")):
        return audio_ref

    audio_bytes = _read_audio_file(audio_ref)
    mime_type, _encoding = mimetypes.guess_type(audio_ref)
    if not mime_type:
        mime_type = "audio/wav"
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _load_voices() -> dict[str, dict[str, Any]]:
    voices_file = Path(_env("COSY_TTS_VOICES_FILE", _env("FISH_TTS_VOICES_FILE", _env("QWEN_TTS_VOICES_FILE", "/voices/voices.json"))))
    if voices_file.exists():
        with voices_file.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise HTTPException(status_code=500, detail="voices.json must be an object")
        return loaded

    ref_audio = _env("COSY_TTS_REF_AUDIO", _env("FISH_TTS_REF_AUDIO", _env("QWEN_TTS_REF_AUDIO", "")))
    if ref_audio:
        return {
            "alloy": {
                "ref_audio": ref_audio,
                "ref_text": _env("COSY_TTS_REF_TEXT", _env("FISH_TTS_REF_TEXT", _env("QWEN_TTS_REF_TEXT", ""))),
                "language": _env("COSY_TTS_LANGUAGE", _env("FISH_TTS_LANGUAGE", _env("QWEN_TTS_LANGUAGE", "Japanese"))),
            }
        }
    return {}


def _voice_config(voice: str) -> dict[str, Any]:
    voices = _load_voices()
    if voice in voices:
        return voices[voice]
    if voices:
        first_name = next(iter(voices))
        return voices[first_name]
    return {}


def _voice_name(voice: str | dict[str, str]) -> str:
    if isinstance(voice, dict):
        return voice.get("id") or voice.get("name") or "alloy"
    return voice


def _fish_reference_payload(voice: str | dict[str, str]) -> tuple[str | None, str, bytes | None]:
    voice_name = _voice_name(voice)
    voice_cfg = _voice_config(voice_name)
    if not voice_cfg:
        return None, "", None

    references = voice_cfg.get("references")
    if references:
        reference = references[0]
        return voice_name, reference.get("ref_text", ""), _read_audio_file(reference["ref_audio"])

    ref_audio = voice_cfg.get("ref_audio")
    if not ref_audio:
        return voice_name, voice_cfg.get("ref_text", ""), None

    return voice_name, voice_cfg.get("ref_text", ""), _read_audio_file(ref_audio)


def _cosy_prompt_text(prompt_text: str) -> str:
    prompt_text = prompt_text.strip()
    if not prompt_text:
        return prompt_text
    if "<|endofprompt|>" in prompt_text:
        return prompt_text

    default_prefix = ""
    if "Fun-CosyVoice3" in _fish_model_name():
        default_prefix = "You are a helpful assistant.<|endofprompt|>"
    prefix = _env("COSY_TTS_PROMPT_PREFIX", default_prefix)
    return f"{prefix}{prompt_text}" if prefix else prompt_text


def _cosy_instruct_text(instruct_text: str) -> str:
    instruct_text = instruct_text.strip()
    if not instruct_text:
        return instruct_text
    if "<|endofprompt|>" in instruct_text:
        return instruct_text

    default_prefix = ""
    if "Fun-CosyVoice3" in _fish_model_name():
        default_prefix = "You are a helpful assistant. "
    prefix = _env("COSY_TTS_INSTRUCT_PREFIX", default_prefix)
    return f"{prefix}{instruct_text}<|endofprompt|>"


def _fish_payload(req: "SpeechRequest") -> tuple[str, dict[str, Any], dict[str, tuple[str, bytes, str]]]:
    endpoint = _env("COSY_TTS_ENDPOINT", "inference_zero_shot")
    voice_name, prompt_text, prompt_audio = _fish_reference_payload(req.voice)
    data: dict[str, Any] = {"tts_text": req.input}
    files: dict[str, tuple[str, bytes, str]] = {}

    if endpoint == "inference_sft":
        data["spk_id"] = voice_name or _env("COSY_TTS_SPK_ID", "中文女")
    elif endpoint == "inference_instruct":
        data["spk_id"] = voice_name or _env("COSY_TTS_SPK_ID", "中文女")
        data["instruct_text"] = _cosy_instruct_text(req.instructions or _env("COSY_TTS_INSTRUCT_TEXT", ""))
    elif endpoint == "inference_instruct2":
        if not prompt_audio:
            raise HTTPException(status_code=400, detail="CosyVoice instruct2 requires a prompt audio")
        data["instruct_text"] = _cosy_instruct_text(req.instructions or prompt_text or _env("COSY_TTS_INSTRUCT_TEXT", ""))
        files["prompt_wav"] = ("prompt.wav", prompt_audio, "audio/wav")
    elif endpoint == "inference_cross_lingual":
        if not prompt_audio:
            raise HTTPException(status_code=400, detail="CosyVoice cross_lingual requires a prompt audio")
        files["prompt_wav"] = ("prompt.wav", prompt_audio, "audio/wav")
    else:
        if not prompt_audio:
            raise HTTPException(status_code=400, detail="CosyVoice zero_shot requires a prompt audio")
        data["prompt_text"] = _cosy_prompt_text(prompt_text)
        files["prompt_wav"] = ("prompt.wav", prompt_audio, "audio/wav")

    return endpoint, data, files


async def _fish_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{_fish_base_url()}/docs")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def _fish_complete_audio(req: "SpeechRequest", fmt: str) -> bytes:
    try:
        endpoint, data, files = _fish_payload(req)
        async with httpx.AsyncClient(timeout=_fish_timeout()) as client:
            response = await client.post(
                f"{_fish_base_url()}/{endpoint}",
                data=data,
                files=files,
            )
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"CosyVoice TTS error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"CosyVoice backend unreachable: {exc}") from exc


def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    backend_format = _env("COSY_TTS_BACKEND_FORMAT", "pcm16")
    if backend_format in {"pcm16", "raw", "s16le"}:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, int(_env("COSY_TTS_SAMPLE_RATE", "24000"))

    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sample_rate)


async def _pcm_stream_from_audio(audio: np.ndarray, chunk_size: int) -> AsyncGenerator[bytes, None]:
    step = max(1, chunk_size)
    for start in range(0, len(audio), step):
        yield _to_pcm16(audio[start : start + step])


def _realtime_event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def _response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _item_id() -> str:
    return f"item_{uuid.uuid4().hex}"


def _extract_text_from_message(message: dict[str, Any]) -> str:
    item = message.get("item") or {}
    content = item.get("content") or []
    parts: list[str] = []
    for entry in content:
        if isinstance(entry, dict):
            text = entry.get("text") or entry.get("input_text") or ""
            if text:
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _extract_text_from_response(message: dict[str, Any], fallback: str) -> str:
    response = message.get("response") or {}
    direct = response.get("instructions") or response.get("input_text") or response.get("text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    inputs = response.get("input") or []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        for entry in content:
            if isinstance(entry, dict):
                text = entry.get("text") or entry.get("input_text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return fallback


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(..., min_length=1)
    voice: str | dict[str, str] = Field(default_factory=lambda: _env("TTS_DEFAULT_VOICE", _env("QWEN_TTS_DEFAULT_VOICE", "Ono_Anna")))
    instructions: str = ""
    response_format: str = "wav"
    stream_format: Literal["audio", "sse"] = "audio"
    speed: float = 1.0
    stream: bool = True
    chunk_size: int | None = Field(default=None, ge=1)
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] = Field(default_factory=_default_task_type)
    language: str = Field(default_factory=lambda: _env("TTS_DEFAULT_LANGUAGE", _env("QWEN_TTS_DEFAULT_LANGUAGE", "Japanese")))
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool = False
    max_new_tokens: int | None = Field(default=None, ge=1)


class RealtimeSession(BaseModel):
    model: str = "tts-1"
    voice: str = Field(default_factory=lambda: _env("TTS_DEFAULT_VOICE", _env("QWEN_TTS_DEFAULT_VOICE", "Ono_Anna")))
    instructions: str = ""
    input_text: str = ""
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] = Field(default_factory=_default_task_type)
    language: str = Field(default_factory=lambda: _env("TTS_DEFAULT_LANGUAGE", _env("QWEN_TTS_DEFAULT_LANGUAGE", "Japanese")))


def _tts_request_payload(
    req: "SpeechRequest",
    *,
    response_format: str | None = None,
    stream: bool | None = None,
) -> dict[str, Any]:
    use_stream = req.stream if stream is None else stream
    payload: dict[str, Any] = {
        "input": req.input,
        "response_format": response_format or req.response_format.lower(),
    }

    model_name = req.model.strip()
    if model_name and model_name != "tts-1":
        payload["model"] = model_name

    if use_stream:
        payload["stream"] = True
    elif req.speed != 1.0:
        payload["speed"] = req.speed

    if req.task_type:
        payload["task_type"] = req.task_type
    if req.language.strip():
        payload["language"] = req.language.strip()
    if req.instructions.strip():
        payload["instructions"] = req.instructions.strip()
    if req.max_new_tokens is not None:
        payload["max_new_tokens"] = req.max_new_tokens

    voice_name = _voice_name(req.voice)
    if req.task_type != "VoiceDesign" and voice_name:
        payload["voice"] = voice_name

    if req.task_type == "Base":
        ref_audio = req.ref_audio
        ref_text = req.ref_text or ""
        if not ref_audio:
            voice_cfg = _voice_config(voice_name)
            ref_audio = voice_cfg.get("ref_audio")
            ref_text = ref_text or voice_cfg.get("ref_text", "")

        if ref_audio:
            payload["ref_audio"] = _audio_reference_to_data_url(ref_audio)
        if ref_text:
            payload["ref_text"] = ref_text
        if req.x_vector_only_mode:
            payload["x_vector_only_mode"] = True

    return payload


async def _tts_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{_tts_upstream_base_url()}/health", headers=_tts_upstream_headers())
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def _warmup_tts_upstream() -> None:
    if not _tts_warmup_enabled():
        logger.info("TTS upstream warmup disabled")
        return

    for _attempt in range(60):
        if await _tts_health():
            break
        await asyncio.sleep(2)
    else:
        logger.warning("TTS upstream warmup skipped because upstream health never became ready")
        return

    payload = {
        "model": _tts_upstream_model_name(),
        "input": _tts_warmup_text(),
        "voice": _env("TTS_DEFAULT_VOICE", _env("QWEN_TTS_DEFAULT_VOICE", "Ono_Anna")),
        "task_type": _default_task_type(),
        "language": _env("TTS_DEFAULT_LANGUAGE", _env("QWEN_TTS_DEFAULT_LANGUAGE", "Japanese")),
        "response_format": "wav",
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=_tts_upstream_timeout()) as client:
            response = await client.post(
                f"{_tts_upstream_base_url()}/v1/audio/speech",
                json=payload,
                headers=_tts_upstream_headers(),
            )
            response.raise_for_status()
        logger.info("TTS upstream warmup completed")
    except httpx.HTTPError as exc:
        logger.warning("TTS upstream warmup failed: %s", exc)


async def _tts_request_with_failover(method: str, path: str, **kwargs: Any) -> httpx.Response:
    headers = {**_tts_upstream_headers(), **(kwargs.pop("headers", {}) or {})}
    try:
        async with httpx.AsyncClient(timeout=_tts_upstream_timeout()) as client:
            response = await client.request(method, f"{_tts_upstream_base_url()}{path}", headers=headers, **kwargs)
            response.raise_for_status()
            return response
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Qwen3-TTS upstream error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Qwen3-TTS upstream unreachable: {exc}") from exc


async def _tts_complete_audio(req: "SpeechRequest", response_format: str | None = None) -> bytes:
    payload = _tts_request_payload(req, response_format=response_format, stream=False)
    response = await _tts_request_with_failover("POST", "/v1/audio/speech", json=payload)
    return response.content


async def _tts_stream_audio(req: "SpeechRequest") -> AsyncGenerator[bytes, None]:
    payload = _tts_request_payload(req, response_format="pcm", stream=True)
    headers = _tts_upstream_headers()
    chunk_bytes = int(_env("TTS_UPSTREAM_STREAM_CHUNK_BYTES", str(_fish_chunk_bytes())))
    try:
        async with httpx.AsyncClient(timeout=_tts_upstream_timeout()) as client:
            async with client.stream(
                "POST",
                f"{_tts_upstream_base_url()}/v1/audio/speech",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=chunk_bytes):
                    if chunk:
                        yield chunk
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Qwen3-TTS upstream error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Qwen3-TTS upstream unreachable: {exc}") from exc


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_model_loaded": _stt_model is not None,
        "tts_upstream_backend": _tts_upstream_base_url(),
        "tts_upstream_model": _tts_upstream_model_name(),
        "tts_upstream_healthy": await _tts_health(),
    }


@app.get("/v1/models")
async def models():
    data = [
        {"id": "whisper-1", "object": "model", "owned_by": "local"},
        {"id": "tts-1", "object": "model", "owned_by": "local"},
    ]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{_tts_upstream_base_url()}/v1/models", headers=_tts_upstream_headers())
            response.raise_for_status()
            payload = response.json()
            upstream_models = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(upstream_models, list):
                data.extend(upstream_models)
    except (httpx.HTTPError, ValueError):
        data.append({"id": _tts_upstream_model_name(), "object": "model", "owned_by": "local"})

    return {"object": "list", "data": data}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: Literal["json", "text", "verbose_json"] = Form(default="json"),
    temperature: float = Form(default=0),
    timestamp_granularities: list[str] | None = Form(default=None),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    word_timestamps = bool(timestamp_granularities and "word" in timestamp_granularities)
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            segments_iter, info = get_stt_model().transcribe(
                tmp.name,
                language=language or None,
                initial_prompt=prompt,
                temperature=temperature,
                word_timestamps=word_timestamps,
            )
            segments = []
            text_parts = []
            for segment in segments_iter:
                item = {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob,
                }
                if word_timestamps and segment.words:
                    item["words"] = [
                        {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                        for w in segment.words
                    ]
                segments.append(item)
                text_parts.append(segment.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    text = "".join(text_parts).strip()
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": info.language, "duration": info.duration, "text": text, "segments": segments}
    return {"text": text}


@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    prompt: str | None = Form(default=None),
    response_format: Literal["json", "text"] = Form(default="json"),
    temperature: float = Form(default=0),
):
    result = await create_transcription(
        file=file,
        model=model,
        language=None,
        prompt=prompt,
        response_format="json",
        temperature=temperature,
        timestamp_granularities=None,
    )
    if response_format == "text":
        return PlainTextResponse(result["text"])
    return result


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    fmt = req.response_format.lower()
    content_types = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "aac": "audio/aac",
        "opus": "audio/ogg",
    }

    if fmt not in content_types:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {fmt}")

    if not req.stream:
        audio_bytes = await _tts_complete_audio(req, response_format=fmt)
        return Response(audio_bytes, media_type=content_types[fmt])

    async def stream_audio():
        if fmt == "pcm":
            async for chunk in _tts_stream_audio(req):
                yield chunk
            return

        audio_bytes = await _tts_complete_audio(req, response_format=fmt)
        yield audio_bytes

    async def stream_sse():
        async for chunk in stream_audio():
            yield b"data: "
            yield json.dumps({"type": "audio.delta", "audio": base64.b64encode(chunk).decode("utf-8")}).encode("utf-8")
            yield b"\n\n"
        yield b"data: {\"type\":\"audio.done\"}\n\n"

    if req.stream_format == "sse":
        return StreamingResponse(stream_sse(), media_type="text/event-stream")
    return StreamingResponse(stream_audio(), media_type=content_types[fmt])


@app.websocket("/v1/realtime")
async def realtime_socket(websocket: WebSocket):
    await websocket.accept()
    session = RealtimeSession()

    await websocket.send_json(
        {
            "type": "session.created",
            "event_id": _realtime_event_id(),
            "session": session.model_dump(),
        }
    )

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "session.update":
                payload = message.get("session") or {}
                if isinstance(payload.get("model"), str) and payload["model"].strip():
                    session.model = payload["model"].strip()
                if isinstance(payload.get("voice"), str) and payload["voice"].strip():
                    session.voice = payload["voice"].strip()
                if isinstance(payload.get("instructions"), str):
                    session.instructions = payload["instructions"]
                if isinstance(payload.get("task_type"), str) and payload["task_type"] in {"CustomVoice", "VoiceDesign", "Base"}:
                    session.task_type = payload["task_type"]
                if isinstance(payload.get("language"), str) and payload["language"].strip():
                    session.language = payload["language"].strip()
                await websocket.send_json(
                    {
                        "type": "session.updated",
                        "event_id": _realtime_event_id(),
                        "session": session.model_dump(),
                    }
                )
                continue

            if msg_type == "conversation.item.create":
                text = _extract_text_from_message(message)
                if text:
                    session.input_text = text
                await websocket.send_json(
                    {
                        "type": "conversation.item.created",
                        "event_id": _realtime_event_id(),
                        "item": message.get("item") or {},
                    }
                )
                continue

            if msg_type == "response.create":
                text = _extract_text_from_response(message, session.input_text)
                if not text:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "event_id": _realtime_event_id(),
                            "error": {"type": "invalid_request_error", "message": "No input text provided for TTS."},
                        }
                    )
                    continue

                response_id = _response_id()
                item_id = _item_id()
                await websocket.send_json(
                    {
                        "type": "response.created",
                        "event_id": _realtime_event_id(),
                        "response": {
                            "id": response_id,
                            "object": "realtime.response",
                            "status": "in_progress",
                            "output": [],
                        },
                    }
                )
                await websocket.send_json(
                    {
                        "type": "response.output_item.added",
                        "event_id": _realtime_event_id(),
                        "response_id": response_id,
                        "output_index": 0,
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [{"type": "audio", "audio": ""}],
                        },
                    }
                )

                speech_req = SpeechRequest(
                    model=session.model,
                    input=text,
                    voice=session.voice,
                    instructions=session.instructions,
                    response_format="pcm",
                    stream=True,
                    task_type=session.task_type,
                    language=session.language,
                )

                try:
                    transcript = text

                    async for chunk in _tts_stream_audio(speech_req):
                        await websocket.send_json(
                            {
                                "type": "response.audio.delta",
                                "event_id": _realtime_event_id(),
                                "response_id": response_id,
                                "item_id": item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": base64.b64encode(chunk).decode("utf-8"),
                            }
                        )

                    await websocket.send_json(
                        {
                            "type": "response.audio.done",
                            "event_id": _realtime_event_id(),
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "response.output_item.done",
                            "event_id": _realtime_event_id(),
                            "response_id": response_id,
                            "output_index": 0,
                            "item": {
                                "id": item_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {"type": "audio", "transcript": transcript},
                                ],
                            },
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "response.done",
                            "event_id": _realtime_event_id(),
                            "response": {
                                "id": response_id,
                                "object": "realtime.response",
                                "status": "completed",
                                "output": [
                                    {
                                        "id": item_id,
                                        "type": "message",
                                        "role": "assistant",
                                        "status": "completed",
                                    }
                                ],
                            },
                        }
                    )
                except HTTPException as exc:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "event_id": _realtime_event_id(),
                            "error": {"type": "server_error", "message": str(exc.detail)},
                        }
                    )
                continue

            await websocket.send_json(
                {
                    "type": "error",
                    "event_id": _realtime_event_id(),
                    "error": {"type": "invalid_request_error", "message": f"Unsupported event type: {msg_type}"},
                }
            )
    except WebSocketDisconnect:
        return


import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import re
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


app = FastAPI(title="OpenAI-compatible TTS + Faster Whisper Server")

_stt_model: WhisperModel | None = None
_stt_lock = threading.Lock()
_stt_warmup_task: asyncio.Task[None] | None = None
_tts_warmup_task: asyncio.Task[None] | None = None

logger = logging.getLogger("tts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

QWEN_SUPPORTED_VOICES = {
    "aiden",
    "dylan",
    "eric",
    "ono_anna",
    "ryan",
    "serena",
    "sohee",
    "uncle_fu",
    "vivian",
}

_tts_warmup_started = False
_stt_warmup_started = False


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed >= 1 else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _stt_default_language() -> str | None:
    value = os.environ.get("STT_DEFAULT_LANGUAGE", "").strip()
    if not value or value.lower() == "auto":
        return None
    return value


def _stt_beam_size() -> int:
    return _env_int("STT_BEAM_SIZE", 1)


def _stt_best_of() -> int:
    return _env_int("STT_BEST_OF", 1)


def _stt_condition_on_previous_text() -> bool:
    return _env_bool("STT_CONDITION_ON_PREVIOUS_TEXT", False)


def _stt_warmup_enabled() -> bool:
    return _env("STT_WARMUP_ENABLED", "1") != "0"


def _stt_model_name() -> str:
    return _env("STT_MODEL_SIZE", "RoachLin/kotoba-whisper-v2.2-faster")


def _stt_public_model_name() -> str:
    return _env("STT_MODEL_ID", _stt_model_name())


@app.on_event("startup")
async def startup_event() -> None:
    global _stt_warmup_started, _stt_warmup_task, _tts_warmup_started, _tts_warmup_task
    if not _stt_warmup_started:
        _stt_warmup_started = True
        _stt_warmup_task = asyncio.create_task(_warmup_stt_model())
    if not _tts_warmup_started:
        _tts_warmup_started = True
        _tts_warmup_task = asyncio.create_task(_warmup_tts_upstream())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _stt_warmup_task, _tts_warmup_task
    if _stt_warmup_task is not None and not _stt_warmup_task.done():
        _stt_warmup_task.cancel()
        try:
            await _stt_warmup_task
        except asyncio.CancelledError:
            logger.info("Cancelled pending STT warmup task during shutdown")
        finally:
            _stt_warmup_task = None
    if _tts_warmup_task is not None and not _tts_warmup_task.done():
        _tts_warmup_task.cancel()
        try:
            await _tts_warmup_task
        except asyncio.CancelledError:
            logger.info("Cancelled pending TTS warmup task during shutdown")
        finally:
            _tts_warmup_task = None


def get_stt_model() -> WhisperModel:
    global _stt_model
    if _stt_model is None:
        with _stt_lock:
            if _stt_model is None:
                stt_model_size = _stt_model_name()
                stt_device = _env("STT_DEVICE", "cuda")
                stt_compute_type = _env("STT_COMPUTE_TYPE", "float16")
                stt_cpu_threads = _env_int("STT_CPU_THREADS", 8)
                stt_num_workers = _env_int("STT_NUM_WORKERS", 1)
                stt_default_language = _stt_default_language() or "auto"
                stt_beam_size = _stt_beam_size()
                stt_best_of = _stt_best_of()
                stt_condition_on_previous_text = _stt_condition_on_previous_text()
                logger.info(
                    "Loading Faster Whisper model=%s device=%s compute_type=%s cpu_threads=%s num_workers=%s default_language=%s beam_size=%s best_of=%s condition_on_previous_text=%s",
                    stt_model_size,
                    stt_device,
                    stt_compute_type,
                    stt_cpu_threads,
                    stt_num_workers,
                    stt_default_language,
                    stt_beam_size,
                    stt_best_of,
                    stt_condition_on_previous_text,
                )
                _stt_model = WhisperModel(
                    stt_model_size,
                    device=stt_device,
                    compute_type=stt_compute_type,
                    cpu_threads=stt_cpu_threads,
                    num_workers=stt_num_workers,
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


def _tts_backend() -> str:
    backend = _env("TTS_BACKEND", "qwen3-tts").strip().lower()
    aliases = {
        "qwen": "qwen3-tts",
        "qwen3": "qwen3-tts",
        "qwen3-tts": "qwen3-tts",
        "cosy": "cosyvoice",
        "cosyvoice": "cosyvoice",
        "fish": "cosyvoice",
        "fish-speech": "cosyvoice",
        "tsukasa": "tsukasa-speech",
        "tsukasa-speech": "tsukasa-speech",
        "tsukasa_speech": "tsukasa-speech",
        "respair-tsukasa": "tsukasa-speech",
    }
    return aliases.get(backend, backend)


def _contains_japanese(text: str) -> bool:
    return bool(re.search(r"[ぁ-んァ-ン一-龯]", text))


def _looks_english_text(text: str) -> bool:
    latin_chars = re.findall(r"[A-Za-z]", text)
    if len(latin_chars) < 3:
        return False
    if _contains_japanese(text):
        return False
    return True


def _tts_language_from_transcription_language(language: str | None) -> str | None:
    normalized = (language or "").strip().lower()
    if not normalized:
        return None
    if normalized.startswith("en"):
        return "English"
    if normalized.startswith("ja"):
        return "Japanese"
    return None


def _apply_detected_tts_language(session: "RealtimeSession", detected_language: str | None) -> None:
    mapped_language = _tts_language_from_transcription_language(detected_language)
    if not mapped_language:
        return
    if session.language != mapped_language:
        logger.info("Updating realtime TTS language from transcription: %s -> %s", session.language, mapped_language)
        session.language = mapped_language


def _tts_backend_for_request(req: "SpeechRequest") -> str:
    backend = _tts_backend()
    if backend != "tsukasa-speech":
        return backend

    language = req.language.strip().lower()
    if language in {"english", "en", "en-us", "en-gb"} or _looks_english_text(req.input):
        logger.info("Routing English TTS request to qwen3-tts instead of tsukasa-speech")
        return "qwen3-tts"
    return backend


def _qwen_default_voice() -> str:
    raw = _env("QWEN_TTS_DEFAULT_VOICE", "ono_anna").strip()
    lowered = raw.lower()
    if lowered in QWEN_SUPPORTED_VOICES:
        return lowered
    return "ono_anna"


def _default_voice_for_backend(backend: str) -> str:
    if backend == "tsukasa-speech":
        return _tsukasa_default_voice()
    if backend == "qwen3-tts":
        return _qwen_default_voice()
    return _env("TTS_DEFAULT_VOICE", _env("QWEN_TTS_DEFAULT_VOICE", "ono_anna"))


def _normalize_voice_for_backend(voice: str | dict[str, str], backend: str) -> str | dict[str, str]:
    if isinstance(voice, dict):
        return voice

    normalized = voice.strip()
    if backend != "qwen3-tts":
        return normalized

    lowered = normalized.lower()
    if lowered in QWEN_SUPPORTED_VOICES:
        return lowered
    return _qwen_default_voice()


def _tsukasa_base_url() -> str:
    return _env("TSUKASA_SPEECH_BASE_URL", "http://tsukasa-speech:5001").rstrip("/")


def _tsukasa_model_name() -> str:
    return _env("TSUKASA_SPEECH_MODEL", "Respair/Tsukasa_Speech")


def _tsukasa_public_model_name() -> str:
    return _env("TSUKASA_SPEECH_PUBLIC_MODEL_NAME", _tsukasa_model_name())


def _tsukasa_timeout() -> float:
    return float(_env("TSUKASA_SPEECH_TIMEOUT", _env("TTS_UPSTREAM_TIMEOUT", "600")))


def _tsukasa_chunk_bytes() -> int:
    return int(_env("TSUKASA_SPEECH_STREAM_CHUNK_BYTES", _env("TTS_UPSTREAM_STREAM_CHUNK_BYTES", "8192")))


def _tsukasa_sample_rate() -> int:
    return _env_int("TSUKASA_SPEECH_SAMPLE_RATE", 24000)


def _tsukasa_default_voice() -> str:
    return _env("TSUKASA_SPEECH_DEFAULT_VOICE", "audio_ref")


def _tsukasa_default_diffusion_steps() -> int:
    return _env_int("TSUKASA_SPEECH_DEFAULT_DIFFUSION_STEPS", 5)


def _tsukasa_default_embedding_scale() -> float:
    return float(_env("TSUKASA_SPEECH_DEFAULT_EMBEDDING_SCALE", "1.0"))


def _tsukasa_default_alpha() -> float:
    return float(_env("TSUKASA_SPEECH_DEFAULT_ALPHA", "0.3"))


def _tsukasa_default_beta() -> float:
    return float(_env("TSUKASA_SPEECH_DEFAULT_BETA", "0.7"))


def _tts_upstream_base_url(backend: str | None = None) -> str:
    explicit = os.environ.get("TTS_UPSTREAM_BASE_URL", "").strip()
    if explicit and (backend is None or backend == _tts_backend()):
        return explicit.rstrip("/")
    backend = backend or _tts_backend()
    if backend == "tsukasa-speech":
        return _tsukasa_base_url()
    if backend == "cosyvoice":
        return _fish_base_url()
    return _env("QWEN_TTS_BASE_URL", "http://qwen3-tts:8091").rstrip("/")


def _tts_upstream_model_name(backend: str | None = None) -> str:
    explicit = os.environ.get("TTS_UPSTREAM_MODEL", "").strip()
    if explicit and (backend is None or backend == _tts_backend()):
        return explicit
    backend = backend or _tts_backend()
    if backend == "tsukasa-speech":
        return _tsukasa_model_name()
    if backend == "cosyvoice":
        return _fish_model_name()
    return _env("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")


def _tts_public_model_name() -> str:
    explicit = os.environ.get("TTS_PUBLIC_MODEL_NAME", "").strip()
    if explicit:
        return explicit
    if _tts_backend() == "tsukasa-speech":
        return _tsukasa_public_model_name()
    return _tts_upstream_model_name()


def _dedupe_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in models:
        model_id = model.get("id")
        if not isinstance(model_id, str) or model_id in seen:
            continue
        seen.add(model_id)
        deduped.append(model)
    return deduped


def _resolve_tts_request_model(model_name: str | None, backend: str | None = None) -> str:
    normalized = (model_name or "").strip()
    target_backend = backend or _tts_backend()
    aliases = {
        "tts-1",
        _tts_public_model_name(),
        _tts_upstream_model_name(),
        _tsukasa_public_model_name(),
        _tsukasa_model_name(),
        _env("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    }
    if not normalized or normalized in aliases:
        return _tts_upstream_model_name(target_backend)
    return normalized


def _tts_upstream_timeout(backend: str | None = None) -> float:
    backend = backend or _tts_backend()
    if backend == "tsukasa-speech":
        return _tsukasa_timeout()
    if backend == "cosyvoice":
        return _fish_timeout()
    return float(_env("TTS_UPSTREAM_TIMEOUT", _env("QWEN_TTS_TIMEOUT", _env("COSY_TTS_TIMEOUT", "600"))))


def _tts_upstream_api_key() -> str:
    return _env("TTS_UPSTREAM_API_KEY", _env("QWEN_TTS_API_KEY", "EMPTY"))


def _tts_upstream_headers(backend: str | None = None) -> dict[str, str]:
    if (backend or _tts_backend()) == "tsukasa-speech":
        return {}
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


def _audio_bytes_from_realtime_input(audio_buffer: bytes, audio_format: str, sample_rate: int) -> tuple[bytes, str]:
    if audio_format == "pcm16":
        return _wav_header(sample_rate, len(audio_buffer)) + audio_buffer, ".wav"
    if audio_format == "wav":
        return audio_buffer, ".wav"
    raise HTTPException(status_code=400, detail=f"Unsupported realtime input_audio_format: {audio_format}")


def _transcription_segment(segment: Any, *, word_timestamps: bool) -> dict[str, Any]:
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
    return item


def _transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    suffix: str,
    language: str | None,
    prompt: str | None,
    temperature: float,
    word_timestamps: bool,
) -> tuple[str, list[dict[str, Any]], Any]:
    effective_language = language or _stt_default_language()
    with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments_iter, info = get_stt_model().transcribe(
            tmp.name,
            language=effective_language,
            initial_prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            beam_size=_stt_beam_size(),
            best_of=_stt_best_of(),
            condition_on_previous_text=_stt_condition_on_previous_text(),
        )
        segments = []
        text_parts = []
        for segment in segments_iter:
            segments.append(_transcription_segment(segment, word_timestamps=word_timestamps))
            text_parts.append(segment.text)

    return "".join(text_parts).strip(), segments, info


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


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    target_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
    source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _tsukasa_voice_settings(req: "SpeechRequest") -> dict[str, Any]:
    voice_cfg = _voice_config(_voice_name(req.voice))
    requested_voice = req.voice if isinstance(req.voice, str) else ""
    ref_audio = str(
        voice_cfg.get("voice_ref")
        or voice_cfg.get("reference_wav")
        or voice_cfg.get("reference_audio")
        or voice_cfg.get("ref_audio")
        or ""
    ).strip()
    voice_name = str(
        voice_cfg.get("voice")
        or voice_cfg.get("speaker")
        or voice_cfg.get("speaker_name")
        or requested_voice
        or _tsukasa_default_voice()
    ).strip()
    if not voice_name and ref_audio:
        voice_name = Path(ref_audio).stem
    return {
        "voice": voice_name or _tsukasa_default_voice(),
        "diffusion_steps": int(voice_cfg.get("diffusion_steps", _tsukasa_default_diffusion_steps())),
        "embedding_scale": float(voice_cfg.get("embedding_scale", _tsukasa_default_embedding_scale())),
        "alpha": float(voice_cfg.get("alpha", _tsukasa_default_alpha())),
        "beta": float(voice_cfg.get("beta", _tsukasa_default_beta())),
    }


def _tsukasa_payload(req: "SpeechRequest") -> dict[str, Any]:
    settings = _tsukasa_voice_settings(req)
    speed = req.speed if req.speed > 0 else 1.0
    payload = {
        "text": req.input,
        "voice": settings["voice"],
        "speed": speed,
        "diffusion_steps": settings["diffusion_steps"],
        "embedding_scale": settings["embedding_scale"],
        "alpha": settings["alpha"],
        "beta": settings["beta"],
        "language": req.language,
    }
    if req.instructions.strip():
        payload["instructions"] = req.instructions.strip()
    return payload


async def _tsukasa_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{_tsukasa_base_url()}/health")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def _tsukasa_complete_audio(req: "SpeechRequest", fmt: str) -> bytes:
    payload = _tsukasa_payload(req)
    try:
        async with httpx.AsyncClient(timeout=_tsukasa_timeout()) as client:
            response = await client.post(f"{_tsukasa_base_url()}/synthesize", json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Tsukasa Speech upstream error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Tsukasa Speech upstream unreachable: {exc}") from exc

    source_audio, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
    if isinstance(source_audio, np.ndarray) and source_audio.ndim > 1:
        source_audio = source_audio.mean(axis=1)
    audio = np.asarray(source_audio, dtype=np.float32)

    target_rate = _tsukasa_sample_rate()
    if sample_rate != target_rate:
        audio = _resample_audio(audio, int(sample_rate), target_rate)
        sample_rate = target_rate

    return _encode_audio(audio, int(sample_rate), fmt)


async def _tsukasa_stream_audio(req: "SpeechRequest") -> AsyncGenerator[bytes, None]:
    wav_bytes = await _tsukasa_complete_audio(req, "wav")
    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    mono = np.asarray(audio, dtype=np.float32)
    target_rate = _tsukasa_sample_rate()
    if sample_rate != target_rate:
        mono = _resample_audio(mono, int(sample_rate), target_rate)
    chunk_size = req.chunk_size or _tsukasa_chunk_bytes() // 2
    async for chunk in _pcm_stream_from_audio(mono, chunk_size):
        yield chunk


def _encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return _to_pcm16(audio)
    if fmt == "wav":
        return _complete_wav(audio, sample_rate)
    if fmt == "mp3":
        return _mp3_bytes(audio, sample_rate)

    buffer = io.BytesIO()
    subtype = None
    format_name = fmt.upper()
    if fmt == "flac":
        format_name = "FLAC"
    elif fmt == "opus":
        format_name = "OGG"
        subtype = "OPUS"
    elif fmt == "aac":
        from pydub import AudioSegment

        segment = AudioSegment(_to_pcm16(audio), frame_rate=sample_rate, sample_width=2, channels=1)
        segment.export(buffer, format="adts")
        return buffer.getvalue()

    sf.write(buffer, audio, sample_rate, format=format_name, subtype=subtype)
    return buffer.getvalue()


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
    model: str = Field(default_factory=_tts_public_model_name)
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
    model: str = Field(default_factory=_tts_public_model_name)
    voice: str = Field(default_factory=lambda: _env("TTS_DEFAULT_VOICE", _env("QWEN_TTS_DEFAULT_VOICE", "Ono_Anna")))
    instructions: str = ""
    input_text: str = ""
    task_type: Literal["CustomVoice", "VoiceDesign", "Base"] = Field(default_factory=_default_task_type)
    language: str = Field(default_factory=lambda: _env("TTS_DEFAULT_LANGUAGE", _env("QWEN_TTS_DEFAULT_LANGUAGE", "Japanese")))
    transcription_model: str = Field(default_factory=_stt_public_model_name)
    input_audio_transcription: bool = False
    input_audio_format: Literal["pcm16", "wav"] = "pcm16"
    input_audio_sample_rate: int = Field(default=16000, ge=1)
    transcription_language: str = ""
    transcription_prompt: str = ""
    transcription_response_format: Literal["text", "json", "verbose_json"] = "text"
    transcription_word_timestamps: bool = False


def _tts_request_payload(
    req: "SpeechRequest",
    *,
    response_format: str | None = None,
    stream: bool | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    use_stream = req.stream if stream is None else stream
    target_backend = backend or _tts_backend_for_request(req)
    payload: dict[str, Any] = {
        "input": req.input,
        "response_format": response_format or req.response_format.lower(),
    }

    payload["model"] = _resolve_tts_request_model(req.model, target_backend)

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

    voice_name = _voice_name(_normalize_voice_for_backend(req.voice, target_backend))
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
    backend = _tts_backend()
    if backend == "tsukasa-speech":
        return await _tsukasa_health()
    if backend == "cosyvoice":
        return await _fish_health()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{_tts_upstream_base_url()}/health", headers=_tts_upstream_headers())
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def _warmup_stt_model() -> None:
    if not _stt_warmup_enabled():
        logger.info("STT warmup disabled")
        return

    logger.info("Starting STT warmup in background")

    try:
        await asyncio.to_thread(get_stt_model)
        logger.info("STT warmup completed")
    except Exception as exc:
        logger.warning("STT warmup failed: %s", exc)


async def _warmup_tts_upstream() -> None:
    if not _tts_warmup_enabled():
        logger.info("TTS upstream warmup disabled")
        return

    logger.info("Starting TTS upstream warmup in background")

    for _attempt in range(60):
        if await _tts_health():
            break
        await asyncio.sleep(2)
    else:
        logger.warning("TTS upstream warmup skipped because upstream health never became ready")
        return

    try:
        backend = _tts_backend()
        await _tts_complete_audio(
            SpeechRequest(
                model=_tts_public_model_name(),
                input=_tts_warmup_text(),
                voice=_default_voice_for_backend(backend),
                task_type=_default_task_type(),
                language=_env("TTS_DEFAULT_LANGUAGE", _env("QWEN_TTS_DEFAULT_LANGUAGE", "Japanese")),
                response_format="wav",
                stream=False,
            ),
            response_format="wav",
        )
        logger.info("TTS upstream warmup completed")
    except HTTPException as exc:
        logger.warning("TTS upstream warmup failed: %s", exc)


async def _tts_request_with_failover(method: str, path: str, *, backend: str | None = None, **kwargs: Any) -> httpx.Response:
    headers = {**_tts_upstream_headers(backend), **(kwargs.pop("headers", {}) or {})}
    try:
        async with httpx.AsyncClient(timeout=_tts_upstream_timeout(backend)) as client:
            response = await client.request(method, f"{_tts_upstream_base_url(backend)}{path}", headers=headers, **kwargs)
            response.raise_for_status()
            return response
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"TTS upstream error: {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"TTS upstream unreachable: {exc}") from exc


async def _tts_complete_audio(req: "SpeechRequest", response_format: str | None = None) -> bytes:
    fmt = (response_format or req.response_format).lower()
    backend = _tts_backend_for_request(req)
    if backend == "tsukasa-speech":
        return await _tsukasa_complete_audio(req, fmt)
    if backend == "cosyvoice":
        audio_bytes = await _fish_complete_audio(req, fmt)
        audio, sample_rate = _decode_audio(audio_bytes)
        return _encode_audio(audio, sample_rate, fmt)

    payload = _tts_request_payload(req, response_format=fmt, stream=False, backend=backend)
    response = await _tts_request_with_failover("POST", "/v1/audio/speech", backend=backend, json=payload)
    return response.content


async def _tts_stream_audio(req: "SpeechRequest") -> AsyncGenerator[bytes, None]:
    backend = _tts_backend_for_request(req)
    if backend == "tsukasa-speech":
        async for chunk in _tsukasa_stream_audio(req):
            yield chunk
        return
    if backend == "cosyvoice":
        audio_bytes = await _fish_complete_audio(req, "pcm")
        audio, sample_rate = _decode_audio(audio_bytes)
        async for chunk in _pcm_stream_from_audio(audio, req.chunk_size or _fish_chunk_bytes() // 2):
            yield chunk
        return

    payload = _tts_request_payload(req, response_format="pcm", stream=True, backend=backend)
    headers = _tts_upstream_headers(backend)
    chunk_bytes = int(_env("TTS_UPSTREAM_STREAM_CHUNK_BYTES", str(_fish_chunk_bytes())))
    try:
        async with httpx.AsyncClient(timeout=_tts_upstream_timeout(backend)) as client:
            async with client.stream(
                "POST",
                f"{_tts_upstream_base_url(backend)}/v1/audio/speech",
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=chunk_bytes):
                    if chunk:
                        yield chunk
    except httpx.HTTPStatusError as exc:
        detail_bytes = await exc.response.aread()
        detail = detail_bytes.decode("utf-8", errors="replace") if detail_bytes else str(exc)
        raise HTTPException(status_code=502, detail=f"TTS upstream error: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"TTS upstream unreachable: {exc}") from exc


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tts_backend": _tts_backend(),
        "stt_model_loaded": _stt_model is not None,
        "stt_model": _stt_model_name(),
        "stt_public_model": _stt_public_model_name(),
        "tts_upstream_backend": _tts_upstream_base_url(),
        "tts_upstream_model": _tts_upstream_model_name(),
        "tts_public_model": _tts_public_model_name(),
        "tts_upstream_healthy": await _tts_health(),
    }


@app.get("/v1/models")
async def models():
    data = [
        {
            "id": _stt_public_model_name(),
            "object": "model",
            "owned_by": "local",
            "metadata": {"capabilities": ["transcription"], "default": True, "backend_model": _stt_model_name()},
        },
        {
            "id": "whisper-1",
            "object": "model",
            "owned_by": "local",
            "metadata": {"capabilities": ["transcription"], "alias_for": _stt_public_model_name()},
        },
        {
            "id": _tts_public_model_name(),
            "object": "model",
            "owned_by": "local",
            "metadata": {"capabilities": ["speech"], "default": True, "backend_model": _tts_upstream_model_name()},
        },
        {
            "id": "tts-1",
            "object": "model",
            "owned_by": "local",
            "metadata": {"capabilities": ["speech"], "alias_for": _tts_public_model_name()},
        },
    ]

    backend = _tts_backend()
    if backend == "tsukasa-speech":
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{_tsukasa_base_url()}/voices")
                response.raise_for_status()
                payload = response.json()
                voices = payload.get("voices") if isinstance(payload, dict) else None
                data.append(
                    {
                        "id": _tts_upstream_model_name(),
                        "object": "model",
                        "owned_by": "local",
                        "metadata": {
                            "capabilities": ["speech"],
                            "backend": "tsukasa-speech",
                            "sample_rate": _tsukasa_sample_rate(),
                            "voices": voices if isinstance(voices, list) else [],
                        },
                    }
                )
        except (httpx.HTTPError, ValueError):
            data.append(
                {
                    "id": _tts_upstream_model_name(),
                    "object": "model",
                    "owned_by": "local",
                    "metadata": {
                        "capabilities": ["speech"],
                        "backend": "tsukasa-speech",
                        "sample_rate": _tsukasa_sample_rate(),
                    },
                }
            )
    else:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{_tts_upstream_base_url()}/v1/models", headers=_tts_upstream_headers())
                response.raise_for_status()
                payload = response.json()
                upstream_models = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(upstream_models, list):
                    data.extend(upstream_models)
        except (httpx.HTTPError, ValueError):
            data.append(
                {
                    "id": _tts_upstream_model_name(),
                    "object": "model",
                    "owned_by": "local",
                    "metadata": {"capabilities": ["speech"], "backend": backend},
                }
            )

    return {"object": "list", "data": _dedupe_models(data)}


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: Literal["json", "text", "verbose_json"] = Form(default="json"),
    temperature: float = Form(default=0),
    timestamp_granularities: list[str] | None = Form(default=None),
):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    word_timestamps = bool(timestamp_granularities and "word" in timestamp_granularities)
    try:
        text, segments, info = _transcribe_audio_bytes(
            await file.read(),
            suffix=suffix,
            language=language,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": info.language, "duration": info.duration, "text": text, "segments": segments}
    return {"text": text}


@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
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
    input_audio_buffer = bytearray()
    input_audio_item_id: str | None = None

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
                if payload.get("input_audio_transcription") is False:
                    session.input_audio_transcription = False
                elif isinstance(payload.get("input_audio_transcription"), bool):
                    session.input_audio_transcription = payload["input_audio_transcription"]
                elif isinstance(payload.get("input_audio_transcription"), dict):
                    transcription = payload["input_audio_transcription"]
                    session.input_audio_transcription = True
                    if isinstance(transcription.get("model"), str) and transcription["model"].strip():
                        session.transcription_model = transcription["model"].strip()
                    if isinstance(transcription.get("language"), str):
                        session.transcription_language = transcription["language"].strip()
                    if isinstance(transcription.get("prompt"), str):
                        session.transcription_prompt = transcription["prompt"]
                    if transcription.get("response_format") in {"text", "json", "verbose_json"}:
                        session.transcription_response_format = transcription["response_format"]
                    granularities = transcription.get("timestamp_granularities")
                    if isinstance(granularities, list):
                        session.transcription_word_timestamps = "word" in granularities
                if payload.get("input_audio_format") in {"pcm16", "wav"}:
                    session.input_audio_format = payload["input_audio_format"]
                sample_rate = payload.get("input_audio_sample_rate")
                if isinstance(sample_rate, int) and sample_rate >= 1:
                    session.input_audio_sample_rate = sample_rate
                await websocket.send_json(
                    {
                        "type": "session.updated",
                        "event_id": _realtime_event_id(),
                        "session": session.model_dump(),
                    }
                )
                continue

            if msg_type == "input_audio_buffer.append":
                audio_chunk = message.get("audio")
                if not isinstance(audio_chunk, str) or not audio_chunk:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "event_id": _realtime_event_id(),
                            "error": {"type": "invalid_request_error", "message": "Missing realtime audio chunk."},
                        }
                    )
                    continue
                try:
                    input_audio_buffer.extend(base64.b64decode(audio_chunk))
                except ValueError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "event_id": _realtime_event_id(),
                            "error": {"type": "invalid_request_error", "message": "Invalid base64 audio chunk."},
                        }
                    )
                    continue
                input_audio_item_id = input_audio_item_id or _item_id()
                continue

            if msg_type == "input_audio_buffer.clear":
                input_audio_buffer.clear()
                input_audio_item_id = None
                await websocket.send_json({"type": "input_audio_buffer.cleared", "event_id": _realtime_event_id()})
                continue

            if msg_type == "input_audio_buffer.commit":
                if not input_audio_buffer:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "event_id": _realtime_event_id(),
                            "error": {"type": "invalid_request_error", "message": "No audio buffered for transcription."},
                        }
                    )
                    continue
                input_audio_item_id = input_audio_item_id or _item_id()
                await websocket.send_json(
                    {
                        "type": "input_audio_buffer.committed",
                        "event_id": _realtime_event_id(),
                        "item_id": input_audio_item_id,
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
                if session.input_audio_transcription or input_audio_buffer:
                    if not input_audio_buffer:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "event_id": _realtime_event_id(),
                                "error": {"type": "invalid_request_error", "message": "No input audio provided for transcription."},
                            }
                        )
                        continue

                    response_id = _response_id()
                    output_item_id = _item_id()
                    input_audio_item_id = input_audio_item_id or _item_id()
                    response = message.get("response") or {}
                    prompt = session.transcription_prompt
                    if isinstance(response.get("instructions"), str) and response["instructions"].strip():
                        prompt = response["instructions"].strip()

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

                    try:
                        audio_bytes, suffix = _audio_bytes_from_realtime_input(
                            bytes(input_audio_buffer),
                            session.input_audio_format,
                            session.input_audio_sample_rate,
                        )
                        text, segments, info = _transcribe_audio_bytes(
                            audio_bytes,
                            suffix=suffix,
                            language=session.transcription_language or None,
                            prompt=prompt or None,
                            temperature=0,
                            word_timestamps=session.transcription_word_timestamps,
                        )

                        await websocket.send_json(
                            {
                                "type": "conversation.item.created",
                                "event_id": _realtime_event_id(),
                                "item": {
                                    "id": input_audio_item_id,
                                    "type": "message",
                                    "role": "user",
                                    "status": "completed",
                                    "content": [{"type": "input_audio", "audio": ""}],
                                },
                            }
                        )
                        await websocket.send_json(
                            {
                                "type": "conversation.item.input_audio_transcription.completed",
                                "event_id": _realtime_event_id(),
                                "item_id": input_audio_item_id,
                                "transcript": text,
                                "language": info.language,
                                "duration": info.duration,
                                "segments": segments,
                            }
                        )
                        _apply_detected_tts_language(session, info.language)
                        await websocket.send_json(
                            {
                                "type": "response.output_item.added",
                                "event_id": _realtime_event_id(),
                                "response_id": response_id,
                                "output_index": 0,
                                "item": {
                                    "id": output_item_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [{"type": "text", "text": ""}],
                                },
                            }
                        )
                        if text:
                            await websocket.send_json(
                                {
                                    "type": "response.output_text.delta",
                                    "event_id": _realtime_event_id(),
                                    "response_id": response_id,
                                    "item_id": output_item_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": text,
                                }
                            )
                        await websocket.send_json(
                            {
                                "type": "response.output_text.done",
                                "event_id": _realtime_event_id(),
                                "response_id": response_id,
                                "item_id": output_item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "text": text,
                            }
                        )
                        await websocket.send_json(
                            {
                                "type": "response.output_item.done",
                                "event_id": _realtime_event_id(),
                                "response_id": response_id,
                                "output_index": 0,
                                "item": {
                                    "id": output_item_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [{"type": "text", "text": text}],
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
                                            "id": output_item_id,
                                            "type": "message",
                                            "role": "assistant",
                                            "status": "completed",
                                            "content": [{"type": "text", "text": text}],
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
                    except Exception as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "event_id": _realtime_event_id(),
                                "error": {"type": "server_error", "message": str(exc)},
                            }
                        )
                    finally:
                        input_audio_buffer.clear()
                        input_audio_item_id = None
                    continue

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


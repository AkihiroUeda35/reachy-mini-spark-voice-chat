from __future__ import annotations

import asyncio
import importlib
import io
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field


app = FastAPI(title="Tsukasa Speech API")
logger = logging.getLogger("tsukasa_speech.api")

_runtime_lock = threading.Lock()
_inference_lock = threading.Lock()
_runtime: dict[str, Any] | None = None


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _repo_dir() -> Path:
    return Path(_env("TSUKASA_SPEECH_REPO_DIR", "/models/tsukasa_speech/repo"))


def _repo_id() -> str:
    return _env("TSUKASA_SPEECH_REPO_ID", "Respair/Tsukasa_Speech")


def _repo_ref() -> str:
    return _env("TSUKASA_SPEECH_REPO_REF", "main")


def _sample_rate() -> int:
    return int(_env("TSUKASA_SPEECH_SAMPLE_RATE", "24000"))


def _default_voice() -> str:
    return _env("TSUKASA_SPEECH_DEFAULT_VOICE", "audio_ref")


def _ensure_repo_available() -> Path:
    repo_dir = _repo_dir()
    repo_dir.mkdir(parents=True, exist_ok=True)
    if (repo_dir / "importable.py").exists():
        return repo_dir
    snapshot_download(
        repo_id=_repo_id(),
        revision=_repo_ref(),
        local_dir=str(repo_dir),
    )
    return repo_dir


def _load_runtime() -> dict[str, Any]:
    global _runtime
    if _runtime is not None:
        return _runtime

    with _runtime_lock:
        if _runtime is not None:
            return _runtime

        repo_dir = _ensure_repo_available()
        os.chdir(repo_dir)
        sys.path.insert(0, str(repo_dir))

        importable_module = importlib.import_module("importable")
        mixed_phon_module = importlib.import_module("Utils.phonemize.mixed_phon")

        _runtime = {
            "repo_dir": repo_dir,
            "importable": importable_module,
            "smart_phonemize": mixed_phon_module.smart_phonemize,
        }
        return _runtime


def _voice_dir() -> Path:
    return _ensure_repo_available() / "reference_sample_wavs"


def _voice_paths() -> list[Path]:
    voice_dir = _voice_dir()
    if not voice_dir.exists():
        raise HTTPException(status_code=500, detail=f"reference_sample_wavs not found under {voice_dir}")
    voices = sorted(path for path in voice_dir.iterdir() if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"})
    if not voices:
        raise HTTPException(status_code=500, detail=f"No reference voices found under {voice_dir}")
    return voices


def _resolve_voice_path(voice: str | None) -> Path:
    requested = (voice or "").strip() or _default_voice()
    voices = _voice_paths()
    if requested.isdigit():
        index = int(requested)
        if 0 <= index < len(voices):
            return voices[index]
    for path in voices:
        if requested in {path.name, path.stem}:
            return path
    available = ", ".join(path.stem for path in voices[:10])
    raise HTTPException(status_code=400, detail=f"Unknown Tsukasa voice '{requested}'. Available voices include: {available}")


def _looks_japanese(text: str) -> bool:
    return bool(re.search(r"[ぁ-んァ-ン一-龯]", text))


def _prompt_text_for_style(importable_module: Any, smart_phonemize: Any, text: str) -> str:
    prompt_text = text.strip()
    if _looks_japanese(prompt_text):
        return prompt_text

    try:
        return importable_module.p2g(smart_phonemize(prompt_text))
    except Exception as exc:
        logger.warning("Falling back to raw non-Japanese prompt text for style conditioning: %s", exc)
        return prompt_text


def _wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), _sample_rate(), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    instructions: str = Field(default="")
    speed: float = Field(default=1.0, gt=0)
    diffusion_steps: int = Field(default=5, ge=1)
    embedding_scale: float = Field(default=1.0, gt=0)
    alpha: float = Field(default=0.3, ge=0)
    beta: float = Field(default=0.7, ge=0)
    language: str = Field(default="Japanese")


def _synthesize_sync(req: SynthesizeRequest) -> bytes:
    runtime = _load_runtime()
    importable_module = runtime["importable"]
    smart_phonemize = runtime["smart_phonemize"]
    voice_path = _resolve_voice_path(req.voice)
    phonemes = smart_phonemize(req.text.strip())

    with _inference_lock:
        voice_style = importable_module.compute_style_through_clip(str(voice_path))
        if req.instructions.strip():
            prompt_text = _prompt_text_for_style(importable_module, smart_phonemize, req.text)
            prompt = f"{req.instructions.strip()}\ntext: {prompt_text}"
            prompt_style = importable_module.Kotodama_Prompter(
                importable_module.model,
                text=prompt,
                device=importable_module.device,
            )
            audio = importable_module.inference(
                phonemes,
                prompt_style,
                alpha=req.alpha,
                beta=req.beta,
                diffusion_steps=req.diffusion_steps,
                embedding_scale=req.embedding_scale,
                rate_of_speech=req.speed,
            )
            audio = importable_module.trim_long_silences(audio)
        else:
            audio = importable_module.inference(
                phonemes,
                voice_style,
                alpha=req.alpha,
                beta=req.beta,
                diffusion_steps=req.diffusion_steps,
                embedding_scale=req.embedding_scale,
                rate_of_speech=req.speed,
            )

    return _wav_bytes(np.asarray(audio, dtype=np.float32))


@app.get("/health")
async def health() -> dict[str, Any]:
    voices = [path.stem for path in _voice_paths()]
    return {
        "status": "ok",
        "repo_id": _repo_id(),
        "repo_ref": _repo_ref(),
        "repo_dir": str(_repo_dir()),
        "default_voice": _default_voice(),
        "sample_rate": _sample_rate(),
        "voice_count": len(voices),
        "voices": voices[:20],
        "model_loaded": _runtime is not None,
    }


@app.get("/voices")
async def voices() -> dict[str, Any]:
    return {"voices": [path.stem for path in _voice_paths()], "default_voice": _default_voice()}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest) -> Response:
    wav_bytes = await asyncio.to_thread(_synthesize_sync, req)
    return Response(content=wav_bytes, media_type="audio/wav")
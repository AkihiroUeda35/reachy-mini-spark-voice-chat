from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import wave
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr
from websockets import connect as ws_connect

from config import project_root, service_url
from openai_model_registry import resolve_model

CHAT_BASE_URL = ""
CHAT_API_KEY = ""
CHAT_MODEL: str | None = None
CHAT_MODEL_FALLBACK = ""
TTS_BASE_URL = ""
TTS_API_KEY = ""
VOICE = ""
TTS_MODEL: str | None = None
TTS_MODEL_FALLBACK = ""
TTS_TASK_TYPE = ""
TTS_LANGUAGE = ""
TTS_TEMPERATURE: float | None = None
TTS_ALPHA: float | None = None
TTS_BETA: float | None = None
TTS_SAMPLE_RATE = 24000
TTS_USE_REALTIME_STREAMING = True
TTS_INSTRUCTIONS = ""
OUT_DIR = project_root() / "data"
OUT_PATH = OUT_DIR / "spark_voice_chat.wav"


def refresh_settings() -> None:
    global CHAT_BASE_URL, CHAT_API_KEY, CHAT_MODEL, CHAT_MODEL_FALLBACK
    global TTS_BASE_URL, TTS_API_KEY, VOICE, TTS_MODEL, TTS_MODEL_FALLBACK
    global TTS_TASK_TYPE, TTS_LANGUAGE, TTS_TEMPERATURE, TTS_ALPHA, TTS_BETA, TTS_SAMPLE_RATE, TTS_USE_REALTIME_STREAMING
    global TTS_INSTRUCTIONS, OUT_DIR, OUT_PATH

    CHAT_BASE_URL = service_url("llm", "http://localhost:8010/v1")
    CHAT_API_KEY = os.environ.get("CHAT_API_KEY", os.environ.get("CHAT_DEEPSEEK_API_KEY", "token-abc"))
    CHAT_MODEL = os.environ.get("CHAT_MODEL")
    CHAT_MODEL_FALLBACK = os.environ.get("CHAT_MODEL_FALLBACK", "spark")
    TTS_BASE_URL = service_url("tts", "http://localhost:8020/v1")
    TTS_API_KEY = os.environ.get("TTS_API_KEY", "local")
    VOICE = os.environ.get("TTS_VOICE", "default")
    TTS_MODEL = os.environ.get("TTS_MODEL")
    TTS_MODEL_FALLBACK = os.environ.get("TTS_MODEL_FALLBACK", "Respair/Tsukasa_Speech")
    TTS_TASK_TYPE = os.environ.get("TTS_TASK_TYPE", "CustomVoice")
    TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "Japanese")
    raw_tts_temperature = os.environ.get("TTS_TEMPERATURE", "").strip()
    TTS_TEMPERATURE = float(raw_tts_temperature) if raw_tts_temperature else None
    raw_tts_alpha = os.environ.get("TTS_ALPHA", "").strip()
    TTS_ALPHA = float(raw_tts_alpha) if raw_tts_alpha else None
    raw_tts_beta = os.environ.get("TTS_BETA", "").strip()
    TTS_BETA = float(raw_tts_beta) if raw_tts_beta else None
    TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
    TTS_USE_REALTIME_STREAMING = os.environ.get("TTS_USE_REALTIME_STREAMING", "1") != "0"
    TTS_INSTRUCTIONS = os.environ.get(
        "TTS_INSTRUCTIONS",
        "Speak in natural Japanese with clear emotion, smooth pacing, and a vivid but easy-to-listen tone.",
    )
    OUT_DIR = project_root() / os.environ.get("TTS_OUTPUT_DIR", "data")
    OUT_PATH = OUT_DIR / os.environ.get("TTS_OUTPUT_NAME", "spark_voice_chat.wav")


refresh_settings()


def resolve_chat_model() -> str:
    return resolve_model(
        base_url=CHAT_BASE_URL,
        api_key=CHAT_API_KEY,
        explicit_model=CHAT_MODEL,
        capability="chat",
        fallback_model=CHAT_MODEL_FALLBACK,
    )


def resolve_tts_model() -> str:
    return resolve_model(
        base_url=TTS_BASE_URL,
        api_key=TTS_API_KEY,
        explicit_model=TTS_MODEL,
        capability="speech",
        fallback_model=TTS_MODEL_FALLBACK,
    )


def _realtime_url() -> str:
    parsed = urlparse(TTS_BASE_URL)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + "/realtime"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _llm() -> ChatDeepSeek:
    return ChatDeepSeek(
        model=resolve_chat_model(),
        api_base=CHAT_BASE_URL,
        api_key=SecretStr(CHAT_API_KEY),
        temperature=0.2,
        max_tokens=1000,
        use_responses_api=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def _script_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an assistant that generates short, natural Japanese scripts for TTS.\n"
                    "Use ひらがな, not 漢字 for difficult words.\n"
                    "Do not output any explanations or non-script content."
                    " Only output the Japanese script itself."
                ),
            ),
            ("human", "{topic} をテーマに、音声読み上げ用の自然な文章を1つ作ってください。"),
        ]
    )
    return prompt | _llm() | StrOutputParser()


def _drain_ready_segments(buffer: str, *, final: bool) -> tuple[list[str], str]:
    segments: list[str] = []
    start = 0
    for index, char in enumerate(buffer):
        if char in "。！？!?\n":
            segment = buffer[start : index + 1].strip()
            if segment:
                segments.append(segment)
            start = index + 1

    remainder = buffer[start:]
    if final:
        tail = remainder.strip()
        if tail:
            segments.append(tail)
        remainder = ""

    return segments, remainder


KANJI_TO_HIRAGANA = {
    "清水": "きよみず",
    "祇園": "ぎおん",
    "竹林": "ちくりん",
}


def convert_kanji_to_hiragana(text: str) -> str:
    for kanji, hira in KANJI_TO_HIRAGANA.items():
        text = text.replace(kanji, hira)
    return text


def build_script(topic: str) -> str:
    chain = _script_chain()
    script = chain.invoke({"topic": topic}).strip()
    return convert_kanji_to_hiragana(script)


def stream_script(topic: str):
    chain = _script_chain()
    for chunk in chain.stream({"topic": topic}):
        yield convert_kanji_to_hiragana(str(chunk))


async def _consume_realtime_audio(websocket, text: str) -> bytes:
    await websocket.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
    )
    await websocket.send(json.dumps({"type": "response.create"}))

    audio = bytearray()
    while True:
        message = await websocket.recv()
        payload = json.loads(message)
        msg_type = payload.get("type")

        if msg_type == "response.audio.delta":
            delta = payload.get("delta") or ""
            if delta:
                audio.extend(base64.b64decode(delta))
            continue
        if msg_type == "response.done":
            return bytes(audio)
        if msg_type == "error":
            detail = payload.get("error") or {}
            raise RuntimeError(detail.get("message") or "Realtime TTS error")


async def synthesize_realtime_streaming(topic: str) -> tuple[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_text = ""
    sentence_buffer = ""
    pcm_audio = bytearray()

    async with ws_connect(_realtime_url(), max_size=None) as websocket:
        payload = json.loads(await websocket.recv())
        if payload.get("type") != "session.created":
            raise RuntimeError(f"Unexpected realtime event: {payload}")

        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": resolve_tts_model(),
                        "voice": VOICE,
                        "instructions": TTS_INSTRUCTIONS,
                        "task_type": TTS_TASK_TYPE,
                        "language": TTS_LANGUAGE,
                    },
                }
            )
        )

        while True:
            payload = json.loads(await websocket.recv())
            if payload.get("type") == "session.updated":
                break
            if payload.get("type") == "error":
                detail = payload.get("error") or {}
                raise RuntimeError(detail.get("message") or "Realtime session update failed")

        for chunk in stream_script(topic):
            token = str(chunk)
            if not token:
                continue
            sys.stdout.write(token)
            sys.stdout.flush()
            full_text += token
            sentence_buffer += token
            ready_segments, sentence_buffer = _drain_ready_segments(sentence_buffer, final=False)
            for segment in ready_segments:
                pcm_audio.extend(await _consume_realtime_audio(websocket, segment))

        if full_text and not full_text.endswith("\n"):
            sys.stdout.write("\n")

        ready_segments, _remainder = _drain_ready_segments(sentence_buffer, final=True)
        for segment in ready_segments:
            pcm_audio.extend(await _consume_realtime_audio(websocket, segment))

    with wave.open(str(OUT_PATH), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TTS_SAMPLE_RATE)
        wav_file.writeframes(bytes(pcm_audio))

    return full_text.strip(), OUT_PATH


def synthesize(text: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    response = httpx.post(
        f"{TTS_BASE_URL}/audio/speech",
        headers={"Authorization": f"Bearer {TTS_API_KEY}"},
        json={
            "model": resolve_tts_model(),
            "task_type": TTS_TASK_TYPE,
            "language": TTS_LANGUAGE,
            "voice": VOICE,
            "input": text,
            "instructions": TTS_INSTRUCTIONS,
            "response_format": "wav",
            "stream": False,
        },
        timeout=600,
    )
    response.raise_for_status()
    OUT_PATH.write_bytes(response.content)
    return OUT_PATH


def main() -> None:
    topic = os.environ.get("TTS_TOPIC", "京都の旅行スポット")
    if TTS_USE_REALTIME_STREAMING:
        script, output_path = asyncio.run(synthesize_realtime_streaming(topic))
        print(f"script: {script}")
        print(f"audio: {output_path}")
        return

    script = build_script(topic)
    print(f"script: {script}")
    output_path = synthesize(script)
    print(f"audio: {output_path}")
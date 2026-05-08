from __future__ import annotations

import argparse
import asyncio
import base64
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import uuid
import wave
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import httpx
import gradio as gr
import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pipecat.frames.frames import EndFrame, ErrorFrame, FunctionCallInProgressFrame, FunctionCallResultFrame, LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pydantic import SecretStr
from reachy_mini import ReachyMini
from rich.logging import RichHandler
from websockets import connect as ws_connect


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

importlib.import_module("lib.config").load_env()

asr_tools = importlib.import_module("lib.whisper_asr")
local_tts = importlib.import_module("lib.local_tts")
model_registry = importlib.import_module("lib.openai_model_registry")
reachy_tools = importlib.import_module("lib.reachy_conversation_tools")

build_langchain_tools = reachy_tools.build_langchain_tools
GUI_TOOL_NAMES = reachy_tools.GUI_TOOL_NAMES
ReachyToolRuntime = reachy_tools.ReachyToolRuntime
resolve_model = model_registry.resolve_model


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_DIR = APP_DIR / "profiles" / "default"
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "conversation"
COMMON_INSTRUCTIONS_FILE = APP_DIR / "profiles" / "common_instructions.txt"
DEFAULT_CHARACTER_FILE = DEFAULT_PROFILE_DIR / "character.txt"
DEFAULT_VOICE_FILE = DEFAULT_PROFILE_DIR / "voice.txt"
DEFAULT_TOOLS_FILE = DEFAULT_PROFILE_DIR / "tools.txt"
DEFAULT_VOICE = "Ono_Anna"

VOICE_CHOICES: list[tuple[str, str]] = [
    ("Vivian", "Bright, slightly edgy young female voice. [Chinese]"),
    ("Serena", "Warm, gentle young female voice. [Chinese]"),
    ("Uncle_Fu", "Seasoned male voice with a low, mellow timbre. [Chinese]"),
    ("Dylan", "Youthful Beijing male voice with a clear, natural timbre. [Chinese - Beijing Dialect]"),
    ("Eric", "Lively Chengdu male voice with a slightly husky brightness. [Chinese - Sichuan Dialect]"),
    ("Ryan", "Dynamic male voice with strong rhythmic drive. [English]"),
    ("Aiden", "Sunny American male voice with a clear midrange. [English]"),
    ("Ono_Anna", "Playful Japanese female voice with a light, nimble timbre. [Japanese]"),
    ("Sohee", "Warm Korean female voice with rich emotion. [Korean]"),
]


def _read_text_file(path: Path, fallback: str = "") -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return fallback


COMMON_SYSTEM_PROMPT = _read_text_file(COMMON_INSTRUCTIONS_FILE)
DEFAULT_CHARACTER_PROMPT = _read_text_file(DEFAULT_CHARACTER_FILE)
DEFAULT_SYSTEM_PROMPT = "\n\n".join(part for part in (COMMON_SYSTEM_PROMPT, DEFAULT_CHARACTER_PROMPT) if part).strip()

CHAT_BASE_URL = local_tts.CHAT_BASE_URL
CHAT_API_KEY = local_tts.CHAT_API_KEY
CHAT_MODEL = local_tts.CHAT_MODEL
CHAT_MODEL_FALLBACK = local_tts.CHAT_MODEL_FALLBACK
TTS_BASE_URL = local_tts.TTS_BASE_URL
TTS_API_KEY = local_tts.TTS_API_KEY
TTS_MODEL = local_tts.TTS_MODEL
TTS_MODEL_FALLBACK = local_tts.TTS_MODEL_FALLBACK
TTS_TASK_TYPE = local_tts.TTS_TASK_TYPE
TTS_LANGUAGE = local_tts.TTS_LANGUAGE
TTS_INSTRUCTIONS = local_tts.TTS_INSTRUCTIONS
TTS_SAMPLE_RATE = local_tts.TTS_SAMPLE_RATE
VOICE = local_tts.VOICE

TTS_TIMEOUT = float(importlib.import_module("os").environ.get("TTS_TIMEOUT", "600"))
TTS_STREAM_CHUNK_BYTES = int(importlib.import_module("os").environ.get("TTS_STREAM_CHUNK_BYTES", "8192"))
TTS_TRANSPORT = importlib.import_module("os").environ.get("TTS_TRANSPORT", "realtime")
SHUTDOWN_STEP_TIMEOUT_S = 2.0


@dataclass
class SynthesizedAudio:
    pcm16_bytes: bytes
    sample_rate: int
    num_channels: int


@dataclass
class PipelineResult:
    assistant_text: str
    audio: SynthesizedAudio


@dataclass
class RuntimeSettings:
    profiles_dir: Path
    active_profile: str
    enabled_tools: list[str]
    active_character_prompt: str
    active_instructions: str
    active_voice: str

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0

    def snapshot(self) -> tuple[str, list[str], str, str, str, int]:
        with self._lock:
            return (
                self.active_profile,
                list(self.enabled_tools),
                self.active_character_prompt,
                self.active_instructions,
                self.active_voice,
                self._version,
            )

    def update(
        self,
        profile: str,
        enabled_tools: list[str],
        character_prompt: str,
        instructions: str,
        voice: str,
    ) -> tuple[str, list[str], str, str, str, int]:
        normalized = [tool for tool in GUI_TOOL_NAMES if tool in enabled_tools]
        with self._lock:
            self.active_profile = profile
            self.enabled_tools = normalized
            self.active_character_prompt = character_prompt.strip()
            self.active_instructions = instructions.strip()
            self.active_voice = voice
            self._version += 1
            return (
                self.active_profile,
                list(self.enabled_tools),
                self.active_character_prompt,
                self.active_instructions,
                self.active_voice,
                self._version,
            )


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reachy Mini conversation app powered by local ASR, LLM, and TTS servers.")
    parser.add_argument("--robot-name", help="Optional Reachy Mini robot name when multiple robots are available.")
    parser.add_argument("--profile", default="default", help="Profile name under apps/conversation/profiles.")
    parser.add_argument("--profiles-dir", default=str(APP_DIR / "profiles"), help="Directory containing profile folders.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory for captured audio and snapshots.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--gradio", action=argparse.BooleanOptionalAction, default=True, help="Launch a simplified Gradio GUI for character and tool selection.")
    parser.add_argument("--gradio-host", default="127.0.0.1", help="Host interface for the Gradio GUI.")
    parser.add_argument("--gradio-port", type=int, default=7860, help="Port for the Gradio GUI.")
    parser.add_argument("--auto-install-optional-deps", action=argparse.BooleanOptionalAction, default=True, help="Automatically install missing optional dependencies such as dance and vision backends.")
    parser.add_argument("--wake-up", action=argparse.BooleanOptionalAction, default=True, help="Wake the robot up on startup.")
    parser.add_argument("--history-turns", type=int, default=6, help="How many previous user+assistant turns to keep.")
    parser.add_argument("--motion-duration", type=float, default=0.8, help="Duration for simple head movements in seconds.")

    parser.add_argument("--base-url", default=asr_tools.ASR_BASE_URL, help="OpenAI-compatible STT base URL.")
    parser.add_argument("--api-key", default=asr_tools.ASR_API_KEY, help="Bearer token for the STT endpoint.")
    parser.add_argument("--model", default=asr_tools.ASR_MODEL, help="STT model name.")
    parser.add_argument("--transport", choices=["realtime", "http"], default=asr_tools.ASR_TRANSPORT, help="STT transport to use.")
    parser.add_argument("--language", default=asr_tools.ASR_LANGUAGE, help="Language hint for STT.")
    parser.add_argument("--prompt", help="Optional STT transcription prompt.")
    parser.add_argument("--response-format", choices=["text", "json", "verbose_json"], default="text", help="STT response format.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for HTTP STT.")
    parser.add_argument("--word-timestamps", action="store_true", help="Request word timestamps when supported.")
    parser.add_argument("--sample-rate", type=int, default=asr_tools.ASR_SAMPLE_RATE, help="Sample rate sent to the STT service.")
    parser.add_argument("--realtime-chunk-ms", type=int, default=asr_tools.ASR_REALTIME_CHUNK_MS, help="Realtime STT chunk size in milliseconds.")
    parser.add_argument("--vad-threshold", type=float, default=asr_tools.ASR_VAD_THRESHOLD, help="RMS threshold for robot audio VAD.")
    parser.add_argument("--vad-start-ms", type=int, default=asr_tools.ASR_VAD_START_MS, help="Speech duration required to trigger capture start.")
    parser.add_argument("--vad-end-ms", type=int, default=asr_tools.ASR_VAD_END_MS, help="Silence duration required to trigger capture end.")
    parser.add_argument("--vad-preroll-ms", type=int, default=asr_tools.ASR_VAD_PREROLL_MS, help="Audio to keep before VAD start.")
    parser.add_argument("--vad-max-seconds", type=float, default=asr_tools.ASR_VAD_MAX_SECONDS, help="Maximum capture duration per turn.")
    parser.add_argument("--listen-timeout-seconds", type=float, default=20.0, help="Maximum time to wait for a new utterance.")
    parser.add_argument("--audio-poll-interval-ms", type=float, default=10.0, help="Polling interval when Reachy media has no pending audio frame.")
    parser.add_argument("--save-transcripts", action=argparse.BooleanOptionalAction, default=False, help="Save STT payloads under the data directory.")

    parser.add_argument("--chat-base-url", default=CHAT_BASE_URL, help="OpenAI-compatible chat base URL.")
    parser.add_argument("--chat-api-key", default=CHAT_API_KEY, help="Bearer token for the chat endpoint.")
    parser.add_argument("--chat-model", default=CHAT_MODEL, help="Chat model name.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for the conversation model.")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM sampling temperature.")
    parser.add_argument("--max-completion-tokens", type=int, default=400, help="Maximum completion tokens for the LLM response.")
    parser.add_argument("--max-tool-rounds", type=int, default=5, help="Maximum number of LLM/tool rounds per turn.")

    parser.add_argument("--tts-base-url", default=TTS_BASE_URL, help="OpenAI-compatible TTS base URL.")
    parser.add_argument("--tts-api-key", default=TTS_API_KEY, help="Bearer token for the TTS endpoint.")
    parser.add_argument("--tts-model", default=TTS_MODEL, help="TTS model name.")
    parser.add_argument("--tts-task-type", default=TTS_TASK_TYPE, help="TTS task type.")
    parser.add_argument("--tts-language", default=TTS_LANGUAGE, help="TTS language.")
    parser.add_argument("--voice", default=VOICE, help="TTS voice name.")
    parser.add_argument("--tts-instructions", default=TTS_INSTRUCTIONS, help="Instructions sent to the TTS wrapper.")
    parser.add_argument("--tts-sample-rate", type=int, default=TTS_SAMPLE_RATE, help="Expected TTS sample rate.")
    parser.add_argument("--tts-transport", choices=["realtime", "http"], default=TTS_TRANSPORT, help="TTS transport used for streaming.")
    parser.add_argument("--save-replies", action=argparse.BooleanOptionalAction, default=False, help="Save synthesized assistant replies as WAV files.")
    return parser.parse_args()


def parse_tools_file(profile_dir: Path) -> list[str]:
    tools_file = profile_dir / "tools.txt"
    if not tools_file.is_file():
        tools_file = DEFAULT_TOOLS_FILE
    names: list[str] = []
    for raw_line in tools_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def list_profile_names(profiles_dir: Path) -> list[str]:
    names: list[str] = []
    for entry in sorted(profiles_dir.iterdir() if profiles_dir.is_dir() else []):
        if entry.is_dir() and ((entry / "character.txt").is_file() or (entry / "instructions.txt").is_file()):
            names.append(entry.name)
    if "default" not in names and DEFAULT_PROFILE_DIR.is_dir():
        names.append("default")
    return sorted(set(names))


def active_tools_for_profile(profiles_dir: Path, profile: str) -> list[str]:
    return [tool for tool in parse_tools_file(profiles_dir / profile) if tool in GUI_TOOL_NAMES]


def compose_system_prompt(character_prompt: str) -> str:
    parts = [COMMON_SYSTEM_PROMPT.strip(), character_prompt.strip()]
    return "\n\n".join(part for part in parts if part).strip()


def normalize_profile_name(profile: str) -> str:
    normalized = profile.strip().replace("/", "-").replace("\\", "-")
    normalized = "_".join(normalized.split())
    if normalized in {"", ".", ".."}:
        return ""
    return normalized


def load_profile_character_prompt_by_name(profiles_dir: Path, profile: str, fallback: str) -> str:
    profile_dir = profiles_dir / profile
    character_file = profile_dir / "character.txt"
    if character_file.is_file():
        return character_file.read_text(encoding="utf-8").strip()
    prompt_file = profiles_dir / profile / "instructions.txt"
    if prompt_file.is_file():
        return prompt_file.read_text(encoding="utf-8").strip()
    return fallback


def load_profile_prompt_by_name(profiles_dir: Path, profile: str, fallback: str) -> str:
    character_prompt = load_profile_character_prompt_by_name(profiles_dir, profile, fallback)
    return compose_system_prompt(character_prompt)


def load_profile_voice_by_name(profiles_dir: Path, profile: str, fallback: str = DEFAULT_VOICE) -> str:
    profile_dir = profiles_dir / profile
    voice = _read_text_file(profile_dir / "voice.txt", fallback).strip()
    if any(voice == name for name, _description in VOICE_CHOICES):
        return voice
    return fallback


def save_profile_definition(
    profiles_dir: Path,
    profile: str,
    character_prompt: str,
    selected_tools: list[str],
    voice: str,
) -> Path:
    profile_dir = profiles_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "character.txt").write_text(character_prompt.strip() + "\n", encoding="utf-8")
    ordered_tools = [tool for tool in GUI_TOOL_NAMES if tool in selected_tools]
    (profile_dir / "tools.txt").write_text("\n".join(ordered_tools) + "\n", encoding="utf-8")
    (profile_dir / "voice.txt").write_text(voice.strip() + "\n", encoding="utf-8")
    return profile_dir


def load_profile_prompt(args: argparse.Namespace) -> str:
    profiles_dir = Path(args.profiles_dir).expanduser().resolve()
    return load_profile_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content or "")


def _to_langchain_messages(messages: list[Any]) -> list[Any]:
    converted: list[Any] = []
    for message in messages:
        if isinstance(message, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
            converted.append(message)
            continue
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content", "")
        if role == "user":
            converted.append(HumanMessage(content=_content_text(content)))
        elif role == "assistant":
            converted.append(AIMessage(content=_content_text(content)))
        elif role == "system":
            converted.append(SystemMessage(content=_content_text(content)))
    return converted


def _build_llm_context(history: list[dict[str, str]], transcript_text: str) -> LLMContext:
    messages: list[dict[str, Any]] = [*history, {"role": "user", "content": transcript_text}]
    return LLMContext(messages=cast(Any, messages))


def _realtime_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + "/realtime"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


async def _wait_for_event(websocket, event_types: set[str]) -> dict[str, Any]:
    while True:
        payload = json.loads(await websocket.recv())
        if payload.get("type") == "error":
            detail = payload.get("error") or {}
            raise RuntimeError(detail.get("message") or "Realtime TTS error")
        if payload.get("type") in event_types:
            return payload


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


def _pcm16_to_float32(audio_bytes: bytes, num_channels: int) -> np.ndarray:
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        return audio.reshape(-1, num_channels)
    return audio


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    if audio.ndim == 1:
        target_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
        source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)

    channels = [
        _resample_audio(audio[:, channel_index], source_rate, target_rate)
        for channel_index in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 2 and normalized.shape[1] > normalized.shape[0]:
        normalized = normalized.T
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim == 2 and normalized.shape[1] == 1:
        return normalized[:, 0]
    return normalized.mean(axis=1)


def _prepared_audio_from_pcm16(audio_bytes: bytes, *, sample_rate: int, stem: str) -> Any:
    return asr_tools.PreparedAudio(
        source_stem=stem,
        filename=f"{stem}.wav",
        mime_type="audio/wav",
        upload_bytes=asr_tools._wav_bytes(audio_bytes, sample_rate),
        pcm16_bytes=audio_bytes,
        sample_rate=sample_rate,
    )


class RichTraceProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="RichTraceProcessor")
        self._logger = logging.getLogger("conversation.trace")

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._logger.info("[bold cyan]LLM[/] response started")
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._logger.info(
                "[bold magenta]LLM tool[/] %s %s",
                frame.function_name,
                _json_preview(frame.arguments),
            )
        elif isinstance(frame, FunctionCallResultFrame):
            self._logger.info(
                "[bold magenta]Tool result[/] %s %s",
                frame.function_name,
                _json_preview(frame.result),
            )
        elif isinstance(frame, ErrorFrame):
            self._logger.error("[bold red]Pipeline error[/] %s", frame.error)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._logger.info("[bold cyan]LLM[/] response finished")
        await self.push_frame(frame, direction)


def _json_preview(payload: Any, max_len: int = 240) -> str:
    raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


TOOL_CHECK_SENTINEL = "No"


class ReachyLangChainProcessor(FrameProcessor):
    def __init__(self, args: argparse.Namespace, tools: list[BaseTool]):
        super().__init__(name="ReachyLangChainProcessor")
        self._logger = logging.getLogger("conversation.llm")
        self._tools_by_name = {tool.name: tool for tool in tools}
        self._model = ChatOpenAI(
            model=args.chat_model,
            base_url=args.chat_base_url,
            api_key=SecretStr(args.chat_api_key),
            temperature=args.llm_temperature,
            max_completion_tokens=args.max_completion_tokens,
            use_responses_api=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).bind_tools(tools)
        self._system_prompt = args.system_prompt
        self._max_tool_rounds = args.max_tool_rounds

    async def _follow_up_for_tool_use(self, messages: list[Any]):
        return await self._model.ainvoke(
            [
                *messages,
                HumanMessage(
                    content=(
                        "Call a tool now only if one is still needed to continue this reply. "
                        f"If no tool call is needed, reply with exactly {TOOL_CHECK_SENTINEL}."
                    )
                ),
            ]
        )

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        messages = [SystemMessage(content=self._system_prompt), *_to_langchain_messages(frame.context.messages)]
        await self.push_frame(LLMFullResponseStartFrame(), direction)

        try:
            for round_index in range(self._max_tool_rounds):
                self._logger.info("[bold cyan]LLM[/] round %d", round_index + 1)
                response = await self._model.ainvoke(messages)
                text = _content_text(response.content).strip()
                tool_calls = getattr(response, "tool_calls", None) or []
                messages.append(response)

                if not tool_calls and text and self._tools_by_name:
                    self._logger.info("[bold cyan]LLM[/] tool follow-up check")
                    follow_up = await self._follow_up_for_tool_use(messages)
                    follow_up_tool_calls = getattr(follow_up, "tool_calls", None) or []
                    if follow_up_tool_calls:
                        response = follow_up
                        tool_calls = follow_up_tool_calls
                        messages.append(follow_up)
                    else:
                        self._logger.debug("Tool follow-up check ended without a tool call")

                if not tool_calls:
                    if text:
                        await self.push_frame(LLMTextFrame(text=text), direction)
                    break

                group_id = uuid.uuid4().hex
                for tool_call in tool_calls:
                    tool_name = str(tool_call.get("name") or "")
                    tool_args = tool_call.get("args") or {}
                    tool_call_id = str(tool_call.get("id") or uuid.uuid4().hex)
                    tool = self._tools_by_name.get(tool_name)
                    if tool is None:
                        result = {"error": f"unknown tool: {tool_name}"}
                    else:
                        await self.push_frame(
                            FunctionCallInProgressFrame(
                                function_name=tool_name,
                                tool_call_id=tool_call_id,
                                arguments=tool_args,
                                group_id=group_id,
                            ),
                            direction,
                        )
                        try:
                            result = await tool.ainvoke(tool_args)
                        except Exception as exc:
                            result = {"error": f"{type(exc).__name__}: {exc}"}
                    await self.push_frame(
                        FunctionCallResultFrame(
                            function_name=tool_name,
                            tool_call_id=tool_call_id,
                            arguments=tool_args,
                            result=result,
                        ),
                        direction,
                    )
                    messages.append(ToolMessage(content=_json_preview(result, max_len=4000), tool_call_id=tool_call_id))
            else:
                raise RuntimeError("LLM exceeded the maximum tool-call rounds.")
        except Exception as exc:
            await self.push_frame(ErrorFrame(error=f"LangChain agent failed: {exc}"), direction)
        finally:
            await self.push_frame(LLMFullResponseEndFrame(), direction)


class ReachyTTSProcessor(FrameProcessor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(name="ReachyTTSProcessor")
        self._args = args
        self._logger = logging.getLogger("conversation.tts")
        self._pending_text = ""
        self._segment_queue: asyncio.Queue[str | None] | None = None
        self._worker_task: asyncio.Task[None] | None = None

    async def _stop_worker(self) -> None:
        if self._segment_queue is not None:
            await self._segment_queue.put(None)
        if self._worker_task is not None:
            with suppress(asyncio.CancelledError):
                await self._worker_task
        self._worker_task = None
        self._segment_queue = None

    async def _tts_worker(self, direction: FrameDirection) -> None:
        if self._segment_queue is None:
            return
        try:
            if self._args.tts_transport == "realtime":
                async with ReachyRealtimeTTSSession(self._args) as websocket:
                    while True:
                        segment = await self._segment_queue.get()
                        try:
                            if segment is None:
                                return
                            self._logger.info("[bold green]TTS[/] synthesizing segment %s", segment)
                            async for audio_frame in synthesize_realtime_audio_stream(self._args, websocket, segment):
                                await self.push_frame(audio_frame, direction)
                        finally:
                            self._segment_queue.task_done()
            else:
                while True:
                    segment = await self._segment_queue.get()
                    try:
                        if segment is None:
                            return
                        self._logger.info("[bold green]TTS[/] synthesizing segment %s", segment)
                        async for audio_frame in synthesize_audio_stream(self._args, segment):
                            await self.push_frame(audio_frame, direction)
                    finally:
                        self._segment_queue.task_done()
        except Exception as exc:
            await self.push_frame(ErrorFrame(error=f"Streaming TTS failed: {exc}"), direction)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._logger.info("[bold green]TTS[/] response stream started")
            self._pending_text = ""
            self._segment_queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._tts_worker(direction))
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            if frame.text:
                self._pending_text += frame.text
                segments, remainder = _drain_ready_segments(self._pending_text, final=False)
                self._pending_text = remainder
                if self._segment_queue is not None:
                    for segment in segments:
                        await self._segment_queue.put(segment)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._segment_queue is not None:
                segments, remainder = _drain_ready_segments(self._pending_text, final=True)
                self._pending_text = remainder
                for segment in segments:
                    await self._segment_queue.put(segment)
                await self._segment_queue.join()
                await self._stop_worker()
            self._logger.info("[bold green]TTS[/] response stream finished")
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


class ReachyRealtimeTTSSession:
    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._websocket = None

    async def __aenter__(self):
        self._websocket = await ws_connect(_realtime_url(self._args.tts_base_url), max_size=None)
        payload = json.loads(await self._websocket.recv())
        if payload.get("type") != "session.created":
            raise RuntimeError(f"Unexpected realtime event: {payload}")
        await self._websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": self._args.tts_model,
                        "voice": self._args.voice,
                        "instructions": self._args.tts_instructions,
                        "task_type": self._args.tts_task_type,
                        "language": self._args.tts_language,
                    },
                }
            )
        )
        await _wait_for_event(self._websocket, {"session.updated"})
        return self._websocket

    async def __aexit__(self, exc_type, exc, tb):
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None


class ReachyAudioPlayer(FrameProcessor):
    def __init__(self, robot: ReachyMini):
        super().__init__(name="ReachyAudioPlayer")
        self._robot = robot
        self._logger = logging.getLogger("conversation.audio")
        self._started = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
            waveform = _pcm16_to_float32(frame.audio, frame.num_channels)
            output_rate = self._robot.media.get_output_audio_samplerate()
            if frame.sample_rate != output_rate:
                waveform = _resample_audio(waveform, frame.sample_rate, output_rate)
            if not self._started:
                self._logger.info("[bold yellow]Audio[/] streaming assistant reply to Reachy speaker")
                self._started = True
            await asyncio.to_thread(self._robot.media.push_audio_sample, waveform)
        elif isinstance(frame, EndFrame):
            self._started = False
        await self.push_frame(frame, direction)


class ResultCollector(FrameProcessor):
    def __init__(self):
        super().__init__(name="ResultCollector")
        self.assistant_chunks: list[str] = []
        self.audio_chunks: list[bytes] = []
        self.sample_rate: int | None = None
        self.num_channels: int | None = None
        self.errors: list[str] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame) and frame.text:
            self.assistant_chunks.append(frame.text)
        elif isinstance(frame, OutputAudioRawFrame):
            if self.sample_rate is None:
                self.sample_rate = frame.sample_rate
                self.num_channels = frame.num_channels
            if self.sample_rate == frame.sample_rate and self.num_channels == frame.num_channels:
                self.audio_chunks.append(frame.audio)
        elif isinstance(frame, ErrorFrame):
            self.errors.append(frame.error)
        await self.push_frame(frame, direction)


class PipelineTerminator(FrameProcessor):
    def __init__(self):
        super().__init__(name="PipelineTerminator")
        self._task: PipelineTask | None = None
        self._tool_calls_in_progress = 0
        self._end_queued = False

    def bind_task(self, task: PipelineTask) -> None:
        self._task = task

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, FunctionCallInProgressFrame):
            self._tool_calls_in_progress += 1
        elif isinstance(frame, FunctionCallResultFrame):
            self._tool_calls_in_progress = max(0, self._tool_calls_in_progress - 1)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if not self._end_queued and self._tool_calls_in_progress == 0 and self._task is not None:
                self._end_queued = True
                await self._task.queue_frame(EndFrame())
        await self.push_frame(frame, direction)


async def synthesize_audio_stream(args: argparse.Namespace, text: str):
    payload = {
        "model": args.tts_model,
        "task_type": args.tts_task_type,
        "language": args.tts_language,
        "voice": args.voice,
        "input": text,
        "instructions": args.tts_instructions,
        "response_format": "pcm",
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=TTS_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{args.tts_base_url.rstrip('/')}/audio/speech",
            headers={"Authorization": f"Bearer {args.tts_api_key}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=TTS_STREAM_CHUNK_BYTES):
                if chunk:
                    yield OutputAudioRawFrame(audio=chunk, sample_rate=args.tts_sample_rate, num_channels=1)


async def synthesize_realtime_audio_stream(args: argparse.Namespace, websocket, text: str):
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
    await _wait_for_event(websocket, {"conversation.item.created"})
    await websocket.send(json.dumps({"type": "response.create"}))
    while True:
        payload = json.loads(await websocket.recv())
        msg_type = payload.get("type")
        if msg_type == "error":
            detail = payload.get("error") or {}
            raise RuntimeError(detail.get("message") or "Realtime TTS error")
        if msg_type == "response.audio.delta":
            delta = payload.get("delta") or ""
            if delta:
                yield OutputAudioRawFrame(audio=base64.b64decode(delta), sample_rate=args.tts_sample_rate, num_channels=1)
            continue
        if msg_type == "response.done":
            return


def _create_turn_pipeline_runner() -> PipelineRunner:
    return PipelineRunner(handle_sigint=False, handle_sigterm=False)


async def run_pipeline(
    args: argparse.Namespace,
    runtime: ReachyToolRuntime,
    transcript_text: str,
    history: list[dict[str, str]],
    enabled_tool_names: list[str],
) -> PipelineResult:
    tools = build_langchain_tools(runtime, enabled_tool_names)
    llm = ReachyLangChainProcessor(args, tools)
    trace = RichTraceProcessor()
    tts = ReachyTTSProcessor(args)
    collector = ResultCollector()
    audio_player = ReachyAudioPlayer(runtime.robot)
    terminator = PipelineTerminator()
    pipeline = Pipeline([llm, trace, tts, collector, audio_player, terminator])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=args.sample_rate, audio_out_sample_rate=args.tts_sample_rate),
        idle_timeout_secs=60,
    )
    terminator.bind_task(task)
    runner = _create_turn_pipeline_runner()
    runner_task = asyncio.create_task(runner.run(task))
    await task.queue_frame(LLMContextFrame(_build_llm_context(history, transcript_text)))
    await runner_task

    if collector.errors:
        raise RuntimeError("; ".join(collector.errors))
    assistant_text = "".join(collector.assistant_chunks).strip()
    if not assistant_text:
        raise RuntimeError("LLM returned an empty response.")
    if not collector.audio_chunks or collector.sample_rate is None or collector.num_channels is None:
        raise RuntimeError("TTS returned no audio.")
    return PipelineResult(
        assistant_text=assistant_text,
        audio=SynthesizedAudio(
            pcm16_bytes=b"".join(collector.audio_chunks),
            sample_rate=collector.sample_rate,
            num_channels=collector.num_channels,
        ),
    )


async def capture_robot_utterance(robot: ReachyMini, args: argparse.Namespace) -> Any | None:
    logger = logging.getLogger("conversation.asr")
    input_rate = robot.media.get_input_audio_samplerate()
    poll_interval_s = max(0.001, args.audio_poll_interval_ms / 1000.0)
    listen_started_at = perf_counter()
    capture_started_at: float | None = None

    speech_ms = 0.0
    silence_ms = 0.0
    speech_active = False
    preroll: deque[tuple[float, bytes]] = deque()
    preroll_ms = 0.0
    captured_chunks: list[bytes] = []

    logger.info(
        "[bold blue]ASR[/] waiting for speech threshold=%.4f start=%dms end=%dms preroll=%dms",
        args.vad_threshold,
        args.vad_start_ms,
        args.vad_end_ms,
        args.vad_preroll_ms,
    )

    while True:
        if capture_started_at is None and perf_counter() - listen_started_at >= args.listen_timeout_seconds:
            logger.debug("No speech detected before listen timeout")
            return None

        chunk = await asyncio.to_thread(robot.media.get_audio_sample)
        if chunk is None:
            await asyncio.sleep(poll_interval_s)
            continue

        mono = _mono_audio(chunk)
        mono = _resample_audio(mono, input_rate, args.sample_rate)
        duration_ms = (len(mono) / args.sample_rate) * 1000.0
        rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float32))) if mono.size else 0.0
        pcm_chunk = asr_tools._pcm16_bytes(mono)

        if not speech_active:
            preroll.append((duration_ms, pcm_chunk))
            preroll_ms += duration_ms
            while preroll and preroll_ms > max(args.vad_preroll_ms, args.vad_start_ms):
                old_duration, _old_chunk = preroll.popleft()
                preroll_ms -= old_duration

        if rms >= args.vad_threshold:
            speech_ms += duration_ms
            silence_ms = 0.0
        else:
            silence_ms += duration_ms
            if not speech_active:
                speech_ms = 0.0

        if not speech_active and speech_ms >= args.vad_start_ms:
            speech_active = True
            capture_started_at = perf_counter()
            logger.info("[bold blue]ASR[/] speech detected")
            for _duration, buffered in preroll:
                captured_chunks.append(buffered)
            preroll.clear()
            preroll_ms = 0.0

        if speech_active:
            captured_chunks.append(pcm_chunk)
            if silence_ms >= args.vad_end_ms:
                logger.info("[bold blue]ASR[/] end of utterance detected")
                break
            if capture_started_at is not None and perf_counter() - capture_started_at >= args.vad_max_seconds:
                logger.info("[bold blue]ASR[/] forcing commit after %.1fs", args.vad_max_seconds)
                break

    if not captured_chunks:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _prepared_audio_from_pcm16(b"".join(captured_chunks), sample_rate=args.sample_rate, stem=f"reachy_turn_{timestamp}")


async def transcribe_captured_audio(args: argparse.Namespace, audio: Any) -> str | dict:
    logger = logging.getLogger("conversation.asr")
    logger.info("[bold blue]ASR[/] sending %.2fs of audio to %s", len(audio.pcm16_bytes) / 2 / audio.sample_rate, args.base_url)
    started_at = perf_counter()
    if args.transport == "realtime":
        payload = await asr_tools.transcribe_realtime(args, audio)
    else:
        payload = await asyncio.to_thread(asr_tools.transcribe_http, args, audio)
    logger.info("[bold blue]ASR[/] transcription finished in %.2fs", perf_counter() - started_at)
    return payload


def resolve_runtime_models(args: argparse.Namespace) -> None:
    args.model = resolve_model(
        base_url=args.base_url,
        api_key=args.api_key,
        explicit_model=args.model,
        capability="transcription",
        fallback_model=asr_tools.ASR_MODEL_FALLBACK,
    )
    args.chat_model = resolve_model(
        base_url=args.chat_base_url,
        api_key=args.chat_api_key,
        explicit_model=args.chat_model,
        capability="chat",
        fallback_model=CHAT_MODEL_FALLBACK,
    )
    args.tts_model = resolve_model(
        base_url=args.tts_base_url,
        api_key=args.tts_api_key,
        explicit_model=args.tts_model,
        capability="speech",
        fallback_model=TTS_MODEL_FALLBACK,
    )


def trim_history(history: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    max_messages = max(0, max_turns * 2)
    if max_messages <= 0:
        return []
    return history[-max_messages:]


def save_reply_audio(data_dir: Path, audio: SynthesizedAudio) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(audio.num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(audio.pcm16_bytes)
    return output_path


def _subprocess_stdout(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command failed: {' '.join(command)}")
    return result.stdout.strip()


def _list_listening_pids(port: int) -> list[int]:
    output = _subprocess_stdout(["lsof", "-ti", f"tcp:{port}"])
    pids: list[int] = []
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return sorted(set(pids))


def _read_process_command(pid: int) -> str:
    return _subprocess_stdout(["ps", "-o", "command=", "-p", str(pid)])


def _is_same_conversation_app_process(pid: int, app_path: Path) -> bool:
    if pid == os.getpid():
        return False
    command = _read_process_command(pid)
    app_markers = {
        str(app_path),
        str(app_path.relative_to(ROOT_DIR)),
        "apps/conversation/main.py",
    }
    if not any(marker in command for marker in app_markers):
        return False
    return "uv" in command or ".venv/bin/python" in command


def _free_gradio_port_if_same_uv_app(port: int, app_path: Path) -> list[int]:
    killed_pids: list[int] = []
    for pid in _list_listening_pids(port):
        with suppress(Exception):
            if _is_same_conversation_app_process(pid, app_path):
                os.kill(pid, signal.SIGKILL)
                killed_pids.append(pid)
    return killed_pids


async def _run_blocking_cleanup_step(logger: logging.Logger, name: str, func: Any, timeout_s: float = SHUTDOWN_STEP_TIMEOUT_S) -> None:
    error: list[BaseException] = []
    finished = threading.Event()

    def runner() -> None:
        try:
            func()
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=runner, name=f"cleanup:{name}", daemon=True)
    thread.start()

    deadline = perf_counter() + timeout_s
    while not finished.is_set():
        if perf_counter() >= deadline:
            logger.warning("Cleanup step timed out: %s", name)
            return
        await asyncio.sleep(0.05)

    if error:
        logger.exception("Cleanup step failed: %s", name, exc_info=(type(error[0]), error[0], error[0].__traceback__))


async def _run_async_cleanup_step(logger: logging.Logger, name: str, awaitable: Any, timeout_s: float = SHUTDOWN_STEP_TIMEOUT_S) -> None:
    try:
        await asyncio.wait_for(awaitable, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("Cleanup step timed out: %s", name)
    except Exception:
        logger.exception("Cleanup step failed: %s", name)


async def conversation_loop(args: argparse.Namespace) -> int:
    app_logger = logging.getLogger("conversation.app")
    asr_logger = logging.getLogger("conversation.asr")
    llm_logger = logging.getLogger("conversation.llm")

    profiles_dir = Path(args.profiles_dir).expanduser().resolve()
    args.system_prompt = load_profile_prompt(args)
    args.voice = load_profile_voice_by_name(profiles_dir, args.profile, args.voice or DEFAULT_VOICE)
    resolve_runtime_models(args)
    runtime_settings = RuntimeSettings(
        profiles_dir=profiles_dir,
        active_profile=args.profile,
        enabled_tools=active_tools_for_profile(profiles_dir, args.profile),
        active_character_prompt=load_profile_character_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT),
        active_instructions=load_profile_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT),
        active_voice=load_profile_voice_by_name(profiles_dir, args.profile, args.voice or DEFAULT_VOICE),
    )

    robot_kwargs: dict[str, Any] = {}
    if args.robot_name:
        robot_kwargs["robot_name"] = args.robot_name

    history: list[dict[str, str]] = []
    data_dir = Path(args.data_dir).expanduser().resolve()
    last_settings_version = 0
    stop_event = asyncio.Event()
    interrupted = False
    loop = asyncio.get_running_loop()
    conversation_task = asyncio.current_task()
    installed_signal_handlers: list[signal.Signals] = []

    def request_shutdown(signame: str) -> None:
        nonlocal interrupted
        interrupted = True
        stop_event.set()
        app_logger.info("[bold]Received %s, shutting down[/]", signame)
        if conversation_task is not None:
            conversation_task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, request_shutdown, signum.name)
            installed_signal_handlers.append(signum)

    gradio_handle = launch_gradio_ui(args, runtime_settings) if args.gradio else None

    app_logger.info("[bold]Starting Reachy Mini conversation app[/]")
    app_logger.info("Profile=%s chat_model=%s stt_model=%s tts_model=%s", args.profile, args.chat_model, args.model, args.tts_model)

    robot = ReachyMini(**robot_kwargs)
    robot.enable_motors()
    robot.wake_up()
    runtime = ReachyToolRuntime(
        robot,
        data_dir=data_dir,
        motion_duration_s=args.motion_duration,
        chat_base_url=args.chat_base_url,
        chat_api_key=args.chat_api_key,
        chat_model=args.chat_model,
        auto_install_optional_deps=args.auto_install_optional_deps,
    )
    try:
        if args.wake_up:
            app_logger.info("[bold]Waking up Reachy Mini[/]")
            await _run_blocking_cleanup_step(app_logger, "wake_up", robot.wake_up, timeout_s=3.0)

        await _run_blocking_cleanup_step(app_logger, "media.start_recording", robot.media.start_recording)
        await _run_blocking_cleanup_step(app_logger, "media.start_playing", robot.media.start_playing)
        app_logger.info("[bold]Reachy media pipelines started[/]")

        while not stop_event.is_set():
            active_profile, active_tools, _active_character_prompt, active_instructions, active_voice, settings_version = runtime_settings.snapshot()
            if settings_version != last_settings_version:
                history.clear()
                last_settings_version = settings_version
                app_logger.info("[bold]Conversation history reset[/] profile=%s", active_profile)
            args.profile = active_profile
            args.system_prompt = active_instructions
            args.voice = active_voice
            captured_audio = await capture_robot_utterance(robot, args)
            if captured_audio is None or stop_event.is_set():
                continue

            payload = await transcribe_captured_audio(args, captured_audio)
            transcript_text = payload if isinstance(payload, str) else str(payload.get("text", "")).strip()
            if not transcript_text:
                asr_logger.warning("[bold blue]ASR[/] empty transcript, skipping turn")
                continue

            asr_logger.info("[bold blue]ASR[/] transcript %s", transcript_text)

            if args.save_transcripts:
                saved = asr_tools.save_output(payload, argparse.Namespace(**{**vars(args), "output": None, "save_output": True}), captured_audio)
                if saved is not None:
                    asr_logger.info("[bold blue]ASR[/] transcript saved to %s", saved)

            llm_logger.info("[bold cyan]LLM[/] user %s", transcript_text)
            result = await run_pipeline(args, runtime, transcript_text, history, active_tools)
            llm_logger.info("[bold cyan]LLM[/] assistant %s", result.assistant_text)

            history.extend(
                [
                    {"role": "user", "content": transcript_text},
                    {"role": "assistant", "content": result.assistant_text},
                ]
            )
            history = trim_history(history, args.history_turns)

            if args.save_replies:
                saved_audio = save_reply_audio(data_dir, result.audio)
                logging.getLogger("conversation.tts").info("[bold green]TTS[/] reply saved to %s", saved_audio)
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        app_logger.info("[bold]Interrupted, shutting down[/]")
    finally:
        for signum in installed_signal_handlers:
            with suppress(NotImplementedError):
                loop.remove_signal_handler(signum)

        await _run_async_cleanup_step(app_logger, "runtime.shutdown", runtime.shutdown())
        if gradio_handle is not None:
            await _run_blocking_cleanup_step(app_logger, "gradio.close", gradio_handle.close)
        await _run_blocking_cleanup_step(app_logger, "media.stop_recording", robot.media.stop_recording)
        await _run_blocking_cleanup_step(app_logger, "media.stop_playing", robot.media.stop_playing)
        if args.wake_up and not interrupted:
            await _run_blocking_cleanup_step(app_logger, "goto_sleep", robot.goto_sleep, timeout_s=3.0)
        await _run_blocking_cleanup_step(app_logger, "media_manager.close", robot.media_manager.close)
        await _run_blocking_cleanup_step(app_logger, "client.disconnect", robot.client.disconnect)
    return 0


def build_gradio_ui(args: argparse.Namespace, runtime_settings: RuntimeSettings) -> gr.Blocks:
    profiles_dir = runtime_settings.profiles_dir
    profile_names = list_profile_names(profiles_dir)
    initial_profile, initial_tools, initial_character_prompt, _initial_prompt, initial_voice, _initial_version = runtime_settings.snapshot()
    voice_dropdown_choices = [(f"{name} - {description}", name) for name, description in VOICE_CHOICES]

    def on_profile_change(profile: str) -> tuple[str, list[str], str, str]:
        selected_profile = profile if profile in profile_names else initial_profile
        selected_tools = active_tools_for_profile(profiles_dir, selected_profile)
        character_prompt = load_profile_character_prompt_by_name(profiles_dir, selected_profile, DEFAULT_CHARACTER_PROMPT)
        voice = load_profile_voice_by_name(profiles_dir, selected_profile, DEFAULT_VOICE)
        status = f"Loaded character '{selected_profile}'. You can edit the character prompt, voice, save, or apply live."
        return character_prompt, selected_tools, voice, status

    def on_apply(profile: str, character_prompt: str, selected_tools: list[str], voice: str) -> str:
        if profile not in profile_names:
            return f"Unknown character '{profile}'."
        active_profile, enabled_tools, _character_prompt, _instructions, active_voice, _version = runtime_settings.update(
            profile,
            selected_tools,
            character_prompt,
            compose_system_prompt(character_prompt),
            voice,
        )
        return (
            f"Live runtime updated: character={active_profile}, voice={active_voice}, "
            f"tools={', '.join(enabled_tools) if enabled_tools else 'none'}"
        )

    def on_save(profile: str, character_prompt: str, selected_tools: list[str], voice: str) -> str:
        if profile not in profile_names:
            return f"Unknown character '{profile}'. Create it first."
        profile_dir = save_profile_definition(profiles_dir, profile, character_prompt, selected_tools, voice)
        runtime_settings.update(profile, selected_tools, character_prompt, compose_system_prompt(character_prompt), voice)
        return f"Saved character '{profile}' to {profile_dir}."

    def on_create(profile_name: str, character_prompt: str, selected_tools: list[str], voice: str):
        nonlocal profile_names
        normalized_name = normalize_profile_name(profile_name)
        if not normalized_name:
            return (
                gr.update(),
                character_prompt,
                selected_tools,
                voice,
                "Enter a valid character name.",
                profile_name,
            )

        profile_dir = save_profile_definition(
            profiles_dir,
            normalized_name,
            character_prompt or DEFAULT_CHARACTER_PROMPT,
            selected_tools or active_tools_for_profile(profiles_dir, initial_profile),
            voice or DEFAULT_VOICE,
        )
        profile_names = sorted(set([*profile_names, normalized_name]))
        saved_character_prompt = load_profile_character_prompt_by_name(profiles_dir, normalized_name, DEFAULT_CHARACTER_PROMPT)
        saved_tools = active_tools_for_profile(profiles_dir, normalized_name)
        saved_voice = load_profile_voice_by_name(profiles_dir, normalized_name, DEFAULT_VOICE)
        runtime_settings.update(
            normalized_name,
            saved_tools,
            saved_character_prompt,
            compose_system_prompt(saved_character_prompt),
            saved_voice,
        )
        return (
            gr.update(choices=profile_names, value=normalized_name),
            saved_character_prompt,
            saved_tools,
            saved_voice,
            f"Created character '{normalized_name}' at {profile_dir}.",
            "",
        )

    with gr.Blocks(title="Reachy Mini Conversation Settings") as demo:
        gr.Markdown(
            "# Reachy Mini Conversation\n"
            "Use this panel to switch character, edit only the character-specific prompt, choose a voice, and save while the local conversation loop is running."
        )
        with gr.Row():
            profile_dropdown = gr.Dropdown(label="Character", choices=profile_names, value=initial_profile)
            new_profile_box = gr.Textbox(label="New Character Name", placeholder="new_character", interactive=True)
            create_button = gr.Button("Create Character")
        character_box = gr.Textbox(label="Character Prompt", value=initial_character_prompt, lines=12, interactive=True)
        voice_dropdown = gr.Dropdown(label="Voice", choices=voice_dropdown_choices, value=initial_voice)
        tool_checkboxes = gr.CheckboxGroup(label="Enabled Tools", choices=GUI_TOOL_NAMES, value=initial_tools)
        status_box = gr.Textbox(label="Status", value="Ready.", interactive=False)
        with gr.Row():
            apply_button = gr.Button("Apply Live", variant="primary")
            save_button = gr.Button("Save Character")

        profile_dropdown.change(
            on_profile_change,
            inputs=[profile_dropdown],
            outputs=[character_box, tool_checkboxes, voice_dropdown, status_box],
        )
        apply_button.click(
            on_apply,
            inputs=[profile_dropdown, character_box, tool_checkboxes, voice_dropdown],
            outputs=[status_box],
        )
        save_button.click(
            on_save,
            inputs=[profile_dropdown, character_box, tool_checkboxes, voice_dropdown],
            outputs=[status_box],
        )
        create_button.click(
            on_create,
            inputs=[new_profile_box, character_box, tool_checkboxes, voice_dropdown],
            outputs=[profile_dropdown, character_box, tool_checkboxes, voice_dropdown, status_box, new_profile_box],
        )

    return demo


def launch_gradio_ui(args: argparse.Namespace, runtime_settings: RuntimeSettings):
    demo = build_gradio_ui(args, runtime_settings)
    app_path = APP_DIR / "main.py"
    killed_pids = _free_gradio_port_if_same_uv_app(args.gradio_port, app_path)
    if killed_pids:
        logging.getLogger("conversation.app").warning(
            "Killed existing conversation app process(es) on port %d: %s",
            args.gradio_port,
            ", ".join(str(pid) for pid in killed_pids),
        )
    logging.getLogger("conversation.app").info(
        "[bold]Starting Gradio GUI[/] http://%s:%d",
        args.gradio_host,
        args.gradio_port,
    )
    try:
        demo.launch(
            server_name=args.gradio_host,
            server_port=args.gradio_port,
            prevent_thread_lock=True,
            quiet=not args.debug,
            show_error=True,
            inbrowser=False,
        )
    except Exception:
        with suppress(Exception):
            demo.close()
        raise
    return demo


def main() -> int:
    args = parse_args()
    configure_logging(args.debug)
    try:
        return asyncio.run(conversation_loop(args))
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        logging.getLogger("conversation.app").error("HTTP error: %s %s", exc.response.status_code, detail)
        return 1
    except httpx.HTTPError as exc:
        logging.getLogger("conversation.app").error("HTTP error: %s", exc)
        return 1
    except Exception as exc:
        logging.getLogger("conversation.app").exception("Conversation app failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
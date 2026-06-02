from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import httpx
import numpy as np
try:
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
except ImportError:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    AIMessageChunk = AIMessage
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
from reachy_audio import HeadWobbler, get_wobble_origin_pose
from state import AssistantSpeechState, RuntimeSettings
from websockets import connect as ws_connect

from reachy_conversation_tools import ReachyToolRuntime, build_langchain_tools
from tts_pronunciation_overrides import TTS_PRONUNCIATION_OVERRIDES

TTS_TIMEOUT = float(os.environ.get("TTS_TIMEOUT", "600"))
TTS_STREAM_CHUNK_BYTES = int(os.environ.get("TTS_STREAM_CHUNK_BYTES", "8192"))
REACHY_AUDIO_PUSH_CHUNK_MS = int(
    os.environ.get("REACHY_AUDIO_PUSH_CHUNK_MS", os.environ.get("REACHY_AUDIO_BATCH_MS", "20"))
)
REACHY_AUDIO_MAX_AHEAD_MS = int(os.environ.get("REACHY_AUDIO_MAX_AHEAD_MS", "500"))


@dataclass
class SynthesizedAudio:
    pcm16_bytes: bytes
    sample_rate: int
    num_channels: int


@dataclass
class PipelineResult:
    assistant_text: str
    audio: SynthesizedAudio


class SpeechInterruptedError(RuntimeError):
    pass


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


def _extract_chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
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
    return ""


def _to_langchain_message_content(content: Any) -> Any:
    if isinstance(content, list):
        return content
    return _content_text(content)


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
            converted.append(HumanMessage(content=_to_langchain_message_content(content)))
        elif role == "assistant":
            converted.append(AIMessage(content=_content_text(content)))
        elif role == "system":
            converted.append(SystemMessage(content=_content_text(content)))
    return converted


def _build_user_turn_content(transcript_text: str, vision_context: dict[str, Any] | None) -> str | list[dict[str, Any]]:
    if not vision_context:
        return transcript_text

    family_references = [
        reference
        for reference in vision_context.get("family_references", [])
        if isinstance(reference, dict) and reference.get("name") and reference.get("image_url")
    ]
    speaker_image_url = str(vision_context.get("speaker_image_url") or "").strip()
    if not family_references and not speaker_image_url:
        return transcript_text

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "User transcript:\n"
                f"{transcript_text}\n\n"
                "People recognition context:\n"
                "- Family reference images are labeled by file name. Use these labels as candidate family names.\n"
                "- The final speaker image, if present, was captured when this utterance started.\n"
                "- Compare the speaker image with the family references and infer who is speaking only when the visual match is clear.\n"
                "- Choose a natural Japanese form of address from the prompt, the relationship, and the visual evidence; if uncertain, avoid using a name."
            ),
        }
    ]
    for reference in family_references:
        name = str(reference["name"])
        content.extend(
            [
                {"type": "text", "text": f"Family reference: {name}"},
                {"type": "image_url", "image_url": {"url": str(reference["image_url"])}},
            ]
        )
    if speaker_image_url:
        content.extend(
            [
                {"type": "text", "text": "Current speaker image captured at utterance start:"},
                {"type": "image_url", "image_url": {"url": speaker_image_url}},
            ]
        )
    return content


def _build_llm_context(history: list[dict[str, Any]], transcript_text: str, vision_context: dict[str, Any] | None = None) -> LLMContext:
    messages: list[dict[str, Any]] = [
        *history,
        {"role": "user", "content": _build_user_turn_content(transcript_text, vision_context)},
    ]
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

def _contains_japanese(text: str) -> bool:
    return any(
        ("ぁ" <= char <= "ん")
        or ("ァ" <= char <= "ン")
        or ("一" <= char <= "龯")
        for char in text
    )

def _is_probably_english_text(text: str) -> bool:
    if _contains_japanese(text):
        return False
    latin_count = sum(1 for char in text if char.isascii() and char.isalpha())
    return latin_count >= 3


def _drain_ready_segments(
    buffer: str,
    *,
    final: bool,
    newline_threshold: int,
    min_chars: int,
    max_chars: int,
) -> tuple[list[str], str]:
    segments: list[str] = []
    start = 0
    consecutive_newlines = 0
    english_like = _is_probably_english_text(buffer)
    effective_min_chars = min_chars * 2 if english_like and min_chars > 0 else min_chars
    effective_max_chars = max_chars * 3 if english_like and max_chars > 0 else max_chars
    sentence_punctuation = {"。", "！", "？", ".", "!", "?", ")"}
    hard_split_punctuation = {"。", ".", ")"}
    enforce_min_chars = True
    for index, char in enumerate(buffer):
        if char == "\n":
            consecutive_newlines += 1
        else:
            consecutive_newlines = 0

        segment = buffer[start : index + 1]
        should_split = False
        if newline_threshold > 0 and consecutive_newlines >= newline_threshold:
            should_split = True
        elif char in sentence_punctuation and (
            not enforce_min_chars or effective_min_chars <= 0 or len(segment.strip()) >= effective_min_chars
        ):
            should_split = True
        elif effective_max_chars > 0 and len(segment.strip()) >= effective_max_chars and char in hard_split_punctuation:
            should_split = True

        if should_split:
            segment = segment.strip()
            if segment:
                segments.append(segment)
                enforce_min_chars = False
            start = index + 1
            consecutive_newlines = 0

    remainder = buffer[start:]
    if final:
        tail = remainder.strip()
        if tail:
            segments.append(tail)
        remainder = ""
    return segments, remainder


def _normalize_tts_text(text: str) -> str:
    normalized = text
    for source in sorted(TTS_PRONUNCIATION_OVERRIDES, key=len, reverse=True):
        normalized = normalized.replace(source, TTS_PRONUNCIATION_OVERRIDES[source])
    return normalized


def _pcm16_to_float32(audio_bytes: bytes, num_channels: int) -> np.ndarray:
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        return audio.reshape(-1, num_channels)
    return audio


def _pcm16_frame_array(audio_bytes: bytes, num_channels: int) -> np.ndarray:
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
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


def _json_preview(payload: Any, max_len: int = 240) -> str:
    raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


def _tts_backend_override(args: argparse.Namespace) -> str:
    backend = getattr(args, "tts_backend", None)
    if not isinstance(backend, str):
        return ""
    return backend.strip()


def _tts_ref_audio_override(args: argparse.Namespace) -> str | dict[str, str] | None:
    ref_audio = getattr(args, "tts_ref_audio", None)
    if isinstance(ref_audio, str):
        return ref_audio.strip() or None
    if isinstance(ref_audio, dict):
        normalized = {
            key.strip(): value.strip()
            for key, value in ref_audio.items()
            if isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
        }
        return normalized or None
    return None


def _tts_ref_text_override(args: argparse.Namespace) -> str | dict[str, str] | None:
    ref_text = getattr(args, "tts_ref_text", None)
    if isinstance(ref_text, str):
        return ref_text.strip() or None
    if isinstance(ref_text, dict):
        normalized = {
            key.strip(): value.strip()
            for key, value in ref_text.items()
            if isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
        }
        return normalized or None
    return None


def _tts_x_vector_only_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "tts_x_vector_only_mode", False))


def _tts_temperature_override(args: argparse.Namespace) -> float | None:
    temperature = getattr(args, "tts_temperature", None)
    if isinstance(temperature, (int, float)):
        return float(temperature)
    return None


def _tts_alpha_override(args: argparse.Namespace) -> float | None:
    alpha = getattr(args, "tts_alpha", None)
    if isinstance(alpha, (int, float)):
        return float(alpha)
    return None


def _tts_beta_override(args: argparse.Namespace) -> float | None:
    beta = getattr(args, "tts_beta", None)
    if isinstance(beta, (int, float)):
        return float(beta)
    return None


def _tts_session_update_payload(args: argparse.Namespace) -> dict[str, Any]:
    session: dict[str, Any] = {
        "model": args.tts_model,
        "voice": args.voice,
        "instructions": args.tts_instructions,
        "task_type": args.tts_task_type,
        "language": args.tts_language,
    }
    backend = _tts_backend_override(args)
    if backend:
        session["backend"] = backend
    ref_audio = _tts_ref_audio_override(args)
    if ref_audio:
        session["ref_audio"] = ref_audio
    ref_text = _tts_ref_text_override(args)
    if ref_text:
        session["ref_text"] = ref_text
    if _tts_x_vector_only_mode(args):
        session["x_vector_only_mode"] = True
    temperature = _tts_temperature_override(args)
    if temperature is not None:
        session["temperature"] = temperature
    alpha = _tts_alpha_override(args)
    if alpha is not None:
        session["alpha"] = alpha
    beta = _tts_beta_override(args)
    if beta is not None:
        session["beta"] = beta
    return session


def _tts_http_request_payload(args: argparse.Namespace, text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.tts_model,
        "task_type": args.tts_task_type,
        "language": args.tts_language,
        "voice": args.voice,
        "input": text,
        "instructions": args.tts_instructions,
        "response_format": "pcm",
        "stream": True,
    }
    backend = _tts_backend_override(args)
    if backend:
        payload["backend"] = backend
    ref_audio = _tts_ref_audio_override(args)
    if ref_audio:
        payload["ref_audio"] = ref_audio
    ref_text = _tts_ref_text_override(args)
    if ref_text:
        payload["ref_text"] = ref_text
    if _tts_x_vector_only_mode(args):
        payload["x_vector_only_mode"] = True
    temperature = _tts_temperature_override(args)
    if temperature is not None:
        payload["temperature"] = temperature
    alpha = _tts_alpha_override(args)
    if alpha is not None:
        payload["alpha"] = alpha
    beta = _tts_beta_override(args)
    if beta is not None:
        payload["beta"] = beta
    return payload


TOOL_CHECK_SENTINEL = "No"


class RichTraceProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="RichTraceProcessor")
        self._logger = logging.getLogger("conversation.trace")

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._logger.info("[bold cyan]LLM[/] response started")
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._logger.info("[bold magenta]LLM tool[/] %s %s", frame.function_name, _json_preview(frame.arguments))
        elif isinstance(frame, FunctionCallResultFrame):
            self._logger.info("[bold magenta]Tool result[/] %s %s", frame.function_name, _json_preview(frame.result))
        elif isinstance(frame, ErrorFrame):
            self._logger.error("[bold red]Pipeline error[/] %s", frame.error)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._logger.info("[bold cyan]LLM[/] response finished")
        await self.push_frame(frame, direction)


class LLMCompletionSignalProcessor(FrameProcessor):
    def __init__(self, llm_finished_event: asyncio.Event | None = None):
        super().__init__(name="LLMCompletionSignalProcessor")
        self._llm_finished_event = llm_finished_event

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame) and self._llm_finished_event is not None:
            self._llm_finished_event.set()
        await self.push_frame(frame, direction)


class ReachyLangChainProcessor(FrameProcessor):
    def __init__(self, args: argparse.Namespace, tools: list[BaseTool]):
        super().__init__(name="ReachyLangChainProcessor")
        self._logger = logging.getLogger("conversation.llm")
        self._tools_by_name = {tool.name: tool for tool in tools}
        base_model = ChatOpenAI(
            model=args.chat_model,
            base_url=args.chat_base_url,
            api_key=SecretStr(args.chat_api_key),
            temperature=args.llm_temperature,
            max_completion_tokens=args.max_completion_tokens,
            use_responses_api=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        self._tool_model = base_model.bind_tools(tools)
        self._response_model = base_model
        self._system_prompt = args.system_prompt
        self._max_tool_rounds = args.max_tool_rounds

    async def _select_tool_use(self, messages: list[Any]):
        return await self._tool_model.ainvoke(
            [
                *messages,
                HumanMessage(
                    content=(
                        "Decide whether a tool call is required to answer the user's last request. "
                        "If a tool is needed, call it now. "
                        f"If no tool call is needed, reply with exactly {TOOL_CHECK_SENTINEL}."
                    )
                ),
            ]
        )

    async def _stream_final_response(self, messages: list[Any], direction: FrameDirection) -> None:
        emitted_text = False
        try:
            async for chunk in self._response_model.astream(messages):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = _extract_chunk_text(chunk)
                if not text:
                    continue
                emitted_text = True
                await self.push_frame(LLMTextFrame(text=text), direction)
        except Exception as exc:
            self._logger.warning("[bold yellow]LLM[/] streaming failed, falling back to buffered response: %s", exc)
            response = await self._response_model.ainvoke(messages)
            text = _content_text(response.content).strip()
            if text:
                await self.push_frame(LLMTextFrame(text=text), direction)
            return

        if emitted_text:
            return

        response = await self._response_model.ainvoke(messages)
        text = _content_text(response.content).strip()
        if text:
            await self.push_frame(LLMTextFrame(text=text), direction)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        messages = [SystemMessage(content=self._system_prompt), *_to_langchain_messages(frame.context.messages)]
        await self.push_frame(LLMFullResponseStartFrame(), direction)

        try:
            if not self._tools_by_name:
                await self._stream_final_response(messages, direction)
                return

            for round_index in range(self._max_tool_rounds):
                self._logger.info("[bold cyan]LLM[/] round %d", round_index + 1)
                response = await self._select_tool_use(messages)
                text = _content_text(response.content).strip()
                tool_calls = getattr(response, "tool_calls", None) or []

                if not tool_calls:
                    if text and text != TOOL_CHECK_SENTINEL:
                        self._logger.debug("Tool decision returned non-sentinel text without tool call: %r", text)
                    await self._stream_final_response(messages, direction)
                    break

                messages.append(response)
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
        self._interrupt_event = asyncio.Event()
        self._stream_has_started = False

    async def request_interrupt(self) -> None:
        self._interrupt_event.set()
        if self._segment_queue is None:
            return
        while True:
            try:
                item = self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._segment_queue.task_done()
                if item is None:
                    return
        await self._segment_queue.put(None)

    async def _stop_worker(self) -> None:
        if self._segment_queue is not None:
            await self.request_interrupt()
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
                            if segment is None or self._interrupt_event.is_set():
                                return
                            self._logger.info("[bold green]TTS[/] synthesizing segment %s", segment)
                            async for audio_frame in synthesize_realtime_audio_stream(self._args, websocket, segment):
                                if self._interrupt_event.is_set():
                                    return
                                await self.push_frame(audio_frame, direction)
                        finally:
                            self._segment_queue.task_done()
            else:
                while True:
                    segment = await self._segment_queue.get()
                    try:
                        if segment is None or self._interrupt_event.is_set():
                            return
                        self._logger.info("[bold green]TTS[/] synthesizing segment %s", segment)
                        async for audio_frame in synthesize_audio_stream(self._args, segment):
                            if self._interrupt_event.is_set():
                                return
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
            self._interrupt_event.clear()
            self._stream_has_started = False
            self._segment_queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._tts_worker(direction))
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            if self._interrupt_event.is_set():
                await self.push_frame(frame, direction)
                return
            if frame.text:
                self._pending_text += frame.text
                min_chars = self._args.tts_segment_min_chars if not self._stream_has_started else 0
                segments, remainder = _drain_ready_segments(
                    self._pending_text,
                    final=False,
                    newline_threshold=self._args.tts_segment_newline_threshold,
                    min_chars=min_chars,
                    max_chars=self._args.tts_segment_max_chars,
                )
                self._pending_text = remainder
                if self._segment_queue is not None:
                    for segment in segments:
                        normalized_segment = _normalize_tts_text(segment)
                        await self._segment_queue.put(normalized_segment)
                    if segments:
                        self._stream_has_started = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._segment_queue is not None:
                if not self._interrupt_event.is_set():
                    min_chars = self._args.tts_segment_min_chars if not self._stream_has_started else 0
                    segments, remainder = _drain_ready_segments(
                        self._pending_text,
                        final=True,
                        newline_threshold=self._args.tts_segment_newline_threshold,
                        min_chars=min_chars,
                        max_chars=self._args.tts_segment_max_chars,
                    )
                    self._pending_text = remainder
                    for segment in segments:
                        normalized_segment = _normalize_tts_text(segment)
                        await self._segment_queue.put(normalized_segment)
                    if segments:
                        self._stream_has_started = True
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
                    "session": _tts_session_update_payload(self._args),
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
    def __init__(self, robot: ReachyMini, *, enable_head_wobble: bool = True, assistant_speech_state: AssistantSpeechState | None = None):
        super().__init__(name="ReachyAudioPlayer")
        self._robot = robot
        self._logger = logging.getLogger("conversation.audio")
        self._started = False
        self._head_wobbler = HeadWobbler(
            robot.set_target_head_pose,
            robot.get_current_head_pose,
            lambda: get_wobble_origin_pose(robot),
        ) if enable_head_wobble else None
        self._assistant_speech_state = assistant_speech_state
        self._closed = False
        self._playback_chunk_ms = max(10, REACHY_AUDIO_PUSH_CHUNK_MS)
        max_ahead_ms = max(self._playback_chunk_ms, REACHY_AUDIO_MAX_AHEAD_MS)
        self._playback_queue_limit = max(1, (max_ahead_ms + self._playback_chunk_ms - 1) // self._playback_chunk_ms)
        self._playback_queue: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue(maxsize=self._playback_queue_limit)
        self._playback_thread: threading.Thread | None = None
        self._playback_stop = threading.Event()

    def abort(self) -> None:
        if self._playback_thread is None:
            return
        self._playback_stop.set()
        while True:
            try:
                item = self._playback_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._playback_queue.task_done()
                if item is None:
                    break
        with suppress(queue.Full):
            self._playback_queue.put_nowait(None)
        self._playback_thread.join(timeout=0.25)
        self._playback_thread = None
        with suppress(Exception):
            self._robot.media.stop_playing()
            self._robot.media.start_playing()
        if self._head_wobbler is not None:
            self._head_wobbler.stop()

    def close(self, timeout_s: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_playback_worker()
        if self._head_wobbler is None:
            return
        try:
            finished = self._head_wobbler.finish(timeout_s=timeout_s)
        except ConnectionError as exc:
            self._logger.warning("Head wobble reset skipped after robot disconnect: %s", exc)
            finished = True
        if timeout_s is not None and not finished:
            self._logger.warning("Head wobble reset timed out")
        try:
            self._head_wobbler.stop()
        except ConnectionError as exc:
            self._logger.warning("Head wobble stop skipped after robot disconnect: %s", exc)

    def _ensure_playback_worker(self) -> None:
        if self._playback_thread is not None and self._playback_thread.is_alive():
            return
        self._playback_stop.clear()
        self._playback_thread = threading.Thread(target=self._playback_loop, name="reachy-audio-playback", daemon=True)
        self._playback_thread.start()

    def _stop_playback_worker(self) -> None:
        if self._playback_thread is None:
            return
        self._playback_queue.put(None)
        self._playback_queue.join()
        self._playback_stop.set()
        self._playback_thread.join(timeout=1.0)
        self._playback_thread = None

    def _playback_loop(self) -> None:
        next_push_at: float | None = None
        while not self._playback_stop.is_set():
            try:
                item = self._playback_queue.get(timeout=0.05)
            except queue.Empty:
                next_push_at = None
                continue

            if item is None:
                self._playback_queue.task_done()
                return

            waveform, sample_rate = item
            chunk_duration_s = max(0.001, waveform.shape[0] / max(1, sample_rate))

            try:
                current_time = time.monotonic()
                if next_push_at is None or current_time > next_push_at + chunk_duration_s:
                    next_push_at = current_time
                elif next_push_at > current_time:
                    self._playback_stop.wait(next_push_at - current_time)
                    if self._playback_stop.is_set():
                        return
                self._robot.media.push_audio_sample(waveform)
                next_push_at = max(next_push_at, time.monotonic()) + chunk_duration_s
            finally:
                self._playback_queue.task_done()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
            if self._playback_stop.is_set():
                return
            if self._assistant_speech_state is not None:
                self._assistant_speech_state.note_output_audio(
                    sample_count=len(frame.audio) // (2 * max(1, frame.num_channels)),
                    sample_rate=frame.sample_rate,
                )
            if self._head_wobbler is not None:
                self._head_wobbler.feed_pcm(_pcm16_frame_array(frame.audio, frame.num_channels), frame.sample_rate)
            waveform = _pcm16_to_float32(frame.audio, frame.num_channels)
            output_rate = self._robot.media.get_output_audio_samplerate()
            if frame.sample_rate != output_rate:
                waveform = _resample_audio(waveform, frame.sample_rate, output_rate)
            if not self._started:
                self._logger.info("[bold yellow]Audio[/] streaming assistant reply to Reachy speaker")
                self._started = True
            self._ensure_playback_worker()
            chunk_frames = max(1, int(round(output_rate * self._playback_chunk_ms / 1000.0)))
            for start in range(0, waveform.shape[0], chunk_frames):
                if self._playback_stop.is_set():
                    return
                chunk = np.array(waveform[start : start + chunk_frames], copy=True)
                await asyncio.to_thread(self._playback_queue.put, (chunk, output_rate))
        elif isinstance(frame, EndFrame):
            self._started = False
            if self._head_wobbler is not None:
                self._head_wobbler.request_reset_after_current_audio()
        await self.push_frame(frame, direction)


async def _wait_for_settings_interrupt(runtime_settings: RuntimeSettings, expected_settings_version: int) -> None:
    while runtime_settings.snapshot()[-1] == expected_settings_version:
        await asyncio.sleep(0.05)


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
    payload = _tts_http_request_payload(args, text)
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
    history: list[dict[str, Any]],
    enabled_tool_names: list[str],
    vision_context: dict[str, Any] | None = None,
    assistant_speech_state: AssistantSpeechState | None = None,
    runtime_settings: RuntimeSettings | None = None,
    expected_settings_version: int | None = None,
    interrupt_event: asyncio.Event | None = None,
    llm_finished_event: asyncio.Event | None = None,
    interrupt_reason: str = "assistant speech interrupted",
) -> PipelineResult:
    tools = build_langchain_tools(runtime, enabled_tool_names)
    llm = ReachyLangChainProcessor(args, tools)
    completion_signal = LLMCompletionSignalProcessor(llm_finished_event)
    trace = RichTraceProcessor()
    tts = ReachyTTSProcessor(args)
    collector = ResultCollector()
    audio_player = ReachyAudioPlayer(
        runtime.robot,
        enable_head_wobble=getattr(args, "head_wobble", True),
        assistant_speech_state=assistant_speech_state,
    )
    terminator = PipelineTerminator()
    pipeline = Pipeline([llm, completion_signal, trace, tts, collector, audio_player, terminator])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=args.sample_rate, audio_out_sample_rate=args.tts_sample_rate),
        idle_timeout_secs=max(1.0, float(getattr(args, "turn_idle_timeout_seconds", 120.0))),
    )
    terminator.bind_task(task)
    runner = _create_turn_pipeline_runner()
    runner_task = asyncio.create_task(runner.run(task))
    interrupt_task: asyncio.Task[None] | None = None
    external_interrupt_task: asyncio.Task[bool] | None = None
    if runtime_settings is not None and expected_settings_version is not None:
        interrupt_task = asyncio.create_task(_wait_for_settings_interrupt(runtime_settings, expected_settings_version))
    if interrupt_event is not None:
        external_interrupt_task = asyncio.create_task(interrupt_event.wait())
    await task.queue_frame(LLMContextFrame(_build_llm_context(history, transcript_text, vision_context)))
    interrupted = False
    try:
        if interrupt_task is None and external_interrupt_task is None:
            await runner_task
        else:
            wait_tasks: set[asyncio.Task[Any]] = {runner_task}
            if interrupt_task is not None:
                wait_tasks.add(interrupt_task)
            if external_interrupt_task is not None:
                wait_tasks.add(external_interrupt_task)
            done, _pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
            if runner_task in done:
                await runner_task
            else:
                interrupted = True
                reason = interrupt_reason if external_interrupt_task is not None and external_interrupt_task in done else "live settings changed"
                logging.getLogger("conversation.tts").info("[bold green]TTS[/] interrupting live reply because %s", reason)
                await tts.request_interrupt()
                audio_player.abort()
                with suppress(Exception):
                    await task.queue_frame(EndFrame())
                try:
                    await asyncio.wait_for(runner_task, timeout=1.0)
                except asyncio.TimeoutError:
                    runner_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await runner_task
    finally:
        if interrupt_task is not None:
            interrupt_task.cancel()
            with suppress(asyncio.CancelledError):
                await interrupt_task
        if external_interrupt_task is not None:
            external_interrupt_task.cancel()
            with suppress(asyncio.CancelledError):
                await external_interrupt_task
        await asyncio.to_thread(audio_player.close)

    if interrupted:
        raise SpeechInterruptedError(interrupt_reason)
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

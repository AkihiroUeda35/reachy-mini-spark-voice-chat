from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import httpx
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
from reachy_audio import HeadWobbler
from state import AssistantSpeechState
from websockets import connect as ws_connect

from reachy_conversation_tools import ReachyToolRuntime, build_langchain_tools

TTS_TIMEOUT = float(os.environ.get("TTS_TIMEOUT", "600"))
TTS_STREAM_CHUNK_BYTES = int(os.environ.get("TTS_STREAM_CHUNK_BYTES", "8192"))


@dataclass
class SynthesizedAudio:
    pcm16_bytes: bytes
    sample_rate: int
    num_channels: int


@dataclass
class PipelineResult:
    assistant_text: str
    audio: SynthesizedAudio


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
    def __init__(self, robot: ReachyMini, *, enable_head_wobble: bool = True, assistant_speech_state: AssistantSpeechState | None = None):
        super().__init__(name="ReachyAudioPlayer")
        self._robot = robot
        self._logger = logging.getLogger("conversation.audio")
        self._started = False
        self._head_wobbler = HeadWobbler(robot.set_target_head_pose, robot.get_current_head_pose) if enable_head_wobble else None
        self._assistant_speech_state = assistant_speech_state
        self._closed = False

    def close(self, timeout_s: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._head_wobbler is None:
            return
        finished = self._head_wobbler.finish(timeout_s=timeout_s)
        if timeout_s is not None and not finished:
            self._logger.warning("Head wobble reset timed out")
        self._head_wobbler.stop()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
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
            await asyncio.to_thread(self._robot.media.push_audio_sample, waveform)
        elif isinstance(frame, EndFrame):
            self._started = False
            if self._head_wobbler is not None:
                self._head_wobbler.request_reset_after_current_audio()
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
    assistant_speech_state: AssistantSpeechState | None = None,
) -> PipelineResult:
    tools = build_langchain_tools(runtime, enabled_tool_names)
    llm = ReachyLangChainProcessor(args, tools)
    trace = RichTraceProcessor()
    tts = ReachyTTSProcessor(args)
    collector = ResultCollector()
    audio_player = ReachyAudioPlayer(
        runtime.robot,
        enable_head_wobble=getattr(args, "head_wobble", True),
        assistant_speech_state=assistant_speech_state,
    )
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
    try:
        await runner_task
    finally:
        await asyncio.to_thread(audio_player.close)

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
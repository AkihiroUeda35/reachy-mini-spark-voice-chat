from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import logging
import os
import signal
import sys
import threading
import time
import wave
import json
from collections import deque
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pipecat.frames.frames import EndFrame
from pipecat.processors.frame_processor import FrameDirection
from pydantic import SecretStr
from reachy_mini import ReachyMini
from rich.logging import RichHandler

import local_tts
import reachy_conversation_tools as reachy_tools
import whisper_asr as asr_tools
from config import load_entrypoint_env
from openai_model_registry import resolve_model

from gradio_ui import launch_gradio_ui
from pipeline import PipelineResult, ReachyAudioPlayer, ReachyRealtimeTTSSession, SpeechInterruptedError, SynthesizedAudio, _drain_ready_segments, _resample_audio, run_pipeline, synthesize_audio_stream, synthesize_realtime_audio_stream
from state import APP_DIR, DEFAULT_CHARACTER_PROMPT, DEFAULT_DATA_DIR, DEFAULT_SYSTEM_PROMPT, DEFAULT_VOICE, DEFAULT_TTS_INSTRUCTIONS, AssistantSpeechState, RuntimeSettings, active_tools_for_profile as _active_tools_for_profile, build_tts_request_voice, effective_tts_transport, listening_gate_settings, load_profile_character_prompt_by_name, load_profile_prompt as _load_profile_prompt, load_profile_prompt_by_name, load_profile_qwen_voice_by_name, load_profile_tts_instructions_by_name, load_profile_tts_ref_audio_by_name, load_profile_tts_ref_text_by_name, load_profile_voice_by_name, load_selected_profile_name, normalize_tts_backend_name, overlap_turn_rejection_reason, save_profile_definition as _save_profile_definition

load_entrypoint_env(local_tts, asr_tools)

GUI_TOOL_NAMES = reachy_tools.GUI_TOOL_NAMES

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
TTS_TEMPERATURE = local_tts.TTS_TEMPERATURE
TTS_INSTRUCTIONS = local_tts.TTS_INSTRUCTIONS
TTS_SAMPLE_RATE = local_tts.TTS_SAMPLE_RATE
TTS_BACKEND = os.environ.get("TTS_BACKEND", "")
TTS_DEBUG_CAPTURE = os.environ.get("TTS_DEBUG_CAPTURE", "0") not in {"", "0", "false", "False", "no", "off"}
VOICE = local_tts.VOICE

TTS_TRANSPORT = os.environ.get("TTS_TRANSPORT", "realtime")
TTS_SEGMENT_NEWLINE_THRESHOLD = int(os.environ.get("TTS_SEGMENT_NEWLINE_THRESHOLD", "2"))
TTS_SEGMENT_MIN_CHARS = int(os.environ.get("TTS_SEGMENT_MIN_CHARS", "0"))
TTS_SEGMENT_MAX_CHARS = int(os.environ.get("TTS_SEGMENT_MAX_CHARS", "100"))
TURN_IDLE_TIMEOUT_SECONDS = float(os.environ.get("TURN_IDLE_TIMEOUT_SECONDS", "120"))
SHUTDOWN_STEP_TIMEOUT_S = 2.0
REACHY_HOST = os.environ.get("REACHY_HOST", "reachy-mini.local")
REACHY_DAEMON_START_TIMEOUT_S = float(os.environ.get("REACHY_DAEMON_START_TIMEOUT_S", "5.0"))
ASR_MIN_SEGMENT_AVG_LOGPROB = float(os.environ.get("ASR_MIN_SEGMENT_AVG_LOGPROB", "-0.75"))
ASR_MAX_SEGMENT_NO_SPEECH_PROB = float(os.environ.get("ASR_MAX_SEGMENT_NO_SPEECH_PROB", "0.6"))
ASR_ALLOWED_LANGUAGES = os.environ.get("ASR_ALLOWED_LANGUAGES", "")
ASR_EXCLUDED_LANGUAGES = os.environ.get("ASR_EXCLUDED_LANGUAGES", "")
ASSISTANT_SPEAKING_INTERRUPT_MIN_VAD_MS = int(
    os.environ.get(
        "ASSISTANT_SPEAKING_INTERRUPT_MIN_VAD_MS",
        os.environ.get("ASSISTANT_SPEAKING_MIN_VAD_MS", "500"),
    )
)
ASSISTANT_SPEAKING_INTERRUPT_MIN_CHARS = int(
    os.environ.get(
        "ASSISTANT_SPEAKING_INTERRUPT_MIN_CHARS",
        os.environ.get("ASSISTANT_SPEAKING_MIN_CHARS", "4"),
    )
)


@dataclass
class CapturedUtterance:
    audio: Any
    duration_ms: float
    overlap_gate_active: bool
    barge_in_candidate: bool = False
    transcription_payload: str | dict[str, Any] | None = None
    transcript_text: str = ""


def _normalize_transcript_language(value: Any) -> str:
    normalized = str(value or "").strip().replace("_", "-").lower()
    if not normalized or normalized == "auto":
        return ""
    return normalized.split("-", 1)[0]


def _parse_transcript_languages(raw_value: str | None) -> set[str]:
    languages: set[str] = set()
    for item in str(raw_value or "").split(","):
        normalized = _normalize_transcript_language(item)
        if normalized:
            languages.add(normalized)
    return languages


def _conversation_transcription_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(args), "response_format": "verbose_json"})


def _reachy_daemon_http_base_url(robot_host: str | None) -> str | None:
    host = str(robot_host or "").strip()
    if not host:
        return None
    if "://" not in host:
        return f"http://{host}:8000"

    parsed = urlparse(host)
    if parsed.netloc:
        return f"{parsed.scheme or 'http'}://{parsed.netloc}"
    if parsed.path:
        return f"{parsed.scheme or 'http'}://{parsed.path}"
    return None


def _request_remote_daemon_start(daemon_base_url: str, *, wake_up: bool, timeout_s: float = 5.0) -> None:
    response = httpx.post(
        f"{daemon_base_url.rstrip('/')}/api/daemon/start",
        params={"wake_up": str(wake_up).lower()},
        timeout=timeout_s,
    )
    response.raise_for_status()


def _is_transient_reachy_startup_error(exc: Exception) -> bool:
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    if isinstance(exc, KeyError) and "Producer reachymini not found." in str(exc):
        return True
    return False


async def _connect_robot(
    args: argparse.Namespace,
    robot_kwargs: dict[str, Any],
    app_logger: logging.Logger,
) -> ReachyMini:
    try:
        return ReachyMini(**robot_kwargs)
    except Exception as exc:
        if not _is_transient_reachy_startup_error(exc):
            raise
        daemon_base_url = _reachy_daemon_http_base_url(getattr(args, "robot_host", None))
        if not args.wake_up or daemon_base_url is None:
            raise

        app_logger.warning(
            "Reachy daemon was not reachable over SDK; requesting remote daemon start via %s",
            daemon_base_url,
        )

        try:
            await asyncio.to_thread(_request_remote_daemon_start, daemon_base_url, wake_up=True)
        except Exception:
            app_logger.exception("Failed to request remote Reachy daemon start")
            raise exc

        deadline = time.monotonic() + REACHY_DAEMON_START_TIMEOUT_S
        last_error: Exception = exc
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            try:
                robot = ReachyMini(**robot_kwargs)
                app_logger.info("Remote Reachy daemon became available after wake request")
                return robot
            except Exception as retry_exc:
                if not _is_transient_reachy_startup_error(retry_exc):
                    raise
                last_error = retry_exc

        raise last_error


def _segment_metric_summary(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return None, None

    avg_logprobs: list[float] = []
    no_speech_probs: list[float] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        avg_logprob = segment.get("avg_logprob")
        no_speech_prob = segment.get("no_speech_prob")
        if isinstance(avg_logprob, int | float):
            avg_logprobs.append(float(avg_logprob))
        if isinstance(no_speech_prob, int | float):
            no_speech_probs.append(float(no_speech_prob))

    mean_avg_logprob = sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else None
    max_no_speech_prob = max(no_speech_probs) if no_speech_probs else None
    return mean_avg_logprob, max_no_speech_prob


def _transcript_observation_fields(payload: str | dict[str, Any], transcript_text: str | None = None) -> tuple[str, str, str, str]:
    normalized_text = str(transcript_text or "").strip()
    if isinstance(payload, dict):
        language = _normalize_transcript_language(payload.get("language")) or "unknown"
        mean_avg_logprob, max_no_speech_prob = _segment_metric_summary(payload)
        if not normalized_text:
            normalized_text = str(payload.get("text") or "").strip()
    else:
        language = "unknown"
        mean_avg_logprob = None
        max_no_speech_prob = None
        if not normalized_text:
            normalized_text = str(payload or "").strip()

    avg_logprob_text = f"{mean_avg_logprob:.2f}" if mean_avg_logprob is not None else "n/a"
    no_speech_prob_text = f"{max_no_speech_prob:.2f}" if max_no_speech_prob is not None else "n/a"
    return language, avg_logprob_text, no_speech_prob_text, normalized_text


def _log_transcript_judgement(
    logger: logging.Logger,
    *,
    label: str,
    payload: str | dict[str, Any],
    transcript_text: str | None,
    accepted: bool,
    rejection_reason: str | None = None,
) -> None:
    language, avg_logprob_text, no_speech_prob_text, normalized_text = _transcript_observation_fields(payload, transcript_text)
    logger.info(
        "[bold blue]ASR[/] %s result=%s language=%s avg_logprob=%s max_no_speech_prob=%s text=%s%s",
        label,
        "accepted" if accepted else "rejected",
        language,
        avg_logprob_text,
        no_speech_prob_text,
        normalized_text,
        f" reason={rejection_reason}" if rejection_reason else "",
    )


def transcript_quality_rejection_reason(payload: dict[str, Any], args: argparse.Namespace) -> str | None:
    excluded_languages = _parse_transcript_languages(getattr(args, "excluded_transcript_languages", ""))
    allowed_languages = _parse_transcript_languages(getattr(args, "allowed_transcript_languages", ""))
    if not allowed_languages:
        explicit_language = _normalize_transcript_language(getattr(args, "language", ""))
        if explicit_language:
            allowed_languages.add(explicit_language)

    detected_language = _normalize_transcript_language(payload.get("language"))
    if excluded_languages and detected_language and detected_language in excluded_languages:
        return f"language {detected_language} excluded by {','.join(sorted(excluded_languages))}"
    if allowed_languages and detected_language and detected_language not in allowed_languages:
        return f"language {detected_language} not in {','.join(sorted(allowed_languages))}"

    mean_avg_logprob, max_no_speech_prob = _segment_metric_summary(payload)
    if mean_avg_logprob is not None and mean_avg_logprob < float(args.min_segment_avg_logprob):
        return f"avg_logprob {mean_avg_logprob:.2f} < {args.min_segment_avg_logprob:.2f}"
    if max_no_speech_prob is not None and max_no_speech_prob > float(args.max_segment_no_speech_prob):
        return f"no_speech_prob {max_no_speech_prob:.2f} > {args.max_segment_no_speech_prob:.2f}"
    return None


def active_tools_for_profile(profiles_dir: Path, profile: str) -> list[str]:
    return _active_tools_for_profile(profiles_dir, profile, GUI_TOOL_NAMES)


def save_profile_definition(
    profiles_dir: Path,
    profile: str,
    character_prompt: str,
    selected_tools: list[str],
    voice: str,
    qwen_voice: str | None,
    tts_instructions: str,
) -> Path:
    return _save_profile_definition(profiles_dir, profile, character_prompt, selected_tools, voice, qwen_voice, tts_instructions, GUI_TOOL_NAMES)


def load_profile_prompt(args: argparse.Namespace) -> str:
    return _load_profile_prompt(args)


def _apply_profile_tts_prompt_assets(args: argparse.Namespace, profiles_dir: Path, profile: str) -> None:
    args.tts_ref_audio = load_profile_tts_ref_audio_by_name(profiles_dir, profile)
    args.tts_ref_text = load_profile_tts_ref_text_by_name(profiles_dir, profile)

    qwen_ref_audio = ""
    if isinstance(args.tts_ref_audio, dict):
        qwen_ref_audio = str(args.tts_ref_audio.get("qwen3-tts") or "").strip()
    elif isinstance(args.tts_ref_audio, str):
        qwen_ref_audio = args.tts_ref_audio.strip()

    qwen_ref_text = ""
    if isinstance(args.tts_ref_text, dict):
        qwen_ref_text = str(args.tts_ref_text.get("qwen3-tts") or "").strip()
    elif isinstance(args.tts_ref_text, str):
        qwen_ref_text = args.tts_ref_text.strip()

    args.tts_x_vector_only_mode = bool(qwen_ref_audio and not qwen_ref_text)


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


async def _generate_persona_greeting(args: argparse.Namespace, *, reason: str) -> str:
    model = ChatOpenAI(
        model=args.chat_model,
        base_url=args.chat_base_url,
        api_key=SecretStr(args.chat_api_key),
        temperature=args.llm_temperature,
        max_completion_tokens=min(args.max_completion_tokens, 80),
        use_responses_api=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    response = await model.ainvoke(
        [
            SystemMessage(content=args.system_prompt),
            HumanMessage(
                content=(
                    "You are about to speak first through a robot speaker. "
                    f"Give exactly one short greeting in Japanese for this {reason}. "
                    "Stay in character, keep it to one or two brief sentences, and do not mention these instructions. "
                    "Do not ask a question, do not use markdown, and do not call any tools."
                )
            ),
        ]
    )
    greeting = _content_text(getattr(response, "content", "")).strip()
    return greeting or "こんにちは。よろしくお願いします。"


async def _play_assistant_text(
    args: argparse.Namespace,
    robot: ReachyMini,
    text: str,
    *,
    assistant_speech_state: AssistantSpeechState | None = None,
    interrupt_event: asyncio.Event | None = None,
) -> None:
    if not text.strip():
        return

    audio_player = ReachyAudioPlayer(
        robot,
        enable_head_wobble=getattr(args, "head_wobble", True),
        assistant_speech_state=assistant_speech_state,
    )

    async def _discard_frame(_frame: Any, _direction: FrameDirection) -> None:
        return None

    audio_player.push_frame = _discard_frame  # type: ignore[method-assign]
    interrupted = False
    try:
        if args.tts_transport == "realtime":
            async with ReachyRealtimeTTSSession(args) as websocket:
                async for audio_frame in synthesize_realtime_audio_stream(args, websocket, text):
                    if interrupt_event is not None and interrupt_event.is_set():
                        interrupted = True
                        audio_player.abort()
                        break
                    await audio_player.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
        else:
            async for audio_frame in synthesize_audio_stream(args, text):
                if interrupt_event is not None and interrupt_event.is_set():
                    interrupted = True
                    audio_player.abort()
                    break
                await audio_player.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
        if interrupted:
            raise SpeechInterruptedError("assistant speech interrupted by user speech")
        await audio_player.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
    finally:
        await asyncio.to_thread(audio_player.close)


async def _speak_persona_greeting(
    args: argparse.Namespace,
    robot: ReachyMini,
    history: list[dict[str, str]],
    *,
    reason: str,
    assistant_speech_state: AssistantSpeechState | None = None,
    interrupt_event: asyncio.Event | None = None,
    barge_in_ready_event: asyncio.Event | None = None,
) -> list[dict[str, str]]:
    greeting = await _generate_persona_greeting(args, reason=reason)
    if barge_in_ready_event is not None:
        barge_in_ready_event.set()
    logging.getLogger("conversation.llm").info("[bold cyan]LLM[/] greeting %s", greeting)
    await _play_assistant_text(
        args,
        robot,
        greeting,
        assistant_speech_state=assistant_speech_state,
        interrupt_event=interrupt_event,
    )
    updated_history = [*history, {"role": "assistant", "content": greeting}]
    return trim_history(updated_history, args.history_turns)


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
    parser.add_argument("--robot-host", default=REACHY_HOST or None, help="Reachy Mini host or IP address. Defaults to REACHY_HOST from the environment when set.")
    parser.add_argument("--robot-name", help="Optional Reachy Mini robot name when multiple robots are available.")
    parser.add_argument("--profile", help="Profile name under apps/conversation/profiles. Defaults to the last saved selection, or 'default'.")
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
    parser.add_argument("--head-wobble", action=argparse.BooleanOptionalAction, default=True, help="Move Reachy's head while speaking synthesized replies.")

    parser.add_argument("--base-url", default=asr_tools.ASR_BASE_URL, help="OpenAI-compatible STT base URL.")
    parser.add_argument("--api-key", default=asr_tools.ASR_API_KEY, help="Bearer token for the STT endpoint.")
    parser.add_argument("--model", default=asr_tools.ASR_MODEL, help="STT model name.")
    parser.add_argument("--transport", choices=["realtime", "http"], default=asr_tools.ASR_TRANSPORT, help="STT transport to use.")
    parser.add_argument("--language", "--lang", default=asr_tools.ASR_LANGUAGE, help="Language hint for STT. Use 'auto' to enable autodetection.")
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
    parser.add_argument("--assistant-speaking-threshold-boost", type=float, default=0.005, help="Additional VAD RMS threshold applied while Reachy's own reply is still playing.")
    parser.add_argument("--assistant-speaking-vad-start-ms", type=int, default=400, help="Minimum continuous speech required to start capture while Reachy's reply is still playing.")
    parser.add_argument("--assistant-speaking-min-vad-ms", type=int, default=500, help="Minimum captured speech duration required to keep a turn while Reachy's reply is still playing.")
    parser.add_argument("--assistant-speaking-min-chars", type=int, default=4, help="Minimum transcript length required to keep a turn while Reachy's reply is still playing.")
    parser.add_argument("--assistant-speaking-interrupt-min-vad-ms", type=int, default=ASSISTANT_SPEAKING_INTERRUPT_MIN_VAD_MS, help="Minimum captured speech duration required to interrupt Reachy's current reply and keep it as a barge-in turn.")
    parser.add_argument("--assistant-speaking-interrupt-min-chars", type=int, default=ASSISTANT_SPEAKING_INTERRUPT_MIN_CHARS, help="Minimum transcript length required to keep a barge-in turn captured while Reachy is speaking.")
    parser.add_argument("--assistant-speaking-tail-ms", type=int, default=350, help="Extra time after queued TTS audio where overlap protection stays active.")
    parser.add_argument("--listen-timeout-seconds", type=float, default=20.0, help="Maximum time to wait for a new utterance.")
    parser.add_argument("--audio-poll-interval-ms", type=float, default=10.0, help="Polling interval when Reachy media has no pending audio frame.")
    parser.add_argument("--save-transcripts", action=argparse.BooleanOptionalAction, default=False, help="Save STT payloads under the data directory.")
    parser.add_argument("--min-segment-avg-logprob", type=float, default=ASR_MIN_SEGMENT_AVG_LOGPROB, help="Reject STT turns when the mean Whisper segment avg_logprob falls below this threshold.")
    parser.add_argument("--max-segment-no-speech-prob", type=float, default=ASR_MAX_SEGMENT_NO_SPEECH_PROB, help="Reject STT turns when any Whisper segment no_speech_prob exceeds this threshold.")
    parser.add_argument("--allowed-transcript-languages", default=ASR_ALLOWED_LANGUAGES, help="Optional comma-separated detected STT languages to accept, for example 'ja,en'. When empty, an explicit --language hint is used as the only allowed language.")
    parser.add_argument("--excluded-transcript-languages", default=ASR_EXCLUDED_LANGUAGES, help="Optional comma-separated detected STT languages to reject before allow-list checks, for example 'ru,es,fr'.")

    parser.add_argument("--chat-base-url", default=CHAT_BASE_URL, help="OpenAI-compatible chat base URL.")
    parser.add_argument("--chat-api-key", default=CHAT_API_KEY, help="Bearer token for the chat endpoint.")
    parser.add_argument("--chat-model", default=CHAT_MODEL, help="Chat model name.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for the conversation model.")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM sampling temperature.")
    parser.add_argument("--max-completion-tokens", type=int, default=400, help="Maximum completion tokens for the LLM response.")
    parser.add_argument("--max-tool-rounds", type=int, default=5, help="Maximum number of LLM/tool rounds per turn.")

    parser.add_argument("--tts-base-url", default=TTS_BASE_URL, help="OpenAI-compatible TTS base URL.")
    parser.add_argument("--tts-api-key", default=TTS_API_KEY, help="Bearer token for the TTS endpoint.")
    parser.add_argument("--tts-backend", default=TTS_BACKEND or None, help="Optional TTS backend override. Leave unset to use the TTS wrapper backend from its environment.")
    parser.add_argument("--tts-model", default=TTS_MODEL, help="TTS model name.")
    parser.add_argument("--tts-task-type", default=TTS_TASK_TYPE, help="TTS task type.")
    parser.add_argument("--tts-language", default=TTS_LANGUAGE, help="TTS language.")
    parser.add_argument("--tts-temperature", type=float, default=TTS_TEMPERATURE, help="Optional TTS sampling temperature override. Leave unset to use the backend default.")
    parser.add_argument("--voice", default=VOICE, help="TTS voice name.")
    parser.add_argument("--tts-instructions", default=TTS_INSTRUCTIONS, help="Instructions sent to the TTS wrapper.")
    parser.add_argument("--tts-sample-rate", type=int, default=TTS_SAMPLE_RATE, help="Expected TTS sample rate.")
    parser.add_argument("--tts-transport", choices=["realtime", "http"], default=TTS_TRANSPORT, help="TTS transport used for streaming.")
    parser.add_argument("--tts-segment-newline-threshold", type=int, default=TTS_SEGMENT_NEWLINE_THRESHOLD, help="Flush streamed TTS text only after this many consecutive newlines. Set to 0 to disable newline-based flushing.")
    parser.add_argument("--tts-segment-min-chars", type=int, default=TTS_SEGMENT_MIN_CHARS, help="Flush streamed TTS text at sentence punctuation once the buffered segment reaches this many characters. Set to 0 to disable this gate and flush at sentence punctuation immediately.")
    parser.add_argument("--tts-segment-max-chars", type=int, default=TTS_SEGMENT_MAX_CHARS, help="Flush streamed TTS text once the buffered segment reaches this many characters. Set to 0 to disable length-based flushing.")
    parser.add_argument("--turn-idle-timeout-seconds", type=float, default=TURN_IDLE_TIMEOUT_SECONDS, help="Pipeline idle timeout for a single conversation turn. Increase this if a long TTS segment can take a while before producing audio.")
    parser.add_argument("--tts-debug-capture", action=argparse.BooleanOptionalAction, default=TTS_DEBUG_CAPTURE, help="When enabled, save synthesized reply audio plus a reverse-transcription debug manifest for TTS dropout diagnosis.")
    parser.add_argument("--save-replies", action=argparse.BooleanOptionalAction, default=False, help="Save synthesized assistant replies as WAV files.")
    args = parser.parse_args()
    args.tts_transport_explicit = any(
        argument == "--tts-transport" or argument.startswith("--tts-transport=")
        for argument in sys.argv[1:]
    )
    args.tts_transport = effective_tts_transport(
        args.tts_transport,
        args.tts_backend,
        transport_explicit=args.tts_transport_explicit,
    )
    return args


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 2 and normalized.shape[1] > normalized.shape[0]:
        normalized = normalized.T
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim == 2 and normalized.shape[1] == 1:
        return normalized[:, 0]
    return normalized.mean(axis=1)


def is_recoverable_llm_turn_error(error: Exception) -> bool:
    message = str(error)
    return (
        "LangChain agent failed:" in message
        or "LLM exceeded the maximum tool-call rounds." in message
    )


def _prepared_audio_from_pcm16(audio_bytes: bytes, *, sample_rate: int, stem: str) -> Any:
    return asr_tools.PreparedAudio(
        source_stem=stem,
        filename=f"{stem}.wav",
        mime_type="audio/wav",
        upload_bytes=asr_tools._wav_bytes(audio_bytes, sample_rate),
        pcm16_bytes=audio_bytes,
        sample_rate=sample_rate,
    )


async def capture_robot_utterance(
    robot: ReachyMini,
    args: argparse.Namespace,
    assistant_speech_state: AssistantSpeechState | None = None,
    stop_when_assistant_stops: bool = False,
    overlap_min_vad_ms: int | None = None,
    barge_in_candidate: bool = False,
) -> CapturedUtterance | None:
    logger = logging.getLogger("conversation.asr")
    input_rate = robot.media.get_input_audio_samplerate()
    poll_interval_s = max(0.001, args.audio_poll_interval_ms / 1000.0)
    listen_started_at = perf_counter()
    capture_started_at: float | None = None

    speech_ms = 0.0
    silence_ms = 0.0
    speech_active = False
    overlap_gate_active = False
    assistant_speaking_observed = False
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
        assistant_speaking = assistant_speech_state.is_speaking() if assistant_speech_state is not None else False
        assistant_speaking_observed = assistant_speaking_observed or assistant_speaking
        active_vad_threshold, active_vad_start_ms = listening_gate_settings(
            args.vad_threshold,
            args.vad_start_ms,
            assistant_speaking=assistant_speaking,
            speaking_threshold_boost=args.assistant_speaking_threshold_boost,
            speaking_vad_start_ms=args.assistant_speaking_vad_start_ms,
        )
        overlap_gate_active = overlap_gate_active or assistant_speaking
        if stop_when_assistant_stops and assistant_speaking_observed and not assistant_speaking and not speech_active:
            logger.debug("Assistant speech ended before barge-in capture started")
            return None

        if not speech_active:
            preroll.append((duration_ms, pcm_chunk))
            preroll_ms += duration_ms
            while preroll and preroll_ms > max(args.vad_preroll_ms, active_vad_start_ms):
                old_duration, _old_chunk = preroll.popleft()
                preroll_ms -= old_duration

        if rms >= active_vad_threshold:
            speech_ms += duration_ms
            silence_ms = 0.0
        else:
            silence_ms += duration_ms
            if not speech_active:
                speech_ms = 0.0

        if not speech_active and speech_ms >= active_vad_start_ms:
            speech_active = True
            capture_started_at = perf_counter()
            if overlap_gate_active:
                logger.info(
                    "[bold blue]ASR[/] speech detected with overlap gate threshold=%.4f start=%dms",
                    active_vad_threshold,
                    active_vad_start_ms,
                )
            else:
                logger.info("[bold blue]ASR[/] speech detected")
            for _duration, buffered in preroll:
                captured_chunks.append(buffered)
            preroll.clear()
            preroll_ms = 0.0

        if speech_active:
            captured_chunks.append(pcm_chunk)
            if silence_ms >= args.vad_end_ms:
                detected_at = datetime.now().isoformat(timespec="milliseconds")
                capture_elapsed_s = perf_counter() - capture_started_at if capture_started_at is not None else 0.0
                logger.debug(
                    "ASR end-of-utterance decision at %s silence=%.0fms elapsed=%.3fs",
                    detected_at,
                    silence_ms,
                    capture_elapsed_s,
                )
                logger.info("[bold blue]ASR[/] end of utterance detected")
                break
            if capture_started_at is not None and perf_counter() - capture_started_at >= args.vad_max_seconds:
                logger.debug(
                    "ASR end-of-utterance decision at %s forced_commit_elapsed=%.3fs",
                    datetime.now().isoformat(timespec="milliseconds"),
                    perf_counter() - capture_started_at,
                )
                logger.info("[bold blue]ASR[/] forcing commit after %.1fs", args.vad_max_seconds)
                break

    if not captured_chunks:
        return None

    captured_duration_ms = sum(len(chunk) for chunk in captured_chunks) / 2 / args.sample_rate * 1000.0
    min_overlap_vad_ms = args.assistant_speaking_min_vad_ms if overlap_min_vad_ms is None else max(0, int(overlap_min_vad_ms))
    if overlap_gate_active and captured_duration_ms < min_overlap_vad_ms:
        logger.info(
            "[bold blue]ASR[/] ignoring overlapping short utterance vad=%.0fms < %dms",
            captured_duration_ms,
            min_overlap_vad_ms,
        )
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CapturedUtterance(
        audio=_prepared_audio_from_pcm16(b"".join(captured_chunks), sample_rate=args.sample_rate, stem=f"reachy_turn_{timestamp}"),
        duration_ms=captured_duration_ms,
        overlap_gate_active=overlap_gate_active,
        barge_in_candidate=barge_in_candidate,
    )


async def transcribe_captured_audio(args: argparse.Namespace, utterance: CapturedUtterance) -> str | dict:
    logger = logging.getLogger("conversation.asr")
    audio = utterance.audio
    request_args = _conversation_transcription_args(args)
    applied_language = asr_tools.apply_language_hint(request_args)
    logger.info(
        "[bold blue]ASR[/] sending %.2fs of audio to %s language=%s",
        len(audio.pcm16_bytes) / 2 / audio.sample_rate,
        request_args.base_url,
        applied_language or "auto",
    )
    started_at = perf_counter()
    if request_args.transport == "realtime":
        payload = await asr_tools.transcribe_realtime(request_args, audio)
    else:
        payload = await asyncio.to_thread(asr_tools.transcribe_http, request_args, audio)
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


def _debug_reply_manifest_path(saved_audio: Path) -> Path:
    return saved_audio.with_name(f"{saved_audio.stem}_debug.json")


def _debug_reply_segments(text: str, args: argparse.Namespace) -> list[str]:
    first_min_chars = args.tts_segment_min_chars
    segments, remainder = _drain_ready_segments(
        text,
        final=False,
        newline_threshold=args.tts_segment_newline_threshold,
        min_chars=first_min_chars,
        max_chars=args.tts_segment_max_chars,
    )
    tail_segments, _tail_remainder = _drain_ready_segments(
        remainder,
        final=True,
        newline_threshold=args.tts_segment_newline_threshold,
        min_chars=0 if segments else first_min_chars,
        max_chars=args.tts_segment_max_chars,
    )
    return [*segments, *tail_segments]


def _prepared_reply_audio(saved_audio: Path, audio: SynthesizedAudio) -> asr_tools.PreparedAudio:
    if audio.num_channels != 1:
        raise ValueError(f"TTS debug capture only supports mono audio, got {audio.num_channels} channels")
    return asr_tools.PreparedAudio(
        source_stem=saved_audio.stem,
        filename=saved_audio.name,
        mime_type="audio/wav",
        upload_bytes=asr_tools._wav_bytes(audio.pcm16_bytes, audio.sample_rate),
        pcm16_bytes=audio.pcm16_bytes,
        sample_rate=audio.sample_rate,
    )


def _reverse_transcription_args(args: argparse.Namespace) -> argparse.Namespace:
    language_hint = args.language
    if isinstance(args.tts_language, str) and args.tts_language.strip().lower().startswith("japanese"):
        language_hint = "ja"
    elif isinstance(args.tts_language, str) and args.tts_language.strip().lower().startswith("english"):
        language_hint = "en"
    return argparse.Namespace(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        transport="http",
        language=language_hint,
        prompt=None,
        response_format="json",
        temperature=0.0,
        word_timestamps=False,
        sample_rate=args.sample_rate,
        realtime_chunk_ms=args.realtime_chunk_ms,
    )


async def capture_tts_debug_artifacts(
    args: argparse.Namespace,
    data_dir: Path,
    audio: SynthesizedAudio,
    assistant_text: str,
    *,
    saved_audio: Path | None = None,
) -> None:
    tts_logger = logging.getLogger("conversation.tts")
    reply_audio_path = saved_audio or save_reply_audio(data_dir, audio)
    manifest_path = _debug_reply_manifest_path(reply_audio_path)
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "assistant_text": assistant_text,
        "tts_transport": args.tts_transport,
        "tts_backend": normalize_tts_backend_name(args.tts_backend or TTS_BACKEND),
        "tts_model": args.tts_model,
        "tts_language": args.tts_language,
        "tts_voice": args.voice,
        "tts_segments": _debug_reply_segments(assistant_text, args),
        "reply_audio_path": str(reply_audio_path),
    }

    try:
        reverse_args = _reverse_transcription_args(args)
        prepared_audio = _prepared_reply_audio(reply_audio_path, audio)
        started_at = perf_counter()
        reverse_payload = await asyncio.to_thread(asr_tools.transcribe_http, reverse_args, prepared_audio)
        payload["reverse_transcription"] = reverse_payload
        tts_logger.info(
            "[bold green]TTS[/] debug reverse transcription finished in %.2fs",
            perf_counter() - started_at,
        )
    except Exception as exc:
        payload["reverse_transcription_error"] = f"{type(exc).__name__}: {exc}"
        tts_logger.warning("[bold green]TTS[/] debug reverse transcription failed: %s", exc)

    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tts_logger.info("[bold green]TTS[/] debug capture saved to %s", manifest_path)


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


def _settings_changed_during_turn(runtime_settings: RuntimeSettings, expected_version: int) -> bool:
    return runtime_settings.snapshot()[-1] != expected_version


async def _capture_barge_in_utterance(
    robot: ReachyMini,
    args: argparse.Namespace,
    *,
    assistant_speech_state: AssistantSpeechState,
    start_event: asyncio.Event | None = None,
) -> CapturedUtterance | None:
    asr_logger = logging.getLogger("conversation.asr")
    if start_event is not None:
        await start_event.wait()
    while True:
        utterance = await capture_robot_utterance(
            robot,
            args,
            assistant_speech_state=assistant_speech_state,
            stop_when_assistant_stops=True,
            overlap_min_vad_ms=args.assistant_speaking_interrupt_min_vad_ms,
            barge_in_candidate=True,
        )
        if utterance is None:
            return None

        payload = await transcribe_captured_audio(args, utterance)
        transcript_text = payload if isinstance(payload, str) else str(payload.get("text", "")).strip()
        if not transcript_text:
            _log_transcript_judgement(
                asr_logger,
                label="barge-in",
                payload=payload,
                transcript_text=transcript_text,
                accepted=False,
                rejection_reason="empty transcript",
            )
            continue

        if isinstance(payload, dict):
            rejection_reason = transcript_quality_rejection_reason(payload, args)
            if rejection_reason is not None:
                _log_transcript_judgement(
                    asr_logger,
                    label="barge-in",
                    payload=payload,
                    transcript_text=transcript_text,
                    accepted=False,
                    rejection_reason=rejection_reason,
                )
                continue

        rejection_reason = overlap_turn_rejection_reason(
            captured_duration_ms=utterance.duration_ms,
            transcript_text=transcript_text,
            min_duration_ms=args.assistant_speaking_interrupt_min_vad_ms,
            min_chars=args.assistant_speaking_interrupt_min_chars,
        )
        if rejection_reason is not None:
            _log_transcript_judgement(
                asr_logger,
                label="barge-in",
                payload=payload,
                transcript_text=transcript_text,
                accepted=False,
                rejection_reason=rejection_reason,
            )
            continue

        utterance.transcription_payload = payload
        utterance.transcript_text = transcript_text
        _log_transcript_judgement(
            asr_logger,
            label="barge-in",
            payload=payload,
            transcript_text=transcript_text,
            accepted=True,
        )
        return utterance


async def _speak_persona_greeting_with_barge_in(
    args: argparse.Namespace,
    robot: ReachyMini,
    history: list[dict[str, str]],
    *,
    reason: str,
    assistant_speech_state: AssistantSpeechState,
) -> tuple[list[dict[str, str]], CapturedUtterance | None]:
    interrupt_event = asyncio.Event()
    barge_in_ready_event = asyncio.Event()
    greeting_task = asyncio.create_task(
        _speak_persona_greeting(
            args,
            robot,
            history,
            reason=reason,
            assistant_speech_state=assistant_speech_state,
            interrupt_event=interrupt_event,
            barge_in_ready_event=barge_in_ready_event,
        )
    )
    barge_in_task = asyncio.create_task(
        _capture_barge_in_utterance(
            robot,
            args,
            assistant_speech_state=assistant_speech_state,
            start_event=barge_in_ready_event,
        )
    )
    try:
        done, _pending = await asyncio.wait({greeting_task, barge_in_task}, return_when=asyncio.FIRST_COMPLETED)
        if greeting_task in done:
            barge_in_task.cancel()
            with suppress(asyncio.CancelledError):
                await barge_in_task
            return await greeting_task, None
        barge_in_utterance = await barge_in_task
        if barge_in_utterance is None:
            return await greeting_task, None
        interrupt_event.set()
        try:
            await greeting_task
        except SpeechInterruptedError:
            pass
        return history, barge_in_utterance
    finally:
        if not greeting_task.done():
            greeting_task.cancel()
            with suppress(asyncio.CancelledError):
                await greeting_task
        if not barge_in_task.done():
            barge_in_task.cancel()
            with suppress(asyncio.CancelledError):
                await barge_in_task


async def _run_pipeline_with_barge_in(
    args: argparse.Namespace,
    runtime: reachy_tools.ReachyToolRuntime,
    transcript_text: str,
    history: list[dict[str, str]],
    active_tools: list[str],
    *,
    robot: ReachyMini,
    assistant_speech_state: AssistantSpeechState,
) -> tuple[PipelineResult | None, CapturedUtterance | None]:
    interrupt_event = asyncio.Event()
    barge_in_ready_event = asyncio.Event()
    reply_task = asyncio.create_task(
        run_pipeline(
            args,
            runtime,
            transcript_text,
            history,
            active_tools,
            assistant_speech_state=assistant_speech_state,
            interrupt_event=interrupt_event,
            llm_finished_event=barge_in_ready_event,
            interrupt_reason="assistant speech interrupted by user speech",
        )
    )
    barge_in_task = asyncio.create_task(
        _capture_barge_in_utterance(
            robot,
            args,
            assistant_speech_state=assistant_speech_state,
            start_event=barge_in_ready_event,
        )
    )
    try:
        done, _pending = await asyncio.wait({reply_task, barge_in_task}, return_when=asyncio.FIRST_COMPLETED)
        if reply_task in done:
            barge_in_task.cancel()
            with suppress(asyncio.CancelledError):
                await barge_in_task
            return await reply_task, None
        barge_in_utterance = await barge_in_task
        if barge_in_utterance is None:
            return await reply_task, None
        interrupt_event.set()
        try:
            await reply_task
        except SpeechInterruptedError:
            pass
        return None, barge_in_utterance
    finally:
        if not reply_task.done():
            reply_task.cancel()
            with suppress(asyncio.CancelledError):
                await reply_task
        if not barge_in_task.done():
            barge_in_task.cancel()
            with suppress(asyncio.CancelledError):
                await barge_in_task


async def conversation_loop(args: argparse.Namespace) -> int:
    app_logger = logging.getLogger("conversation.app")
    asr_logger = logging.getLogger("conversation.asr")
    llm_logger = logging.getLogger("conversation.llm")

    profiles_dir = Path(args.profiles_dir).expanduser().resolve()
    args.profile = args.profile or load_selected_profile_name(profiles_dir)
    args.system_prompt = load_profile_prompt(args)
    primary_voice = load_profile_voice_by_name(profiles_dir, args.profile, args.voice or DEFAULT_VOICE)
    args.voice = build_tts_request_voice(
        primary_voice,
        load_profile_qwen_voice_by_name(profiles_dir, args.profile),
    )
    args.tts_instructions = load_profile_tts_instructions_by_name(profiles_dir, args.profile, args.tts_instructions or DEFAULT_TTS_INSTRUCTIONS)
    _apply_profile_tts_prompt_assets(args, profiles_dir, args.profile)
    resolve_runtime_models(args)
    runtime_settings = RuntimeSettings(
        profiles_dir=profiles_dir,
        active_profile=args.profile,
        enabled_tools=active_tools_for_profile(profiles_dir, args.profile),
        active_character_prompt=load_profile_character_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT),
        active_instructions=load_profile_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT),
        active_voice=primary_voice,
        active_qwen_voice=load_profile_qwen_voice_by_name(profiles_dir, args.profile),
        active_tts_instructions=load_profile_tts_instructions_by_name(profiles_dir, args.profile, args.tts_instructions or DEFAULT_TTS_INSTRUCTIONS),
        gui_tool_names=GUI_TOOL_NAMES,
    )

    robot_kwargs: dict[str, Any] = {}
    if args.robot_host:
        robot_kwargs["host"] = args.robot_host
    if args.robot_name:
        robot_kwargs["robot_name"] = args.robot_name

    history: list[dict[str, str]] = []
    data_dir = Path(args.data_dir).expanduser().resolve()
    last_settings_version = 0
    current_persona_signature = (args.profile, args.system_prompt)
    pending_utterance: CapturedUtterance | None = None
    assistant_speech_state = AssistantSpeechState(tail_hold_s=args.assistant_speaking_tail_ms / 1000.0)
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

    robot = await _connect_robot(args, robot_kwargs, app_logger)
    robot.enable_motors()
    runtime = reachy_tools.ReachyToolRuntime(
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
        history, pending_utterance = await _speak_persona_greeting_with_barge_in(
            args,
            robot,
            history,
            reason="startup",
            assistant_speech_state=assistant_speech_state,
        )

        while not stop_event.is_set():
            active_profile, active_tools, _active_character_prompt, active_instructions, active_voice, active_qwen_voice, active_tts_instructions, settings_version = runtime_settings.snapshot()
            if settings_version != last_settings_version:
                history.clear()
                last_settings_version = settings_version
                app_logger.info("[bold]Conversation history reset[/] profile=%s", active_profile)
            args.profile = active_profile
            args.system_prompt = active_instructions
            args.voice = build_tts_request_voice(
                active_voice,
                active_qwen_voice,
            )
            args.tts_instructions = active_tts_instructions
            _apply_profile_tts_prompt_assets(args, profiles_dir, active_profile)
            next_persona_signature = (active_profile, active_instructions)
            if next_persona_signature != current_persona_signature:
                history, pending_utterance = await _speak_persona_greeting_with_barge_in(
                    args,
                    robot,
                    history,
                    reason="persona switch",
                    assistant_speech_state=assistant_speech_state,
                )
                current_persona_signature = next_persona_signature
            if pending_utterance is not None:
                captured_utterance = pending_utterance
                pending_utterance = None
            else:
                captured_utterance = await capture_robot_utterance(robot, args, assistant_speech_state=assistant_speech_state)
            if captured_utterance is None or stop_event.is_set():
                continue
            if _settings_changed_during_turn(runtime_settings, settings_version):
                app_logger.info("[bold]Discarding pending utterance because live settings changed[/]")
                continue

            if captured_utterance.transcription_payload is not None:
                payload = captured_utterance.transcription_payload
                transcript_text = captured_utterance.transcript_text
            else:
                payload = await transcribe_captured_audio(args, captured_utterance)
                if _settings_changed_during_turn(runtime_settings, settings_version):
                    app_logger.info("[bold]Discarding transcript because live settings changed[/]")
                    continue
                transcript_text = payload if isinstance(payload, str) else str(payload.get("text", "")).strip()
            if not transcript_text:
                _log_transcript_judgement(
                    asr_logger,
                    label="turn",
                    payload=payload,
                    transcript_text=transcript_text,
                    accepted=False,
                    rejection_reason="empty transcript",
                )
                continue

            if isinstance(payload, dict):
                rejection_reason = transcript_quality_rejection_reason(payload, args)
                if rejection_reason is not None:
                    _log_transcript_judgement(
                        asr_logger,
                        label="turn",
                        payload=payload,
                        transcript_text=transcript_text,
                        accepted=False,
                        rejection_reason=rejection_reason,
                    )
                    continue

            if captured_utterance.overlap_gate_active:
                min_overlap_vad_ms = args.assistant_speaking_interrupt_min_vad_ms if captured_utterance.barge_in_candidate else args.assistant_speaking_min_vad_ms
                min_overlap_chars = args.assistant_speaking_interrupt_min_chars if captured_utterance.barge_in_candidate else args.assistant_speaking_min_chars
                rejection_reason = overlap_turn_rejection_reason(
                    captured_duration_ms=captured_utterance.duration_ms,
                    transcript_text=transcript_text,
                    min_duration_ms=min_overlap_vad_ms,
                    min_chars=min_overlap_chars,
                )
                if rejection_reason is not None:
                    _log_transcript_judgement(
                        asr_logger,
                        label="barge-in" if captured_utterance.barge_in_candidate else "turn",
                        payload=payload,
                        transcript_text=transcript_text,
                        accepted=False,
                        rejection_reason=rejection_reason,
                    )
                    continue

            _log_transcript_judgement(
                asr_logger,
                label="barge-in" if captured_utterance.barge_in_candidate else "turn",
                payload=payload,
                transcript_text=transcript_text,
                accepted=True,
            )

            if args.save_transcripts:
                saved = asr_tools.save_output(payload, argparse.Namespace(**{**vars(_conversation_transcription_args(args)), "output": None, "save_output": True}), captured_utterance.audio)
                if saved is not None:
                    asr_logger.info("[bold blue]ASR[/] transcript saved to %s", saved)

            llm_logger.info("[bold cyan]LLM[/] user %s", transcript_text)
            try:
                result, pending_utterance = await _run_pipeline_with_barge_in(
                    args,
                    runtime,
                    transcript_text,
                    history,
                    active_tools,
                    robot=robot,
                    assistant_speech_state=assistant_speech_state,
                )
                if pending_utterance is not None:
                    llm_logger.info("[bold cyan]LLM[/] interrupted live reply because user speech was detected")
                    continue
            except RuntimeError as exc:
                if is_recoverable_llm_turn_error(exc):
                    llm_logger.warning("[bold cyan]LLM[/] turn aborted without reply: %s", exc)
                    continue
                raise
            if result is None:
                continue
            llm_logger.info("[bold cyan]LLM[/] assistant %s", result.assistant_text)

            history.extend(
                [
                    {"role": "user", "content": transcript_text},
                    {"role": "assistant", "content": result.assistant_text},
                ]
            )
            history = trim_history(history, args.history_turns)

            saved_audio: Path | None = None
            if args.save_replies:
                saved_audio = save_reply_audio(data_dir, result.audio)
                logging.getLogger("conversation.tts").info("[bold green]TTS[/] reply saved to %s", saved_audio)
            if args.tts_debug_capture:
                await capture_tts_debug_artifacts(
                    args,
                    data_dir,
                    result.audio,
                    result.assistant_text,
                    saved_audio=saved_audio,
                )
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
        await _run_blocking_cleanup_step(app_logger, "media_manager.close", robot.media_manager.close)
        await _run_blocking_cleanup_step(app_logger, "client.disconnect", robot.client.disconnect)
    return 0


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
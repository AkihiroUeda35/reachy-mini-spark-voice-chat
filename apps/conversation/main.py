from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import logging
import os
import signal
import threading
import wave
from collections import deque
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import numpy as np
from reachy_mini import ReachyMini
from rich.logging import RichHandler

import local_tts
import reachy_conversation_tools as reachy_tools
import whisper_asr as asr_tools
from config import load_entrypoint_env
from openai_model_registry import resolve_model

from gradio_ui import launch_gradio_ui
from pipeline import SynthesizedAudio, _resample_audio, run_pipeline
from state import APP_DIR, DEFAULT_CHARACTER_PROMPT, DEFAULT_DATA_DIR, DEFAULT_SYSTEM_PROMPT, DEFAULT_VOICE, DEFAULT_TTS_INSTRUCTIONS, AssistantSpeechState, RuntimeSettings, active_tools_for_profile as _active_tools_for_profile, listening_gate_settings, load_profile_character_prompt_by_name, load_profile_prompt as _load_profile_prompt, load_profile_prompt_by_name, load_profile_tts_instructions_by_name, load_profile_voice_by_name, load_selected_profile_name, overlap_turn_rejection_reason, save_profile_definition as _save_profile_definition

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
TTS_INSTRUCTIONS = local_tts.TTS_INSTRUCTIONS
TTS_SAMPLE_RATE = local_tts.TTS_SAMPLE_RATE
VOICE = local_tts.VOICE

TTS_TRANSPORT = os.environ.get("TTS_TRANSPORT", "realtime")
TTS_SEGMENT_NEWLINE_THRESHOLD = int(os.environ.get("TTS_SEGMENT_NEWLINE_THRESHOLD", "2"))
TTS_SEGMENT_MAX_CHARS = int(os.environ.get("TTS_SEGMENT_MAX_CHARS", "100"))
SHUTDOWN_STEP_TIMEOUT_S = 2.0


@dataclass
class CapturedUtterance:
    audio: Any
    duration_ms: float
    overlap_gate_active: bool


def active_tools_for_profile(profiles_dir: Path, profile: str) -> list[str]:
    return _active_tools_for_profile(profiles_dir, profile, GUI_TOOL_NAMES)


def save_profile_definition(
    profiles_dir: Path,
    profile: str,
    character_prompt: str,
    selected_tools: list[str],
    voice: str,
    tts_instructions: str,
) -> Path:
    return _save_profile_definition(profiles_dir, profile, character_prompt, selected_tools, voice, tts_instructions, GUI_TOOL_NAMES)


def load_profile_prompt(args: argparse.Namespace) -> str:
    return _load_profile_prompt(args)


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
    parser.add_argument("--assistant-speaking-tail-ms", type=int, default=350, help="Extra time after queued TTS audio where overlap protection stays active.")
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
    parser.add_argument("--tts-segment-newline-threshold", type=int, default=TTS_SEGMENT_NEWLINE_THRESHOLD, help="Flush streamed TTS text only after this many consecutive newlines. Set to 0 to disable newline-based flushing.")
    parser.add_argument("--tts-segment-max-chars", type=int, default=TTS_SEGMENT_MAX_CHARS, help="Flush streamed TTS text once the buffered segment reaches this many characters. Set to 0 to disable length-based flushing.")
    parser.add_argument("--save-replies", action=argparse.BooleanOptionalAction, default=False, help="Save synthesized assistant replies as WAV files.")
    return parser.parse_args()


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
        active_vad_threshold, active_vad_start_ms = listening_gate_settings(
            args.vad_threshold,
            args.vad_start_ms,
            assistant_speaking=assistant_speaking,
            speaking_threshold_boost=args.assistant_speaking_threshold_boost,
            speaking_vad_start_ms=args.assistant_speaking_vad_start_ms,
        )
        overlap_gate_active = overlap_gate_active or assistant_speaking

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
    if overlap_gate_active and captured_duration_ms < args.assistant_speaking_min_vad_ms:
        logger.info(
            "[bold blue]ASR[/] ignoring overlapping short utterance vad=%.0fms < %dms",
            captured_duration_ms,
            args.assistant_speaking_min_vad_ms,
        )
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CapturedUtterance(
        audio=_prepared_audio_from_pcm16(b"".join(captured_chunks), sample_rate=args.sample_rate, stem=f"reachy_turn_{timestamp}"),
        duration_ms=captured_duration_ms,
        overlap_gate_active=overlap_gate_active,
    )


async def transcribe_captured_audio(args: argparse.Namespace, utterance: CapturedUtterance) -> str | dict:
    logger = logging.getLogger("conversation.asr")
    audio = utterance.audio
    applied_language = asr_tools.apply_language_hint(args)
    logger.info(
        "[bold blue]ASR[/] sending %.2fs of audio to %s language=%s",
        len(audio.pcm16_bytes) / 2 / audio.sample_rate,
        args.base_url,
        applied_language or "auto",
    )
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
    args.profile = args.profile or load_selected_profile_name(profiles_dir)
    args.system_prompt = load_profile_prompt(args)
    args.voice = load_profile_voice_by_name(profiles_dir, args.profile, args.voice or DEFAULT_VOICE)
    args.tts_instructions = load_profile_tts_instructions_by_name(profiles_dir, args.profile, args.tts_instructions or DEFAULT_TTS_INSTRUCTIONS)
    resolve_runtime_models(args)
    runtime_settings = RuntimeSettings(
        profiles_dir=profiles_dir,
        active_profile=args.profile,
        enabled_tools=active_tools_for_profile(profiles_dir, args.profile),
        active_character_prompt=load_profile_character_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT),
        active_instructions=load_profile_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT),
        active_voice=load_profile_voice_by_name(profiles_dir, args.profile, args.voice or DEFAULT_VOICE),
        active_tts_instructions=load_profile_tts_instructions_by_name(profiles_dir, args.profile, args.tts_instructions or DEFAULT_TTS_INSTRUCTIONS),
        gui_tool_names=GUI_TOOL_NAMES,
    )

    robot_kwargs: dict[str, Any] = {}
    if args.robot_name:
        robot_kwargs["robot_name"] = args.robot_name

    history: list[dict[str, str]] = []
    data_dir = Path(args.data_dir).expanduser().resolve()
    last_settings_version = 0
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

    robot = ReachyMini(**robot_kwargs)
    robot.enable_motors()
    robot.wake_up()
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

        while not stop_event.is_set():
            active_profile, active_tools, _active_character_prompt, active_instructions, active_voice, active_tts_instructions, settings_version = runtime_settings.snapshot()
            if settings_version != last_settings_version:
                history.clear()
                last_settings_version = settings_version
                app_logger.info("[bold]Conversation history reset[/] profile=%s", active_profile)
            args.profile = active_profile
            args.system_prompt = active_instructions
            args.voice = active_voice
            args.tts_instructions = active_tts_instructions
            captured_utterance = await capture_robot_utterance(robot, args, assistant_speech_state=assistant_speech_state)
            if captured_utterance is None or stop_event.is_set():
                continue

            payload = await transcribe_captured_audio(args, captured_utterance)
            transcript_text = payload if isinstance(payload, str) else str(payload.get("text", "")).strip()
            if not transcript_text:
                asr_logger.warning("[bold blue]ASR[/] empty transcript, skipping turn")
                continue

            if captured_utterance.overlap_gate_active:
                rejection_reason = overlap_turn_rejection_reason(
                    captured_duration_ms=captured_utterance.duration_ms,
                    transcript_text=transcript_text,
                    min_duration_ms=args.assistant_speaking_min_vad_ms,
                    min_chars=args.assistant_speaking_min_chars,
                )
                if rejection_reason is not None:
                    asr_logger.info("[bold blue]ASR[/] ignoring overlapping short utterance (%s)", rejection_reason)
                    continue

            asr_logger.info("[bold blue]ASR[/] transcript %s", transcript_text)

            if args.save_transcripts:
                saved = asr_tools.save_output(payload, argparse.Namespace(**{**vars(args), "output": None, "save_output": True}), captured_utterance.audio)
                if saved is not None:
                    asr_logger.info("[bold blue]ASR[/] transcript saved to %s", saved)

            llm_logger.info("[bold cyan]LLM[/] user %s", transcript_text)
            try:
                result = await run_pipeline(args, runtime, transcript_text, history, active_tools, assistant_speech_state=assistant_speech_state)
            except RuntimeError as exc:
                if is_recoverable_llm_turn_error(exc):
                    llm_logger.warning("[bold cyan]LLM[/] turn aborted without reply: %s", exc)
                    continue
                raise
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
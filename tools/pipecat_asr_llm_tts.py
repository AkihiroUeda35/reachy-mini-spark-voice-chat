import argparse
import asyncio
import io
import os
import sys
import wave
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from langgraph_agent import LangGraphLLMProcessor
from openai_model_registry import resolve_model
from pipecat.frames.frames import EndFrame, ErrorFrame, FunctionCallInProgressFrame, FunctionCallResultFrame, LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import langchain_openai_tts as tts_defaults
import whisper_asr_test as asr_tools


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant Reachy(リーチー) in a spoken Japanese conversation. "
    "Your reply will be read aloud, so keep it concise, natural, and easy to speak. "
    "Avoid markdown, bullets, emojis, and long enumerations. "
    "If you need a tool, call it immediately without filler like '確認します'. "
    "Only produce user-facing answer text after you have the tool result."
)
TTS_TIMEOUT = float(os.environ.get("TTS_TIMEOUT", "600"))
TTS_STREAM_CHUNK_BYTES = int(os.environ.get("TTS_STREAM_CHUNK_BYTES", "8192"))
TTS_PLAYBACK_LEAD_IN_MS = int(os.environ.get("TTS_PLAYBACK_LEAD_IN_MS", "5"))
TTS_PLAYBACK_FADE_IN_MS = int(os.environ.get("TTS_PLAYBACK_FADE_IN_MS", "5"))


@dataclass
class SynthesizedAudio:
    pcm16_bytes: bytes
    sample_rate: int
    num_channels: int


@dataclass
class PipelineResult:
    assistant_text: str
    audio: SynthesizedAudio


class LocalOpenAITTSProcessor(FrameProcessor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(name="LocalOpenAITTSProcessor")
        self._args = args
        self._text_chunks: list[str] = []
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
            while True:
                segment = await self._segment_queue.get()
                try:
                    if segment is None:
                        return

                    async for audio_frame in synthesize_audio_stream(self._args, segment):
                        await self.push_frame(audio_frame, direction)
                finally:
                    self._segment_queue.task_done()
        except Exception as exc:
            await self.push_frame(ErrorFrame(error=f"Streaming TTS failed: {exc}"), direction)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._text_chunks = []
            self._pending_text = ""
            self._segment_queue = asyncio.Queue()
            self._worker_task = asyncio.create_task(self._tts_worker(direction))
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            if frame.text:
                self._text_chunks.append(frame.text)
                self._pending_text += frame.text
                segments, remainder = drain_ready_tts_segments(self._pending_text, final=False)
                self._pending_text = remainder
            await self.push_frame(frame, direction)
            if frame.text and self._segment_queue is not None:
                for segment in segments:
                    await self._segment_queue.put(segment)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._segment_queue is not None:
                segments, remainder = drain_ready_tts_segments(self._pending_text, final=True)
                self._pending_text = remainder
                for segment in segments:
                    await self._segment_queue.put(segment)
                await self._segment_queue.join()
                await self._stop_worker()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


class LiveAudioPlayer(FrameProcessor):
    def __init__(self, device: str | None):
        super().__init__(name="LiveAudioPlayer")
        self._device = device
        self._stream = None
        self._playback_started = False
        self.errors: list[str] = []

    def _ensure_stream(self, sample_rate: int, num_channels: int):
        if self._stream is not None:
            return self._stream

        try:
            import sounddevice as sd
        except OSError as exc:
            raise RuntimeError("Sound output requires PortAudio to be installed on the host.") from exc

        self._stream = sd.RawOutputStream(
            samplerate=sample_rate,
            channels=num_channels,
            dtype="int16",
            device=self._device,
        )
        self._stream.start()
        return self._stream

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame):
            try:
                stream = self._ensure_stream(frame.sample_rate, frame.num_channels)
                if not self._playback_started:
                    lead_in_bytes = silence_pcm16(
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        duration_ms=TTS_PLAYBACK_LEAD_IN_MS,
                    )
                    if lead_in_bytes:
                        await asyncio.to_thread(stream.write, lead_in_bytes)
                    audio_bytes = apply_fade_in_pcm16(
                        frame.audio,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        fade_ms=TTS_PLAYBACK_FADE_IN_MS,
                    )
                    self._playback_started = True
                else:
                    audio_bytes = frame.audio

                await asyncio.to_thread(stream.write, audio_bytes)
            except Exception as exc:
                message = f"Live audio playback failed: {exc}"
                self.errors.append(message)
                await self.push_frame(ErrorFrame(error=message), direction)

        if isinstance(frame, EndFrame) and self._stream is not None:
            await asyncio.to_thread(self._stream.stop)
            await asyncio.to_thread(self._stream.close)
            self._stream = None
            self._playback_started = False

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

        if isinstance(frame, LLMTextFrame):
            if frame.text:
                self.assistant_chunks.append(frame.text)
        elif isinstance(frame, OutputAudioRawFrame):
            if self.sample_rate is None:
                self.sample_rate = frame.sample_rate
                self.num_channels = frame.num_channels
            elif self.sample_rate != frame.sample_rate or self.num_channels != frame.num_channels:
                self.errors.append("TTS returned inconsistent audio stream settings.")
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
            is_final = frame.properties.is_final if frame.properties is not None else True
            if is_final:
                self._tool_calls_in_progress = max(0, self._tool_calls_in_progress - 1)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if not self._end_queued and self._tool_calls_in_progress == 0 and self._task is not None:
                self._end_queued = True
                await self._task.queue_frame(EndFrame())

        await self.push_frame(frame, direction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run microphone or audio-file input through ASR, local LLM, and TTS using a Pipecat pipeline."
    )
    parser.add_argument("file", nargs="?", help="Audio file to transcribe. If omitted, records from the microphone.")
    parser.add_argument("--microphone", action="store_true", help="Record from the default microphone even if a file path is provided.")

    parser.add_argument("--base-url", default=asr_tools.ASR_BASE_URL, help="OpenAI-compatible STT base URL.")
    parser.add_argument("--api-key", default=asr_tools.ASR_API_KEY, help="Bearer token for the STT endpoint.")
    parser.add_argument("--model", default=asr_tools.ASR_MODEL, help="STT model name sent to the API.")
    parser.add_argument("--transport", choices=["realtime", "http"], default=asr_tools.ASR_TRANSPORT, help="STT transport to use.")
    parser.add_argument("--language", default=asr_tools.ASR_LANGUAGE, help="Language hint, for example ja or en. Use 'auto' to enable autodetection.")
    parser.add_argument("--prompt", help="Optional STT transcription prompt.")
    parser.add_argument(
        "--response-format",
        choices=["text", "json", "verbose_json"],
        default="text",
        help="STT response format.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for HTTP transcription mode.")
    parser.add_argument("--word-timestamps", action="store_true", help="Request word timestamps when supported.")
    parser.add_argument("--sample-rate", type=int, default=asr_tools.ASR_SAMPLE_RATE, help="Target sample rate for microphone capture and realtime audio.")
    parser.add_argument("--record-seconds", type=float, default=asr_tools.ASR_RECORD_SECONDS, help="Microphone recording length in seconds for non-realtime capture.")
    parser.add_argument("--realtime-chunk-ms", type=int, default=asr_tools.ASR_REALTIME_CHUNK_MS, help="Realtime audio chunk size in milliseconds.")
    parser.add_argument("--vad-threshold", type=float, default=asr_tools.ASR_VAD_THRESHOLD, help="RMS threshold for local VAD in microphone realtime mode.")
    parser.add_argument("--vad-frame-ms", type=int, default=asr_tools.ASR_VAD_FRAME_MS, help="Frame size in milliseconds for local VAD.")
    parser.add_argument("--vad-start-ms", type=int, default=asr_tools.ASR_VAD_START_MS, help="Speech duration in milliseconds required to trigger input start.")
    parser.add_argument("--vad-end-ms", type=int, default=asr_tools.ASR_VAD_END_MS, help="Silence duration in milliseconds required to trigger input end.")
    parser.add_argument("--vad-preroll-ms", type=int, default=asr_tools.ASR_VAD_PREROLL_MS, help="Audio to keep before VAD start so the first syllable is preserved.")
    parser.add_argument("--vad-max-seconds", type=float, default=asr_tools.ASR_VAD_MAX_SECONDS, help="Maximum microphone wait or capture time before aborting or forcing commit.")
    parser.add_argument("--output", help="Optional transcript output path. Defaults to ./data/<source>_<timestamp>_transcript.(txt|json).")
    parser.add_argument(
        "--save-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the transcription result to a file.",
    )

    parser.add_argument("--chat-base-url", default=tts_defaults.CHAT_BASE_URL, help="OpenAI-compatible chat base URL.")
    parser.add_argument("--chat-api-key", default=tts_defaults.CHAT_API_KEY, help="Bearer token for the chat endpoint.")
    parser.add_argument("--chat-model", default=tts_defaults.CHAT_MODEL, help="Chat model name.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt sent to the LLM.")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM sampling temperature.")
    parser.add_argument("--max-completion-tokens", type=int, default=1000, help="Maximum completion tokens for the LLM response.")

    parser.add_argument("--tts-base-url", default=tts_defaults.TTS_BASE_URL, help="OpenAI-compatible TTS base URL.")
    parser.add_argument("--tts-api-key", default=tts_defaults.TTS_API_KEY, help="Bearer token for the TTS endpoint.")
    parser.add_argument("--tts-model", default=tts_defaults.TTS_MODEL, help="TTS model name.")
    parser.add_argument("--tts-task-type", default=tts_defaults.TTS_TASK_TYPE, help="Task type sent to the TTS wrapper.")
    parser.add_argument("--tts-language", default=tts_defaults.TTS_LANGUAGE, help="Language sent to the TTS wrapper.")
    parser.add_argument("--voice", default=tts_defaults.VOICE, help="Voice name sent to the TTS wrapper.")
    parser.add_argument("--tts-instructions", default=tts_defaults.TTS_INSTRUCTIONS, help="Instructions sent to the TTS wrapper.")
    parser.add_argument("--tts-sample-rate", type=int, default=tts_defaults.TTS_SAMPLE_RATE, help="Expected TTS sample rate.")

    parser.add_argument("--output-mode", choices=["sound", "file"], default="sound", help="Assistant audio destination. Defaults to local sound output.")
    parser.add_argument("--output-audio", help="Output WAV path when --output-mode file is used.")
    parser.add_argument("--playback-device", help="Optional sounddevice output device name or index.")
    return parser.parse_args()


def extract_transcript_text(payload: str | dict) -> str:
    if isinstance(payload, str):
        return payload.strip()
    return str(payload.get("text", "")).strip()


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
        fallback_model=tts_defaults.CHAT_MODEL_FALLBACK,
    )
    args.tts_model = resolve_model(
        base_url=args.tts_base_url,
        api_key=args.tts_api_key,
        explicit_model=args.tts_model,
        capability="speech",
        fallback_model=tts_defaults.TTS_MODEL_FALLBACK,
    )


def build_llm_context(args: argparse.Namespace, transcript_text: str) -> LLMContext:
    return LLMContext(
        messages=[
            {"role": "user", "content": transcript_text},
        ]
    )


def default_audio_output_path() -> Path:
    asr_tools.DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return asr_tools.DATA_DIR / f"pipecat_reply_{timestamp}.wav"


def drain_ready_tts_segments(buffer: str, *, final: bool) -> tuple[list[str], str]:
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


def pcm16_to_float32(audio_bytes: bytes, num_channels: int) -> np.ndarray:
    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels)
    return audio.astype(np.float32) / 32768.0


def silence_pcm16(*, sample_rate: int, num_channels: int, duration_ms: int) -> bytes:
    if duration_ms <= 0 or sample_rate <= 0 or num_channels <= 0:
        return b""

    frame_count = max(1, int(round(sample_rate * duration_ms / 1000)))
    return b"\x00\x00" * frame_count * num_channels


def apply_fade_in_pcm16(audio_bytes: bytes, *, sample_rate: int, num_channels: int, fade_ms: int) -> bytes:
    if fade_ms <= 0 or sample_rate <= 0 or num_channels <= 0 or not audio_bytes:
        return audio_bytes

    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio.size == 0:
        return audio_bytes

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels)

    frame_count = audio.shape[0]
    fade_frames = min(frame_count, max(1, int(round(sample_rate * fade_ms / 1000))))
    if fade_frames <= 0:
        return audio_bytes

    faded = audio.astype(np.float32, copy=True)
    envelope = np.linspace(0.0, 1.0, num=fade_frames, endpoint=True, dtype=np.float32)
    if num_channels > 1:
        faded[:fade_frames, :] *= envelope[:, None]
    else:
        faded[:fade_frames] *= envelope

    return np.clip(faded, -32768, 32767).astype(np.int16).tobytes()


def write_wav_file(audio: SynthesizedAudio, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(audio.num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(audio.pcm16_bytes)
    return output_path


def play_audio(audio: SynthesizedAudio, device: str | None) -> None:
    try:
        import sounddevice as sd
    except OSError as exc:
        raise RuntimeError("Sound output requires PortAudio to be installed on the host.") from exc

    waveform = pcm16_to_float32(audio.pcm16_bytes, audio.num_channels)
    sd.play(waveform, samplerate=audio.sample_rate, device=device)
    sd.wait()


async def transcribe_input(args: argparse.Namespace) -> tuple[str | dict, asr_tools.PreparedAudio]:
    if args.transport == "realtime" and (args.microphone or not args.file):
        return await asr_tools.transcribe_realtime_microphone(args)

    audio = asr_tools.prepare_audio(args)
    if args.transport == "realtime":
        payload = await asr_tools.transcribe_realtime(args, audio)
    else:
        payload = asr_tools.transcribe_http(args, audio)
    return payload, audio


async def synthesize_audio(args: argparse.Namespace, text: str) -> SynthesizedAudio:
    payload = {
        "model": args.tts_model,
        "task_type": args.tts_task_type,
        "language": args.tts_language,
        "voice": args.voice,
        "input": text,
        "instructions": args.tts_instructions,
        "response_format": "wav",
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=TTS_TIMEOUT) as client:
        response = await client.post(
            f"{args.tts_base_url.rstrip('/')}/audio/speech",
            headers={"Authorization": f"Bearer {args.tts_api_key}"},
            json=payload,
        )
        response.raise_for_status()

    audio_data, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=True)
    pcm16 = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16).tobytes()
    return SynthesizedAudio(
        pcm16_bytes=pcm16,
        sample_rate=int(sample_rate),
        num_channels=int(audio_data.shape[1]),
    )


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
                    yield OutputAudioRawFrame(
                        audio=chunk,
                        sample_rate=args.tts_sample_rate,
                        num_channels=1,
                    )


async def run_pipeline(args: argparse.Namespace, transcript_text: str) -> PipelineResult:
    llm = LangGraphLLMProcessor(args)
    tts = LocalOpenAITTSProcessor(args)
    collector = ResultCollector()
    terminator = PipelineTerminator()
    processors: list[FrameProcessor] = [llm, tts, collector]
    audio_player: LiveAudioPlayer | None = None
    if args.output_mode == "sound":
        audio_player = LiveAudioPlayer(args.playback_device)
        processors.append(audio_player)
    processors.append(terminator)
    pipeline = Pipeline(processors)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=args.sample_rate,
            audio_out_sample_rate=args.tts_sample_rate,
        ),
        idle_timeout_secs=60,
    )
    terminator.bind_task(task)
    runner = PipelineRunner()

    runner_task = asyncio.create_task(runner.run(task))
    await task.queue_frame(LLMContextFrame(build_llm_context(args, transcript_text)))
    await runner_task

    if collector.errors:
        raise RuntimeError("; ".join(collector.errors))
    if audio_player is not None and audio_player.errors:
        raise RuntimeError("; ".join(audio_player.errors))

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


async def main_async() -> int:
    args = parse_args()

    try:
        resolve_runtime_models(args)
        payload, prepared_audio = await transcribe_input(args)
        transcript_output = asr_tools.save_output(payload, args, prepared_audio)
        transcript_text = extract_transcript_text(payload)
        if not transcript_text:
            raise RuntimeError("STT returned an empty transcript.")

        result = await run_pipeline(args, transcript_text)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"Voice pipeline failed: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        print(f"Voice pipeline failed: {exc.response.status_code} {detail}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"Voice pipeline failed: {exc}", file=sys.stderr)
        return 1

    print(f"transcript: {transcript_text}")
    print(f"assistant: {result.assistant_text}")

    if transcript_output is not None:
        print(f"transcript_saved: {transcript_output}", file=sys.stderr)

    if args.output_mode == "file":
        output_path = Path(args.output_audio).expanduser().resolve() if args.output_audio else default_audio_output_path()
        saved_path = write_wav_file(result.audio, output_path)
        print(f"audio_saved: {saved_path}")
    else:
        print("audio_played: default-output" if args.playback_device is None else f"audio_played: {args.playback_device}")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
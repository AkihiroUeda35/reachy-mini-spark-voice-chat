import argparse
import asyncio
import base64
import math
import json
import mimetypes
import os
import struct
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse, urlunparse

import httpx
import numpy as np
import soundfile as sf
from websockets import connect as ws_connect

from openai_model_registry import resolve_model


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / os.environ.get("ASR_OUTPUT_DIR", "data")
ASR_BASE_URL = os.environ.get("ASR_BASE_URL", "http://localhost:8020/v1").rstrip("/")
ASR_API_KEY = os.environ.get("ASR_API_KEY", "local")
ASR_MODEL = os.environ.get("ASR_MODEL")
ASR_MODEL_FALLBACK = os.environ.get("ASR_MODEL_FALLBACK", "whisper-1")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "ja")
ASR_TIMEOUT = float(os.environ.get("ASR_TIMEOUT", "600"))
ASR_TRANSPORT = os.environ.get("ASR_TRANSPORT", "realtime")
ASR_SAMPLE_RATE = int(os.environ.get("ASR_SAMPLE_RATE", "16000"))
ASR_RECORD_SECONDS = float(os.environ.get("ASR_RECORD_SECONDS", "5"))
ASR_REALTIME_CHUNK_MS = int(os.environ.get("ASR_REALTIME_CHUNK_MS", "250"))
ASR_VAD_THRESHOLD = float(os.environ.get("ASR_VAD_THRESHOLD", "0.012"))
ASR_VAD_FRAME_MS = int(os.environ.get("ASR_VAD_FRAME_MS", "30"))
ASR_VAD_START_MS = int(os.environ.get("ASR_VAD_START_MS", "90"))
ASR_VAD_END_MS = int(os.environ.get("ASR_VAD_END_MS", "900"))
ASR_VAD_PREROLL_MS = int(os.environ.get("ASR_VAD_PREROLL_MS", "450"))
ASR_VAD_MAX_SECONDS = float(os.environ.get("ASR_VAD_MAX_SECONDS", "20"))


@dataclass
class PreparedAudio:
    source_stem: str
    filename: str
    mime_type: str
    upload_bytes: bytes
    pcm16_bytes: bytes
    sample_rate: int


@dataclass
class LocalVAD:
    threshold: float
    start_frames: int
    end_frames: int
    speech_frames: int = 0
    silence_frames: int = 0
    active: bool = False

    def process(self, audio_frame: np.ndarray) -> tuple[bool, bool, bool, float]:
        rms = float(np.sqrt(np.mean(np.square(audio_frame), dtype=np.float32))) if audio_frame.size else 0.0
        is_speech = rms >= self.threshold
        started_now = False
        ended_now = False

        if not self.active:
            self.speech_frames = self.speech_frames + 1 if is_speech else 0
            if self.speech_frames >= self.start_frames:
                self.active = True
                self.silence_frames = 0
                started_now = True
        else:
            if is_speech:
                self.silence_frames = 0
            else:
                self.silence_frames += 1
                if self.silence_frames >= self.end_frames:
                    self.active = False
                    self.speech_frames = 0
                    self.silence_frames = 0
                    ended_now = True

        return is_speech, started_now, ended_now, rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record or load audio and send it to the local Whisper-compatible STT endpoint.")
    parser.add_argument("file", nargs="?", help="Audio file to transcribe. If omitted, records from the microphone.")
    parser.add_argument("--microphone", action="store_true", help="Record from the default microphone even if a file path is provided.")
    parser.add_argument("--base-url", default=ASR_BASE_URL, help="OpenAI-compatible STT base URL.")
    parser.add_argument("--api-key", default=ASR_API_KEY, help="Bearer token for the STT endpoint.")
    parser.add_argument("--model", default=ASR_MODEL, help="Model name sent to the API.")
    parser.add_argument("--transport", choices=["realtime", "http"], default=ASR_TRANSPORT, help="STT transport to use. Defaults to realtime.")
    parser.add_argument("--language", default=ASR_LANGUAGE, help="Language hint, for example ja or en. Use 'auto' to enable language autodetection.")
    parser.add_argument("--prompt", help="Optional transcription prompt.")
    parser.add_argument(
        "--response-format",
        choices=["text", "json", "verbose_json"],
        default="text",
        help="Output format written to stdout and the saved transcript file.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for HTTP transcription mode.")
    parser.add_argument("--word-timestamps", action="store_true", help="Request word timestamps when supported.")
    parser.add_argument("--sample-rate", type=int, default=ASR_SAMPLE_RATE, help="Target sample rate for microphone capture and realtime audio.")
    parser.add_argument("--record-seconds", type=float, default=ASR_RECORD_SECONDS, help="Microphone recording length in seconds.")
    parser.add_argument("--realtime-chunk-ms", type=int, default=ASR_REALTIME_CHUNK_MS, help="Realtime audio chunk size in milliseconds.")
    parser.add_argument("--vad-threshold", type=float, default=ASR_VAD_THRESHOLD, help="RMS threshold for local VAD when using the microphone with realtime transport.")
    parser.add_argument("--vad-frame-ms", type=int, default=ASR_VAD_FRAME_MS, help="Frame size in milliseconds for local VAD.")
    parser.add_argument("--vad-start-ms", type=int, default=ASR_VAD_START_MS, help="Speech duration in milliseconds required to trigger input start.")
    parser.add_argument("--vad-end-ms", type=int, default=ASR_VAD_END_MS, help="Silence duration in milliseconds required to trigger input end.")
    parser.add_argument("--vad-preroll-ms", type=int, default=ASR_VAD_PREROLL_MS, help="Audio to keep before VAD start so the first syllable is preserved.")
    parser.add_argument("--vad-max-seconds", type=float, default=ASR_VAD_MAX_SECONDS, help="Maximum microphone wait/capture time before aborting or forcing commit.")
    parser.add_argument("--output", help="Optional transcript output path. Defaults to ./data/<source>_<timestamp>_transcript.(txt|json).")
    parser.add_argument(
        "--save-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the transcription result to a file. Defaults to on.",
    )
    return parser.parse_args()


def _realtime_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/") + "/realtime"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _normalized_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    return np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_bytes(pcm16_bytes: bytes, sample_rate: int) -> bytes:
    channels = 1
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    riff_size = 36 + len(pcm16_bytes)
    header = bytearray()
    header.extend(b"RIFF")
    header.extend(struct.pack("<I", riff_size))
    header.extend(b"WAVEfmt ")
    header.extend(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits))
    header.extend(b"data")
    header.extend(struct.pack("<I", len(pcm16_bytes)))
    return bytes(header) + pcm16_bytes


def _mono_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    target_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
    source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _prepared_audio_from_file(audio_path: Path, sample_rate: int) -> PreparedAudio:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_bytes = audio_path.read_bytes()
    mime_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    audio, source_rate = sf.read(str(audio_path), dtype="float32")
    mono_audio = _mono_audio(audio)
    normalized = _resample_audio(mono_audio, int(source_rate), sample_rate)
    return PreparedAudio(
        source_stem=audio_path.stem,
        filename=audio_path.name,
        mime_type=mime_type,
        upload_bytes=file_bytes,
        pcm16_bytes=_pcm16_bytes(normalized),
        sample_rate=sample_rate,
    )


def _prepared_audio_from_microphone(sample_rate: int, record_seconds: float) -> PreparedAudio:
    if record_seconds <= 0:
        raise ValueError("record-seconds must be greater than zero")

    try:
        import sounddevice as sd
    except OSError as exc:
        raise RuntimeError("Microphone recording requires PortAudio to be installed on the host.") from exc

    frame_count = max(1, int(round(sample_rate * record_seconds)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Recording {record_seconds:.1f}s from microphone at {sample_rate} Hz...", file=sys.stderr)
    captured = sd.rec(frame_count, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    mono_audio = np.squeeze(captured, axis=1).astype(np.float32)
    pcm16_bytes = _pcm16_bytes(mono_audio)
    return PreparedAudio(
        source_stem=f"microphone_{timestamp}",
        filename=f"microphone_{timestamp}.wav",
        mime_type="audio/wav",
        upload_bytes=_wav_bytes(pcm16_bytes, sample_rate),
        pcm16_bytes=pcm16_bytes,
        sample_rate=sample_rate,
    )


def prepare_audio(args: argparse.Namespace) -> PreparedAudio:
    if args.microphone or not args.file:
        return _prepared_audio_from_microphone(args.sample_rate, args.record_seconds)
    return _prepared_audio_from_file(Path(args.file).expanduser().resolve(), args.sample_rate)


def resolve_asr_model(args: argparse.Namespace) -> str:
    return resolve_model(
        base_url=args.base_url,
        api_key=args.api_key,
        explicit_model=args.model,
        capability="transcription",
        fallback_model=ASR_MODEL_FALLBACK,
    )


def _http_form_data(args: argparse.Namespace) -> dict[str, str]:
    form_data = {
        "model": args.model,
        "response_format": args.response_format,
        "temperature": str(args.temperature),
    }
    language = _normalized_language(args.language)
    if language:
        form_data["language"] = language
    if args.prompt:
        form_data["prompt"] = args.prompt
    if args.word_timestamps:
        form_data["timestamp_granularities"] = "word"
    return form_data


def transcribe_http(args: argparse.Namespace, audio: PreparedAudio) -> str | dict:
    response = httpx.post(
        f"{args.base_url.rstrip('/')}/audio/transcriptions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        data=_http_form_data(args),
        files={"file": (audio.filename, audio.upload_bytes, audio.mime_type)},
        timeout=ASR_TIMEOUT,
    )
    response.raise_for_status()

    if args.response_format == "text":
        return response.text.strip()
    return response.json()


async def _recv_json(websocket) -> dict:
    payload = json.loads(await websocket.recv())
    if payload.get("type") == "error":
        detail = payload.get("error") or {}
        raise RuntimeError(detail.get("message") or "Realtime transcription error")
    return payload


async def _wait_for_event(websocket, event_types: set[str]) -> dict:
    while True:
        payload = await _recv_json(websocket)
        if payload.get("type") in event_types:
            return payload


def _iter_audio_chunks(audio_bytes: bytes, chunk_size: int):
    step = max(1, chunk_size)
    for start in range(0, len(audio_bytes), step):
        yield audio_bytes[start : start + step]


def _realtime_chunk_size(sample_rate: int, chunk_ms: int) -> int:
    samples = max(1, int(round(sample_rate * chunk_ms / 1000)))
    return samples * 2


def _timing_payload(session_started_at: float, *, input_started_at: float | None, input_ended_at: float | None, transcription_ended_at: float | None) -> dict[str, float | None]:
    def elapsed(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value - session_started_at, 3)

    return {
        "input_started_s": elapsed(input_started_at),
        "input_ended_s": elapsed(input_ended_at),
        "transcription_ended_s": elapsed(transcription_ended_at),
    }


def _log_timing(label: str, session_started_at: float) -> float:
    now = perf_counter()
    wall_clock = datetime.now().isoformat(timespec="milliseconds")
    print(f"{label}: {wall_clock} (+{now - session_started_at:.3f}s)", file=sys.stderr)
    return now


def _microphone_audio_name() -> tuple[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"microphone_{timestamp}"
    return stem, f"{stem}.wav"


async def _append_audio_chunk(websocket, audio_chunk: bytes) -> None:
    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_chunk).decode("utf-8"),
            }
        )
    )


async def transcribe_realtime_microphone(args: argparse.Namespace) -> tuple[str | dict, PreparedAudio]:
    try:
        import sounddevice as sd
    except OSError as exc:
        raise RuntimeError("Microphone recording requires PortAudio to be installed on the host.") from exc

    if args.vad_frame_ms <= 0:
        raise ValueError("vad-frame-ms must be greater than zero")
    if args.vad_threshold < 0:
        raise ValueError("vad-threshold must be zero or greater")
    if args.vad_max_seconds <= 0:
        raise ValueError("vad-max-seconds must be greater than zero")

    block_frames = max(1, int(round(args.sample_rate * args.vad_frame_ms / 1000)))
    start_frames = max(1, math.ceil(args.vad_start_ms / args.vad_frame_ms))
    end_frames = max(1, math.ceil(args.vad_end_ms / args.vad_frame_ms))
    preroll_frames = max(start_frames, math.ceil(max(0, args.vad_preroll_ms) / args.vad_frame_ms))
    max_blocks = max(1, math.ceil(args.vad_max_seconds * args.sample_rate / block_frames))
    vad = LocalVAD(threshold=args.vad_threshold, start_frames=start_frames, end_frames=end_frames)
    session_started_at = perf_counter()
    input_started_at: float | None = None
    input_ended_at: float | None = None
    transcription_ended_at: float | None = None
    preroll_buffer: deque[bytes] = deque(maxlen=max(1, preroll_frames) if preroll_frames else 1)
    captured_chunks: list[bytes] = []
    source_stem, filename = _microphone_audio_name()

    session = {
        "input_audio_transcription": {
            "model": args.model,
            "response_format": args.response_format,
            "timestamp_granularities": ["word"] if args.word_timestamps else [],
        },
        "input_audio_format": "pcm16",
        "input_audio_sample_rate": args.sample_rate,
    }
    language = _normalized_language(args.language)
    if language:
        session["input_audio_transcription"]["language"] = language
    if args.prompt:
        session["input_audio_transcription"]["prompt"] = args.prompt

    headers = {"Authorization": f"Bearer {args.api_key}"}
    transcript_event: dict = {}
    output_text = ""
    deltas: list[str] = []

    print(
        (
            f"Listening for speech with VAD threshold={args.vad_threshold:.4f}, frame={args.vad_frame_ms}ms, "
            f"start={args.vad_start_ms}ms, end={args.vad_end_ms}ms, preroll={args.vad_preroll_ms}ms"
        ),
        file=sys.stderr,
    )

    async with ws_connect(_realtime_url(args.base_url), additional_headers=headers, max_size=None) as websocket:
        await _wait_for_event(websocket, {"session.created"})
        await websocket.send(json.dumps({"type": "session.update", "session": session}))
        await _wait_for_event(websocket, {"session.updated"})

        with sd.InputStream(samplerate=args.sample_rate, channels=1, dtype="float32", blocksize=block_frames) as stream:
            for _ in range(max_blocks):
                audio_frame, overflowed = stream.read(block_frames)
                if overflowed:
                    print("warning: microphone input overflow detected", file=sys.stderr)
                mono_audio = np.squeeze(audio_frame, axis=1).astype(np.float32)
                pcm_chunk = _pcm16_bytes(mono_audio)

                if not vad.active:
                    preroll_buffer.append(pcm_chunk)

                _is_speech, started_now, ended_now, _rms = vad.process(mono_audio)
                if started_now:
                    input_started_at = _log_timing("input_started", session_started_at)
                    for buffered_chunk in preroll_buffer:
                        await _append_audio_chunk(websocket, buffered_chunk)
                        captured_chunks.append(buffered_chunk)
                    preroll_buffer.clear()
                elif vad.active:
                    await _append_audio_chunk(websocket, pcm_chunk)
                    captured_chunks.append(pcm_chunk)

                if ended_now:
                    input_ended_at = _log_timing("input_ended", session_started_at)
                    break
            else:
                if not vad.active:
                    raise RuntimeError("No speech detected before VAD timeout.")
                input_ended_at = _log_timing("input_ended", session_started_at)
                print("warning: forcing input end because vad-max-seconds was reached", file=sys.stderr)

        if not captured_chunks:
            raise RuntimeError("No speech audio was captured by VAD.")

        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await _wait_for_event(websocket, {"input_audio_buffer.committed"})
        await websocket.send(json.dumps({"type": "response.create", "response": {"modalities": ["text"]}}))

        while True:
            payload = await _recv_json(websocket)
            payload_type = payload.get("type")
            if payload_type == "conversation.item.input_audio_transcription.completed":
                transcript_event = payload
            elif payload_type == "response.output_text.delta":
                deltas.append(payload.get("delta", ""))
            elif payload_type == "response.output_text.done":
                output_text = payload.get("text", "")
            elif payload_type == "response.done":
                transcription_ended_at = _log_timing("transcription_ended", session_started_at)
                break

    pcm16_bytes = b"".join(captured_chunks)
    audio = PreparedAudio(
        source_stem=source_stem,
        filename=filename,
        mime_type="audio/wav",
        upload_bytes=_wav_bytes(pcm16_bytes, args.sample_rate),
        pcm16_bytes=pcm16_bytes,
        sample_rate=args.sample_rate,
    )

    text = (output_text or transcript_event.get("transcript") or "".join(deltas)).strip()
    timing = _timing_payload(
        session_started_at,
        input_started_at=input_started_at,
        input_ended_at=input_ended_at,
        transcription_ended_at=transcription_ended_at,
    )
    if args.response_format == "text":
        return text, audio

    payload = {
        "text": text,
        "language": transcript_event.get("language"),
        "duration": transcript_event.get("duration"),
        "segments": transcript_event.get("segments", []),
        "transport": "realtime",
        "timing": timing,
    }
    if args.response_format == "verbose_json":
        return payload, audio
    return {"text": text, "timing": timing}, audio


async def transcribe_realtime(args: argparse.Namespace, audio: PreparedAudio) -> str | dict:
    session = {
        "input_audio_transcription": {
            "model": args.model,
            "response_format": args.response_format,
            "timestamp_granularities": ["word"] if args.word_timestamps else [],
        },
        "input_audio_format": "pcm16",
        "input_audio_sample_rate": audio.sample_rate,
    }
    language = _normalized_language(args.language)
    if language:
        session["input_audio_transcription"]["language"] = language
    if args.prompt:
        session["input_audio_transcription"]["prompt"] = args.prompt

    headers = {"Authorization": f"Bearer {args.api_key}"}
    transcript_event: dict = {}
    output_text = ""
    deltas: list[str] = []

    async with ws_connect(_realtime_url(args.base_url), additional_headers=headers, max_size=None) as websocket:
        await _wait_for_event(websocket, {"session.created"})
        await websocket.send(json.dumps({"type": "session.update", "session": session}))
        await _wait_for_event(websocket, {"session.updated"})

        chunk_size = _realtime_chunk_size(audio.sample_rate, args.realtime_chunk_ms)
        for chunk in _iter_audio_chunks(audio.pcm16_bytes, chunk_size):
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )

        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await _wait_for_event(websocket, {"input_audio_buffer.committed"})
        await websocket.send(json.dumps({"type": "response.create", "response": {"modalities": ["text"]}}))

        while True:
            payload = await _recv_json(websocket)
            payload_type = payload.get("type")
            if payload_type == "conversation.item.input_audio_transcription.completed":
                transcript_event = payload
            elif payload_type == "response.output_text.delta":
                deltas.append(payload.get("delta", ""))
            elif payload_type == "response.output_text.done":
                output_text = payload.get("text", "")
            elif payload_type == "response.done":
                break

    text = (output_text or transcript_event.get("transcript") or "".join(deltas)).strip()
    if args.response_format == "text":
        return text

    payload = {
        "text": text,
        "language": transcript_event.get("language"),
        "duration": transcript_event.get("duration"),
        "segments": transcript_event.get("segments", []),
        "transport": "realtime",
    }
    if args.response_format == "verbose_json":
        return payload
    return {"text": text}


def _default_output_path(audio: PreparedAudio, response_format: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ".txt" if response_format == "text" else ".json"
    return DATA_DIR / f"{audio.source_stem}_{timestamp}_transcript{suffix}"


def save_output(payload: str | dict, args: argparse.Namespace, audio: PreparedAudio) -> Path | None:
    if not args.save_output:
        return None

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(audio, args.response_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, str):
        output_path.write_text(payload + "\n", encoding="utf-8")
        return output_path

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def print_payload(payload: str | dict) -> None:
    if isinstance(payload, str):
        print(payload)
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    try:
        args.model = resolve_asr_model(args)
        if args.transport == "realtime" and (args.microphone or not args.file):
            payload, audio = asyncio.run(transcribe_realtime_microphone(args))
        else:
            audio = prepare_audio(args)
            if args.transport == "realtime":
                payload = asyncio.run(transcribe_realtime(args, audio))
            else:
                payload = transcribe_http(args, audio)
        output_path = save_output(payload, args, audio)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"STT request failed: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        print(f"STT request failed: {exc.response.status_code} {detail}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"STT request failed: {exc}", file=sys.stderr)
        return 1

    print_payload(payload)
    if output_path is not None:
        print(f"saved: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
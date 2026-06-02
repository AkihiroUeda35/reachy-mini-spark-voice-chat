from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import threading
import time
from urllib import error, request
from dataclasses import dataclass
from pathlib import Path

import local_tts


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_DIR = APP_DIR / "profiles" / "default"
ROOT_DIR = APP_DIR.parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "conversation"
DEFAULT_SELECTED_PROFILE_FILE = APP_DIR / "profiles" / ".selected_profile"
COMMON_INSTRUCTIONS_FILE = APP_DIR / "profiles" / "common_instructions.txt"
DEFAULT_CHARACTER_FILE = DEFAULT_PROFILE_DIR / "character.txt"
DEFAULT_TOOLS_FILE = DEFAULT_PROFILE_DIR / "tools.txt"
DEFAULT_VOICE = "default"
CUSTOM_VOICE = "custom"
DEFAULT_TTS_INSTRUCTIONS = local_tts.TTS_INSTRUCTIONS
logger = logging.getLogger("conversation.state")
PROFILE_TTS_PROMPT_AUDIO_STEMS: dict[str, str] = {
    "tsukasa-speech": "japanese",
    "qwen3-tts": "english",
}
PROFILE_TTS_PROMPT_TEXT_STEMS: dict[str, str] = {
    "qwen3-tts": "english",
}
PROFILE_TTS_PROMPT_TEXT_LANGUAGE_HINTS: dict[str, str] = {
    "english": "en",
}
PROFILE_TTS_PROMPT_AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
FAMILY_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

VOICE_CHOICES: list[tuple[str, str]] = [
    ("audio_ref", "Bundled Tsukasa reference voice. [Tsukasa]"),
    ("kaede_san", "Gentle Tsukasa reference voice. [Tsukasa]"),
    ("shiki_fine05", "Refined Tsukasa reference voice. [Tsukasa]"),
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

QWEN_VOICE_CHOICES: dict[str, str] = {
    "aiden": "Aiden",
    "dylan": "Dylan",
    "eric": "Eric",
    "ono_anna": "Ono_Anna",
    "ryan": "Ryan",
    "serena": "Serena",
    "sohee": "Sohee",
    "uncle_fu": "Uncle_Fu",
    "vivian": "Vivian",
}
DEFAULT_QWEN_VOICE = "Ono_Anna"

TSUKASA_SPEECH_BASE_URL = os.environ.get("TSUKASA_SPEECH_BASE_URL", "http://localhost:5001").rstrip("/")


def normalize_tts_backend_name(backend: str | None) -> str:
    normalized = (backend or "").strip().lower()
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
    return aliases.get(normalized, normalized)


def effective_tts_transport(tts_transport: str, tts_backend: str | None, *, transport_explicit: bool = False) -> str:
    if transport_explicit:
        return tts_transport
    if normalize_tts_backend_name(tts_backend) == "tsukasa-speech":
        return "http"
    return tts_transport


def _fetch_tsukasa_voice_catalog(base_url: str | None = None, timeout_s: float = 1.0) -> tuple[list[str], str]:
    voice_base_url = (base_url or TSUKASA_SPEECH_BASE_URL).rstrip("/")
    if not voice_base_url:
        return [], ""
    try:
        with request.urlopen(f"{voice_base_url}/voices", timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, error.URLError):
        return [], ""
    voices = payload.get("voices") if isinstance(payload, dict) else None
    if not isinstance(voices, list):
        return [], ""
    default_voice = str(payload.get("default_voice") or "").strip() if isinstance(payload, dict) else ""
    normalized_voices = [str(voice).strip() for voice in voices if str(voice).strip()]
    return normalized_voices, default_voice


def _fetch_tsukasa_voice_names(base_url: str | None = None, timeout_s: float = 1.0) -> list[str]:
    voices, _default_voice = _fetch_tsukasa_voice_catalog(base_url=base_url, timeout_s=timeout_s)
    return voices


def normalize_tsukasa_voice(voice: str, base_url: str | None = None) -> str:
    requested_voice = voice.strip()
    if not requested_voice:
        return requested_voice

    voices, default_voice = _fetch_tsukasa_voice_catalog(base_url=base_url)
    if not voices:
        return requested_voice
    if requested_voice in voices:
        return requested_voice
    if default_voice and default_voice in voices:
        return default_voice
    return voices[0]


def is_custom_voice_choice(voice: str | None) -> bool:
    return (voice or "").strip().lower() == CUSTOM_VOICE


def _profile_has_prompt_audio(profile_dir: Path, backend: str) -> bool:
    stem = PROFILE_TTS_PROMPT_AUDIO_STEMS.get(backend)
    return bool(stem and _find_profile_prompt_audio_file(profile_dir, stem) is not None)


def profile_has_prompt_audio_by_name(profiles_dir: Path, profile: str, backend: str) -> bool:
    return _profile_has_prompt_audio(profiles_dir / profile, backend)


def _resolve_profile_tsukasa_voice_choice(profile_dir: Path, voice: str, fallback: str = DEFAULT_VOICE) -> str:
    resolved = voice.strip()
    if not resolved:
        return fallback
    if is_custom_voice_choice(resolved):
        return CUSTOM_VOICE if _profile_has_prompt_audio(profile_dir, "tsukasa-speech") else fallback
    return resolved


def _normalize_qwen_voice_choice(voice: str) -> str:
    raw = voice.strip()
    if is_custom_voice_choice(raw):
        return CUSTOM_VOICE
    return _canonical_qwen_voice(raw)


def _resolve_profile_qwen_voice_choice(profile_dir: Path, voice: str, fallback: str = DEFAULT_QWEN_VOICE) -> str:
    normalized = _normalize_qwen_voice_choice(voice)
    if normalized == CUSTOM_VOICE:
        return CUSTOM_VOICE if _profile_has_prompt_audio(profile_dir, "qwen3-tts") else fallback
    return normalized or fallback


def tsukasa_voice_choices(current_voice: str = "", base_url: str | None = None, *, include_custom: bool = False) -> list[tuple[str, str]]:
    live_voices = _fetch_tsukasa_voice_names(base_url=base_url)
    if live_voices:
        choices = [(voice, "Live Tsukasa voice from /voices. [Tsukasa]") for voice in live_voices]
    else:
        choices = [
            (name, description)
            for name, description in VOICE_CHOICES
            if "[Tsukasa]" in description
        ]

    if include_custom:
        choices.insert(0, (CUSTOM_VOICE, "Use profile japanese.wav when available, else fall back to the default Tsukasa voice. [Tsukasa]"))

    current_value = current_voice.strip()
    if current_value and all(name != current_value for name, _description in choices):
        choices.insert(0, (current_value, "Saved profile value not advertised by current Tsukasa service. [Tsukasa]"))
    return choices


def qwen_voice_choices(*, include_custom: bool = False) -> list[str]:
    choices = list(QWEN_VOICE_CHOICES.values())
    if include_custom:
        return [CUSTOM_VOICE, *choices]
    return choices


def _read_text_file(path: Path, fallback: str = "") -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def _audio_file_to_data_url(path: Path) -> str:
    audio_bytes = path.read_bytes()
    mime_type, _encoding = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "audio/wav"
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _image_file_to_data_url(path: Path) -> str:
    image_bytes = path.read_bytes()
    mime_type, _encoding = mimetypes.guess_type(path.name)
    if mime_type not in {"image/png", "image/jpeg"}:
        mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def load_family_image_references(family_dir: Path) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    if not family_dir.is_dir():
        return references

    for path in sorted(family_dir.iterdir(), key=lambda candidate: candidate.name.lower()):
        if not path.is_file() or path.suffix.lower() not in FAMILY_IMAGE_SUFFIXES:
            continue
        name = path.stem.strip()
        if not name:
            continue
        try:
            data_url = _image_file_to_data_url(path)
        except OSError as exc:
            logger.warning("Failed to load family image %s: %s", path, exc)
            continue
        references.append({"name": name, "image_url": data_url, "path": str(path)})
    return references


def _find_profile_prompt_audio_file(profile_dir: Path, stem: str) -> Path | None:
    for suffix in PROFILE_TTS_PROMPT_AUDIO_SUFFIXES:
        candidate = profile_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _transcribe_profile_prompt_audio(audio_path: Path, *, language: str | None = None) -> str:
    import whisper_asr

    whisper_asr.refresh_settings()
    args = argparse.Namespace(
        base_url=whisper_asr.ASR_BASE_URL,
        api_key=whisper_asr.ASR_API_KEY,
        model=whisper_asr.ASR_MODEL or whisper_asr.ASR_MODEL_FALLBACK,
        language=language or "auto",
        response_format="text",
        temperature=0.0,
        prompt=None,
        word_timestamps=False,
    )
    prepared_audio = whisper_asr._prepared_audio_from_file(audio_path, whisper_asr.ASR_SAMPLE_RATE)
    return str(whisper_asr.transcribe_http(args, prepared_audio)).strip()


def _ensure_profile_prompt_text_file(profile_dir: Path, stem: str) -> None:
    text_path = profile_dir / f"{stem}.txt"
    if text_path.is_file():
        return

    audio_path = _find_profile_prompt_audio_file(profile_dir, stem)
    if audio_path is None:
        return

    try:
        transcript = _transcribe_profile_prompt_audio(
            audio_path,
            language=PROFILE_TTS_PROMPT_TEXT_LANGUAGE_HINTS.get(stem),
        )
    except Exception as exc:
        logger.warning("Failed to auto-generate %s from %s: %s", text_path.name, audio_path.name, exc)
        return

    if transcript:
        text_path.write_text(transcript + "\n", encoding="utf-8")


def ensure_profile_tts_ref_text_by_name(profiles_dir: Path, profile: str) -> None:
    profile_dir = profiles_dir / profile
    for _backend, stem in PROFILE_TTS_PROMPT_TEXT_STEMS.items():
        _ensure_profile_prompt_text_file(profile_dir, stem)


COMMON_SYSTEM_PROMPT = _read_text_file(COMMON_INSTRUCTIONS_FILE)
DEFAULT_CHARACTER_PROMPT = _read_text_file(DEFAULT_CHARACTER_FILE)
DEFAULT_SYSTEM_PROMPT = "\n\n".join(part for part in (COMMON_SYSTEM_PROMPT, DEFAULT_CHARACTER_PROMPT) if part).strip()


@dataclass
class RuntimeSettings:
    profiles_dir: Path
    active_profile: str
    enabled_tools: list[str]
    active_character_prompt: str
    active_instructions: str
    active_voice: str
    active_qwen_voice: str
    active_tts_instructions: str
    gui_tool_names: list[str]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._pending_greeting_reason: str | None = None

    def snapshot(self) -> tuple[str, list[str], str, str, str, str, str, int]:
        with self._lock:
            return (
                self.active_profile,
                list(self.enabled_tools),
                self.active_character_prompt,
                self.active_instructions,
                self.active_voice,
                self.active_qwen_voice,
                self.active_tts_instructions,
                self._version,
            )

    def consume_pending_greeting_reason(self) -> str | None:
        with self._lock:
            reason = self._pending_greeting_reason
            self._pending_greeting_reason = None
            return reason

    def update(
        self,
        profile: str,
        enabled_tools: list[str],
        character_prompt: str,
        instructions: str,
        voice: str,
        qwen_voice: str,
        tts_instructions: str,
        *,
        greeting_reason: str | None = None,
    ) -> tuple[str, list[str], str, str, str, str, str, int]:
        normalized = [tool for tool in self.gui_tool_names if tool in enabled_tools]
        with self._lock:
            self.active_profile = profile
            self.enabled_tools = normalized
            self.active_character_prompt = character_prompt.strip()
            self.active_instructions = instructions.strip()
            profile_dir = self.profiles_dir / profile
            self.active_voice = _resolve_profile_tsukasa_voice_choice(profile_dir, voice, DEFAULT_VOICE)
            self.active_qwen_voice = _resolve_profile_qwen_voice_choice(profile_dir, qwen_voice, DEFAULT_QWEN_VOICE)
            self.active_tts_instructions = tts_instructions.strip()
            self._pending_greeting_reason = str(greeting_reason or "").strip() or None
            self._version += 1
            return (
                self.active_profile,
                list(self.enabled_tools),
                self.active_character_prompt,
                self.active_instructions,
                self.active_voice,
                self.active_qwen_voice,
                self.active_tts_instructions,
                self._version,
            )


class AssistantSpeechState:
    def __init__(self, tail_hold_s: float = 0.35) -> None:
        self._lock = threading.Lock()
        self._playback_end_s = 0.0
        self._tail_hold_s = max(0.0, tail_hold_s)

    def note_output_audio(self, *, sample_count: int, sample_rate: int) -> None:
        if sample_count <= 0 or sample_rate <= 0:
            return
        duration_s = sample_count / sample_rate
        now = time.monotonic()
        with self._lock:
            base = max(now, self._playback_end_s)
            self._playback_end_s = base + duration_s

    def is_speaking(self) -> bool:
        now = time.monotonic()
        with self._lock:
            deadline = self._playback_end_s + self._tail_hold_s
        return now < deadline


def listening_gate_settings(
    base_vad_threshold: float,
    base_vad_start_ms: int,
    *,
    assistant_speaking: bool,
    speaking_threshold_boost: float,
    speaking_vad_start_ms: int,
) -> tuple[float, int]:
    if not assistant_speaking:
        return base_vad_threshold, base_vad_start_ms
    return base_vad_threshold + max(0.0, speaking_threshold_boost), max(base_vad_start_ms, speaking_vad_start_ms)


def transcript_char_count(text: str) -> int:
    ignored_chars = {"、", "。", "！", "？", "!", "?", ".", ",", "，", "．", "…", "ー", "〜", "-"}
    return sum(1 for char in text.strip() if not char.isspace() and char not in ignored_chars)


def overlap_turn_rejection_reason(
    *,
    captured_duration_ms: float,
    transcript_text: str,
    min_duration_ms: int,
    min_chars: int,
) -> str | None:
    if captured_duration_ms < max(0, min_duration_ms):
        return f"vad {captured_duration_ms:.0f}ms < {min_duration_ms}ms"
    char_count = transcript_char_count(transcript_text)
    if char_count < max(0, min_chars):
        return f"chars {char_count} < {min_chars}"
    return None


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


def selected_profile_file(profiles_dir: Path) -> Path:
    return profiles_dir / DEFAULT_SELECTED_PROFILE_FILE.name


def load_selected_profile_name(profiles_dir: Path, fallback: str = "default") -> str:
    normalized_fallback = normalize_profile_name(fallback) or "default"
    stored_name = normalize_profile_name(_read_text_file(selected_profile_file(profiles_dir)))
    if not stored_name:
        return normalized_fallback
    if stored_name in list_profile_names(profiles_dir):
        return stored_name
    return normalized_fallback


def save_selected_profile_name(profiles_dir: Path, profile: str) -> Path:
    normalized_profile = normalize_profile_name(profile)
    if not normalized_profile:
        raise ValueError("profile must be a valid name")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    profile_file = selected_profile_file(profiles_dir)
    profile_file.write_text(normalized_profile + "\n", encoding="utf-8")
    return profile_file


def active_tools_for_profile(profiles_dir: Path, profile: str, gui_tool_names: list[str]) -> list[str]:
    return [tool for tool in parse_tools_file(profiles_dir / profile) if tool in gui_tool_names]


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


def _profile_voice_config_path(profile_dir: Path) -> Path:
    return profile_dir / "voice.json"


def _profile_qwen_voice_path(profile_dir: Path) -> Path:
    return profile_dir / "voice.txt"


def _canonical_qwen_voice(voice: str) -> str:
    return QWEN_VOICE_CHOICES.get(voice.strip().lower(), "")


def _load_profile_voice_config(profile_dir: Path) -> dict[str, str]:
    config_path = _profile_voice_config_path(profile_dir)
    legacy_qwen_voice = _normalize_qwen_voice_choice(_read_text_file(_profile_qwen_voice_path(profile_dir), ""))
    if config_path.is_file():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            voice = str(payload.get("voice") or "").strip()
            qwen_voice = _normalize_qwen_voice_choice(str(payload.get("qwen_voice") or "")) or legacy_qwen_voice
            tts_instructions = str(payload.get("tts_instructions") or "").strip()
            return {
                "voice": voice,
                "qwen_voice": qwen_voice,
                "tts_instructions": tts_instructions,
            }

    legacy_voice = _read_text_file(_profile_qwen_voice_path(profile_dir), "").strip()
    return {
        "voice": legacy_voice,
        "qwen_voice": _normalize_qwen_voice_choice(legacy_voice),
        "tts_instructions": "",
    }


def load_profile_voice_by_name(profiles_dir: Path, profile: str, fallback: str = DEFAULT_VOICE) -> str:
    profile_dir = profiles_dir / profile
    voice = _load_profile_voice_config(profile_dir).get("voice", "").strip()
    return _resolve_profile_tsukasa_voice_choice(profile_dir, voice, fallback)


def load_profile_tts_instructions_by_name(
    profiles_dir: Path,
    profile: str,
    fallback: str = DEFAULT_TTS_INSTRUCTIONS,
) -> str:
    profile_dir = profiles_dir / profile
    tts_instructions = _load_profile_voice_config(profile_dir).get("tts_instructions", "").strip()
    return tts_instructions or fallback


def load_profile_qwen_voice_by_name(profiles_dir: Path, profile: str, fallback: str = DEFAULT_QWEN_VOICE) -> str:
    profile_dir = profiles_dir / profile
    qwen_voice = _load_profile_voice_config(profile_dir).get("qwen_voice", "").strip()
    return _resolve_profile_qwen_voice_choice(profile_dir, qwen_voice, fallback)


def load_profile_tts_ref_audio_by_name(profiles_dir: Path, profile: str) -> str | dict[str, str] | None:
    profile_dir = profiles_dir / profile
    ref_audio: dict[str, str] = {}
    for backend, stem in PROFILE_TTS_PROMPT_AUDIO_STEMS.items():
        prompt_path = _find_profile_prompt_audio_file(profile_dir, stem)
        if prompt_path is not None:
            ref_audio[backend] = _audio_file_to_data_url(prompt_path)
    return ref_audio or None


def load_profile_tts_ref_text_by_name(profiles_dir: Path, profile: str) -> str | dict[str, str] | None:
    profile_dir = profiles_dir / profile
    ensure_profile_tts_ref_text_by_name(profiles_dir, profile)
    ref_text: dict[str, str] = {}
    for backend, stem in PROFILE_TTS_PROMPT_TEXT_STEMS.items():
        prompt_text_path = profile_dir / f"{stem}.txt"
        text = _read_text_file(prompt_text_path, "")
        if text:
            ref_text[backend] = text
    return ref_text or None


def build_tts_request_voice(voice: str, qwen_voice: str) -> str | dict[str, str]:
    primary_voice = DEFAULT_VOICE if is_custom_voice_choice(voice) else normalize_tsukasa_voice(voice)
    normalized_qwen_voice = DEFAULT_QWEN_VOICE if is_custom_voice_choice(qwen_voice) else _canonical_qwen_voice(qwen_voice)
    if primary_voice and normalized_qwen_voice:
        return {
            "tsukasa-speech": primary_voice,
            "qwen3-tts": normalized_qwen_voice,
        }
    if normalized_qwen_voice:
        return normalized_qwen_voice
    return primary_voice or DEFAULT_VOICE


def save_profile_definition(
    profiles_dir: Path,
    profile: str,
    character_prompt: str,
    selected_tools: list[str],
    voice: str,
    qwen_voice: str | None,
    tts_instructions: str,
    gui_tool_names: list[str],
) -> Path:
    profile_dir = profiles_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    existing_qwen_voice = _load_profile_voice_config(profile_dir).get("qwen_voice", "").strip()
    (profile_dir / "character.txt").write_text(character_prompt.strip() + "\n", encoding="utf-8")
    ordered_tools = [tool for tool in gui_tool_names if tool in selected_tools]
    (profile_dir / "tools.txt").write_text("\n".join(ordered_tools) + "\n", encoding="utf-8")
    normalized_qwen_voice = _normalize_qwen_voice_choice(qwen_voice or "") or _canonical_qwen_voice(voice) or _normalize_qwen_voice_choice(existing_qwen_voice) or DEFAULT_QWEN_VOICE
    _profile_voice_config_path(profile_dir).write_text(
        json.dumps(
            {
                "voice": voice.strip(),
                "qwen_voice": normalized_qwen_voice,
                "tts_instructions": tts_instructions.strip(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ensure_profile_tts_ref_text_by_name(profiles_dir, profile)
    return profile_dir


def load_profile_prompt(args: argparse.Namespace) -> str:
    profiles_dir = Path(args.profiles_dir).expanduser().resolve()
    return load_profile_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT)

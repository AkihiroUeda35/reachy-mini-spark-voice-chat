from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_DIR = APP_DIR / "profiles" / "default"
ROOT_DIR = APP_DIR.parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "conversation"
DEFAULT_SELECTED_PROFILE_FILE = APP_DIR / "profiles" / ".selected_profile"
COMMON_INSTRUCTIONS_FILE = APP_DIR / "profiles" / "common_instructions.txt"
DEFAULT_CHARACTER_FILE = DEFAULT_PROFILE_DIR / "character.txt"
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


@dataclass
class RuntimeSettings:
    profiles_dir: Path
    active_profile: str
    enabled_tools: list[str]
    active_character_prompt: str
    active_instructions: str
    active_voice: str
    gui_tool_names: list[str]

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

    def update(
        self,
        profile: str,
        enabled_tools: list[str],
        character_prompt: str,
        instructions: str,
        voice: str,
    ) -> tuple[str, list[str], str, str, str, int]:
        normalized = [tool for tool in self.gui_tool_names if tool in enabled_tools]
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
    gui_tool_names: list[str],
) -> Path:
    profile_dir = profiles_dir / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "character.txt").write_text(character_prompt.strip() + "\n", encoding="utf-8")
    ordered_tools = [tool for tool in gui_tool_names if tool in selected_tools]
    (profile_dir / "tools.txt").write_text("\n".join(ordered_tools) + "\n", encoding="utf-8")
    (profile_dir / "voice.txt").write_text(voice.strip() + "\n", encoding="utf-8")
    return profile_dir


def load_profile_prompt(args: argparse.Namespace) -> str:
    profiles_dir = Path(args.profiles_dir).expanduser().resolve()
    return load_profile_prompt_by_name(profiles_dir, args.profile, DEFAULT_CHARACTER_PROMPT)
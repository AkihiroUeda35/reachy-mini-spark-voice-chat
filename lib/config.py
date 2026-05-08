from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_ENV_LOADED = False


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[7:].lstrip()

    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def load_env(env_path: str | os.PathLike[str] | None = None, *, override: bool = False) -> Path:
    global _ENV_LOADED
    if _ENV_LOADED and env_path is None and not override:
        return project_root() / ".env"

    candidate = Path(env_path).expanduser().resolve() if env_path else project_root() / ".env"
    if not candidate.is_file():
        _ENV_LOADED = True
        return candidate

    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value

    _ENV_LOADED = True
    return candidate


def refresh_module_settings(*modules: Any) -> None:
    for module in modules:
        refresh = getattr(module, "refresh_settings", None)
        if callable(refresh):
            refresh()


def load_entrypoint_env(
    *modules: Any,
    env_path: str | os.PathLike[str] | None = None,
    override: bool = False,
) -> Path:
    env_file = load_env(env_path, override=override)
    refresh_module_settings(*modules)
    return env_file


def _default_service_urls() -> dict[str, str]:
    return {
        "stt": os.environ.get("STT_SERVER_URL", os.environ.get("ASR_BASE_URL", "http://localhost:8020/v1")).rstrip("/"),
        "llm": os.environ.get("LLM_SERVER_URL", os.environ.get("CHAT_BASE_URL", os.environ.get("CHAT_DEEPSEEK_BASE_URL", "http://localhost:8010/v1"))).rstrip("/"),
        "tts": os.environ.get("TTS_SERVER_URL", os.environ.get("TTS_BASE_URL", "http://localhost:8020/v1")).rstrip("/"),
    }


def service_urls() -> dict[str, str]:
    return _default_service_urls()


def service_url(name: str, default: str) -> str:
    urls = _default_service_urls()
    return urls.get(name, default).rstrip("/")
from __future__ import annotations

import os
from typing import Any

import httpx


MODELS_TIMEOUT = float(os.environ.get("OPENAI_MODELS_TIMEOUT", "10"))


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def fetch_models(base_url: str, api_key: str | None = None) -> list[dict[str, Any]]:
    response = httpx.get(_models_url(base_url), headers=_headers(api_key), timeout=MODELS_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError(f"Unexpected models payload from {base_url}: {payload!r}")
    return [item for item in models if isinstance(item, dict) and isinstance(item.get("id"), str)]


def _capabilities(model: dict[str, Any]) -> set[str]:
    metadata = model.get("metadata")
    if not isinstance(metadata, dict):
        return set()
    capabilities = metadata.get("capabilities")
    if not isinstance(capabilities, list):
        return set()
    return {str(capability).strip().lower() for capability in capabilities if str(capability).strip()}


def _looks_like_capability(model_id: str, capability: str) -> bool:
    normalized = model_id.strip().lower()
    if capability == "transcription":
        return "whisper" in normalized or "asr" in normalized or "transcribe" in normalized
    if capability == "speech":
        return "tts" in normalized or "voice" in normalized or "speech" in normalized
    if capability == "chat":
        return not _looks_like_capability(normalized, "transcription") and not _looks_like_capability(normalized, "speech")
    return False


def select_model(models: list[dict[str, Any]], *, capability: str | None = None, preferred: str | None = None) -> str:
    if preferred:
        for model in models:
            if model.get("id") == preferred:
                return preferred

    if capability:
        normalized_capability = capability.strip().lower()
        defaults = [
            model for model in models if normalized_capability in _capabilities(model) and bool((model.get("metadata") or {}).get("default"))
        ]
        if defaults:
            return str(defaults[0]["id"])

        tagged = [model for model in models if normalized_capability in _capabilities(model)]
        if tagged:
            return str(tagged[0]["id"])

        guessed = [model for model in models if _looks_like_capability(str(model.get("id")), normalized_capability)]
        if guessed:
            return str(guessed[0]["id"])

    if len(models) == 1:
        return str(models[0]["id"])

    raise ValueError(f"No model matched capability={capability!r} preferred={preferred!r}")


def resolve_model(
    *,
    base_url: str,
    api_key: str | None,
    explicit_model: str | None,
    capability: str | None,
    fallback_model: str | None,
) -> str:
    if explicit_model and explicit_model.strip():
        return explicit_model.strip()

    try:
        models = fetch_models(base_url, api_key)
        return select_model(models, capability=capability)
    except Exception:
        if fallback_model and fallback_model.strip():
            return fallback_model.strip()
        raise
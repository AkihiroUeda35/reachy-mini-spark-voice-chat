#!/bin/sh
set -eu

HOST=${TSUKASA_SPEECH_HOST:-0.0.0.0}
PORT=${TSUKASA_SPEECH_PORT:-5001}
REPO_ID=${TSUKASA_SPEECH_REPO_ID:-Respair/Tsukasa_Speech}
REPO_REF=${TSUKASA_SPEECH_REPO_REF:-main}
REPO_DIR=${TSUKASA_SPEECH_REPO_DIR:-/models/tsukasa_speech/repo}

mkdir -p "${REPO_DIR}"

python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_dir = Path(os.environ["TSUKASA_SPEECH_REPO_DIR"])
if not (repo_dir / "importable.py").exists():
    snapshot_download(
        repo_id=os.environ["TSUKASA_SPEECH_REPO_ID"],
        revision=os.environ.get("TSUKASA_SPEECH_REPO_REF", "main"),
        local_dir=str(repo_dir),
    )
PY

exec uvicorn api:app --app-dir /workspace/tts/tsukasa_speech --host "${HOST}" --port "${PORT}"
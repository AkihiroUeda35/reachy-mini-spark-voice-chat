#!/bin/sh
set -eu

export DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-TORCH_SDPA}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

python - <<'PY'
import os

from huggingface_hub import snapshot_download

model_id = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")

snapshot_download(
  repo_id=model_id,
  cache_dir=cache_dir,
  allow_patterns=[
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/preprocessor_config.json",
  ],
  resume_download=True,
)
PY

DEPLOY_CONFIG="${QWEN_TTS_DEPLOY_CONFIG:-}"
if [ -z "$DEPLOY_CONFIG" ]; then
  DEPLOY_CONFIG="$(python - <<'PY' 2>/dev/null | tail -n 1
import inspect
from pathlib import Path
import vllm_omni

print(Path(inspect.getfile(vllm_omni)).resolve().parent / "deploy" / "qwen3_tts.yaml")
PY
)"
fi

set -- \
  vllm-omni serve "${QWEN_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}" \
  --deploy-config "$DEPLOY_CONFIG" \
  --host "${QWEN_TTS_HOST:-0.0.0.0}" \
  --port "${QWEN_TTS_PORT:-8091}" \
  --gpu-memory-utilization "${QWEN_TTS_GPU_MEMORY_UTILIZATION:-0.1}" \
  --trust-remote-code \
  --omni

if [ "${QWEN_TTS_ENFORCE_EAGER:-1}" = "1" ]; then
  set -- "$@" --enforce-eager
fi

if [ "${QWEN_TTS_NO_ASYNC_CHUNK:-0}" = "1" ]; then
  set -- "$@" --no-async-chunk
fi

exec "$@"
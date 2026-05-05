#!/bin/sh
set -eu

set -- \
  vllm serve "${QWEN_LLM_MODEL:-sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP}" \
  --host "${QWEN_LLM_HOST:-0.0.0.0}" \
  --port "${QWEN_LLM_PORT:-8010}" \
  --trust-remote-code \
  --served-model-name "${QWEN_LLM_SERVED_MODEL_NAME:-spark}" \
  --quantization "${QWEN_LLM_QUANTIZATION:-modelopt}" \
  --max-model-len "${QWEN_LLM_MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${QWEN_LLM_MAX_NUM_SEQS:-4}" \
  --kv-cache-dtype "${QWEN_LLM_KV_CACHE_DTYPE:-fp8}" \
  --gpu-memory-utilization "${QWEN_LLM_GPU_MEMORY_UTILIZATION:-0.6}" \
  --reasoning-parser "${QWEN_LLM_REASONING_PARSER:-qwen3}" \
  --tool-call-parser "${QWEN_LLM_TOOL_CALL_PARSER:-qwen3_coder}"

if [ "${QWEN_LLM_ENABLE_AUTO_TOOL_CHOICE:-1}" = "1" ]; then
  set -- "$@" --enable-auto-tool-choice
fi

if [ -n "${QWEN_LLM_SPECULATIVE_CONFIG:-}" ]; then
  set -- "$@" --speculative-config "${QWEN_LLM_SPECULATIVE_CONFIG}"
fi

if [ -n "${QWEN_LLM_EXTRA_ARGS:-}" ]; then
  # Intentionally word-split so operators can append raw CLI flags.
  # shellcheck disable=SC2086
  set -- "$@" ${QWEN_LLM_EXTRA_ARGS}
fi

exec "$@"
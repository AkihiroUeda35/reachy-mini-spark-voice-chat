#!/bin/sh
set -eu

profile="${LLM_PROFILE:-qwen3.6-27b}"

profile_default() {
  key=$1

  case "${profile}:${key}" in
    qwen3.6-27b:model)
      printf '%s' 'sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP'
      ;;
    qwen3.6-27b:host)
      printf '%s' '0.0.0.0'
      ;;
    qwen3.6-27b:port)
      printf '%s' '8010'
      ;;
    qwen3.6-27b:served_model_name)
      printf '%s' 'spark'
      ;;
    qwen3.6-27b:quantization)
      printf '%s' 'modelopt'
      ;;
    qwen3.6-27b:max_model_len)
      printf '%s' '262144'
      ;;
    qwen3.6-27b:max_num_seqs)
      printf '%s' '4'
      ;;
    qwen3.6-27b:kv_cache_dtype)
      printf '%s' 'fp8'
      ;;
    qwen3.6-27b:gpu_memory_utilization)
      printf '%s' '0.6'
      ;;
    qwen3.6-27b:reasoning_parser)
      printf '%s' 'qwen3'
      ;;
    qwen3.6-27b:enable_auto_tool_choice)
      printf '%s' '1'
      ;;
    qwen3.6-27b:tool_call_parser)
      printf '%s' 'qwen3_coder'
      ;;
    qwen3.6-27b:speculative_config)
      printf '%s' '{"method":"qwen3_5_mtp","num_speculative_tokens":4,"moe_backend":"triton"}'
      ;;
    qwen3.6-27b:tensor_parallel_size)
      printf '%s' ''
      ;;
    qwen3.6-27b:enable_prefix_caching)
      printf '%s' '0'
      ;;
    qwen3.6-27b:generation_config)
      printf '%s' ''
      ;;
    qwen3.6-27b:extra_args)
      printf '%s' ''
      ;;
    gemma4-26b-a4b:model)
      printf '%s' 'bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4'
      ;;
    gemma4-26b-a4b:host)
      printf '%s' '0.0.0.0'
      ;;
    gemma4-26b-a4b:port)
      printf '%s' '8010'
      ;;
    gemma4-26b-a4b:served_model_name)
      printf '%s' 'spark'
      ;;
    gemma4-26b-a4b:quantization)
      printf '%s' 'modelopt'
      ;;
    gemma4-26b-a4b:max_model_len)
      printf '%s' '256000'
      ;;
    gemma4-26b-a4b:max_num_seqs)
      printf '%s' '2'
      ;;
    gemma4-26b-a4b:kv_cache_dtype)
      printf '%s' 'fp8'
      ;;
    gemma4-26b-a4b:gpu_memory_utilization)
      printf '%s' '0.65'
      ;;
    gemma4-26b-a4b:reasoning_parser)
      printf '%s' 'gemma4'
      ;;
    gemma4-26b-a4b:enable_auto_tool_choice)
      printf '%s' '1'
      ;;
    gemma4-26b-a4b:tool_call_parser)
      printf '%s' 'gemma4'
      ;;
    gemma4-26b-a4b:speculative_config)
      printf '%s' '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":3}'
      ;;
    gemma4-26b-a4b:tensor_parallel_size)
      printf '%s' '1'
      ;;
    gemma4-26b-a4b:enable_prefix_caching)
      printf '%s' '1'
      ;;
    gemma4-26b-a4b:generation_config)
      printf '%s' 'vllm'
      ;;
    gemma4-26b-a4b:extra_args)
      printf '%s' '--max-num-batched-tokens 4096'
      ;;
    gemma4-31b:model)
      printf '%s' 'nvidia/Gemma-4-31B-IT-NVFP4'
      ;;
    gemma4-31b:host)
      printf '%s' '0.0.0.0'
      ;;
    gemma4-31b:port)
      printf '%s' '8010'
      ;;
    gemma4-31b:served_model_name)
      printf '%s' 'spark'
      ;;
    gemma4-31b:quantization)
      printf '%s' 'modelopt'
      ;;
    gemma4-31b:max_model_len)
      printf '%s' '256000'
      ;;
    gemma4-31b:max_num_seqs)
      printf '%s' '1'
      ;;
    gemma4-31b:kv_cache_dtype)
      printf '%s' 'fp8'
      ;;
    gemma4-31b:gpu_memory_utilization)
      printf '%s' '0.65'
      ;;
    gemma4-31b:reasoning_parser)
      printf '%s' 'gemma4'
      ;;
    gemma4-31b:enable_auto_tool_choice)
      printf '%s' '1'
      ;;
    gemma4-31b:tool_call_parser)
      printf '%s' 'gemma4'
      ;;
    gemma4-31b:speculative_config)
      printf '%s' '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
      ;;
    gemma4-31b:tensor_parallel_size)
      printf '%s' '1'
      ;;
    gemma4-31b:enable_prefix_caching)
      printf '%s' '1'
      ;;
    gemma4-31b:generation_config)
      printf '%s' 'vllm'
      ;;
    gemma4-31b:extra_args)
      printf '%s' '--enforce-eager --max-num-batched-tokens 4096'
      ;;
    gemma4-e4b:model)
      printf '%s' 'bg-digitalservices/Gemma-4-E4B-it-NVFP4'
      ;;
    gemma4-e4b:host)
      printf '%s' '0.0.0.0'
      ;;
    gemma4-e4b:port)
      printf '%s' '8010'
      ;;
    gemma4-e4b:served_model_name)
      printf '%s' 'spark'
      ;;
    gemma4-e4b:quantization)
      printf '%s' 'modelopt'
      ;;
    gemma4-e4b:max_model_len)
      printf '%s' '256000'
      ;;
    gemma4-e4b:max_num_seqs)
      printf '%s' '8'
      ;;
    gemma4-e4b:kv_cache_dtype)
      printf '%s' 'fp8'
      ;;
    gemma4-e4b:gpu_memory_utilization)
      printf '%s' '0.5'
      ;;
    gemma4-e4b:reasoning_parser)
      printf '%s' 'gemma4'
      ;;
    gemma4-e4b:enable_auto_tool_choice)
      printf '%s' '1'
      ;;
    gemma4-e4b:tool_call_parser)
      printf '%s' 'gemma4'
      ;;
    gemma4-e4b:speculative_config)
      printf '%s' '{"method":"mtp","model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":3}'
      ;;
    gemma4-e4b:tensor_parallel_size)
      printf '%s' '1'
      ;;
    gemma4-e4b:enable_prefix_caching)
      printf '%s' '1'
      ;;
    gemma4-e4b:generation_config)
      printf '%s' 'vllm'
      ;;
    gemma4-e4b:extra_args)
      printf '%s' '--enforce-eager --max-num-batched-tokens 4096'
      ;;
    *)
      printf '%s' ''
      ;;
  esac
}

resolve_value() {
  name=$1
  key=$2

  eval "value=\${${name}:-}"
  if [ -n "$value" ]; then
    printf '%s' "$value"
    return
  fi

  profile_default "$key"
}

resolve_optional_value() {
  value=$(resolve_value "$1" "$2")
  if [ "$value" = "none" ]; then
    printf '%s' ''
    return
  fi

  printf '%s' "$value"
}

model=$(resolve_value LLM_MODEL model)
host=$(resolve_value LLM_HOST host)
port=$(resolve_value LLM_PORT port)
served_model_name=$(resolve_value LLM_SERVED_MODEL_NAME served_model_name)
quantization=$(resolve_value LLM_QUANTIZATION quantization)
max_model_len=$(resolve_value LLM_MAX_MODEL_LEN max_model_len)
max_num_seqs=$(resolve_value LLM_MAX_NUM_SEQS max_num_seqs)
kv_cache_dtype=$(resolve_value LLM_KV_CACHE_DTYPE kv_cache_dtype)
gpu_memory_utilization=$(resolve_value LLM_GPU_MEMORY_UTILIZATION gpu_memory_utilization)
reasoning_parser=$(resolve_value LLM_REASONING_PARSER reasoning_parser)
enable_auto_tool_choice=$(resolve_value LLM_ENABLE_AUTO_TOOL_CHOICE enable_auto_tool_choice)
tool_call_parser=$(resolve_value LLM_TOOL_CALL_PARSER tool_call_parser)
speculative_config=$(resolve_optional_value LLM_SPECULATIVE_CONFIG speculative_config)
tensor_parallel_size=$(resolve_optional_value LLM_TENSOR_PARALLEL_SIZE tensor_parallel_size)
enable_prefix_caching=$(resolve_value LLM_ENABLE_PREFIX_CACHING enable_prefix_caching)
generation_config=$(resolve_optional_value LLM_GENERATION_CONFIG generation_config)
extra_args=$(resolve_optional_value LLM_EXTRA_ARGS extra_args)

if [ -z "$model" ]; then
  echo "No LLM model configured. Set LLM_MODEL or use a supported LLM_PROFILE." >&2
  exit 1
fi

set -- \
  vllm serve "$model" \
  --host "$host" \
  --port "$port" \
  --trust-remote-code \
  --served-model-name "$served_model_name" \
  --quantization "$quantization" \
  --max-model-len "$max_model_len" \
  --max-num-seqs "$max_num_seqs" \
  --kv-cache-dtype "$kv_cache_dtype" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --reasoning-parser "$reasoning_parser" \
  --tool-call-parser "$tool_call_parser"

if [ -n "$tensor_parallel_size" ]; then
  set -- "$@" --tensor-parallel-size "$tensor_parallel_size"
fi

if [ "$enable_auto_tool_choice" = "1" ]; then
  set -- "$@" --enable-auto-tool-choice
fi

if [ "$enable_prefix_caching" = "1" ]; then
  set -- "$@" --enable-prefix-caching
fi

if [ -n "$generation_config" ]; then
  set -- "$@" --generation-config "$generation_config"
fi

if [ -n "$speculative_config" ]; then
  set -- "$@" --speculative-config "$speculative_config"
fi

if [ -n "$extra_args" ]; then
  # Intentionally word-split so operators can append raw CLI flags.
  # shellcheck disable=SC2086
  set -- "$@" ${extra_args}
fi

exec "$@"
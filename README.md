# spark-voice-chat

Local voice stack for DGX Spark built around three services managed from the repository root:

- Qwen LLM on vLLM at `http://localhost:8010/v1`
- Qwen3-TTS backend at `http://localhost:8091`
- OpenAI-compatible STT/TTS wrapper at `http://localhost:8020/v1`

The stack is started from the top-level [docker-compose.yml](docker-compose.yml) and built from the shared [Dockerfile](Dockerfile).

## Layout

- [Dockerfile](Dockerfile): shared multi-stage build for `llm-runtime`, `tts-runtime`, and `stt-runtime`
- [llm](llm): LLM runtime entrypoint and configuration wrapper
- [tts](tts): Qwen3-TTS runtime entrypoint, models, and voices
- [stt](stt): FastAPI wrapper, STT app code, and STT model cache
- [lib](lib): reusable STT/LLM/TTS client modules
- [samples](samples): local smoke-test and sample entry points built on top of `lib`
- [apps](apps): planned application entry points such as a Reachy Mini app
- [spark-vllm-docker](spark-vllm-docker): upstream helper repo kept for wheels and Spark-specific vLLM assets

## Services

### `vllm`

Runs the LLM server from the `llm-runtime` target.

- Default model: `sakamakismile/Huihui-Qwen3.6-27B-abliterated-NVFP4-MTP`
- Default served model name: `spark`
- Port: `8010`
- GPU: required

### `qwen3-tts`

Runs Qwen3-TTS through `vllm-omni` from the `tts-runtime` target.

- Default model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- Port: `8091`
- GPU: required

### `voice-server`

Runs the OpenAI-compatible wrapper from the `stt-runtime` target.

- Port: `8020`
- STT backend: Faster Whisper `RoachLin/kotoba-whisper-v2.2-faster` on GPU
- STT warmup: enabled on startup by default
- TTS upstream: `http://qwen3-tts:8091`

## Requirements

- Docker with Compose support
- NVIDIA Container Toolkit / GPU-enabled Docker runtime
- `uv` for local Python tooling
- Access to `nvcr.io/nvidia/vllm:26.04-py3`

If the NVIDIA image is not already accessible on the machine, log in first.

```bash
docker login nvcr.io
```

## Start

Build and start the full stack from the repository root.

```bash
docker compose up -d --build
```

Check service state.

```bash
docker compose ps
```

Follow startup logs when needed.

```bash
docker compose logs -f vllm
docker compose logs -f qwen3-tts
docker compose logs -f voice-server
```

## Health Checks

LLM:

```bash
curl http://localhost:8010/v1/models
```

TTS backend:

```bash
curl http://localhost:8091/health
```

Wrapper health:

```bash
curl http://localhost:8020/health
```

## Local Sample

The sample script in [samples/langchain_openai_tts.py](samples/langchain_openai_tts.py) generates a short Japanese script with the local LLM, sends it to the local TTS wrapper, and writes a WAV file. Shared client code lives in [lib](lib), so future apps can reuse the same STT/LLM/TTS access layer directly.

Install the local environment first. `uv sync` now installs this repository as an editable package, so `lib` can be imported without extra path hacks.

```bash
uv sync
```

If you already have the environment and only want to refresh the editable install, this is equivalent.

```bash
uv pip install -e .
```

If you want local script defaults from a file, create a root `.env`. That file is loaded by script entry points such as `samples/*.py`; `lib` itself does not auto-load it.

The Reachy conversation app under [apps/conversation/main.py](/home/aki/server/apps/conversation/main.py#L44) is one of those entry points, so putting `TAVILY_API_KEY=...` in the repository-root `.env` enables the `web_search` tool to use Tavily automatically. If `TAVILY_API_KEY` is unset, the tool falls back to DuckDuckGo.

```bash
cp .env.example .env
```

Install local dependencies and run it from the repository root.

```bash
uv run python samples/langchain_openai_tts.py
```

Default output:

- [data/spark_voice_chat.wav](data/spark_voice_chat.wav)

## Main Runtime Data

- LLM/HF cache: `~/.cache/huggingface`, `~/.cache/vllm`, `~/.cache/flashinfer`, `~/.triton`
- TTS model cache: [tts/models](tts/models)
- STT model cache: [stt/models](stt/models)
- Voice definitions: [tts/voices/voices.json](tts/voices/voices.json)

## Important Environment Variables

LLM:

- `QWEN_LLM_MODEL`
- `QWEN_LLM_SERVED_MODEL_NAME`
- `QWEN_LLM_GPU_MEMORY_UTILIZATION`
- `QWEN_LLM_MAX_MODEL_LEN`
- `QWEN_LLM_SPECULATIVE_CONFIG`

TTS:

- `QWEN_TTS_MODEL`
- `QWEN_TTS_GPU_MEMORY_UTILIZATION`
- `QWEN_TTS_ENFORCE_EAGER`
- `QWEN_TTS_NO_ASYNC_CHUNK`

Wrapper:

- `STT_MODEL_ID`
- `STT_MODEL_SIZE`
- `STT_DEVICE`
- `STT_COMPUTE_TYPE`
- `STT_CPU_THREADS`
- `STT_NUM_WORKERS`
- `STT_DEFAULT_LANGUAGE`
- `STT_BEAM_SIZE`
- `STT_BEST_OF`
- `STT_CONDITION_ON_PREVIOUS_TEXT`
- `STT_WARMUP_ENABLED`
- `TTS_DEFAULT_TASK_TYPE`
- `TTS_DEFAULT_LANGUAGE`
- `TTS_DEFAULT_VOICE`
- `TTS_PUBLIC_MODEL_NAME`

Script entry point layer:

- `LLM_SERVER_URL`
- `STT_SERVER_URL`
- `TTS_SERVER_URL`
- `CHAT_API_KEY`
- `ASR_API_KEY`
- `TTS_API_KEY`
- `ASR_LANGUAGE`
- `ASR_TRANSPORT`
- `TTS_USE_REALTIME_STREAMING`

## Notes

- The shared top-level Dockerfile keeps the build flow unified, but `llm-runtime` and `tts-runtime` remain separate targets because their runtime dependencies differ.
- The LLM runtime still consumes wheels from [spark-vllm-docker/wheels](spark-vllm-docker/wheels) to match the Spark-tested vLLM stack.
- The top-level Dockerfile copies wheel files from [spark-vllm-docker/wheels](spark-vllm-docker/wheels) by wildcard, so refreshed wheel versions from `spark-vllm-docker/build-and-copy.sh` do not require filename updates in [Dockerfile](Dockerfile).
- First startup can take a long time because both the LLM and TTS models may need to download and initialize.
- Local client samples auto-discover the current STT, chat, and TTS model IDs from each service's `/v1/models` endpoint unless you override them explicitly with environment variables or CLI flags.
- `.env` loading is intentionally limited to script entry points. Reusable modules under [lib](lib) read only the process environment they are given.
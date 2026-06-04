# reachy-mini-spark-voice-chat

Local voice chat stack for DGX Spark and Reachy Mini chat application.

The stack is started from the top-level [docker-compose.yml](docker-compose.yml) and built from the shared [Dockerfile](Dockerfile).

## Start Servers

Build DGX spark community vLLM wheels from the helper repo and copy them to the expected location in this repository.

```bash
cd  spark-vllm-docker
./build-and-copy.sh
cd ..
```

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

## Run Reachy Mini Application

Run Reachy Mini Application.
After starting the stack, run the Reachy Mini application that connects to the services.

```bash
uv run python apps/conversation/main.py
```

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

### `tsukasa-speech`

Runs a dedicated wrapper around [Respair/Tsukasa_Speech](https://huggingface.co/Respair/Tsukasa_Speech) for Japanese synthesis.

- Default port: `5001`
- Model repo: `Respair/Tsukasa_Speech`
- Default sample rate: `24000`
- Voice source: bundled `reference_sample_wavs` from the downloaded repo
- Supports both voice-guided synthesis and prompt-guided synthesis when `instructions` are provided to `/v1/audio/speech`

## Requirements

- DGX spark
- Docker with Compose support
- `uv` for local Python tooling

If the NVIDIA image is not already accessible on the machine, log in first.

```bash
docker login nvcr.io
```


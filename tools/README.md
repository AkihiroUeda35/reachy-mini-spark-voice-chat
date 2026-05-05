# spark-voice-chat

Small local sample that generates a short Japanese script with a Qwen model served by vLLM on `http://localhost:8010/v1`, then sends it to the local Qwen3-TTS wrapper on `http://localhost:8020/v1/audio/speech` and stores the resulting WAV in `./data/`.

## Requirements

- vLLM exposing an OpenAI-compatible chat endpoint on `http://localhost:8010/v1`
- TTS server exposing `http://localhost:8020/v1/audio/speech`
- `uv` for environment management

## Run

```bash
cd ..
uv sync
uv run python tools/langchain_openai_tts.py
```

## Notes

- Default chat model: `spark`
- Default local chat token: `token-abc`
- Thinking is disabled for Qwen requests via `chat_template_kwargs.enable_thinking=false`
- Default task type: `CustomVoice`
- Default voice: `Ono_Anna`
- Default language: `Japanese`
- Output WAV path: `./data/spark_voice_chat.wav`
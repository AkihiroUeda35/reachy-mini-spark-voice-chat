# spark-voice-chat

Top-level workspace for local Qwen + Qwen3-TTS experiments.

## Layout

- `spark-vllm-docker/`: upstream vLLM helper repo, intended to stay as a submodule
- `tts-stt-server/`: local OpenAI-compatible STT/TTS wrapper and compose setup
- `tools/`: local smoke-test and sample scripts


## Sample

The local sample script is in `tools/langchain_openai_tts.py`.
It uses:

- Qwen via local vLLM on `http://localhost:8010/v1`
- `ChatDeepSeek` with `chat_template_kwargs.enable_thinking=false`
- local TTS on `http://localhost:8020/v1/audio/speech`
- default Japanese speaker `Ono_Anna`
- `task_type=CustomVoice`, `language=Japanese`
- WAV output written to `tools/data/spark_voice_chat.wav`
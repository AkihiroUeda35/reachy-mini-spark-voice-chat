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

## Whisper ASR Test

Use the local OpenAI-compatible STT endpoint to transcribe either a microphone capture or a file. The script uses the Realtime API by default, saves the transcript result to `./data/` by default, and can fall back to plain HTTP multipart uploads when needed.

The tool defaults to `--language ja` so Whisper can skip language autodetection for faster Japanese transcription. Pass `--language auto` if you want autodetection back.

```bash
cd ..
uv sync
uv run python tools/whisper_asr_test.py
```

The default invocation listens on the microphone, starts sending audio to `/v1/realtime` when local VAD detects speech, commits the input when silence crosses the VAD end threshold, prints the transcript, and writes a transcript file under `./data/`. It also prints `input_started`, `input_ended`, and `transcription_ended` timings to stderr.

```bash
uv run python tools/whisper_asr_test.py
```

You can tune the local VAD when needed.

```bash
uv run python tools/whisper_asr_test.py --vad-threshold 0.02 --vad-start-ms 120 --vad-end-ms 700 --vad-preroll-ms 200
```

You can also transcribe a file, switch back to HTTP upload mode, or change the save path.

```bash
uv run python tools/whisper_asr_test.py data/your_audio.wav
uv run python tools/whisper_asr_test.py data/your_audio.wav --transport http --response-format verbose_json --word-timestamps
uv run python tools/whisper_asr_test.py --output data/custom_transcript.txt
```

## Pipecat Voice Chain

Use the local STT wrapper, the local OpenAI-compatible chat endpoint, and the local TTS wrapper in one script. The script accepts microphone input by default, can take an audio file instead, and can either play the generated reply through the default sound device or write it to a WAV file.
In `sound` mode, the assistant starts speaking sentence by sentence while the LLM is still generating later tokens.

```bash
cd ..
uv sync
uv run python tools/pipecat_asr_llm_tts.py
```

Use a file as input and save the assistant audio to disk.

```bash
uv run python tools/pipecat_asr_llm_tts.py data/your_audio.wav --output-mode file --output-audio data/reply.wav
```

The script reuses the same STT parameters as `whisper_asr_test.py`, so options such as `--transport`, `--language`, `--vad-threshold`, and `--response-format` behave the same way.

## Notes

- Default chat model: `spark`
- Default local chat token: `token-abc`
- Thinking is disabled for Qwen requests via `chat_template_kwargs.enable_thinking=false`
- Default task type: `CustomVoice`
- Default voice: `Ono_Anna`
- Default language: `Japanese`
- Output WAV path: `./data/spark_voice_chat.wav`
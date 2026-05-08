# Reachy Mini Conversation Clone

This app is a local clone of the official Reachy Mini conversation app adapted to this repository.

It keeps the Reachy-side loop local, but all inference goes through the already running local servers:

- ASR: the OpenAI-compatible STT endpoint
- LLM: the OpenAI-compatible chat endpoint
- TTS: the OpenAI-compatible speech endpoint

You can also launch a simplified Gradio GUI for runtime selection of:

- character profile
- enabled tools
- editable profile instructions

The orchestration path is:

1. Capture audio from Reachy Mini's media pipeline.
2. Run VAD locally and send the utterance to the configured ASR server.
3. Run the LLM + tool-calling loop through LangChain.
4. Stream TTS audio through Pipecat and play it back on Reachy Mini.

## Run

Install the repo as an editable package first so the shared STT/LLM/TTS modules are available as top-level imports.

```bash
uv sync
# or: uv pip install -e .
```

The conversation app itself is still a repository-local entry point. It is run directly from [apps/conversation](apps/conversation), not from the installed package.

```bash
uv run python apps/conversation/main.py
uv run python apps/conversation/main.py --no-gradio
```

Useful flags:

```bash
uv run python apps/conversation/main.py --debug --save-transcripts --save-replies
uv run python apps/conversation/main.py --robot-name reachy-mini-01
uv run python apps/conversation/main.py --profile default
```

## Tooling

The default profile mirrors the official tool surface:

- `dance`
- `stop_dance`
- `play_emotion`
- `stop_emotion`
- `camera`
- `idle_do_nothing`
- `head_tracking`
- `move_head`
- `get_jma_weather_tool`
- `get_current_time_tool`

System tools are also registered:

- `task_status`
- `task_cancel`

Current behavior notes:

- `camera` saves a snapshot, runs local OpenCV face detection, and asks the configured LLM to answer from the structured vision observation.
- `head_tracking` runs a local OpenCV face-tracking loop and drives Reachy's `look_at_image` API.
- `dance` now depends on `reachy-mini-dances-library`, which is included in the project dependencies and can also be auto-installed at runtime if missing.
- Gradio is on by default; use `--no-gradio` when you want the console loop only.
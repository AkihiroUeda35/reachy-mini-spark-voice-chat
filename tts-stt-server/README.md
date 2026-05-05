# OpenAI-compatible TTS/STT server

Qwen3-TTS を `vLLM-Omni` で配信し、その前段に FastAPI wrapper を置く構成です。外向きには OpenAI API 互換の `/v1/audio/*` と簡易 RealTime API 互換の `/v1/realtime` を提供します。

- OpenAI 互換 API: `http://localhost:8020/v1`
- TTS 実体: Qwen3-TTS via `vLLM-Omni` `http://localhost:8091/v1/audio/speech`
- STT 実体: Faster Whisper on CPU

## 構成

- `voice-server`: OpenAI 互換 FastAPI
- `qwen3-tts`: NVIDIA NGC `vllm` image をベースに `vLLM-Omni` を追加した TTS sidecar

`voice-server` は upstream の `/v1/audio/speech` を proxy しつつ、STT と簡易 `/v1/realtime` をまとめて提供します。

## 事前準備

`voices/voices.json` は `Base` タスクで参照音声を使いたい場合だけ使います。`CustomVoice` では不要です。

NGC image を pull できるよう、必要なら先に `docker login nvcr.io` を済ませてください。

参考音声を使う場合はサンプルをコピーして編集します。

```bash
cd /home/aki/server/tts-stt-server
cp voices/voices.example.json voices/voices.json
```

最低限、各 voice に次の 2 つを設定してください。

- `ref_audio`: 参照音声ファイルへのパス
- `ref_text`: その参照音声の正確な読み上げテキスト

既定では sidecar が `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` を起動します。speaker は `Ono_Anna`、language は `Japanese` を既定にしています。

## 起動

デフォルトは `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` を GPU で起動します。

```bash
cd /home/aki/server/tts-stt-server
docker compose up --build
```

VoiceDesign や Base に変えたい場合の例:

```bash
cd /home/aki/server/tts-stt-server
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign TTS_DEFAULT_TASK_TYPE=VoiceDesign docker compose up --build
```

## 動作確認

ヘルスチェック:

```bash
curl -s http://localhost:8020/health | jq
curl -s http://localhost:8091/health
```

HTTP TTS:

```bash
curl -X POST http://localhost:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","task_type":"CustomVoice","language":"Japanese","input":"こんにちは。Qwen3-TTS 経由の音声です。","voice":"Ono_Anna","response_format":"wav","stream":false}' \
  --output speech.wav
```

PCM streaming:

```bash
curl -X POST http://localhost:8020/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","task_type":"CustomVoice","language":"Japanese","input":"こんにちは。会話用のストリーミング音声です。","voice":"Ono_Anna","response_format":"pcm","stream":true}' \
  --output speech.pcm
```

STT:

```bash
curl -s -X POST http://localhost:8020/v1/audio/transcriptions \
  -F "file=@speech.wav" \
  -F "model=whisper-1" \
  -F "language=ja"
```

## RealTime API 互換

`/v1/realtime` は WebSocket で以下のイベントを扱います。

- `session.update`
- `conversation.item.create`
- `response.create`

音声は `response.audio.delta` で base64 PCM16 として返します。完全な OpenAI RealTime 実装ではなく、TTS 用に必要な最小互換です。

## OpenAI SDK 例

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key="local")

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input="こんにちは。",
    response_format="pcm",
) as response:
    for chunk in response.iter_bytes():
        pass
```

## 主な環境変数

- `QWEN_TTS_MODEL`: 起動する Qwen3-TTS モデル ID
- `QWEN_TTS_NGC_VLLM_IMAGE`: ベースに使う NGC vLLM image。既定値は `nvcr.io/nvidia/vllm:26.04-py3`
- `QWEN_TTS_VLLM_VERSION`: image 内で上書きインストールする `vllm` version。既定値は `0.20.0`
- `QWEN_TTS_VLLM_OMNI_REF`: install する `vllm-omni` の git ref。既定値は `main`
- `TTS_UPSTREAM_BASE_URL`: wrapper が最初に使う upstream
- `TTS_DEFAULT_TASK_TYPE`: wrapper の既定 task type。既定値は `CustomVoice`
- `TTS_DEFAULT_LANGUAGE`: wrapper の既定 language。既定値は `Japanese`
- `TTS_DEFAULT_VOICE`: wrapper の既定 speaker。既定値は `Ono_Anna`
- `QWEN_TTS_VOICES_FILE`: `Base` タスク向け参照音声定義ファイル

## 注意

- NGC `26.04` には `vllm 0.19.0` が入っているため、この構成では image 内の `vllm` を `0.20.0` に上書きして `vllm-omni` と version を揃えています。
- `voice-server` の RealTime API は TTS 互換用途の最小実装です。音声入力や双方向会話の完全互換までは含みません。
- `Base` タスクで参照音声を使う場合は、`ref_audio` と `ref_text` の整合が品質に直結します。
- 初回起動では model download と image build に時間がかかります。

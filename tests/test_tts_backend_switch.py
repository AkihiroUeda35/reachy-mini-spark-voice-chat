from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def _load_voice_server_module():
    module_name = "test_voice_server_main"
    module_path = Path(__file__).resolve().parents[1] / "stt" / "app" / "main.py"
    fake_faster_whisper = types.ModuleType("faster_whisper")
    setattr(fake_faster_whisper, "WhisperModel", object)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for stt/app/main.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.modules.setdefault("faster_whisper", fake_faster_whisper)
    spec.loader.exec_module(module)
    return module


VOICE_SERVER = _load_voice_server_module()


class TTSBackendSwitchTests(unittest.TestCase):
    def test_backend_switch_resolves_tsukasa_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TTS_BACKEND": "tsukasa",
                "TSUKASA_SPEECH_BASE_URL": "http://tsukasa-speech:5001",
                "TSUKASA_SPEECH_MODEL": "Respair/Tsukasa_Speech",
                "TSUKASA_SPEECH_PUBLIC_MODEL_NAME": "tsukasa-speech",
            },
            clear=False,
        ):
            self.assertEqual(VOICE_SERVER._tts_backend(), "tsukasa-speech")
            self.assertEqual(VOICE_SERVER._tts_upstream_base_url(), "http://tsukasa-speech:5001")
            self.assertEqual(VOICE_SERVER._tts_upstream_model_name(), "Respair/Tsukasa_Speech")
            self.assertEqual(VOICE_SERVER._tts_public_model_name(), "tsukasa-speech")

    def test_tsukasa_payload_uses_voice_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_path = Path(temp_dir) / "voices.json"
            voices_path.write_text(
                json.dumps(
                    {
                        "tsukasa": {
                            "voice": "audio_ref",
                            "diffusion_steps": 7,
                            "embedding_scale": 1.5,
                            "alpha": 0.2,
                            "beta": 0.4,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "TTS_BACKEND": "tsukasa-speech",
                    "QWEN_TTS_VOICES_FILE": str(voices_path),
                },
                clear=False,
            ):
                request = VOICE_SERVER.SpeechRequest(
                    model="tts-1",
                    input="こんにちは",
                    voice="tsukasa",
                    language="Japanese",
                    speed=1.25,
                    stream=False,
                )
                payload = VOICE_SERVER._tsukasa_payload(request)

            self.assertEqual(payload["voice"], "audio_ref")
            self.assertEqual(payload["diffusion_steps"], 7)
            self.assertAlmostEqual(payload["embedding_scale"], 1.5)
            self.assertAlmostEqual(payload["alpha"], 0.2)
            self.assertAlmostEqual(payload["beta"], 0.4)
            self.assertAlmostEqual(payload["speed"], 1.25)

    def test_tsukasa_payload_uses_backend_specific_voice_from_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            voices_path = Path(temp_dir) / "voices.json"
            voices_path.write_text(
                json.dumps(
                    {
                        "captain": {
                            "voice": "audio_ref",
                            "diffusion_steps": 7,
                            "embedding_scale": 1.5,
                            "alpha": 0.2,
                            "beta": 0.4,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "TTS_BACKEND": "tsukasa-speech",
                    "QWEN_TTS_VOICES_FILE": str(voices_path),
                },
                clear=False,
            ):
                request = VOICE_SERVER.SpeechRequest(
                    model="tts-1",
                    input="こんにちは",
                    voice={"tsukasa-speech": "captain", "qwen3-tts": "Ryan"},
                    language="Japanese",
                    stream=False,
                )
                payload = VOICE_SERVER._tsukasa_payload(request)

            self.assertEqual(payload["voice"], "audio_ref")
            self.assertEqual(payload["diffusion_steps"], 7)

    def test_tsukasa_payload_forwards_prompt_instructions(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TTS_BACKEND": "tsukasa-speech",
            },
            clear=False,
        ):
            request = VOICE_SERVER.SpeechRequest(
                model="tts-1",
                input="テストです",
                voice="default",
                instructions="落ち着いて、でも少し楽しげに話してください。",
                language="Japanese",
                stream=False,
            )
            payload = VOICE_SERVER._tsukasa_payload(request)

        self.assertEqual(payload["instructions"], "落ち着いて、でも少し楽しげに話してください。")

    def test_tsukasa_payload_falls_back_to_default_voice(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TTS_BACKEND": "tsukasa-speech",
                "TSUKASA_SPEECH_DEFAULT_VOICE": "audio_ref",
            },
            clear=False,
        ):
            request = VOICE_SERVER.SpeechRequest(
                model="tts-1",
                input="テストです",
                voice="",
                language="Japanese",
                stream=False,
            )
            payload = VOICE_SERVER._tsukasa_payload(request)

        self.assertEqual(payload["voice"], "audio_ref")
        self.assertEqual(payload["diffusion_steps"], 5)

    def test_english_request_uses_qwen_backend_when_default_is_tsukasa(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TTS_BACKEND": "tsukasa-speech",
                "QWEN_TTS_MODEL": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            },
            clear=False,
        ):
            request = VOICE_SERVER.SpeechRequest(
                model="tts-1",
                input="Hello Reachy, this is an English test.",
                voice="default",
                language="English",
                stream=False,
            )

            self.assertEqual(VOICE_SERVER._tts_backend_for_request(request), "qwen3-tts")
            payload = VOICE_SERVER._tts_request_payload(request, response_format="wav", stream=False, backend="qwen3-tts")

        self.assertEqual(payload["model"], "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    def test_japanese_request_keeps_tsukasa_backend(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TTS_BACKEND": "tsukasa-speech",
            },
            clear=False,
        ):
            request = VOICE_SERVER.SpeechRequest(
                model="tts-1",
                input="こんにちは、リーチー。",
                voice="default",
                language="Japanese",
                stream=False,
            )

            self.assertEqual(VOICE_SERVER._tts_backend_for_request(request), "tsukasa-speech")

    def test_qwen_voice_normalization_falls_back_from_tsukasa_default(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QWEN_TTS_DEFAULT_VOICE": "ono_anna",
            },
            clear=False,
        ):
            self.assertEqual(VOICE_SERVER._normalize_voice_for_backend("default", "qwen3-tts"), "ono_anna")
            self.assertEqual(VOICE_SERVER._normalize_voice_for_backend("Sohee", "qwen3-tts"), "sohee")

    def test_qwen_voice_normalization_uses_backend_specific_mapping(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QWEN_TTS_DEFAULT_VOICE": "ono_anna",
            },
            clear=False,
        ):
            self.assertEqual(
                VOICE_SERVER._normalize_voice_for_backend(
                    {"tsukasa-speech": "captain", "qwen3-tts": "Ryan"},
                    "qwen3-tts",
                ),
                "ryan",
            )
            self.assertEqual(
                VOICE_SERVER._normalize_voice_for_backend(
                    {"tsukasa-speech": "captain", "qwen3-tts": "Ryan"},
                    "tsukasa-speech",
                ),
                "captain",
            )


    def test_warmup_uses_backend_specific_default_voice(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TSUKASA_SPEECH_DEFAULT_VOICE": "shiki_fine05",
                "QWEN_TTS_DEFAULT_VOICE": "ono_anna",
            },
            clear=False,
        ):
            self.assertEqual(VOICE_SERVER._default_voice_for_backend("tsukasa-speech"), "shiki_fine05")
            self.assertEqual(VOICE_SERVER._default_voice_for_backend("qwen3-tts"), "ono_anna")

    def test_transcription_language_maps_to_english_tts_language(self) -> None:
        self.assertEqual(VOICE_SERVER._tts_language_from_transcription_language("en"), "English")
        self.assertEqual(VOICE_SERVER._tts_language_from_transcription_language("en-US"), "English")

    def test_apply_detected_tts_language_updates_realtime_session(self) -> None:
        session = VOICE_SERVER.RealtimeSession(language="Japanese")

        VOICE_SERVER._apply_detected_tts_language(session, "en")
        self.assertEqual(session.language, "English")

        VOICE_SERVER._apply_detected_tts_language(session, "pt")
        self.assertEqual(session.language, "English")

        VOICE_SERVER._apply_detected_tts_language(session, "ja")
        self.assertEqual(session.language, "Japanese")

    def test_realtime_session_update_accepts_voice_mapping(self) -> None:
        with TestClient(VOICE_SERVER.app) as client:
            with client.websocket_connect("/v1/realtime") as websocket:
                created = websocket.receive_json()
                self.assertEqual(created["type"], "session.created")

                websocket.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "voice": {"tsukasa-speech": "captain", "qwen3-tts": "Ryan"},
                            "language": "Japanese",
                        },
                    }
                )
                updated = websocket.receive_json()
                self.assertEqual(updated["type"], "session.updated")
                self.assertEqual(updated["session"]["voice"], {"tsukasa-speech": "captain", "qwen3-tts": "Ryan"})


if __name__ == "__main__":
    unittest.main()
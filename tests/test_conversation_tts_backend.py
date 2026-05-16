from __future__ import annotations

import argparse
import importlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "conversation"))

pipeline = importlib.import_module("pipeline")
effective_tts_transport = importlib.import_module("state").effective_tts_transport
_tts_http_request_payload = pipeline._tts_http_request_payload
_tts_session_update_payload = pipeline._tts_session_update_payload


class ConversationTTSBackendTests(unittest.TestCase):
    @staticmethod
    def _make_args(*, tts_backend: str | None) -> argparse.Namespace:
        return argparse.Namespace(
            tts_model="tts-test",
            tts_task_type="CustomVoice",
            tts_language="Japanese",
            tts_temperature=0.65,
            tts_alpha=0.42,
            tts_beta=0.84,
            voice={"tsukasa-speech": "captain", "qwen3-tts": "Ryan"},
            tts_instructions="Speak clearly.",
            tts_backend=tts_backend,
            tts_ref_audio={"tsukasa-speech": "data:audio/wav;base64,AAA=", "qwen3-tts": "data:audio/wav;base64,BBB="},
            tts_ref_text={"qwen3-tts": "Good evening, sir."},
            tts_x_vector_only_mode=False,
        )

    def test_http_payload_omits_backend_when_unset(self) -> None:
        payload = _tts_http_request_payload(self._make_args(tts_backend=None), "こんにちは")

        self.assertNotIn("backend", payload)

    def test_http_payload_includes_backend_when_set(self) -> None:
        payload = _tts_http_request_payload(self._make_args(tts_backend="qwen3-tts"), "こんにちは")

        self.assertEqual(payload["backend"], "qwen3-tts")
        self.assertEqual(payload["temperature"], 0.65)
        self.assertEqual(payload["alpha"], 0.42)
        self.assertEqual(payload["beta"], 0.84)
        self.assertEqual(payload["ref_audio"]["qwen3-tts"], "data:audio/wav;base64,BBB=")
        self.assertEqual(payload["ref_text"]["qwen3-tts"], "Good evening, sir.")

    def test_realtime_session_payload_omits_backend_when_unset(self) -> None:
        session = _tts_session_update_payload(self._make_args(tts_backend=""))

        self.assertNotIn("backend", session)

    def test_realtime_session_payload_includes_backend_when_set(self) -> None:
        session = _tts_session_update_payload(self._make_args(tts_backend="tsukasa"))

        self.assertEqual(session["backend"], "tsukasa")
        self.assertEqual(session["temperature"], 0.65)
        self.assertEqual(session["alpha"], 0.42)
        self.assertEqual(session["beta"], 0.84)
        self.assertEqual(session["ref_audio"]["tsukasa-speech"], "data:audio/wav;base64,AAA=")
        json.dumps(session)

    def test_payload_omits_temperature_when_unset(self) -> None:
        args = self._make_args(tts_backend="qwen3-tts")
        args.tts_temperature = None
        args.tts_alpha = None
        args.tts_beta = None

        payload = _tts_http_request_payload(args, "こんにちは")
        session = _tts_session_update_payload(args)

        self.assertNotIn("temperature", payload)
        self.assertNotIn("temperature", session)
        self.assertNotIn("alpha", payload)
        self.assertNotIn("alpha", session)
        self.assertNotIn("beta", payload)
        self.assertNotIn("beta", session)

    def test_effective_transport_prefers_http_for_tsukasa_when_not_explicit(self) -> None:
        self.assertEqual(effective_tts_transport("realtime", "tsukasa", transport_explicit=False), "http")

    def test_effective_transport_keeps_explicit_transport_choice(self) -> None:
        self.assertEqual(effective_tts_transport("realtime", "tsukasa-speech", transport_explicit=True), "realtime")


if __name__ == "__main__":
    unittest.main()
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
            voice={"tsukasa-speech": "captain", "qwen3-tts": "Ryan"},
            tts_instructions="Speak clearly.",
            tts_backend=tts_backend,
        )

    def test_http_payload_omits_backend_when_unset(self) -> None:
        payload = _tts_http_request_payload(self._make_args(tts_backend=None), "こんにちは")

        self.assertNotIn("backend", payload)

    def test_http_payload_includes_backend_when_set(self) -> None:
        payload = _tts_http_request_payload(self._make_args(tts_backend="qwen3-tts"), "こんにちは")

        self.assertEqual(payload["backend"], "qwen3-tts")

    def test_realtime_session_payload_omits_backend_when_unset(self) -> None:
        session = _tts_session_update_payload(self._make_args(tts_backend=""))

        self.assertNotIn("backend", session)

    def test_realtime_session_payload_includes_backend_when_set(self) -> None:
        session = _tts_session_update_payload(self._make_args(tts_backend="tsukasa"))

        self.assertEqual(session["backend"], "tsukasa")
        json.dumps(session)

    def test_effective_transport_prefers_http_for_tsukasa_when_not_explicit(self) -> None:
        self.assertEqual(effective_tts_transport("realtime", "tsukasa", transport_explicit=False), "http")

    def test_effective_transport_keeps_explicit_transport_choice(self) -> None:
        self.assertEqual(effective_tts_transport("realtime", "tsukasa-speech", transport_explicit=True), "realtime")


if __name__ == "__main__":
    unittest.main()
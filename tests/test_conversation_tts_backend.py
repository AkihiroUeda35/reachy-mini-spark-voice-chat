from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "conversation"))

from pipeline import _tts_http_request_payload, _tts_session_update_payload


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


if __name__ == "__main__":
    unittest.main()
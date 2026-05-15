from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

CONVERSATION_DIR = Path(__file__).resolve().parents[1] / "apps" / "conversation"
if str(CONVERSATION_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERSATION_DIR))

_drain_ready_segments = importlib.import_module("pipeline")._drain_ready_segments


class ConversationTTSChunkingTests(unittest.TestCase):
    def test_punctuation_does_not_flush_before_thresholds(self) -> None:
        segments, remainder = _drain_ready_segments(
            "こんにちは。元気ですか？はい、元気です。",
            final=False,
            newline_threshold=2,
            max_chars=100,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "こんにちは。元気ですか？はい、元気です。")

    def test_single_newline_does_not_flush_when_threshold_is_two(self) -> None:
        segments, remainder = _drain_ready_segments(
            "一行目\n二行目",
            final=False,
            newline_threshold=2,
            max_chars=100,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "一行目\n二行目")

    def test_consecutive_newlines_flush_segment(self) -> None:
        segments, remainder = _drain_ready_segments(
            "一段落目です。\n\n次の段落",
            final=False,
            newline_threshold=2,
            max_chars=100,
        )

        self.assertEqual(segments, ["一段落目です。"])
        self.assertEqual(remainder, "次の段落")

    def test_max_chars_flushes_without_waiting_for_punctuation(self) -> None:
        segments, remainder = _drain_ready_segments(
            "あいうえおかきくけこ",
            final=False,
            newline_threshold=2,
            max_chars=5,
        )

        self.assertEqual(segments, ["あいうえお", "かきくけこ"])
        self.assertEqual(remainder, "")

    def test_final_flush_returns_tail(self) -> None:
        segments, remainder = _drain_ready_segments(
            "まだ途中です。",
            final=True,
            newline_threshold=2,
            max_chars=100,
        )

        self.assertEqual(segments, ["まだ途中です。"])
        self.assertEqual(remainder, "")


if __name__ == "__main__":
    unittest.main()
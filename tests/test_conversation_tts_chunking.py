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

    def test_max_chars_alone_does_not_flush_without_punctuation(self) -> None:
        segments, remainder = _drain_ready_segments(
            "あいうえおかきくけこ",
            final=False,
            newline_threshold=2,
            max_chars=5,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "あいうえおかきくけこ")

    def test_flushes_when_threshold_and_japanese_period_are_both_met(self) -> None:
        segments, remainder = _drain_ready_segments(
            "あいうえお。かきくけこ",
            final=False,
            newline_threshold=2,
            max_chars=5,
        )

        self.assertEqual(segments, ["あいうえお。"])
        self.assertEqual(remainder, "かきくけこ")

    def test_final_flush_returns_tail(self) -> None:
        segments, remainder = _drain_ready_segments(
            "まだ途中です。",
            final=True,
            newline_threshold=2,
            max_chars=100,
        )

        self.assertEqual(segments, ["まだ途中です。"])
        self.assertEqual(remainder, "")

    def test_english_period_does_not_flush_before_threshold(self) -> None:
        segments, remainder = _drain_ready_segments(
            "Hello world. Next sentence",
            final=False,
            newline_threshold=2,
            max_chars=100,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "Hello world. Next sentence")

    def test_english_max_chars_is_tripled(self) -> None:
        segments, remainder = _drain_ready_segments(
            "abcdefghijklmnopqrstuvwxyz",
            final=False,
            newline_threshold=2,
            max_chars=10,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "abcdefghijklmnopqrstuvwxyz")

    def test_english_flushes_at_period_after_tripled_threshold(self) -> None:
        segments, remainder = _drain_ready_segments(
            "abcdefghijklmnopqrstuvwxyzabcd. trailing",
            final=False,
            newline_threshold=2,
            max_chars=10,
        )

        self.assertEqual(segments, ["abcdefghijklmnopqrstuvwxyzabcd."])
        self.assertEqual(remainder, " trailing")

    def test_closing_paren_flushes_after_threshold(self) -> None:
        segments, remainder = _drain_ready_segments(
            "abcdefghijklmnop)",
            final=False,
            newline_threshold=2,
            max_chars=5,
        )

        self.assertEqual(segments, ["abcdefghijklmnop)"])
        self.assertEqual(remainder, "")

    def test_english_does_not_flush_on_other_punctuation(self) -> None:
        segments, remainder = _drain_ready_segments(
            "Hello, world! How are you?",
            final=False,
            newline_threshold=2,
            max_chars=100,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "Hello, world! How are you?")


if __name__ == "__main__":
    unittest.main()
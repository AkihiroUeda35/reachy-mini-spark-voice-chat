from __future__ import annotations

import argparse
from unittest.mock import MagicMock
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "conversation"))

from main import _conversation_transcription_args, _log_transcript_judgement, _transcript_observation_fields, transcript_quality_rejection_reason


class ConversationASRFilteringTests(unittest.TestCase):
    @staticmethod
    def _make_args(
        *,
        language: str = "auto",
        allowed_transcript_languages: str = "",
        excluded_transcript_languages: str = "",
        min_segment_avg_logprob: float = -1.0,
        max_segment_no_speech_prob: float = 0.6,
        response_format: str = "text",
    ) -> argparse.Namespace:
        return argparse.Namespace(
            language=language,
            allowed_transcript_languages=allowed_transcript_languages,
            excluded_transcript_languages=excluded_transcript_languages,
            min_segment_avg_logprob=min_segment_avg_logprob,
            max_segment_no_speech_prob=max_segment_no_speech_prob,
            response_format=response_format,
        )

    def test_conversation_transcription_args_force_verbose_json(self) -> None:
        request_args = _conversation_transcription_args(self._make_args(response_format="text"))

        self.assertEqual(request_args.response_format, "verbose_json")

    def test_rejects_detected_language_outside_explicit_hint(self) -> None:
        reason = transcript_quality_rejection_reason(
            {
                "text": "privet",
                "language": "ru",
                "segments": [{"avg_logprob": -0.2, "no_speech_prob": 0.02}],
            },
            self._make_args(language="ja"),
        )

        self.assertEqual(reason, "language ru not in ja")

    def test_rejects_detected_language_in_excluded_language_list(self) -> None:
        reason = transcript_quality_rejection_reason(
            {
                "text": "Продолжение следует...",
                "language": "ru",
                "segments": [{"avg_logprob": -0.82, "no_speech_prob": 0.00}],
            },
            self._make_args(language="auto", excluded_transcript_languages="ru,es,fr"),
        )

        self.assertEqual(reason, "language ru excluded by es,fr,ru")

    def test_accepts_region_variant_inside_allowed_language_list(self) -> None:
        reason = transcript_quality_rejection_reason(
            {
                "text": "hello",
                "language": "en-US",
                "segments": [{"avg_logprob": -0.2, "no_speech_prob": 0.02}],
            },
            self._make_args(allowed_transcript_languages="ja,en"),
        )

        self.assertIsNone(reason)

    def test_rejects_low_mean_avg_logprob(self) -> None:
        reason = transcript_quality_rejection_reason(
            {
                "text": "こんにちは",
                "language": "ja",
                "segments": [
                    {"avg_logprob": -1.4, "no_speech_prob": 0.10},
                    {"avg_logprob": -1.2, "no_speech_prob": 0.15},
                ],
            },
            self._make_args(language="ja", min_segment_avg_logprob=-1.0),
        )

        self.assertEqual(reason, "avg_logprob -1.30 < -1.00")

    def test_rejects_high_no_speech_probability(self) -> None:
        reason = transcript_quality_rejection_reason(
            {
                "text": "こんにちは",
                "language": "ja",
                "segments": [{"avg_logprob": -0.3, "no_speech_prob": 0.84}],
            },
            self._make_args(language="ja", max_segment_no_speech_prob=0.6),
        )

        self.assertEqual(reason, "no_speech_prob 0.84 > 0.60")

    def test_skips_score_gate_when_segments_do_not_expose_metrics(self) -> None:
        reason = transcript_quality_rejection_reason(
            {
                "text": "こんにちは",
                "language": "ja",
                "segments": [{"id": 0, "text": "こんにちは"}],
            },
            self._make_args(language="ja"),
        )

        self.assertIsNone(reason)

    def test_transcript_observation_fields_always_return_language_score_and_text(self) -> None:
        fields = _transcript_observation_fields(
            {
                "text": "selamat menikmati",
                "language": "id",
                "segments": [{"avg_logprob": -0.23, "no_speech_prob": 0.04}],
            }
        )

        self.assertEqual(fields, ("id", "-0.23", "0.04", "selamat menikmati"))

    def test_log_transcript_judgement_includes_reason_and_fallback_values(self) -> None:
        logger = MagicMock()

        _log_transcript_judgement(
            logger,
            label="barge-in",
            payload={"text": "", "segments": []},
            transcript_text="",
            accepted=False,
            rejection_reason="empty transcript",
        )

        logger.info.assert_called_once_with(
            "[bold blue]ASR[/] %s result=%s language=%s avg_logprob=%s max_no_speech_prob=%s text=%s%s",
            "barge-in",
            "rejected",
            "unknown",
            "n/a",
            "n/a",
            "",
            " reason=empty transcript",
        )


if __name__ == "__main__":
    unittest.main()
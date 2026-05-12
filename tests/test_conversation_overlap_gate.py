from __future__ import annotations

import time
import unittest

from apps.conversation.state import AssistantSpeechState, listening_gate_settings, overlap_turn_rejection_reason, transcript_char_count


class ConversationOverlapGateTests(unittest.TestCase):
    def test_assistant_speech_state_stays_active_for_audio_duration_and_tail(self) -> None:
        speech_state = AssistantSpeechState(tail_hold_s=0.05)

        speech_state.note_output_audio(sample_count=1600, sample_rate=16000)

        self.assertTrue(speech_state.is_speaking())
        time.sleep(0.03)
        self.assertTrue(speech_state.is_speaking())
        time.sleep(0.13)
        self.assertFalse(speech_state.is_speaking())

    def test_listening_gate_settings_only_tighten_while_assistant_is_speaking(self) -> None:
        self.assertEqual(
            listening_gate_settings(0.012, 90, assistant_speaking=False, speaking_threshold_boost=0.01, speaking_vad_start_ms=400),
            (0.012, 90),
        )
        self.assertEqual(
            listening_gate_settings(0.012, 90, assistant_speaking=True, speaking_threshold_boost=0.01, speaking_vad_start_ms=400),
            (0.022, 400),
        )

    def test_overlap_turn_rejection_reason_checks_vad_then_char_count(self) -> None:
        self.assertEqual(
            overlap_turn_rejection_reason(
                captured_duration_ms=320,
                transcript_text="ごめん",
                min_duration_ms=500,
                min_chars=4,
            ),
            "vad 320ms < 500ms",
        )
        self.assertEqual(
            overlap_turn_rejection_reason(
                captured_duration_ms=700,
                transcript_text="おー",
                min_duration_ms=500,
                min_chars=4,
            ),
            "chars 1 < 4",
        )
        self.assertIsNone(
            overlap_turn_rejection_reason(
                captured_duration_ms=700,
                transcript_text="それはちがうよ",
                min_duration_ms=500,
                min_chars=4,
            )
        )

    def test_transcript_char_count_ignores_spacing_and_punctuation(self) -> None:
        self.assertEqual(transcript_char_count(" おー!? "), 1)
        self.assertEqual(transcript_char_count(" ごめん。 "), 3)


if __name__ == "__main__":
    unittest.main()
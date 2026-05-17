from __future__ import annotations

import argparse
import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

CONVERSATION_DIR = Path(__file__).resolve().parents[1] / "apps" / "conversation"
if str(CONVERSATION_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERSATION_DIR))

from main import _run_pipeline_with_barge_in, _speak_persona_greeting_with_barge_in, CapturedUtterance, conversation_loop
from pipeline import PipelineResult, SpeechInterruptedError, SynthesizedAudio


class ConversationBargeInTests(unittest.TestCase):
    def _make_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            profiles_dir="/home/aki/server/apps/conversation/profiles",
            profile="default",
            voice="Sohee",
            tts_instructions="Be concise.",
            base_url="http://localhost:8000/v1",
            api_key="test-key",
            model="stt-test",
            transport="http",
            chat_base_url="http://localhost:8001/v1",
            chat_api_key="test-key",
            chat_model="chat-test",
            tts_base_url="http://localhost:8002/v1",
            tts_api_key="test-key",
            tts_model="tts-test",
            robot_name=None,
            robot_host=None,
            data_dir="/tmp/conversation-test-data",
            sample_rate=16000,
            audio_poll_interval_ms=10.0,
            listen_timeout_seconds=20.0,
            vad_threshold=0.02,
            vad_start_ms=150,
            vad_end_ms=250,
            vad_preroll_ms=150,
            vad_max_seconds=5.0,
            assistant_speaking_tail_ms=350,
            assistant_speaking_threshold_boost=0.005,
            assistant_speaking_vad_start_ms=400,
            assistant_speaking_min_vad_ms=500,
            assistant_speaking_min_chars=4,
            assistant_speaking_interrupt_min_vad_ms=700,
            assistant_speaking_interrupt_min_chars=6,
            tts_transport="http",
            head_wobble=False,
            gradio=False,
            wake_up=False,
            motion_duration=0.8,
            auto_install_optional_deps=False,
            history_turns=6,
            save_replies=False,
            save_transcripts=False,
            min_segment_avg_logprob=-1.0,
            max_segment_no_speech_prob=0.6,
            language="auto",
            allowed_transcript_languages="",
            excluded_transcript_languages="",
        )

    @staticmethod
    def _make_robot() -> MagicMock:
        robot = MagicMock()
        robot.media = MagicMock()
        robot.media.start_recording = MagicMock()
        robot.media.start_playing = MagicMock()
        robot.media.stop_recording = MagicMock()
        robot.media.stop_playing = MagicMock()
        robot.media_manager = MagicMock()
        robot.media_manager.close = MagicMock()
        robot.client = MagicMock()
        robot.client.disconnect = MagicMock()
        return robot

    def test_run_pipeline_with_barge_in_returns_captured_utterance(self) -> None:
        args = self._make_args()
        robot = self._make_robot()
        runtime = MagicMock()
        captured = CapturedUtterance(audio=MagicMock(), duration_ms=900.0, overlap_gate_active=True, barge_in_candidate=True)

        async def run_turn_side_effect(*_args, **kwargs):
            kwargs["llm_finished_event"].set()
            interrupt_event = kwargs["interrupt_event"]
            while not interrupt_event.is_set():
                await asyncio.sleep(0)
            raise SpeechInterruptedError("assistant speech interrupted by user speech")

        with (
            patch("main.run_pipeline", side_effect=run_turn_side_effect),
            patch("main.capture_robot_utterance", AsyncMock(return_value=captured)),
            patch("main.transcribe_captured_audio", AsyncMock(return_value={"text": "ちょっと待って", "language": "ja", "segments": []})),
        ):
            result, barge_in = asyncio.run(
                _run_pipeline_with_barge_in(
                    args,
                    runtime,
                    "こんにちは",
                    [],
                    [],
                    robot=robot,
                    assistant_speech_state=MagicMock(),
                )
            )

        self.assertIsNone(result)
        self.assertIs(barge_in, captured)

    def test_speak_persona_greeting_with_barge_in_returns_captured_utterance(self) -> None:
        args = self._make_args()
        robot = self._make_robot()
        history = [{"role": "assistant", "content": "既存"}]
        captured = CapturedUtterance(audio=MagicMock(), duration_ms=900.0, overlap_gate_active=True, barge_in_candidate=True)

        async def speak_side_effect(*_args, **kwargs):
            kwargs["barge_in_ready_event"].set()
            interrupt_event = kwargs["interrupt_event"]
            while not interrupt_event.is_set():
                await asyncio.sleep(0)
            raise SpeechInterruptedError("assistant speech interrupted by user speech")

        with (
            patch("main._speak_persona_greeting", side_effect=speak_side_effect),
            patch("main.capture_robot_utterance", AsyncMock(return_value=captured)),
            patch("main.transcribe_captured_audio", AsyncMock(return_value={"text": "ちょっと待って", "language": "ja", "segments": []})),
        ):
            updated_history, barge_in = asyncio.run(
                _speak_persona_greeting_with_barge_in(
                    args,
                    robot,
                    history,
                    reason="startup",
                    assistant_speech_state=MagicMock(),
                )
            )

        self.assertEqual(updated_history, history)
        self.assertIs(barge_in, captured)

    def test_run_pipeline_with_barge_in_ignores_rejected_language_candidate(self) -> None:
        args = self._make_args()
        args.excluded_transcript_languages = "ru"
        robot = self._make_robot()
        runtime = MagicMock()
        captured = CapturedUtterance(audio=MagicMock(), duration_ms=900.0, overlap_gate_active=True, barge_in_candidate=True)
        expected_result = PipelineResult(
            assistant_text="続けます。",
            audio=SynthesizedAudio(pcm16_bytes=b"\x00\x00", sample_rate=24000, num_channels=1),
        )

        with (
            patch("main.run_pipeline", AsyncMock(return_value=expected_result)),
            patch("main.capture_robot_utterance", AsyncMock(side_effect=[captured, None])),
            patch("main.transcribe_captured_audio", AsyncMock(return_value={"text": "Продолжение следует...", "language": "ru", "segments": []})),
        ):
            result, barge_in = asyncio.run(
                _run_pipeline_with_barge_in(
                    args,
                    runtime,
                    "こんにちは",
                    [],
                    [],
                    robot=robot,
                    assistant_speech_state=MagicMock(),
                )
            )

        self.assertEqual(result, expected_result)
        self.assertIsNone(barge_in)

    def test_run_pipeline_with_barge_in_logs_rejected_candidate_details(self) -> None:
        args = self._make_args()
        args.excluded_transcript_languages = "ru"
        robot = self._make_robot()
        runtime = MagicMock()
        captured = CapturedUtterance(audio=MagicMock(), duration_ms=900.0, overlap_gate_active=True, barge_in_candidate=True)
        expected_result = PipelineResult(
            assistant_text="続けます。",
            audio=SynthesizedAudio(pcm16_bytes=b"\x00\x00", sample_rate=24000, num_channels=1),
        )
        asr_logger = MagicMock()

        with (
            patch("main.run_pipeline", AsyncMock(return_value=expected_result)),
            patch("main.capture_robot_utterance", AsyncMock(side_effect=[captured, None])),
            patch("main.transcribe_captured_audio", AsyncMock(return_value={"text": "Продолжение следует...", "language": "ru", "segments": [{"avg_logprob": -0.82, "no_speech_prob": 0.00}]})),
            patch("main.logging.getLogger", side_effect=lambda name: asr_logger if name == "conversation.asr" else MagicMock()),
        ):
            result, barge_in = asyncio.run(
                _run_pipeline_with_barge_in(
                    args,
                    runtime,
                    "こんにちは",
                    [],
                    [],
                    robot=robot,
                    assistant_speech_state=MagicMock(),
                )
            )

        self.assertEqual(result, expected_result)
        self.assertIsNone(barge_in)
        self.assertTrue(
            any(
                call.args[0] == "[bold blue]ASR[/] %s result=%s language=%s avg_logprob=%s max_no_speech_prob=%s text=%s%s"
                and call.args[1:] == (
                    "barge-in",
                    "rejected",
                    "ru",
                    "-0.82",
                    "0.00",
                    "Продолжение следует...",
                    " reason=language ru excluded by ru",
                )
                for call in asr_logger.info.call_args_list
            )
        )

    def test_run_pipeline_with_barge_in_ignores_short_transcript_candidate(self) -> None:
        args = self._make_args()
        robot = self._make_robot()
        runtime = MagicMock()
        captured = CapturedUtterance(audio=MagicMock(), duration_ms=900.0, overlap_gate_active=True, barge_in_candidate=True)
        expected_result = PipelineResult(
            assistant_text="続けます。",
            audio=SynthesizedAudio(pcm16_bytes=b"\x00\x00", sample_rate=24000, num_channels=1),
        )

        with (
            patch("main.run_pipeline", AsyncMock(return_value=expected_result)),
            patch("main.capture_robot_utterance", AsyncMock(side_effect=[captured, None])),
            patch("main.transcribe_captured_audio", AsyncMock(return_value={"text": "え", "language": "ja", "segments": []})),
        ):
            result, barge_in = asyncio.run(
                _run_pipeline_with_barge_in(
                    args,
                    runtime,
                    "こんにちは",
                    [],
                    [],
                    robot=robot,
                    assistant_speech_state=MagicMock(),
                )
            )

        self.assertEqual(result, expected_result)
        self.assertIsNone(barge_in)

    def test_conversation_loop_reuses_barge_in_utterance_as_next_turn(self) -> None:
        args = self._make_args()
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        first_utterance = CapturedUtterance(audio=MagicMock(name="first_audio"), duration_ms=800.0, overlap_gate_active=False)
        barge_in_utterance = CapturedUtterance(audio=MagicMock(name="barge_audio"), duration_ms=900.0, overlap_gate_active=True, barge_in_candidate=True)
        run_results: list[object] = [(None, barge_in_utterance), KeyboardInterrupt]

        async def run_with_barge_in_side_effect(*_args, **_kwargs):
            next_result = run_results.pop(0)
            if next_result is KeyboardInterrupt:
                raise KeyboardInterrupt
            return next_result

        transcribe_audio = AsyncMock(side_effect=[{"text": "最初の質問", "segments": []}, {"text": "ちょっと待って", "segments": []}])

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting_with_barge_in", AsyncMock(return_value=([], None))),
            patch("main.capture_robot_utterance", side_effect=[first_utterance, KeyboardInterrupt]),
            patch("main.transcribe_captured_audio", transcribe_audio),
            patch("main._run_pipeline_with_barge_in", side_effect=run_with_barge_in_side_effect),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        self.assertEqual(transcribe_audio.await_count, 2)
        self.assertIs(transcribe_audio.await_args_list[0].args[1], first_utterance)
        self.assertIs(transcribe_audio.await_args_list[1].args[1], barge_in_utterance)

    def test_conversation_loop_reuses_startup_greeting_barge_in_utterance(self) -> None:
        args = self._make_args()
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        startup_barge_in = CapturedUtterance(audio=MagicMock(name="startup_barge_audio"), duration_ms=900.0, overlap_gate_active=True, barge_in_candidate=True)

        transcribe_audio = AsyncMock(side_effect=[{"text": "ちょっと待って", "segments": []}])

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting_with_barge_in", side_effect=[([], startup_barge_in)]),
            patch("main._run_pipeline_with_barge_in", side_effect=KeyboardInterrupt),
            patch("main.transcribe_captured_audio", transcribe_audio),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        self.assertEqual(transcribe_audio.await_count, 1)
        self.assertIs(transcribe_audio.await_args_list[0].args[1], startup_barge_in)

    def test_conversation_loop_reuses_persona_switch_greeting_barge_in_utterance(self) -> None:
        args = self._make_args()
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        persona_barge_in = CapturedUtterance(audio=MagicMock(name="persona_barge_audio"), duration_ms=900.0, overlap_gate_active=True, barge_in_candidate=True)
        fallback_capture = CapturedUtterance(audio=MagicMock(name="fallback_audio"), duration_ms=900.0, overlap_gate_active=False)
        runtime_settings = MagicMock()
        runtime_settings.snapshot.return_value = ("samurai", [], "Be stoic.", "You are stoic.", "Sohee", "Ono_Anna", "Be concise.", 1)

        transcribe_audio = AsyncMock(return_value={"text": "待った", "segments": []})

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.RuntimeSettings", return_value=runtime_settings),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting_with_barge_in", side_effect=[([], None), ([], persona_barge_in)]),
            patch("main.capture_robot_utterance", AsyncMock(return_value=fallback_capture)),
            patch("main._run_pipeline_with_barge_in", side_effect=KeyboardInterrupt),
            patch("main.transcribe_captured_audio", transcribe_audio),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        self.assertTrue(any(call.args[1] is persona_barge_in for call in transcribe_audio.await_args_list))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import asyncio
import signal
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import OutputAudioRawFrame

from gradio_ui import _free_gradio_port_if_same_uv_app, _is_same_conversation_app_process
from main import _play_assistant_text, capture_robot_utterance, conversation_loop, is_recoverable_llm_turn_error
from pipeline import SpeechInterruptedError, _create_turn_pipeline_runner
from state import RuntimeSettings as StateRuntimeSettings


class ConversationShutdownTests(unittest.TestCase):
    def _make_args(self, *, wake_up: bool) -> argparse.Namespace:
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
            robot_host="reachy-mini.local",
            data_dir="/tmp/conversation-test-data",
            assistant_speaking_tail_ms=350,
            gradio=False,
            wake_up=wake_up,
            motion_duration=0.8,
            auto_install_optional_deps=False,
            history_turns=6,
            save_replies=False,
            save_transcripts=False,
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

    def test_recoverable_llm_turn_error_matches_tool_round_limit(self) -> None:
        self.assertTrue(is_recoverable_llm_turn_error(RuntimeError("LangChain agent failed: LLM exceeded the maximum tool-call rounds.")))

    def test_recoverable_llm_turn_error_ignores_unrelated_failures(self) -> None:
        self.assertFalse(is_recoverable_llm_turn_error(RuntimeError("TTS returned no audio.")))

    @patch("pipeline.PipelineRunner")
    def test_create_turn_pipeline_runner_disables_signal_handlers(self, mock_runner) -> None:
        _create_turn_pipeline_runner()
        mock_runner.assert_called_once_with(handle_sigint=False, handle_sigterm=False)

    @patch("gradio_ui._read_process_command")
    def test_is_same_conversation_app_process_matches_local_conversation_entrypoint(self, mock_read_process_command) -> None:
        mock_read_process_command.return_value = "/home/aki/server/.venv/bin/python apps/conversation/main.py --gradio"
        result = _is_same_conversation_app_process(999999, Path("/home/aki/server/apps/conversation/main.py"))
        self.assertTrue(result)

    @patch("gradio_ui.os.kill")
    @patch("gradio_ui._is_same_conversation_app_process")
    @patch("gradio_ui._list_listening_pids")
    def test_free_gradio_port_kills_only_matching_processes(self, mock_listening_pids, mock_is_same_process, mock_kill) -> None:
        mock_listening_pids.return_value = [111, 222]
        mock_is_same_process.side_effect = [True, False]

        killed = _free_gradio_port_if_same_uv_app(7860, Path("/home/aki/server/apps/conversation/main.py"))

        self.assertEqual(killed, [111])
        mock_kill.assert_called_once_with(111, signal.SIGKILL)

    def test_conversation_loop_wakes_robot_once_on_startup(self) -> None:
        args = self._make_args(wake_up=True)
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        speak_greeting = AsyncMock(return_value=[])

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=KeyboardInterrupt),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        robot.wake_up.assert_called_once_with()
        speak_greeting.assert_awaited_once()

    def test_conversation_loop_skips_wake_up_when_disabled(self) -> None:
        args = self._make_args(wake_up=False)
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        speak_greeting = AsyncMock(return_value=[])

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=KeyboardInterrupt),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        robot.wake_up.assert_not_called()
        speak_greeting.assert_awaited_once()

    def test_conversation_loop_passes_robot_host_to_reachymini(self) -> None:
        args = self._make_args(wake_up=False)
        args.robot_host = "192.168.0.42"
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        speak_greeting = AsyncMock(return_value=[])

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot) as reachy_cls,
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=KeyboardInterrupt),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        reachy_cls.assert_called_once_with(host="192.168.0.42")

    def test_conversation_loop_greets_on_startup(self) -> None:
        args = self._make_args(wake_up=False)
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        speak_greeting = AsyncMock(return_value=[{"role": "assistant", "content": "こんにちは。"}])

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=KeyboardInterrupt),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        self.assertEqual(speak_greeting.await_count, 1)
        self.assertEqual(speak_greeting.await_args.kwargs["reason"], "startup")

    def test_conversation_loop_recovers_when_startup_greeting_is_interrupted(self) -> None:
        args = self._make_args(wake_up=False)
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        created_settings: dict[str, StateRuntimeSettings] = {}

        def make_runtime_settings(*args, **kwargs):
            settings = StateRuntimeSettings(*args, **kwargs)
            created_settings["value"] = settings
            return settings

        async def speak_greeting_side_effect(*_args, **kwargs):
            reason = kwargs["reason"]
            if reason == "startup":
                settings = created_settings["value"]
                profile, tools, character, instructions, _voice, qwen_voice, tts_instructions, _version = settings.snapshot()
                settings.update(
                    profile,
                    tools,
                    character,
                    instructions,
                    "audio_ref",
                    qwen_voice,
                    tts_instructions,
                    greeting_reason="character update",
                )
                raise SpeechInterruptedError("assistant speech interrupted by live settings change")
            return []

        speak_greeting = AsyncMock(side_effect=speak_greeting_side_effect)

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.RuntimeSettings", side_effect=make_runtime_settings),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=KeyboardInterrupt),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        self.assertEqual(speak_greeting.await_count, 2)
        self.assertEqual(speak_greeting.await_args_list[0].kwargs["reason"], "startup")
        self.assertEqual(speak_greeting.await_args_list[1].kwargs["reason"], "character update")

    def test_conversation_loop_greets_when_persona_changes(self) -> None:
        args = self._make_args(wake_up=False)
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        speak_greeting = AsyncMock(side_effect=[
            [{"role": "assistant", "content": "はじめまして。"}],
            [{"role": "assistant", "content": "拙者が参った。"}],
        ])
        runtime_settings = MagicMock()
        runtime_settings.snapshot.return_value = ("samurai", [], "Be stoic.", "You are stoic.", "Sohee", "Ono_Anna", "Be concise.", 1)

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.RuntimeSettings", return_value=runtime_settings),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=[None, KeyboardInterrupt]),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        self.assertEqual(speak_greeting.await_count, 2)
        self.assertEqual(speak_greeting.await_args_list[0].kwargs["reason"], "startup")
        self.assertEqual(speak_greeting.await_args_list[1].kwargs["reason"], "persona switch")

    def test_conversation_loop_discards_pending_turn_when_live_settings_change_during_capture(self) -> None:
        args = self._make_args(wake_up=False)
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        speak_greeting = AsyncMock(side_effect=[
            [{"role": "assistant", "content": "こんにちは。"}],
            [{"role": "assistant", "content": "どうぞ。"}],
            [{"role": "assistant", "content": "拙者が参った。"}],
        ])
        runtime_settings = MagicMock()
        runtime_settings.snapshot.side_effect = [
            ("default", [], "Be helpful.", "You are helpful.", "Sohee", "Ono_Anna", "Be concise.", 0),
            ("default", [], "Be helpful.", "You are helpful.", "Sohee", "Ono_Anna", "Be concise.", 0),
            ("samurai", [], "Be stoic.", "You are stoic.", "Sohee", "Ono_Anna", "Be concise.", 1),
            ("samurai", [], "Be stoic.", "You are stoic.", "Sohee", "Ono_Anna", "Be concise.", 1),
        ]
        transcribe_audio = AsyncMock()
        run_turn = AsyncMock()

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.RuntimeSettings", return_value=runtime_settings),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=[MagicMock(), KeyboardInterrupt]),
            patch("main.transcribe_captured_audio", transcribe_audio),
            patch("main.run_pipeline", run_turn),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        transcribe_audio.assert_not_awaited()
        run_turn.assert_not_awaited()
        self.assertEqual(speak_greeting.await_count, 3)
        self.assertEqual(speak_greeting.await_args_list[-1].kwargs["reason"], "persona switch")

    def test_conversation_loop_greets_after_live_voice_update(self) -> None:
        args = self._make_args(wake_up=False)
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        speak_greeting = AsyncMock(side_effect=[
            [{"role": "assistant", "content": "こんにちは。"}],
            [{"role": "assistant", "content": "声を変えました。"}],
        ])
        created_settings: dict[str, StateRuntimeSettings] = {}

        def make_runtime_settings(*args, **kwargs):
            settings = StateRuntimeSettings(*args, **kwargs)
            created_settings["value"] = settings
            return settings

        async def capture_side_effect(*_args, **_kwargs):
            if "updated" not in created_settings:
                settings = created_settings["value"]
                profile, tools, character, instructions, voice, qwen_voice, tts_instructions, _version = settings.snapshot()
                next_voice = "audio_ref" if voice != "audio_ref" else "kaede_san"
                settings.update(
                    profile,
                    tools,
                    character,
                    instructions,
                    next_voice,
                    qwen_voice,
                    tts_instructions,
                    greeting_reason="character update",
                )
                created_settings["updated"] = settings
                return None
            raise KeyboardInterrupt

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.RuntimeSettings", side_effect=make_runtime_settings),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=capture_side_effect),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        self.assertEqual(speak_greeting.await_count, 2)
        self.assertEqual(speak_greeting.await_args_list[0].kwargs["reason"], "startup")
        self.assertEqual(speak_greeting.await_args_list[1].kwargs["reason"], "character update")

    def test_capture_robot_utterance_aborts_wait_when_live_settings_change(self) -> None:
        args = argparse.Namespace(
            audio_poll_interval_ms=1.0,
            listen_timeout_seconds=20.0,
            vad_threshold=0.02,
            vad_start_ms=150,
            vad_end_ms=250,
            vad_preroll_ms=150,
            vad_max_seconds=5.0,
            assistant_speaking_threshold_boost=0.005,
            assistant_speaking_vad_start_ms=400,
            assistant_speaking_min_vad_ms=500,
            sample_rate=16000,
        )
        robot = MagicMock()
        robot.media = MagicMock()
        robot.media.get_input_audio_samplerate.return_value = 16000
        robot.media.get_audio_sample.return_value = None
        runtime_settings = MagicMock()
        runtime_settings.snapshot.side_effect = [
            ("default", [], "", "", "default", "Ono_Anna", "", 0),
            ("default", [], "", "", "default", "Ono_Anna", "", 1),
        ]

        utterance = asyncio.run(
            capture_robot_utterance(
                robot,
                args,
                runtime_settings=runtime_settings,
                expected_settings_version=0,
            )
        )

        self.assertIsNone(utterance)

    def test_play_assistant_text_interrupts_when_live_settings_change(self) -> None:
        args = argparse.Namespace(
            tts_transport="http",
            head_wobble=False,
            tts_sample_rate=24000,
        )
        robot = self._make_robot()
        runtime_settings = MagicMock()
        runtime_settings.snapshot.side_effect = [
            ("default", [], "", "", "default", "Ono_Anna", "", 0),
            ("default", [], "", "", "default", "Ono_Anna", "", 1),
        ]
        audio_player = MagicMock()
        audio_player.process_frame = AsyncMock()

        async def fake_audio_stream(_args, _text):
            yield OutputAudioRawFrame(audio=b"\x00\x00" * 32, sample_rate=24000, num_channels=1)
            yield OutputAudioRawFrame(audio=b"\x00\x00" * 32, sample_rate=24000, num_channels=1)

        with (
            patch("main.ReachyAudioPlayer", return_value=audio_player),
            patch("main.synthesize_audio_stream", fake_audio_stream),
        ):
            with self.assertRaises(SpeechInterruptedError):
                asyncio.run(
                    _play_assistant_text(
                        args,
                        robot,
                        "こんにちは。",
                        runtime_settings=runtime_settings,
                        expected_settings_version=0,
                    )
                )

        audio_player.abort.assert_called_once_with()
        audio_player.close.assert_called_once_with()
        self.assertEqual(audio_player.process_frame.await_count, 1)

    def test_conversation_loop_discards_live_reply_when_settings_change(self) -> None:
        args = self._make_args(wake_up=False)
        robot = self._make_robot()
        runtime = MagicMock()
        runtime.shutdown = AsyncMock()
        speak_greeting = AsyncMock(side_effect=[
            [{"role": "assistant", "content": "こんにちは。"}],
            [{"role": "assistant", "content": "声を変えました。"}],
        ])
        run_turn = AsyncMock(side_effect=[SpeechInterruptedError("interrupted"), KeyboardInterrupt])
        created_settings: dict[str, StateRuntimeSettings] = {}

        def make_runtime_settings(*args, **kwargs):
            settings = StateRuntimeSettings(*args, **kwargs)
            created_settings["value"] = settings
            return settings

        async def capture_side_effect(*_args, **_kwargs):
            settings = created_settings["value"]
            if "updated" not in created_settings:
                profile, tools, character, instructions, voice, qwen_voice, tts_instructions, _version = settings.snapshot()
                settings.update(
                    profile,
                    tools,
                    character,
                    instructions,
                    voice,
                    qwen_voice,
                    tts_instructions,
                    greeting_reason="character update",
                )
                created_settings["updated"] = True
                return MagicMock(overlap_gate_active=False, duration_ms=800.0)
            raise KeyboardInterrupt

        with (
            patch("main.resolve_runtime_models"),
            patch("main.ReachyMini", return_value=robot),
            patch("main.RuntimeSettings", side_effect=make_runtime_settings),
            patch("main.reachy_tools.ReachyToolRuntime", return_value=runtime),
            patch("main._speak_persona_greeting", speak_greeting),
            patch("main.capture_robot_utterance", side_effect=capture_side_effect),
            patch("main.transcribe_captured_audio", AsyncMock(return_value={"text": "ねえ", "segments": []})),
            patch("main.run_pipeline", run_turn),
        ):
            result = asyncio.run(conversation_loop(args))

        self.assertEqual(result, 0)
        self.assertEqual(speak_greeting.await_count, 2)
        self.assertEqual(speak_greeting.await_args_list[-1].kwargs["reason"], "character update")


if __name__ == "__main__":
    unittest.main()
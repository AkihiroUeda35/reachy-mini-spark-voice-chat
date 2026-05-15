from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import subprocess
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from gradio_ui import _free_gradio_port_if_same_uv_app, _is_same_conversation_app_process
from main import conversation_loop, is_recoverable_llm_turn_error
from pipeline import _create_turn_pipeline_runner


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
        runtime_settings.snapshot.side_effect = [
            ("samurai", [], "Be stoic.", "You are stoic.", "Sohee", "Be concise.", 1),
            ("samurai", [], "Be stoic.", "You are stoic.", "Sohee", "Be concise.", 1),
        ]

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


if __name__ == "__main__":
    unittest.main()
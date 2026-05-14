from __future__ import annotations

import logging
import signal
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from gradio_ui import _free_gradio_port_if_same_uv_app, _is_same_conversation_app_process
from main import is_recoverable_llm_turn_error
from pipeline import _create_turn_pipeline_runner


class ConversationShutdownTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
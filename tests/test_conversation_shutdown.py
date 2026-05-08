from __future__ import annotations

import logging
import signal
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.conversation.main import _create_turn_pipeline_runner, _free_gradio_port_if_same_uv_app, _is_same_conversation_app_process


class ConversationShutdownTests(unittest.TestCase):
    @patch("apps.conversation.main.PipelineRunner")
    def test_create_turn_pipeline_runner_disables_signal_handlers(self, mock_runner) -> None:
        _create_turn_pipeline_runner()
        mock_runner.assert_called_once_with(handle_sigint=False, handle_sigterm=False)

    @patch("apps.conversation.main._read_process_command")
    def test_is_same_conversation_app_process_matches_local_conversation_entrypoint(self, mock_read_process_command) -> None:
        mock_read_process_command.return_value = "/home/aki/server/.venv/bin/python apps/conversation/main.py --gradio"
        result = _is_same_conversation_app_process(999999, Path("/home/aki/server/apps/conversation/main.py"))
        self.assertTrue(result)

    @patch("apps.conversation.main.os.kill")
    @patch("apps.conversation.main._is_same_conversation_app_process")
    @patch("apps.conversation.main._list_listening_pids")
    def test_free_gradio_port_kills_only_matching_processes(self, mock_listening_pids, mock_is_same_process, mock_kill) -> None:
        mock_listening_pids.return_value = [111, 222]
        mock_is_same_process.side_effect = [True, False]

        killed = _free_gradio_port_if_same_uv_app(7860, Path("/home/aki/server/apps/conversation/main.py"))

        self.assertEqual(killed, [111])
        mock_kill.assert_called_once_with(111, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
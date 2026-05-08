from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any

from lib.reachy_conversation_tools import ReachyToolRuntime


class _FakeRobot:
    def __init__(self) -> None:
        self.moves: list[tuple[str, float]] = []

    async def async_play_move(self, move, initial_goto_duration=0.25):
        self.moves.append((type(move).__name__, initial_goto_duration))

    def cancel_move(self) -> None:
        return None


class ReachyConversationToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_dance_runtime_completes_without_llm(self) -> None:
        fake_robot = _FakeRobot()
        runtime = ReachyToolRuntime(
            fake_robot,  # type: ignore[arg-type]
            data_dir=Path(tempfile.mkdtemp()),
            motion_duration_s=0.2,
            chat_base_url="http://spark:8010/v1",
            chat_api_key="dummy",
            chat_model="spark",
            auto_install_optional_deps=False,
        )
        try:
            queued = await runtime.dance("dizzy_spin", 1)
            self.assertEqual(queued["status"], "queued")

            for _ in range(20):
                status = await runtime.task_status(queued["tool_id"])
                if status["status"] != "running":
                    break
                await asyncio.sleep(0.01)

            status = await runtime.task_status(queued["tool_id"])
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["result"]["move"], "dizzy_spin")
            self.assertEqual(fake_robot.moves[0][0], "DanceMove")
        finally:
            await runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from reachy_mini.utils import create_head_pose

from reachy_conversation_tools import ReachyToolRuntime, build_langchain_tools
from reachy_audio import get_wobble_origin_pose


class _FakeRobot:
    def __init__(self) -> None:
        self.moves: list[tuple[str, float]] = []
        self.head_pose = create_head_pose(degrees=False)

    async def async_play_move(self, move, initial_goto_duration=0.25):
        self.moves.append((type(move).__name__, initial_goto_duration))

    def cancel_move(self) -> None:
        return None

    def goto_target(self, *, head, duration, body_yaw=None) -> None:
        del duration, body_yaw
        self.head_pose = np.array(head, copy=True)

    def get_current_head_pose(self) -> np.ndarray:
        return np.array(self.head_pose, copy=True)


class ReachyConversationToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_move_head_updates_speech_wobble_origin_pose(self) -> None:
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
            await runtime.move_head("left")
            origin_pose = get_wobble_origin_pose(fake_robot)
            self.assertIsNotNone(origin_pose)
            if origin_pose is None:
                self.fail("move_head did not store a wobble origin pose")
            self.assertTrue(np.allclose(origin_pose, fake_robot.get_current_head_pose()))
        finally:
            await runtime.shutdown()

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

    async def test_web_search_prefers_tavily_when_api_key_is_set(self) -> None:
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
            with (
                patch.dict(os.environ, {"TAVILY_API_KEY": "secret"}, clear=False),
                patch.object(runtime, "_search_with_tavily", return_value={"status": "ok", "backend": "tavily", "results": []}) as tavily_search,
                patch.object(runtime, "_search_with_duckduckgo", return_value={"status": "ok", "backend": "duckduckgo", "results": []}) as duckduckgo_search,
            ):
                result = await runtime.web_search("Reachy Mini", 3)

            self.assertEqual(result["backend"], "tavily")
            tavily_search.assert_called_once_with("Reachy Mini", 3)
            duckduckgo_search.assert_not_called()
        finally:
            await runtime.shutdown()

    async def test_web_search_falls_back_to_duckduckgo_without_api_key(self) -> None:
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
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(runtime, "_search_with_tavily", return_value={"status": "ok", "backend": "tavily", "results": []}) as tavily_search,
                patch.object(runtime, "_search_with_duckduckgo", return_value={"status": "ok", "backend": "duckduckgo", "results": [{"title": "Reachy", "url": "https://example.com", "snippet": "robot"}]}) as duckduckgo_search,
            ):
                result = await runtime.web_search("Reachy Mini", 3)

            self.assertEqual(result["backend"], "duckduckgo")
            tavily_search.assert_not_called()
            duckduckgo_search.assert_called_once_with("Reachy Mini", 3)
        finally:
            await runtime.shutdown()

    async def test_build_langchain_tools_registers_web_search(self) -> None:
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
            tools = build_langchain_tools(runtime, ["web_search"])
            self.assertIn("web_search", [tool.name for tool in tools])
        finally:
            await runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
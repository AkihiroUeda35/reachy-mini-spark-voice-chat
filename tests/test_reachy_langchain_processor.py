from __future__ import annotations

import argparse
import unittest
from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pipecat.frames.frames import FunctionCallInProgressFrame, FunctionCallResultFrame, LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from apps.conversation.main import ReachyLangChainProcessor


class _StubModel:
    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("Stub model exhausted")
        return self._responses.pop(0)


class ReachyLangChainProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_up_tool_check_converts_text_then_tool_then_final_text(self) -> None:
        tool_invocations: list[str] = []

        async def dummy_tool(city: str) -> dict[str, str]:
            tool_invocations.append(city)
            return {"city": city, "forecast": "sunny"}

        tool = StructuredTool.from_function(coroutine=dummy_tool, name="weather_tool", description="Check weather")
        args = argparse.Namespace(
            chat_model="stub",
            chat_base_url="http://example.invalid/v1",
            chat_api_key="dummy",
            llm_temperature=0.0,
            max_completion_tokens=128,
            system_prompt="You are helpful.",
            max_tool_rounds=4,
        )
        processor = ReachyLangChainProcessor(args, [tool])
        stub_model = _StubModel(
            [
                AIMessage(content="天気を確認しますね。"),
                AIMessage(content="", tool_calls=[{"name": "weather_tool", "args": {"city": "Tokyo"}, "id": "call_weather"}]),
                AIMessage(content="東京は晴れです。"),
                AIMessage(content="No"),
            ]
        )
        processor._model = cast(Any, stub_model)

        frames: list[Any] = []

        async def capture(frame, direction):
            del direction
            frames.append(frame)

        processor.push_frame = capture  # type: ignore[method-assign]

        await processor.process_frame(
            LLMContextFrame(LLMContext(messages=[{"role": "user", "content": "東京の天気を教えて"}])),
            FrameDirection.DOWNSTREAM,
        )

        self.assertEqual(tool_invocations, ["Tokyo"])
        self.assertEqual(len(stub_model.calls), 4)
        self.assertTrue(any(isinstance(frame, LLMFullResponseStartFrame) for frame in frames))
        self.assertTrue(any(isinstance(frame, FunctionCallInProgressFrame) and frame.function_name == "weather_tool" for frame in frames))
        self.assertTrue(any(isinstance(frame, FunctionCallResultFrame) and frame.function_name == "weather_tool" for frame in frames))
        self.assertTrue(any(isinstance(frame, LLMTextFrame) and frame.text == "東京は晴れです。" for frame in frames))
        self.assertFalse(any(isinstance(frame, LLMTextFrame) and frame.text == "天気を確認しますね。" for frame in frames))
        self.assertTrue(any(isinstance(frame, LLMFullResponseEndFrame) for frame in frames))


if __name__ == "__main__":
    unittest.main()
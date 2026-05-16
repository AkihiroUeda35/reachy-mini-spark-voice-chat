from __future__ import annotations

import argparse
import importlib
import sys
import types
import unittest
from typing import Any, cast
from pathlib import Path

CONVERSATION_DIR = Path(__file__).resolve().parents[1] / "apps" / "conversation"
if str(CONVERSATION_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERSATION_DIR))


def _install_pipeline_import_stubs() -> None:
    class _StubFrameProcessor:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def process_frame(self, frame, direction):
            return None

        async def push_frame(self, frame, direction):
            return None

    class _StubChatOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def bind_tools(self, tools):
            del tools
            return self

    class _StubMessage:
        def __init__(self, content="", tool_calls=None, **kwargs) -> None:
            del kwargs
            self.content = content
            self.tool_calls = list(tool_calls or [])

    class _StubStructuredTool:
        def __init__(self, coroutine, name: str, description: str) -> None:
            self._coroutine = coroutine
            self.name = name
            self.description = description

        @classmethod
        def from_function(cls, coroutine, name: str, description: str):
            return cls(coroutine, name, description)

        async def ainvoke(self, arguments: dict[str, Any]):
            return await self._coroutine(**arguments)

    class _StubLLMContext:
        def __init__(self, messages) -> None:
            self.messages = messages

    class _StubLLMContextFrame:
        def __init__(self, context) -> None:
            self.context = context

    class _StubTextFrame:
        def __init__(self, text="", **kwargs) -> None:
            del kwargs
            self.text = text

    class _StubFunctionFrame:
        def __init__(self, function_name="", arguments=None, result=None, **kwargs) -> None:
            del kwargs
            self.function_name = function_name
            self.arguments = arguments
            self.result = result

    def _stub_class(name: str):
        return type(name, (), {"__init__": lambda self, *args, **kwargs: None})

    stub_modules: dict[str, types.ModuleType] = {}

    stub_modules["httpx"] = types.ModuleType("httpx")
    stub_modules["numpy"] = types.ModuleType("numpy")

    langchain_messages = types.ModuleType("langchain_core.messages")
    langchain_messages.AIMessage = _StubMessage
    langchain_messages.AIMessageChunk = _StubMessage
    langchain_messages.HumanMessage = _StubMessage
    langchain_messages.SystemMessage = _StubMessage
    langchain_messages.ToolMessage = _StubMessage
    stub_modules["langchain_core.messages"] = langchain_messages

    langchain_tools = types.ModuleType("langchain_core.tools")
    langchain_tools.BaseTool = _stub_class("BaseTool")
    langchain_tools.StructuredTool = _StubStructuredTool
    stub_modules["langchain_core.tools"] = langchain_tools

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = _StubChatOpenAI
    stub_modules["langchain_openai"] = langchain_openai

    pipecat_frames = types.ModuleType("pipecat.frames.frames")
    pipecat_frames.FunctionCallInProgressFrame = _StubFunctionFrame
    pipecat_frames.FunctionCallResultFrame = _StubFunctionFrame
    pipecat_frames.LLMContextFrame = _StubLLMContextFrame
    pipecat_frames.LLMFullResponseEndFrame = _stub_class("LLMFullResponseEndFrame")
    pipecat_frames.LLMFullResponseStartFrame = _stub_class("LLMFullResponseStartFrame")
    pipecat_frames.LLMTextFrame = _StubTextFrame
    pipecat_frames.EndFrame = _stub_class("EndFrame")
    pipecat_frames.ErrorFrame = _StubTextFrame
    pipecat_frames.OutputAudioRawFrame = _stub_class("OutputAudioRawFrame")
    stub_modules["pipecat.frames.frames"] = pipecat_frames

    pipecat_pipeline = types.ModuleType("pipecat.pipeline.pipeline")
    pipecat_pipeline.Pipeline = _stub_class("Pipeline")
    stub_modules["pipecat.pipeline.pipeline"] = pipecat_pipeline

    pipecat_runner = types.ModuleType("pipecat.pipeline.runner")
    pipecat_runner.PipelineRunner = _stub_class("PipelineRunner")
    stub_modules["pipecat.pipeline.runner"] = pipecat_runner

    pipecat_task = types.ModuleType("pipecat.pipeline.task")
    pipecat_task.PipelineParams = _stub_class("PipelineParams")
    pipecat_task.PipelineTask = _stub_class("PipelineTask")
    stub_modules["pipecat.pipeline.task"] = pipecat_task

    pipecat_context = types.ModuleType("pipecat.processors.aggregators.llm_context")
    pipecat_context.LLMContext = _StubLLMContext
    stub_modules["pipecat.processors.aggregators.llm_context"] = pipecat_context

    pipecat_processor = types.ModuleType("pipecat.processors.frame_processor")
    pipecat_processor.FrameDirection = types.SimpleNamespace(DOWNSTREAM="downstream")
    pipecat_processor.FrameProcessor = _StubFrameProcessor
    stub_modules["pipecat.processors.frame_processor"] = pipecat_processor

    pydantic = types.ModuleType("pydantic")
    pydantic.SecretStr = _stub_class("SecretStr")
    stub_modules["pydantic"] = pydantic

    reachy_mini = types.ModuleType("reachy_mini")
    reachy_mini.ReachyMini = _stub_class("ReachyMini")
    stub_modules["reachy_mini"] = reachy_mini

    reachy_audio = types.ModuleType("reachy_audio")
    reachy_audio.HeadWobbler = _stub_class("HeadWobbler")
    reachy_audio.get_wobble_origin_pose = lambda robot: None
    stub_modules["reachy_audio"] = reachy_audio

    state = types.ModuleType("state")
    state.AssistantSpeechState = _stub_class("AssistantSpeechState")
    state.RuntimeSettings = _stub_class("RuntimeSettings")
    stub_modules["state"] = state

    websockets = types.ModuleType("websockets")
    websockets.connect = lambda *args, **kwargs: None
    stub_modules["websockets"] = websockets

    conversation_tools = types.ModuleType("reachy_conversation_tools")
    conversation_tools.ReachyToolRuntime = _stub_class("ReachyToolRuntime")
    conversation_tools.build_langchain_tools = lambda runtime, enabled_tool_names: []
    stub_modules["reachy_conversation_tools"] = conversation_tools

    for module_name, module in stub_modules.items():
        sys.modules.setdefault(module_name, module)


_install_pipeline_import_stubs()

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import StructuredTool
from pipecat.frames.frames import FunctionCallInProgressFrame, FunctionCallResultFrame, LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from pipeline import ReachyLangChainProcessor


class _StubModel:
    def __init__(self, *, responses: list[AIMessage] | None = None, stream_chunks: list[AIMessageChunk] | None = None):
        self._responses = list(responses or [])
        self._stream_chunks = list(stream_chunks or [])
        self.calls: list[list[Any]] = []
        self.stream_calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("Stub model exhausted")
        return self._responses.pop(0)

    async def astream(self, messages: list[Any]):
        self.stream_calls.append(messages)
        for chunk in self._stream_chunks:
            yield chunk


class ReachyLangChainProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_decision_runs_before_streamed_final_text(self) -> None:
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
        tool_model = _StubModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "weather_tool", "args": {"city": "Tokyo"}, "id": "call_weather"}]),
                AIMessage(content="No"),
            ]
        )
        response_model = _StubModel(
            stream_chunks=[
                AIMessageChunk(content="東京は"),
                AIMessageChunk(content="晴れです。"),
            ]
        )
        processor._tool_model = cast(Any, tool_model)
        processor._response_model = cast(Any, response_model)

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
        self.assertEqual(len(tool_model.calls), 2)
        self.assertEqual(len(response_model.stream_calls), 1)
        self.assertTrue(any(isinstance(frame, LLMFullResponseStartFrame) for frame in frames))
        self.assertTrue(any(isinstance(frame, FunctionCallInProgressFrame) and frame.function_name == "weather_tool" for frame in frames))
        self.assertTrue(any(isinstance(frame, FunctionCallResultFrame) and frame.function_name == "weather_tool" for frame in frames))
        llm_text = "".join(frame.text for frame in frames if isinstance(frame, LLMTextFrame))
        self.assertEqual(llm_text, "東京は晴れです。")
        self.assertFalse(any(isinstance(frame, LLMTextFrame) and frame.text == "天気を確認しますね。" for frame in frames))
        self.assertTrue(any(isinstance(frame, LLMFullResponseEndFrame) for frame in frames))


if __name__ == "__main__":
    unittest.main()
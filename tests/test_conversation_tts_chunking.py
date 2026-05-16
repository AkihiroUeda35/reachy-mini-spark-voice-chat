from __future__ import annotations

import importlib
import sys
import types
import unittest
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
            return self

    def _stub_class(name: str):
        return type(name, (), {"__init__": lambda self, *args, **kwargs: None})

    stub_modules: dict[str, types.ModuleType] = {}

    stub_modules["httpx"] = types.ModuleType("httpx")
    stub_modules["numpy"] = types.ModuleType("numpy")

    langchain_messages = types.ModuleType("langchain_core.messages")
    for name in ("AIMessage", "AIMessageChunk", "HumanMessage", "SystemMessage", "ToolMessage"):
        setattr(langchain_messages, name, _stub_class(name))
    stub_modules["langchain_core.messages"] = langchain_messages

    langchain_tools = types.ModuleType("langchain_core.tools")
    langchain_tools.BaseTool = _stub_class("BaseTool")
    stub_modules["langchain_core.tools"] = langchain_tools

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = _StubChatOpenAI
    stub_modules["langchain_openai"] = langchain_openai

    pipecat_frames = types.ModuleType("pipecat.frames.frames")
    for name in (
        "EndFrame",
        "ErrorFrame",
        "FunctionCallInProgressFrame",
        "FunctionCallResultFrame",
        "LLMContextFrame",
        "LLMFullResponseEndFrame",
        "LLMFullResponseStartFrame",
        "LLMTextFrame",
        "OutputAudioRawFrame",
    ):
        setattr(pipecat_frames, name, _stub_class(name))
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
    pipecat_context.LLMContext = _stub_class("LLMContext")
    stub_modules["pipecat.processors.aggregators.llm_context"] = pipecat_context

    pipecat_processor = types.ModuleType("pipecat.processors.frame_processor")
    pipecat_processor.FrameDirection = _stub_class("FrameDirection")
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

_pipeline = importlib.import_module("pipeline")
_drain_ready_segments = _pipeline._drain_ready_segments
_normalize_tts_text = _pipeline._normalize_tts_text


class ConversationTTSChunkingTests(unittest.TestCase):
    def test_tts_pronunciation_dictionary_replaces_known_words(self) -> None:
        normalized = _normalize_tts_text("清水寺のお側までお願いします。")

        self.assertEqual(normalized, "きよみずでらのおそばまでお願いします。")

    def test_tts_pronunciation_dictionary_leaves_other_text_unchanged(self) -> None:
        normalized = _normalize_tts_text("今日は良い天気ですね。")

        self.assertEqual(normalized, "今日は良い天気ですね。")

    def test_punctuation_does_not_flush_before_thresholds(self) -> None:
        segments, remainder = _drain_ready_segments(
            "こんにちは。元気ですか？はい、元気です。",
            final=False,
            newline_threshold=2,
            min_chars=100,
            max_chars=100,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "こんにちは。元気ですか？はい、元気です。")

    def test_single_newline_does_not_flush_when_threshold_is_two(self) -> None:
        segments, remainder = _drain_ready_segments(
            "一行目\n二行目",
            final=False,
            newline_threshold=2,
            min_chars=100,
            max_chars=100,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "一行目\n二行目")

    def test_consecutive_newlines_flush_segment(self) -> None:
        segments, remainder = _drain_ready_segments(
            "一段落目です。\n\n次の段落",
            final=False,
            newline_threshold=2,
            min_chars=100,
            max_chars=100,
        )

        self.assertEqual(segments, ["一段落目です。"])
        self.assertEqual(remainder, "次の段落")

    def test_max_chars_alone_does_not_flush_without_punctuation(self) -> None:
        segments, remainder = _drain_ready_segments(
            "あいうえおかきくけこ",
            final=False,
            newline_threshold=2,
            min_chars=100,
            max_chars=5,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "あいうえおかきくけこ")

    def test_flushes_when_threshold_and_japanese_period_are_both_met(self) -> None:
        segments, remainder = _drain_ready_segments(
            "あいうえお。かきくけこ",
            final=False,
            newline_threshold=2,
            min_chars=5,
            max_chars=5,
        )

        self.assertEqual(segments, ["あいうえお。"])
        self.assertEqual(remainder, "かきくけこ")

    def test_final_flush_returns_tail(self) -> None:
        segments, remainder = _drain_ready_segments(
            "まだ途中です。",
            final=True,
            newline_threshold=2,
            min_chars=100,
            max_chars=100,
        )

        self.assertEqual(segments, ["まだ途中です。"])
        self.assertEqual(remainder, "")

    def test_english_period_does_not_flush_before_threshold(self) -> None:
        segments, remainder = _drain_ready_segments(
            "Hello world. Next sentence",
            final=False,
            newline_threshold=2,
            min_chars=100,
            max_chars=100,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "Hello world. Next sentence")

    def test_english_max_chars_is_tripled(self) -> None:
        segments, remainder = _drain_ready_segments(
            "abcdefghijklmnopqrstuvwxyz",
            final=False,
            newline_threshold=2,
            min_chars=100,
            max_chars=10,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "abcdefghijklmnopqrstuvwxyz")

    def test_english_flushes_at_period_after_tripled_threshold(self) -> None:
        segments, remainder = _drain_ready_segments(
            "abcdefghijklmnopqrstuvwxyzabcd. trailing",
            final=False,
            newline_threshold=2,
            min_chars=10,
            max_chars=10,
        )

        self.assertEqual(segments, ["abcdefghijklmnopqrstuvwxyzabcd."])
        self.assertEqual(remainder, " trailing")

    def test_closing_paren_flushes_after_threshold(self) -> None:
        segments, remainder = _drain_ready_segments(
            "abcdefghijklmnop)",
            final=False,
            newline_threshold=2,
            min_chars=5,
            max_chars=5,
        )

        self.assertEqual(segments, ["abcdefghijklmnop)"])
        self.assertEqual(remainder, "")

    def test_english_does_not_flush_on_other_punctuation(self) -> None:
        segments, remainder = _drain_ready_segments(
            "Hello, world! How are you?",
            final=False,
            newline_threshold=2,
            min_chars=100,
            max_chars=100,
        )

        self.assertEqual(segments, [])
        self.assertEqual(remainder, "Hello, world! How are you?")

    def test_sentence_punctuation_flushes_once_min_chars_is_reached(self) -> None:
        segments, remainder = _drain_ready_segments(
            "先ほどは少し簡潔すぎましたでしょうか。失礼いたしました。改めてご説明します。",
            final=False,
            newline_threshold=2,
            min_chars=12,
            max_chars=100,
        )

        self.assertEqual(segments, ["先ほどは少し簡潔すぎましたでしょうか。", "失礼いたしました。", "改めてご説明します。"])
        self.assertEqual(remainder, "")

    def test_short_sentence_waits_until_min_chars(self) -> None:
        segments, remainder = _drain_ready_segments(
            "はい。次です。",
            final=False,
            newline_threshold=2,
            min_chars=6,
            max_chars=100,
        )

        self.assertEqual(segments, ["はい。次です。"])
        self.assertEqual(remainder, "")


if __name__ == "__main__":
    unittest.main()
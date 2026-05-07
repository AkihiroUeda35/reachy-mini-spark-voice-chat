from __future__ import annotations

import argparse
from typing import Any, cast

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessageChunk
from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
from pipecat.frames.frames import ErrorFrame, LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from jma_weather_tool import LANGGRAPH_TOOLS


def _extract_chunk_text(chunk) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def build_langgraph_agent(args: argparse.Namespace):
    chat_model_cls = cast(Any, ChatDeepSeek)
    model = chat_model_cls(
        model=args.chat_model,
        api_base=args.chat_base_url,
        api_key=SecretStr(args.chat_api_key),
        temperature=args.llm_temperature,
        max_tokens=args.max_completion_tokens,
        use_responses_api=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).bind_tools(LANGGRAPH_TOOLS, parallel_tool_calls=False)

    return create_react_agent(
        model,
        LANGGRAPH_TOOLS,
        prompt=args.system_prompt,
    )


class LangGraphLLMProcessor(FrameProcessor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(name="LangGraphLLMProcessor")
        self._agent = build_langgraph_agent(args)

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        try:
            await self.push_frame(LLMFullResponseStartFrame(), direction)
            async for chunk, _metadata in self._agent.astream(
                {"messages": frame.context.messages},
                stream_mode="messages",
            ):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = _extract_chunk_text(chunk)
                if text:
                    await self.push_frame(LLMTextFrame(text=text), direction)
            await self.push_frame(LLMFullResponseEndFrame(), direction)
        except Exception as exc:
            await self.push_frame(ErrorFrame(error=f"LangGraph agent failed: {exc}"), direction)

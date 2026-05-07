from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, cast

from langchain_core.messages import AIMessageChunk, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr
from pipecat.frames.frames import ErrorFrame, LLMContextFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from lib.jma_weather_tool import get_jma_weather_tool


@tool
def get_current_time_tool() -> str:
    """Get the current local date and time on this machine."""
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


LANGGRAPH_TOOLS = [
    get_jma_weather_tool,
    get_current_time_tool,
]


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
    ).bind_tools(LANGGRAPH_TOOLS, parallel_tool_calls=True)

    async def call_model(state: MessagesState) -> dict[str, list[Any]]:
        response = await model.ainvoke([
            SystemMessage(content=args.system_prompt),
            *state["messages"],
        ])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", call_model)
    graph.add_node("tools", ToolNode(LANGGRAPH_TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    graph.add_edge("tools", "assistant")
    return graph.compile()


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
                cast(MessagesState, {"messages": frame.context.messages}),
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
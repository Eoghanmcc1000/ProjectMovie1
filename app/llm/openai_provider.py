from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCompletion:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    async def complete_structured(
        self, messages: list[dict[str, str]], schema: type[T]
    ) -> T:
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=schema,
            temperature=0.3,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("LLM returned no structured output")
        return parsed

    async def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ToolCompletion:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3,
        )

        message = response.choices[0].message
        if message.tool_calls:
            tool_calls: list[ToolCall] = []
            for tc in message.tool_calls:
                args = tc.function.arguments or "{}"
                try:
                    parsed_args = json.loads(args)
                except json.JSONDecodeError:
                    parsed_args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=parsed_args,
                    )
                )
            return ToolCompletion(tool_calls=tool_calls)

        return ToolCompletion(content=message.content or "")

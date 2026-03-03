from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.llm.openai_provider import OpenAIProvider, ToolCall
from app.repository.movie_repo import MovieRepository
from app.search.vector_store import VectorStore
from app.services.agent_tools import RetrievalToolResult, retrieve_movies, search_streaming

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are a movie recommendation assistant for a movie database API.

Your job:
1) Understand the user's intent and constraints.
2) Use the `retrieve_movies` tool to fetch evidence.
3) Reply conversationally, but only based on retrieved results.

Tool usage policy:
- Always use `retrieve_movies` before giving movie recommendations or movie facts.
- Use `mode="auto"` by default.
- Populate structured fields when explicit constraints are given:
  - title, genre, year, year_min, year_max, actor, director, min_rating, sort_by, limit
- Populate `semantic_query` when the user asks for mood/tone/vibe/similar/feel/theme-style requests.
- For mixed requests (hard filters + vibe), include BOTH structured filters and `semantic_query` in the same call.
- If results are weak, too broad, or empty, make one follow-up tool call with refined arguments (broaden or tighten as needed).
- Prefer `sort_by="rating"` for "best/top/highest rated" requests.
- Prefer `sort_by="year"` for "latest/newest/recent" requests.
- Otherwise use `sort_by="popularity"`.
- `retrieve_movies` has NO streaming platform information. It cannot tell you what is on Netflix, Hulu, etc.
- When the user mentions a streaming platform (Netflix, Hulu, Prime Video, Disney+), you MUST call `search_streaming` with the matching platform.
- `search_streaming` supports optional `genre`, `year`, `movie_ids`, and `movie_titles` filters. For "comedies on Netflix from 2012", call `search_streaming(platform="netflix", genre="comedy", year=2012)` directly.
- For follow-up streaming questions like "are any of those on Netflix?", pass either `movie_ids` (from a tool observation) or `movie_titles` (from the conversation); we resolve titles to IDs.

Grounding and accuracy rules:
- Only mention movies present in the latest tool observation.
- Do not invent titles, years, ratings, cast, or plot details.
- If no good matches are found, say so clearly and suggest a better query.
- Keep explanations concise and useful.

Response style:
- Friendly, clear, and brief.
- For recommendations, include title, year, short reason, and rating when available.
- If user provided strict constraints, explicitly confirm they were applied."""

RETRIEVE_MOVIES_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_movies",
        "description": (
            "Retrieve movie data using SQL filters, semantic search, or both. "
            "Set mode='auto' unless you have a clear reason to force sql/semantic/hybrid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "genre": {"type": "string"},
                "year": {"type": "integer"},
                "year_min": {"type": "integer"},
                "year_max": {"type": "integer"},
                "actor": {"type": "string"},
                "director": {"type": "string"},
                "min_rating": {"type": "number"},
                "semantic_query": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["auto", "sql", "semantic", "hybrid"],
                    "default": "auto",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["popularity", "rating", "year"],
                    "default": "popularity",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
        },
    },
}


SEARCH_STREAMING_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_streaming",
        "description": (
            "Search for movies available on a specific streaming platform. "
            "Use this when the user mentions Netflix, Hulu, Prime Video, or Disney+. "
            "Supports optional genre and year filters to narrow results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["netflix", "hulu", "prime_video", "disney_plus"],
                },
                "genre": {"type": "string"},
                "year": {"type": "integer"},
                "movie_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Filter to these specific movie IDs (integers only, no titles). Use for follow-up questions like 'are any of those on Netflix?'.",
                },
                "movie_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Movie titles to check (e.g. from the last message). Use for follow-ups like 'are any of those on Netflix?'.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            "required": ["platform"],
        },
    },
}


class GroundedResponse(BaseModel):
    selected_movie_ids: list[int]
    reasoning: str
    response: str


GROUNDING_PROMPT = (
    "You are a strict grounding filter. You receive a user question and a list of "
    "retrieved movies. Your job:\n"
    "1) Select ONLY movie IDs from the provided list that are relevant to the question.\n"
    "2) Explain in 'reasoning' WHY you selected each movie.\n"
    "3) Write a conversational 'response' that ONLY mentions movies from your selected IDs.\n"
    "Do NOT invent any movie not in the provided list."
)


@dataclass
class AgentRunResult:
    response: str
    retrieval: RetrievalToolResult
    reasoning: str | None = None


class MovieAgent:
    def __init__(
        self,
        *,
        llm: OpenAIProvider,
        repo: MovieRepository,
        vector_store: VectorStore,
        db_session: AsyncSession,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._vector_store = vector_store
        self._db_session = db_session
        self._max_results = get_settings().max_retrieval_results

    async def run(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        trace_id: str | None = None,
    ) -> AgentRunResult:
        run_start = time.perf_counter()
        tool_events: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if history:
            messages.extend(history[-MAX_HISTORY_TURNS:])
        messages.append({"role": "user", "content": user_message})

        latest_retrieval = RetrievalToolResult()
        for iteration in range(MAX_ITERATIONS):
            completion = await self._llm.complete_with_tools(
                messages=messages,
                tools=[RETRIEVE_MOVIES_TOOL_SCHEMA, SEARCH_STREAMING_TOOL_SCHEMA],
            )
            if completion.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                            for call in completion.tool_calls
                        ],
                    }
                )
                for call in completion.tool_calls:
                    latest_retrieval = await self._dispatch_tool(call, latest_retrieval)
                    tool_events.append(
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                            "strategy_used": latest_retrieval.strategy_used,
                            "results_found": latest_retrieval.total_found,
                            "latency_ms": round(latest_retrieval.retrieval_time_ms, 1),
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(latest_retrieval.as_observation()),
                        }
                    )
                continue

            if completion.content.strip():
                if not latest_retrieval.movies:
                    result = AgentRunResult(
                        response=completion.content.strip(),
                        retrieval=latest_retrieval,
                    )
                else:
                    result = await self._ground_response(
                        user_message=user_message,
                        retrieval=latest_retrieval,
                        draft_response=completion.content.strip(),
                    )
                self._log_trace(
                    trace_id=trace_id,
                    user_message=user_message,
                    iterations=iteration + 1,
                    tool_events=tool_events,
                    result=result,
                    run_start=run_start,
                )
                return result

        logger.warning("Agent hit max iterations without final response")
        fallback = (
            "I couldn't complete this request confidently. "
            "Please try rephrasing your question."
        )
        result = AgentRunResult(response=fallback, retrieval=latest_retrieval)
        self._log_trace(
            trace_id=trace_id,
            user_message=user_message,
            iterations=MAX_ITERATIONS,
            tool_events=tool_events,
            result=result,
            run_start=run_start,
        )
        return result

    async def _ground_response(
        self,
        *,
        user_message: str,
        retrieval: RetrievalToolResult,
        draft_response: str,
    ) -> AgentRunResult:
        context = json.dumps(retrieval.as_observation()["movies"], indent=2)
        valid_ids = {m.id for m in retrieval.movies}

        try:
            grounded = await self._llm.complete_structured(
                messages=[
                    {"role": "system", "content": GROUNDING_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"User question: {user_message}\n\n"
                            f"Retrieved movies:\n{context}"
                        ),
                    },
                ],
                schema=GroundedResponse,
            )
        except Exception:
            logger.warning("Grounding step failed, falling back to unvalidated response")
            return AgentRunResult(response=draft_response, retrieval=retrieval)

        validated_ids = [mid for mid in grounded.selected_movie_ids if mid in valid_ids]
        invalid_ids = set(grounded.selected_movie_ids) - valid_ids
        if invalid_ids:
            logger.warning("Grounding validator rejected IDs not in context: %s", invalid_ids)

        grounded_movies = [m for m in retrieval.movies if m.id in set(validated_ids)]
        retrieval.movies = grounded_movies
        retrieval.total_found = len(grounded_movies)

        return AgentRunResult(
            response=grounded.response,
            retrieval=retrieval,
            reasoning=grounded.reasoning,
        )

    def _log_trace(
        self,
        *,
        trace_id: str | None,
        user_message: str,
        iterations: int,
        tool_events: list[dict[str, Any]],
        result: AgentRunResult,
        run_start: float,
    ) -> None:
        payload = {
            "trace_id": trace_id,
            "user_message": user_message,
            "iterations": iterations,
            "tool_events": tool_events,
            "final_strategy": result.retrieval.strategy_used,
            "results_found": result.retrieval.total_found,
            "retrieval_time_ms": round(result.retrieval.retrieval_time_ms, 1),
            "total_latency_ms": round((time.perf_counter() - run_start) * 1000, 1),
            "has_reasoning": bool(result.reasoning),
        }
        logger.info("agent_trace %s", json.dumps(payload, ensure_ascii=True))

    async def _dispatch_tool(
        self, call: ToolCall, prev_retrieval: RetrievalToolResult
    ) -> RetrievalToolResult:
        if call.name == "retrieve_movies":
            return await retrieve_movies(
                repo=self._repo,
                vector_store=self._vector_store,
                max_results=self._max_results,
                **call.arguments,
            )
        if call.name == "search_streaming":
            args = dict(call.arguments)
            if prev_retrieval.movies and not args.get("movie_ids"):
                args["movie_ids"] = [m.id for m in prev_retrieval.movies]
            if "movie_titles" in args and args["movie_titles"] is not None:
                args["movie_titles"] = [
                    t for t in args["movie_titles"] if isinstance(t, str)
                ]
            return await search_streaming(
                session=self._db_session,
                **args,
            )
        logger.warning("Unknown tool requested: %s", call.name)
        return RetrievalToolResult()

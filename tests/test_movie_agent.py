from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.movie_repo import MovieRepository
from app.search.vector_store import VectorStore
from app.services.movie_agent import MovieAgent
from tests.conftest import MockLLMProvider


@pytest.mark.asyncio
async def test_agent_runs_tool_then_returns_final_response(
    db_session: AsyncSession,
) -> None:
    agent = MovieAgent(
        llm=MockLLMProvider(),
        repo=MovieRepository(db_session),
        vector_store=VectorStore(),
        db_session=db_session,
    )
    result = await agent.run("Tell me about Inception")
    assert "great movies" in result.response
    assert result.retrieval.total_found >= 1
    assert result.retrieval.strategy_used in {"sql", "sql_fallback"}

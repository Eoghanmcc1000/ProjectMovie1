from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repository.movie_repo import MovieRepository
from app.search.vector_store import VectorStore
from app.services.agent_tools import retrieve_movies


@pytest.mark.asyncio
async def test_sql_retrieval(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    vs = VectorStore()
    result = await retrieve_movies(
        repo=repo,
        vector_store=vs,
        max_results=10,
        genre="Action",
        mode="sql",
    )

    assert result.strategy_used == "sql"
    assert result.total_found == 3
    assert len(result.movies) == 3
    assert result.retrieval_time_ms > 0


@pytest.mark.asyncio
async def test_semantic_fallback_to_sql_when_faiss_unavailable(
    db_session: AsyncSession,
) -> None:
    repo = MovieRepository(db_session)
    vs = VectorStore()
    result = await retrieve_movies(
        repo=repo,
        vector_store=vs,
        max_results=10,
        semantic_query="mind-bending thriller",
        mode="semantic",
    )

    assert result.strategy_used == "sql_fallback"
    assert result.total_found >= 1


@pytest.mark.asyncio
async def test_hybrid_retrieval_without_faiss(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    vs = VectorStore()
    result = await retrieve_movies(
        repo=repo,
        vector_store=vs,
        max_results=10,
        genre="Science Fiction",
        semantic_query="space exploration",
        mode="hybrid",
    )

    assert result.strategy_used == "hybrid"
    assert result.total_found >= 1
    titles = {m.title for m in result.movies}
    assert "Inception" in titles or "The Matrix" in titles or "Interstellar" in titles


@pytest.mark.asyncio
async def test_sql_retrieval_no_results(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    vs = VectorStore()
    result = await retrieve_movies(
        repo=repo,
        vector_store=vs,
        max_results=10,
        genre="Western",
        mode="sql",
    )

    assert result.strategy_used == "sql"
    assert result.total_found == 0
    assert len(result.movies) == 0


@pytest.mark.asyncio
async def test_auto_mode_chooses_semantic_without_structured_filters(
    db_session: AsyncSession,
) -> None:
    repo = MovieRepository(db_session)
    vs = VectorStore()
    result = await retrieve_movies(
        repo=repo,
        vector_store=vs,
        max_results=10,
        semantic_query="movies about dreams",
        mode="auto",
    )
    assert result.strategy_used in {"semantic", "sql_fallback"}



from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import MovieSearchParams
from app.repository.movie_repo import MovieRepository


@pytest.mark.asyncio
async def test_search_by_genre(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(MovieSearchParams(genre="Action"))
    assert total == 3
    titles = {m.title for m in movies}
    assert "Inception" in titles
    assert "The Matrix" in titles
    assert "The Dark Knight" in titles


@pytest.mark.asyncio
async def test_search_by_title(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(MovieSearchParams(title="inception"))
    assert total == 1
    assert movies[0].title == "Inception"


@pytest.mark.asyncio
async def test_search_by_year(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(MovieSearchParams(year=2010))
    assert total == 1
    assert movies[0].title == "Inception"


@pytest.mark.asyncio
async def test_search_by_year_range(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(
        MovieSearchParams(year_min=2008, year_max=2014)
    )
    assert total == 3
    titles = {m.title for m in movies}
    assert "Inception" in titles
    assert "Interstellar" in titles
    assert "The Dark Knight" in titles


@pytest.mark.asyncio
async def test_search_by_actor(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(MovieSearchParams(actor="Leonardo DiCaprio"))
    assert total == 1
    assert movies[0].title == "Inception"


@pytest.mark.asyncio
async def test_search_by_director(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(MovieSearchParams(director="Christopher Nolan"))
    assert total == 3
    titles = {m.title for m in movies}
    assert "Inception" in titles
    assert "Interstellar" in titles
    assert "The Dark Knight" in titles


@pytest.mark.asyncio
async def test_search_by_min_rating(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(MovieSearchParams(min_rating=8.5))
    assert total == 2
    titles = {m.title for m in movies}
    assert "Interstellar" in titles
    assert "The Dark Knight" in titles


@pytest.mark.asyncio
async def test_search_combined_filters(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(
        MovieSearchParams(genre="Action", min_rating=8.4)
    )
    assert total == 2
    titles = {m.title for m in movies}
    assert "Inception" in titles
    assert "The Dark Knight" in titles


@pytest.mark.asyncio
async def test_search_respects_limit(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies, total = await repo.search(MovieSearchParams(limit=2))
    assert total == 5
    assert len(movies) == 2


@pytest.mark.asyncio
async def test_get_by_ids_preserves_order(db_session: AsyncSession) -> None:
    repo = MovieRepository(db_session)
    movies = await repo.get_by_ids([3, 1, 5])
    assert len(movies) == 3
    assert movies[0].id == 3
    assert movies[1].id == 1
    assert movies[2].id == 5



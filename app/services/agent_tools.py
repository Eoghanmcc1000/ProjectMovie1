from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import func as sa_func

from app.models.db import Genre, Movie, MovieGenre, MoviePlatform
from app.models.schemas import MovieSearchParams, MovieSummary
from app.repository.movie_repo import MovieRepository, normalize_genre
from app.search.vector_store import VectorStore

PLATFORM_COLUMNS = {
    "netflix": MoviePlatform.netflix,
    "hulu": MoviePlatform.hulu,
    "prime_video": MoviePlatform.prime_video,
    "disney_plus": MoviePlatform.disney_plus,
}


ToolMode = Literal["auto", "sql", "semantic", "hybrid"]


@dataclass
class RetrievalToolResult:
    movies: list[Movie] = field(default_factory=list)
    strategy_used: str = "none"
    total_found: int = 0
    retrieval_time_ms: float = 0.0

    def as_observation(self) -> dict[str, Any]:
        return {
            "strategy_used": self.strategy_used,
            "total_found": self.total_found,
            "retrieval_time_ms": round(self.retrieval_time_ms, 1),
            "movies": [
                {
                    **MovieSummary.from_movie(movie).model_dump(),
                    "runtime": movie.runtime,
                    "budget": movie.budget,
                    "revenue": movie.revenue,
                }
                for movie in self.movies
            ],
        }


async def retrieve_movies(
    *,
    repo: MovieRepository,
    vector_store: VectorStore,
    max_results: int,
    title: str | None = None,
    genre: str | None = None,
    year: int | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    actor: str | None = None,
    director: str | None = None,
    min_rating: float | None = None,
    semantic_query: str | None = None,
    mode: ToolMode = "auto",
    sort_by: Literal["popularity", "rating", "year"] = "popularity",
    limit: int = 10,
) -> RetrievalToolResult:
    start = time.time()
    safe_limit = max(1, min(limit, max_results))
    strategy = _resolve_mode(
        mode=mode,
        title=title,
        genre=genre,
        year=year,
        year_min=year_min,
        year_max=year_max,
        actor=actor,
        director=director,
        min_rating=min_rating,
        semantic_query=semantic_query,
    )

    if strategy == "sql":
        result = await _run_sql(
            repo=repo,
            title=title,
            genre=genre,
            year=year,
            year_min=year_min,
            year_max=year_max,
            actor=actor,
            director=director,
            min_rating=min_rating,
            sort_by=sort_by,
            limit=safe_limit,
        )
    elif strategy == "semantic":
        result = await _run_semantic(
            repo=repo,
            vector_store=vector_store,
            semantic_query=semantic_query,
            limit=safe_limit,
            fallback_sort_by=sort_by,
        )
    else:
        sql_part = await _run_sql(
            repo=repo,
            title=title,
            genre=genre,
            year=year,
            year_min=year_min,
            year_max=year_max,
            actor=actor,
            director=director,
            min_rating=min_rating,
            sort_by=sort_by,
            limit=safe_limit,
        )
        semantic_part = await _run_semantic(
            repo=repo,
            vector_store=vector_store,
            semantic_query=semantic_query,
            limit=safe_limit,
            fallback_sort_by=sort_by,
        )
        result = _merge_results(sql_part, semantic_part, safe_limit)

    result.retrieval_time_ms = (time.time() - start) * 1000
    return result


def _resolve_mode(
    *,
    mode: ToolMode,
    title: str | None,
    genre: str | None,
    year: int | None,
    year_min: int | None,
    year_max: int | None,
    actor: str | None,
    director: str | None,
    min_rating: float | None,
    semantic_query: str | None,
) -> Literal["sql", "semantic", "hybrid"]:
    if mode in ("sql", "semantic", "hybrid"):
        return mode

    has_structured = any(
        [
            title,
            genre,
            year is not None,
            year_min is not None,
            year_max is not None,
            actor,
            director,
            min_rating is not None,
        ]
    )
    has_semantic = bool(semantic_query and semantic_query.strip())

    if has_structured and has_semantic:
        return "hybrid"
    if has_semantic:
        return "semantic"
    return "sql"


async def _run_sql(
    *,
    repo: MovieRepository,
    title: str | None,
    genre: str | None,
    year: int | None,
    year_min: int | None,
    year_max: int | None,
    actor: str | None,
    director: str | None,
    min_rating: float | None,
    sort_by: Literal["popularity", "rating", "year"],
    limit: int,
) -> RetrievalToolResult:
    params = MovieSearchParams(
        title=title,
        genre=genre,
        year=year,
        year_min=year_min,
        year_max=year_max,
        actor=actor,
        director=director,
        min_rating=min_rating,
        sort_by=sort_by,
        limit=limit,
    )
    movies, total = await repo.search(params)
    return RetrievalToolResult(
        movies=movies,
        strategy_used="sql",
        total_found=total,
    )


async def _run_semantic(
    *,
    repo: MovieRepository,
    vector_store: VectorStore,
    semantic_query: str | None,
    limit: int,
    fallback_sort_by: Literal["popularity", "rating", "year"],
) -> RetrievalToolResult:
    if not vector_store.is_ready:
        fallback = await _run_sql(
            repo=repo,
            title=None,
            genre=None,
            year=None,
            year_min=None,
            year_max=None,
            actor=None,
            director=None,
            min_rating=None,
            sort_by=fallback_sort_by,
            limit=limit,
        )
        fallback.strategy_used = "sql_fallback"
        return fallback

    if not semantic_query:
        return RetrievalToolResult(strategy_used="semantic", total_found=0)

    results = vector_store.search(semantic_query, top_k=limit)
    movie_ids = [r.movie_id for r in results]
    movies = await repo.get_by_ids(movie_ids)
    return RetrievalToolResult(
        movies=movies,
        strategy_used="semantic",
        total_found=len(movies),
    )


def _merge_results(
    sql_result: RetrievalToolResult,
    semantic_result: RetrievalToolResult,
    limit: int,
) -> RetrievalToolResult:
    merged = list(sql_result.movies)
    seen_ids = {movie.id for movie in merged}
    for movie in semantic_result.movies:
        if movie.id in seen_ids:
            continue
        merged.append(movie)
        seen_ids.add(movie.id)
    merged = merged[:limit]
    return RetrievalToolResult(
        movies=merged,
        strategy_used="hybrid",
        total_found=len(merged),
    )


async def search_streaming(
    *,
    session: AsyncSession,
    platform: str,
    genre: str | None = None,
    year: int | None = None,
    movie_ids: list[int] | None = None,
    movie_titles: list[str] | None = None,
    limit: int = 10,
) -> RetrievalToolResult:
    start = time.time()

    col = PLATFORM_COLUMNS.get(platform)
    if col is None:
        return RetrievalToolResult(strategy_used="streaming", total_found=0)

    if movie_titles:
        normalized = [
            str(t).strip().lower()
            for t in movie_titles
            if t is not None and str(t).strip()
        ]
        if normalized:
            stmt = select(Movie.id).where(sa_func.lower(Movie.title).in_(normalized))
            res = await session.execute(stmt)
            resolved_ids = list(res.scalars().all())
            if resolved_ids:
                movie_ids = list(set(movie_ids or []) | set(resolved_ids))

    query = (
        select(Movie)
        .join(MoviePlatform, MoviePlatform.movie_id == Movie.id)
        .where(col == True)  # noqa: E712
    )

    if movie_ids:
        safe_ids = [mid for mid in movie_ids if isinstance(mid, int)]
        if safe_ids:
            query = query.where(Movie.id.in_(safe_ids))

    if genre:
        genre_sub = (
            select(MovieGenre.movie_id)
            .join(Genre, Genre.id == MovieGenre.genre_id)
            .where(sa_func.lower(Genre.name) == normalize_genre(genre))
        )
        query = query.where(Movie.id.in_(genre_sub))

    if year is not None:
        query = query.where(Movie.year == year)

    query = query.order_by(Movie.popularity.desc()).limit(max(1, min(limit, 50)))

    result = await session.execute(query)
    movies = list(result.scalars().unique().all())

    return RetrievalToolResult(
        movies=movies,
        strategy_used="streaming",
        total_found=len(movies),
        retrieval_time_ms=(time.time() - start) * 1000,
    )

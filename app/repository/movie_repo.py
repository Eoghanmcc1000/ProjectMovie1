from __future__ import annotations

import logging

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import CastMember, CrewMember, Genre, Movie, MovieGenre
from app.models.schemas import MovieSearchParams

logger = logging.getLogger(__name__)

GENRE_ALIASES: dict[str, str] = {
    "sci-fi": "science fiction",
    "scifi": "science fiction",
}


def normalize_genre(name: str) -> str:
    return GENRE_ALIASES.get(name.strip().lower(), name.strip().lower())


class MovieRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(self, params: MovieSearchParams) -> tuple[list[Movie], int]:
        """Search movies with dynamic filters. Returns (movies, total_count)."""
        query = select(Movie)
        count_query = select(func.count(Movie.id))

        query, count_query = self._apply_filters(query, count_query, params)

        sort_columns = {
            "popularity": Movie.popularity.desc(),
            "rating": Movie.weighted_rating.desc(),
            "year": Movie.year.desc(),
        }
        query = query.order_by(sort_columns.get(params.sort_by, Movie.popularity.desc()))
        count_result = await self._session.execute(count_query)
        total = count_result.scalar() or 0

        query = query.limit(params.limit)
        result = await self._session.execute(query)
        movies = list(result.scalars().unique().all())

        return movies, total

    async def get_by_ids(self, movie_ids: list[int]) -> list[Movie]:
        if not movie_ids:
            return []
        query = select(Movie).where(Movie.id.in_(movie_ids))
        result = await self._session.execute(query)
        movies = list(result.scalars().unique().all())
        id_order = {mid: i for i, mid in enumerate(movie_ids)}
        movies.sort(key=lambda m: id_order.get(m.id, len(movie_ids)))
        return movies

    def _apply_filters(
        self,
        query: Select,
        count_query: Select,
        params: MovieSearchParams,
    ) -> tuple[Select, Select]:
        if params.title:
            pattern = f"%{params.title}%"
            query = query.where(Movie.title.ilike(pattern))
            count_query = count_query.where(Movie.title.ilike(pattern))

        if params.genre:
            genre_sub = (
                select(MovieGenre.movie_id)
                .join(Genre, Genre.id == MovieGenre.genre_id)
                .where(func.lower(Genre.name) == normalize_genre(params.genre))
            )
            query = query.where(Movie.id.in_(genre_sub))
            count_query = count_query.where(Movie.id.in_(genre_sub))

        if params.year:
            query = query.where(Movie.year == params.year)
            count_query = count_query.where(Movie.year == params.year)

        if params.year_min:
            query = query.where(Movie.year >= params.year_min)
            count_query = count_query.where(Movie.year >= params.year_min)

        if params.year_max:
            query = query.where(Movie.year <= params.year_max)
            count_query = count_query.where(Movie.year <= params.year_max)

        if params.actor:
            actor_sub = select(CastMember.movie_id).where(
                CastMember.name.ilike(f"%{params.actor}%")
            )
            query = query.where(Movie.id.in_(actor_sub))
            count_query = count_query.where(Movie.id.in_(actor_sub))

        if params.director:
            director_sub = (
                select(CrewMember.movie_id)
                .where(CrewMember.name.ilike(f"%{params.director}%"))
                .where(CrewMember.job == "Director")
            )
            query = query.where(Movie.id.in_(director_sub))
            count_query = count_query.where(Movie.id.in_(director_sub))

        if params.min_rating is not None:
            query = query.where(Movie.vote_average >= params.min_rating)
            count_query = count_query.where(Movie.vote_average >= params.min_rating)

        return query, count_query

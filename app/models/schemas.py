from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.db import Movie


# --- Request schemas ---


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = None


class MovieSearchParams(BaseModel):
    title: str | None = None
    genre: str | None = None
    year: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    actor: str | None = None
    director: str | None = None
    min_rating: float | None = Field(None, ge=0, le=10)
    sort_by: Literal["popularity", "rating", "year"] = "popularity"
    limit: int = Field(10, ge=1, le=50)


# --- Response schemas ---


class MovieSummary(BaseModel):
    id: int
    title: str
    year: int | None = None
    vote_average: float
    genres: list[str] = []
    overview: str | None = None
    cast: list[str] = []
    director: str | None = None

    @classmethod
    def from_movie(cls, movie: Movie) -> MovieSummary:
        return cls(
            id=movie.id,
            title=movie.title,
            year=movie.year,
            vote_average=movie.vote_average,
            genres=[g.name for g in movie.genres],
            overview=movie.overview,
            cast=[c.name for c in movie.cast_members[:5]],
            director=next(
                (c.name for c in movie.crew_members if c.job == "Director"),
                None,
            ),
        )


class RetrievalMetadata(BaseModel):
    retrieval_strategy: str
    results_found: int
    retrieval_time_ms: float
    reasoning: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace_id: str
    movies: list[MovieSummary] = []
    metadata: RetrievalMetadata

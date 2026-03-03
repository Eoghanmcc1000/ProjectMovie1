from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    vote_average: Mapped[float] = mapped_column(Float, default=0.0)
    vote_count: Mapped[int] = mapped_column(Integer, default=0)
    popularity: Mapped[float] = mapped_column(Float, default=0.0)
    weighted_rating: Mapped[float] = mapped_column(Float, default=0.0)
    budget: Mapped[int] = mapped_column(Integer, default=0)
    revenue: Mapped[int] = mapped_column(Integer, default=0)
    runtime: Mapped[int | None] = mapped_column(Integer, nullable=True)
    release_date: Mapped[str | None] = mapped_column(String(10), nullable=True)

    genres: Mapped[list[Genre]] = relationship(
        secondary="movie_genres", back_populates="movies", lazy="selectin"
    )
    cast_members: Mapped[list[CastMember]] = relationship(
        back_populates="movie", lazy="selectin"
    )
    crew_members: Mapped[list[CrewMember]] = relationship(
        back_populates="movie", lazy="selectin"
    )
    platforms: Mapped[MoviePlatform | None] = relationship(
        back_populates="movie", uselist=False, lazy="selectin"
    )


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)

    movies: Mapped[list[Movie]] = relationship(
        secondary="movie_genres", back_populates="genres", lazy="selectin"
    )


class MovieGenre(Base):
    __tablename__ = "movie_genres"

    movie_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("movies.id"), primary_key=True
    )
    genre_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("genres.id"), primary_key=True
    )


class CastMember(Base):
    __tablename__ = "cast_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    movie_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("movies.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    character: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cast_order: Mapped[int] = mapped_column(Integer, default=0)

    movie: Mapped[Movie] = relationship(back_populates="cast_members")


class CrewMember(Base):
    __tablename__ = "crew_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    movie_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("movies.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    job: Mapped[str] = mapped_column(String(100), nullable=False)
    department: Mapped[str | None] = mapped_column(String(100), nullable=True)

    movie: Mapped[Movie] = relationship(back_populates="crew_members")


class MoviePlatform(Base):
    __tablename__ = "movie_platforms"

    movie_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("movies.id"), primary_key=True
    )
    netflix: Mapped[bool] = mapped_column(Boolean, default=False)
    hulu: Mapped[bool] = mapped_column(Boolean, default=False)
    prime_video: Mapped[bool] = mapped_column(Boolean, default=False)
    disney_plus: Mapped[bool] = mapped_column(Boolean, default=False)

    movie: Mapped[Movie] = relationship(back_populates="platforms")

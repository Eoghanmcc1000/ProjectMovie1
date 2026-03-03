"""
Data ingestion script for TMDB 5000 dataset.

Parses CSV files, loads into SQLite, generates embeddings, and builds FAISS index.

Usage:
    python -m scripts.ingest --movies data/tmdb_5000_movies.csv --credits data/tmdb_5000_credits.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, func as sa_func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.db import Base, CastMember, CrewMember, Genre, Movie, MovieGenre

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500
TOP_CAST_LIMIT = 10


def parse_json_column(raw: str) -> list[dict]:
    """Safely parse a JSON-encoded CSV column."""
    if not raw or raw.strip() == "":
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse JSON column: %.80s...", raw)
        return []


def safe_int(value: str | None, default: int = 0) -> int:
    """Safely parse a string to int, handling floats and empty strings."""
    if not value or value.strip() == "":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value: str | None, default: float = 0.0) -> float:
    if not value or value.strip() == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def extract_year(release_date: str | None) -> int | None:
    if not release_date or len(release_date) < 4:
        return None
    try:
        return int(release_date[:4])
    except ValueError:
        return None


def load_credits(credits_path: Path) -> dict[int, dict]:
    """Load credits CSV and return a mapping of movie_id -> {cast, crew}."""
    credits_map: dict[int, dict] = {}
    with open(credits_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                movie_id = int(row["movie_id"])
            except (ValueError, KeyError):
                continue
            credits_map[movie_id] = {
                "cast": parse_json_column(row.get("cast", "[]")),
                "crew": parse_json_column(row.get("crew", "[]")),
            }
    logger.info("Loaded credits for %d movies", len(credits_map))
    return credits_map


def ingest_to_database(
    movies_path: Path, credits_path: Path, db_url: str
) -> list[tuple[int, str]]:
    """Parse CSVs and insert into SQLite. Returns list of (movie_id, overview) for embedding."""
    sync_url = db_url.replace("sqlite+aiosqlite", "sqlite")
    engine = create_engine(sync_url, echo=False)
    Base.metadata.create_all(engine)

    credits_map = load_credits(credits_path)
    genre_cache: dict[str, int] = {}
    movie_overviews: list[tuple[int, str]] = []

    with Session(engine) as session:
        with open(movies_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0

            for row in reader:
                try:
                    movie_id = int(row["id"])
                except (ValueError, KeyError):
                    continue

                year = extract_year(row.get("release_date"))
                overview = row.get("overview", "").strip() or None

                runtime_raw = row.get("runtime")
                runtime = safe_int(runtime_raw) if runtime_raw and runtime_raw.strip() else None

                movie = Movie(
                    id=movie_id,
                    title=row.get("title", "Unknown"),
                    year=year,
                    overview=overview,
                    vote_average=safe_float(row.get("vote_average")),
                    vote_count=safe_int(row.get("vote_count")),
                    popularity=safe_float(row.get("popularity")),
                    budget=safe_int(row.get("budget")),
                    revenue=safe_int(row.get("revenue")),
                    runtime=runtime,
                    release_date=row.get("release_date"),
                )
                session.add(movie)

                for g in parse_json_column(row.get("genres", "[]")):
                    genre_name = g.get("name")
                    if not genre_name:
                        continue
                    if genre_name not in genre_cache:
                        genre_obj = Genre(id=g["id"], name=genre_name)
                        session.merge(genre_obj)
                        genre_cache[genre_name] = g["id"]
                    session.add(
                        MovieGenre(movie_id=movie_id, genre_id=genre_cache[genre_name])
                    )

                credits = credits_map.get(movie_id, {})
                for i, c in enumerate(credits.get("cast", [])[:TOP_CAST_LIMIT]):
                    session.add(
                        CastMember(
                            movie_id=movie_id,
                            name=c.get("name", "Unknown"),
                            character=c.get("character"),
                            cast_order=i,
                        )
                    )

                for c in credits.get("crew", []):
                    if c.get("job") == "Director":
                        session.add(
                            CrewMember(
                                movie_id=movie_id,
                                name=c.get("name", "Unknown"),
                                job="Director",
                                department=c.get("department"),
                            )
                        )

                if overview:
                    movie_overviews.append((movie_id, overview))

                count += 1
                if count % BATCH_SIZE == 0:
                    session.flush()
                    logger.info("Processed %d movies...", count)

            session.commit()
            logger.info("Ingested %d movies into database", count)

        _compute_weighted_ratings(session)

    engine.dispose()
    return movie_overviews


def _compute_weighted_ratings(session: Session) -> None:
    """Compute IMDb-style weighted ratings for all movies."""
    c = session.query(sa_func.avg(Movie.vote_average)).scalar() or 0.0

    vote_counts = [
        vc for (vc,) in session.query(Movie.vote_count).all()
    ]
    vote_counts.sort()
    m_index = int(len(vote_counts) * 0.9)
    m = vote_counts[m_index] if vote_counts else 0

    movies = session.query(Movie).all()
    for movie in movies:
        v = movie.vote_count
        r = movie.vote_average
        movie.weighted_rating = (v / (v + m)) * r + (m / (v + m)) * c if (v + m) > 0 else 0.0

    session.commit()
    logger.info(
        "Weighted ratings computed: C=%.2f, m=%d, updated %d movies",
        c, m, len(movies),
    )


def build_faiss_index(
    movie_overviews: list[tuple[int, str]],
    model_name: str,
    index_path: str,
    ids_path: str,
) -> None:
    """Generate embeddings for movie overviews and build a FAISS index."""
    if not movie_overviews:
        logger.warning("No overviews to embed")
        return

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    movie_ids = [mid for mid, _ in movie_overviews]
    texts = [text for _, text in movie_overviews]

    logger.info("Generating embeddings for %d movies...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    np.save(ids_path, np.array(movie_ids))

    logger.info(
        "FAISS index built: %d vectors, %d dimensions -> %s",
        index.ntotal,
        dimension,
        index_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest TMDB 5000 dataset")
    parser.add_argument(
        "--movies", default="data/tmdb_5000_movies.csv", help="Path to movies CSV"
    )
    parser.add_argument(
        "--credits", default="data/tmdb_5000_credits.csv", help="Path to credits CSV"
    )
    args = parser.parse_args()

    settings = get_settings()
    start = time.time()

    logger.info("Starting data ingestion...")
    movie_overviews = ingest_to_database(
        Path(args.movies), Path(args.credits), settings.database_url
    )

    logger.info("Building FAISS index...")
    build_faiss_index(
        movie_overviews,
        settings.embedding_model,
        settings.faiss_index_path,
        settings.movie_ids_path,
    )

    elapsed = time.time() - start
    logger.info("Ingestion complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()

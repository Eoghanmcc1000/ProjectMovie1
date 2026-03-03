"""
Ingest streaming platform availability from MoviesOnStreamingPlatforms CSV.

Only inserts rows for movies that already exist in the DB (matched by title + year).

Usage:
    python -m scripts.ingest_streaming --csv data/MoviesOnStreamingPlatforms.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.db import Base, Movie, MoviePlatform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_streaming(csv_path: Path, db_url: str) -> None:
    sync_url = db_url.replace("sqlite+aiosqlite", "sqlite")
    engine = create_engine(sync_url, echo=False)
    Base.metadata.create_all(engine)

    matched = 0
    skipped = 0

    with Session(engine) as session:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = (row.get("Title") or "").strip()
                year_raw = row.get("Year")
                if not title or not year_raw:
                    skipped += 1
                    continue

                try:
                    year = int(year_raw)
                except ValueError:
                    skipped += 1
                    continue

                movie = (
                    session.query(Movie)
                    .filter(func.lower(Movie.title) == title.lower(), Movie.year == year)
                    .first()
                )
                if movie is None:
                    skipped += 1
                    continue

                existing = session.get(MoviePlatform, movie.id)
                if existing:
                    continue

                session.add(
                    MoviePlatform(
                        movie_id=movie.id,
                        netflix=row.get("Netflix") == "1",
                        hulu=row.get("Hulu") == "1",
                        prime_video=row.get("Prime Video") == "1",
                        disney_plus=row.get("Disney+") == "1",
                    )
                )
                matched += 1

        session.commit()

    engine.dispose()
    logger.info("Streaming ingestion done: %d matched, %d skipped", matched, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest streaming platform data")
    parser.add_argument(
        "--csv",
        default="data/MoviesOnStreamingPlatforms.csv",
        help="Path to streaming platforms CSV",
    )
    args = parser.parse_args()
    settings = get_settings()
    ingest_streaming(Path(args.csv), settings.database_url)


if __name__ == "__main__":
    main()

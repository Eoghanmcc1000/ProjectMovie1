from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    movie_id: int
    score: float


class VectorStore:
    """FAISS-based semantic search over movie overviews."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._movie_ids: np.ndarray | None = None
        self._model: SentenceTransformer | None = None
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def load(self, index_path: str, ids_path: str, model_name: str) -> None:
        index_file = Path(index_path)
        ids_file = Path(ids_path)

        if not index_file.exists() or not ids_file.exists():
            logger.warning(
                "FAISS index or ID mapping not found at %s / %s. "
                "Semantic search will be unavailable. Run ingestion first.",
                index_path,
                ids_path,
            )
            return

        logger.info("Loading FAISS index from %s", index_path)
        self._index = faiss.read_index(str(index_file))
        self._movie_ids = np.load(str(ids_file))

        logger.info("Loading embedding model: %s", model_name)
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning(
                "Failed to load embedding model (%s). "
                "Semantic search disabled; SQL and streaming still available. Error: %s",
                model_name,
                exc,
            )
            self._index = None
            self._movie_ids = None
            self._ready = False
            return

        self._ready = True
        logger.info(
            "Vector store ready: %d vectors indexed", self._index.ntotal
        )

    def search(self, query: str, top_k: int = 10) -> list[VectorSearchResult]:
        if not self._ready or self._model is None or self._index is None:
            logger.warning("Vector store not ready, returning empty results")
            return []

        embedding = self._model.encode([query], normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)

        scores, indices = self._index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._movie_ids):
                continue
            results.append(
                VectorSearchResult(
                    movie_id=int(self._movie_ids[idx]),
                    score=float(score),
                )
            )

        return results

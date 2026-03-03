from __future__ import annotations

from app.config import get_settings
from app.llm.openai_provider import OpenAIProvider
from app.search.vector_store import VectorStore
from app.services.session import SessionManager

_llm: OpenAIProvider | None = None
_vector_store: VectorStore | None = None
_session_mgr: SessionManager | None = None


def startup() -> None:
    """Initialize all components. Called during FastAPI lifespan."""
    global _llm, _vector_store, _session_mgr
    settings = get_settings()

    _llm = OpenAIProvider(settings)
    _vector_store = VectorStore()
    _vector_store.load(
        settings.faiss_index_path, settings.movie_ids_path, settings.embedding_model
    )
    _session_mgr = SessionManager()


def get_llm() -> OpenAIProvider:
    assert _llm is not None, "LLM not initialized. Call startup() first."
    return _llm


def get_vector_store() -> VectorStore:
    assert _vector_store is not None, "VectorStore not initialized. Call startup() first."
    return _vector_store


def get_session_manager() -> SessionManager:
    assert _session_mgr is not None, "SessionManager not initialized. Call startup() first."
    return _session_mgr

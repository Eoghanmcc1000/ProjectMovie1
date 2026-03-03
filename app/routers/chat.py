from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_llm, get_session_manager, get_vector_store
from app.llm.openai_provider import OpenAIProvider
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    MovieSummary,
    RetrievalMetadata,
)
from app.repository.movie_repo import MovieRepository
from app.search.vector_store import VectorStore
from app.services.movie_agent import MovieAgent
from app.services.session import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    llm: OpenAIProvider = Depends(get_llm),
    vector_store: VectorStore = Depends(get_vector_store),
    session_mgr: SessionManager = Depends(get_session_manager),
) -> ChatResponse:
    trace_id = str(uuid.uuid4())
    session_id = session_mgr.get_or_create(request.session_id)
    history = session_mgr.get_history(session_id)

    repo = MovieRepository(db)
    agent = MovieAgent(llm=llm, repo=repo, vector_store=vector_store, db_session=db)

    try:
        result = await agent.run(request.message, history, trace_id=trace_id)
    except Exception:
        logger.exception("Agent failed for trace_id=%s", trace_id)
        raise HTTPException(
            status_code=502,
            detail="Failed to process request. Please try again.",
        )

    session_mgr.add_turn(session_id, "user", request.message)
    session_mgr.add_turn(session_id, "assistant", result.response)

    movie_summaries = [
        MovieSummary.from_movie(m) for m in result.retrieval.movies
    ]

    return ChatResponse(
        response=result.response,
        session_id=session_id,
        trace_id=trace_id,
        movies=movie_summaries,
        metadata=RetrievalMetadata(
            retrieval_strategy=result.retrieval.strategy_used,
            results_found=result.retrieval.total_found,
            retrieval_time_ms=round(result.retrieval.retrieval_time_ms, 1),
            reasoning=result.reasoning,
        ),
    )

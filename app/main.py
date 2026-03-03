from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.config import get_settings
from app.dependencies import startup
from app.routers import chat

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting up — initializing components...")
    startup()
    logger.info("Ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Movie Conversational AI Agent",
    description=(
        "A REST API for a conversational AI agent that answers questions about movies "
        "using hybrid retrieval (SQL + semantic search) and LLM-powered response generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat.router)

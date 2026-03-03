from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db
from app.dependencies import get_llm, get_session_manager, get_vector_store
from app.llm.openai_provider import ToolCall, ToolCompletion
from app.main import app
from app.models.db import Base, CastMember, CrewMember, Genre, Movie, MovieGenre
from app.search.vector_store import VectorStore
from app.services.session import SessionManager


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


TEST_ENGINE = create_async_engine(
    "sqlite+aiosqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = async_sessionmaker(TEST_ENGINE, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with TEST_ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSession() as session:
        await _seed_test_data(session)
        yield session

    async with TEST_ENGINE.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def _seed_test_data(session: AsyncSession) -> None:
    genres = [
        Genre(id=28, name="Action"),
        Genre(id=878, name="Science Fiction"),
        Genre(id=35, name="Comedy"),
        Genre(id=18, name="Drama"),
    ]
    for g in genres:
        session.add(g)

    movies = [
        Movie(
            id=1,
            title="Inception",
            year=2010,
            overview="A thief who steals corporate secrets through dream-sharing technology.",
            vote_average=8.4,
            vote_count=30000,
            popularity=100.0,
            budget=160000000,
            revenue=836800000,
            runtime=148,
            release_date="2010-07-16",
        ),
        Movie(
            id=2,
            title="The Matrix",
            year=1999,
            overview="A computer hacker learns about the true nature of reality.",
            vote_average=8.2,
            vote_count=25000,
            popularity=90.0,
            budget=63000000,
            revenue=463500000,
            runtime=136,
            release_date="1999-03-31",
        ),
        Movie(
            id=3,
            title="Superbad",
            year=2007,
            overview="Two co-dependent high school seniors force themselves to go to different colleges.",
            vote_average=7.0,
            vote_count=5000,
            popularity=40.0,
            budget=20000000,
            revenue=169800000,
            runtime=113,
            release_date="2007-08-17",
        ),
        Movie(
            id=4,
            title="Interstellar",
            year=2014,
            overview="A team of explorers travel through a wormhole in space.",
            vote_average=8.6,
            vote_count=28000,
            popularity=95.0,
            budget=165000000,
            revenue=677500000,
            runtime=169,
            release_date="2014-11-07",
        ),
        Movie(
            id=5,
            title="The Dark Knight",
            year=2008,
            overview="Batman raises the stakes in his war on crime.",
            vote_average=8.5,
            vote_count=27000,
            popularity=85.0,
            budget=185000000,
            revenue=1004600000,
            runtime=152,
            release_date="2008-07-18",
        ),
    ]
    for m in movies:
        session.add(m)

    movie_genres = [
        MovieGenre(movie_id=1, genre_id=28),
        MovieGenre(movie_id=1, genre_id=878),
        MovieGenre(movie_id=2, genre_id=28),
        MovieGenre(movie_id=2, genre_id=878),
        MovieGenre(movie_id=3, genre_id=35),
        MovieGenre(movie_id=4, genre_id=878),
        MovieGenre(movie_id=4, genre_id=18),
        MovieGenre(movie_id=5, genre_id=28),
        MovieGenre(movie_id=5, genre_id=18),
    ]
    for mg in movie_genres:
        session.add(mg)

    cast_data = [
        CastMember(movie_id=1, name="Leonardo DiCaprio", character="Cobb", cast_order=0),
        CastMember(movie_id=1, name="Joseph Gordon-Levitt", character="Arthur", cast_order=1),
        CastMember(movie_id=2, name="Keanu Reeves", character="Neo", cast_order=0),
        CastMember(movie_id=5, name="Christian Bale", character="Batman", cast_order=0),
        CastMember(movie_id=5, name="Heath Ledger", character="Joker", cast_order=1),
    ]
    for c in cast_data:
        session.add(c)

    crew_data = [
        CrewMember(movie_id=1, name="Christopher Nolan", job="Director", department="Directing"),
        CrewMember(movie_id=2, name="Lana Wachowski", job="Director", department="Directing"),
        CrewMember(movie_id=4, name="Christopher Nolan", job="Director", department="Directing"),
        CrewMember(movie_id=5, name="Christopher Nolan", job="Director", department="Directing"),
    ]
    for c in crew_data:
        session.add(c)

    await session.commit()


class MockLLMProvider:
    """Deterministic mock LLM for testing."""

    async def complete_structured(
        self, messages: list[dict[str, str]], schema: type
    ):
        from app.services.movie_agent import GroundedResponse

        if schema is GroundedResponse:
            user_content = ""
            for m in messages:
                if m.get("role") == "user":
                    user_content = m.get("content", "")
            import json

            try:
                movie_data = json.loads(
                    user_content.split("Retrieved movies:\n", 1)[1]
                )
                ids = [m["id"] for m in movie_data]
            except (IndexError, KeyError, json.JSONDecodeError):
                ids = []
            return GroundedResponse(
                selected_movie_ids=ids,
                reasoning="Selected all retrieved movies as relevant.",
                response="Here are some great movies I found for you.",
            )
        return schema.model_validate({})

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> ToolCompletion:
        has_tool_observation = any(m.get("role") == "tool" for m in messages)
        if has_tool_observation:
            return ToolCompletion(
                content="Here are some great movies I found for you."
            )

        user_msg = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_msg = str(message.get("content", "")).lower()
                break

        args: dict[str, object] = {"mode": "auto", "limit": 10}
        if "inception" in user_msg:
            args["title"] = "Inception"
        elif "action" in user_msg:
            args["genre"] = "Action"
        elif "nolan" in user_msg:
            args["director"] = "Christopher Nolan"
        else:
            args["semantic_query"] = user_msg

        return ToolCompletion(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="retrieve_movies",
                    arguments=args,
                )
            ]
        )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def mock_vector_store() -> VectorStore:
    store = VectorStore()
    return store


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    mock_llm = MockLLMProvider()
    mock_vs = VectorStore()
    mock_sm = SessionManager()

    async def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_vector_store] = lambda: mock_vs
    app.dependency_overrides[get_session_manager] = lambda: mock_sm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()

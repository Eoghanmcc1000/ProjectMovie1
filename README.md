# Movie Conversational AI Agent

A REST API for a conversational AI virtual agent that answers questions about movies using hybrid retrieval (SQL + semantic search) and LLM-powered response generation. Built with Python 3.11+, FastAPI, SQLite, FAISS, and OpenAI.

## Approach: Combining Structured Data with LLM

The core idea is that neither pure database queries nor a standalone LLM can solve this well alone. SQL is precise for structured filters (genre, year, director) but can't handle vague requests like "movies with a dark psychological vibe." An LLM can understand natural language but will hallucinate movie facts if left unchecked.

This project combines both through a **ReAct agent loop**:

1. The user's message goes to the LLM alongside tool schemas
2. The LLM decides which tool to call — `retrieve_movies` for structured/semantic search, or `search_streaming` for platform availability
3. The tool fetches real data from SQLite (structured filters) and/or FAISS (semantic similarity over movie overviews)
4. The tool result is returned to the LLM as an observation, which it uses to compose a grounded response
5. A **grounding step** then validates the response — a second LLM call selects only movie IDs that exist in the retrieved set, stripping anything hallucinated

This means the LLM handles intent parsing and natural language generation, while the database and vector index handle factual retrieval. The grounding step closes the loop by ensuring the final response only references movies that were actually retrieved.

## Architecture

```
User ──► POST /chat
              │
              ▼
         MovieAgent (ReAct loop, up to 5 iterations)
              │
              ├─ LLM decides which tool to call
              ├─ retrieve_movies ──► SQL (MovieRepository) + FAISS (VectorStore)
              ├─ search_streaming ──► movie_platforms table
              ├─ Observation returned to LLM → loop or produce final answer
              │
              ▼
         Grounding step: structured LLM call validates movie IDs
              │
              ▼
         JSON response with movies, metadata, trace_id
```

## Quickstart in 60 Seconds

### Prerequisites
- Python 3.11+
- OpenAI API key

### Setup

```bash
# 1. Clone and install
git clone https://github.com/Eoghanmcc1000/ProjectMovie1.git && cd ProjectMovie1
make setup

# 2. Add your OpenAI API key to .env
nano .env

# 3. Download datasets manually into data/
#    TMDB 5000: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
#      → Place tmdb_5000_movies.csv and tmdb_5000_credits.csv in data/
#    Streaming: https://www.kaggle.com/datasets/ruchi798/movies-on-netflix-prime-video-hulu-and-disney
#      → Place MoviesOnStreamingPlatforms.csv in data/

# 4. Ingest TMDB data (~4800 movies into SQLite + FAISS index)
make ingest

# 5. Ingest streaming platform data
make ingest-streaming

# 6. Run the server
make run
```

Interactive API docs (Swagger UI): http://localhost:8000/docs — open in a browser to try the `/chat` endpoint directly, send messages, and inspect responses without needing curl.

## Incomplete Items & Known Limitations

> **Note:** This is a proof-of-concept built under time constraints. The following items were identified as improvements but not implemented due to time. Apologies for the rough edges — I'm happy to discuss these tradeoffs in detail.

- **Manual data ingestion:** The CSV datasets must be downloaded manually from Kaggle and placed in `data/`. Ideally this would be automated via `make download-data` using the Kaggle API with a fallback to clear instructions when credentials are missing. I simply didn't have time to implement this.
- **Two separate ingestion scripts:** Currently `scripts/ingest.py` handles TMDB movies/credits/FAISS and `scripts/ingest_streaming.py` handles streaming platform data. These could be consolidated into a single CLI with subcommands (`ingest core`, `ingest streaming`, `ingest all`) for a cleaner developer experience. The split is functional but adds friction.
- **Ingestion is not idempotent:** Running `make ingest` twice on the same database may cause primary key conflicts or duplicate rows. For a clean re-ingest, run `make clean` first to remove the database and FAISS index, then re-run ingestion. A `--reset` flag or automatic table truncation would be the proper fix.
- **Hallucination metric is heuristic-based:** The evaluation harness (`scripts/eval.py`) checks for hallucinated movie titles by extracting bold-formatted titles from response text. This is an approximation — if the LLM doesn't consistently bold-format titles, the metric may under-report. Results should be read as "heuristic-based hallucination check" rather than an absolute guarantee.
- **No authentication or rate limiting** on the API endpoints.
- **Session storage is in-memory** — sessions are lost on server restart.

## API

### GET /health

Returns `{"status": "ok"}`.

### POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Give me top comedies from 2012"}'
```

Response:

```json
{
  "response": "Here are some top comedies from 2012: ...",
  "session_id": "78c84953-...",
  "trace_id": "b3aad404-...",
  "movies": [
    {
      "id": 82690, "title": "Wreck-It Ralph", "year": 2012, "vote_average": 7.1,
      "genres": ["Adventure", "Animation", "Comedy", "Family"],
      "overview": "A video game villain wants to be a hero...",
      "cast": ["John C. Reilly", "Sarah Silverman", "Jack McBrayer"],
      "director": "Rich Moore"
    }
  ],
  "metadata": {
    "retrieval_strategy": "sql",
    "results_found": 7,
    "retrieval_time_ms": 62.4,
    "reasoning": "I selected these movies because they are all comedies released in 2012..."
  }
}
```

More examples:

```bash
# Follow-up with session context
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Are any of those on Netflix?", "session_id": "78c84953-..."}'

# Semantic query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Recommend movies with a dark psychological vibe"}'

# Director + rating filter
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Best Christopher Nolan movies rated above 8"}'
```

## Data Setup

The ingestion script (`scripts/ingest.py`) parses two TMDB 5000 CSVs into 6 tables:

| Table | Purpose |
|-------|---------|
| `movies` | Title, year, overview, ratings, popularity, weighted rating, runtime |
| `genres` / `movie_genres` | Genre lookup + many-to-many join |
| `cast_members` | Top 10 cast per movie (name, character, order) |
| `crew_members` | Directors only |
| `movie_platforms` | Streaming availability (Netflix, Hulu, Prime Video, Disney+) — ingested separately via `scripts/ingest_streaming.py` |

During ingestion, an **IMDb-style weighted rating** is computed per movie to prevent low-vote-count movies from ranking artificially high.

A **FAISS vector index** is also built: movie overviews are encoded with `all-MiniLM-L6-v2` (384-dim) and stored in an `IndexFlatIP` index for cosine similarity search at query time.

## How It Works

### ReAct Agent

The user's message is sent to the LLM with two tool schemas. The LLM decides which tool to call, receives the result as an observation, and can make follow-up calls to refine. Up to 5 iterations.

### Tools

**`retrieve_movies`** — Accepts structured filters (title, genre, year, actor, director, min_rating, sort_by) and/or a `semantic_query` string. Mode is resolved automatically:
- Structured filters only → SQL
- Semantic query only → FAISS
- Both → hybrid (run both, merge and deduplicate)
- If FAISS is unavailable, falls back to SQL

**`search_streaming`** — Queries `movie_platforms` for a specific platform, with optional genre and year filters.

### Grounding

After the agent produces a draft response, a second structured LLM call selects which movie IDs from the retrieved set are relevant and writes a response using only those. A validator strips any IDs not in the retrieval set. This adds latency (extra LLM call) but reduces hallucination risk — a tradeoff I'd revisit in production (e.g., deterministic templating for low-latency mode).

### Observability

Each response includes `trace_id`, `retrieval_strategy`, `results_found`, `retrieval_time_ms`, and optionally `reasoning` from the grounding step. Structured JSON logs are emitted per request.

### Sessions

Each conversation has a `session_id`. An in-memory `SessionManager` stores the last 10 turns and passes them as history for multi-turn context.

## LLM Provider

`OpenAIProvider` is a concrete class with two methods: `complete_structured()` and `complete_with_tools()`. It runs against `gpt-4o-mini` by default. Swapping providers works via duck typing — as demonstrated by `MockLLMProvider` in the test suite — with no changes to agent code.

## Project Structure

```
app/
├── main.py               # FastAPI app, lifespan
├── config.py              # pydantic-settings
├── database.py            # Async SQLAlchemy engine
├── dependencies.py        # Singleton init
├── models/
│   ├── db.py              # ORM models
│   └── schemas.py         # Request/response schemas
├── routers/
│   └── chat.py            # /health, /chat endpoints
├── services/
│   ├── movie_agent.py     # ReAct loop, grounding, tracing
│   ├── agent_tools.py     # retrieve_movies, search_streaming
│   └── session.py         # In-memory session store
├── llm/
│   └── openai_provider.py
├── repository/
│   └── movie_repo.py      # SQL query builder
└── search/
    └── vector_store.py    # FAISS index
scripts/
├── ingest.py              # TMDB CSV → SQLite + FAISS
├── ingest_streaming.py    # Streaming CSV → movie_platforms
└── eval.py                # Evaluation harness
tests/                     # Unit/integration tests
data/                      # DB, index, CSVs, eval queries
Makefile                   # setup, ingest, run, test, eval, lint, clean
requirements.txt           # Python dependencies
.env.example               # Environment variable template
```

## Testing

```bash
make test    # 28 tests, no API key needed
```

Tests use an in-memory SQLite database seeded with 5 movies and a `MockLLMProvider` that returns deterministic responses. FastAPI's `dependency_overrides` swaps in the mocks.

### Evaluation

An evaluation harness (`scripts/eval.py`) runs 16 predefined queries against the live server and measures strategy accuracy, hallucination rate (heuristic-based), and retrieval latency.

```bash
# Start the server first, then:
make eval
```

Results from the last run (16 queries covering SQL, semantic, hybrid, and streaming):

| Metric | Result |
|--------|--------|
| Strategy accuracy | 100% (all queries used the expected retrieval path) |
| Hallucination rate | 0% (heuristic-based — checks bold-formatted titles against retrieved set; see limitations above) |
| Reasoning rate | 81% (13/16 responses included grounding reasoning) |

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make setup` | Create venv, install dependencies, copy `.env.example` |
| `make ingest` | Ingest TMDB movies + credits into SQLite and build FAISS index |
| `make ingest-streaming` | Ingest streaming platform availability data |
| `make run` | Start the FastAPI server on port 8000 |
| `make test` | Run the test suite (28 tests) |
| `make eval` | Run evaluation harness against live server |
| `make lint` | Run ruff linter and formatter checks |
| `make clean` | Remove database, FAISS index, and `__pycache__` dirs |

## Configuration

All via environment variables (`.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/movies.db` | SQLite path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer |
| `FAISS_INDEX_PATH` | `./data/faiss.index` | FAISS index file |
| `MOVIE_IDS_PATH` | `./data/movie_ids.npy` | NumPy array mapping FAISS index positions to movie IDs |
| `MAX_RETRIEVAL_RESULTS` | `10` | Max movies per query |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
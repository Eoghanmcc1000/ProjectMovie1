.PHONY: setup ingest ingest-streaming run test eval lint clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	cp -n .env.example .env 2>/dev/null || true
	@echo "Setup complete. Edit .env with your API keys, then run: make ingest"

ingest:
	. venv/bin/activate && python -m scripts.ingest \
		--movies data/tmdb_5000_movies.csv \
		--credits data/tmdb_5000_credits.csv

ingest-streaming:
	. venv/bin/activate && python -m scripts.ingest_streaming \
		--csv data/MoviesOnStreamingPlatforms.csv

run:
	. venv/bin/activate && uvicorn app.main:app --reload --port 8000

test:
	. venv/bin/activate && python -m pytest tests/ -v

eval:
	. venv/bin/activate && python -m scripts.eval \
		--cases data/eval_queries.json \
		--base-url http://127.0.0.1:8000

lint:
	. venv/bin/activate && ruff check . && ruff format --check .

clean:
	rm -f data/movies.db data/faiss.index data/movie_ids.npy
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

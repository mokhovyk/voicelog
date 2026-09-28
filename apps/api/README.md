# API

The voicelog FastAPI service. It currently provides a `/health` liveness endpoint
and text notes stored in PostgreSQL, enriched by OpenAI through LangChain. Audio
and search follow the root README's implementation order.

## Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness check; does not touch the database. |
| `POST` | `/api/v1/notes` | Create a text note: `raw_transcript`, optional `category` and `tags`. |
| `GET` | `/api/v1/notes` | Newest first; `category`, `limit` (1-100, default 20), `offset`. |
| `GET` | `/api/v1/notes/{id}` | One note, or 404. |

## AI enrichment

When a note is created, two OpenAI calls run concurrently on the transcript:

- **Extraction** (`app/ai/extraction.py`, `OPENAI_CHAT_MODEL`): summary, category
  (`Work`, `Personal`, `Ideas`, `Other`), up to 5 tags, and action items.
- **Embedding** (`app/ai/embeddings.py`, `OPENAI_EMBEDDING_MODEL`): a 1536-dimension
  vector stored in `notes.embedding` for semantic search. It is not returned by the API.

A `category` or `tags` sent by the client take precedence over extracted values.
If `OPENAI_API_KEY` is empty or a call fails, the failure is logged and the note is
still saved with that part empty. Notes with `embedding IS NULL` can be reprocessed
once background processing exists.

Tests never call OpenAI: they use fakes, or run the real clients against a mocked
OpenAI HTTP API.

## Development

Prerequisite: [uv](https://docs.astral.sh/uv/). It installs Python 3.12 from
`.python-version` if needed.

Start the database first (from the repository root, see the root README), then:

```sh
cp .env.example .env
uv sync
uv run alembic upgrade head        # apply migrations
uv run fastapi dev app/main.py     # http://127.0.0.1:8000, docs at /docs
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests need the local database: they drop and recreate a separate `voicelog_test`
database and migrate it with Alembic. Override it with `TEST_DATABASE_URL`.

After changing models, generate a migration and review it before applying:

```sh
uv run alembic revision --autogenerate -m "describe the change"
uv run alembic check               # fails if models and migrations disagree
```

Container build:

```sh
docker build -t voicelog-api .
docker run --rm -p 8000:8000 -e ENVIRONMENT=production -e DATABASE_URL=... voicelog-api
```

The image includes the migrations; run `alembic upgrade head` in it as the deploy
release step, before starting the new version.

Settings are read from environment variables and `.env` (see `app/core/config.py`).
Interactive docs are disabled when `ENVIRONMENT=production`.

## Structure

Code is grouped by feature rather than by layer:

```text
apps/api/
├── app/
│   ├── main.py      # app factory; mounts routers under /api/v1
│   ├── core/        # settings, async database session, logging
│   ├── notes/       # router, models, schemas, service for note CRUD and upload
│   ├── ai/          # metadata extraction and embeddings (transcription later)
│   └── search/      # router, schemas, service for hybrid retrieval and answers
├── migrations/      # Alembic migrations
└── tests/           # API, service, and PostgreSQL integration tests
```

`app/health.py` holds the liveness route. `core/`, `notes/`, and `ai/` exist;
`search/` is added with retrieval.

The first migration enables `vector` itself, so it works on managed databases
without the local init script in `infra/postgres/`. The `embedding` column and its
HNSW cosine index are filled when notes are created with an OpenAI key.

See the [root README](../../README.md) to start the local database and the
[specification](../../docs/specification.md) for the API requirements. The
specification's layer-based paths (`models/`, `schemas/`, `services/`) map onto
the feature modules above.

# API

The voicelog FastAPI service. It currently provides a `/health` liveness endpoint
and text notes stored in PostgreSQL, enriched by OpenAI through LangChain. Audio
and search follow the root README's implementation order.

## Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Liveness check; does not touch the database. |
| `GET` | `/health/ready` | Readiness check: 200 if the database answers, else 503. |
| `POST` | `/api/v1/notes` | Create a text note: `raw_transcript`, optional `category` and `tags`. Returns it as `pending`. |
| `GET` | `/api/v1/notes` | Newest first; `category`, `tag`, `created_after`, `created_before` (with timezone), `limit` (1-100, default 20), `offset`. |
| `GET` | `/api/v1/notes/{id}` | One note, or 404. |

Categories are `Work`, `Personal`, `Ideas`, `Other` (any casing is accepted; other
values are rejected). Tags are stored lowercase.

## AI enrichment

A note is saved and returned immediately with `status: "pending"`. After the
response is sent, a background task runs two OpenAI calls concurrently on the
transcript:

- **Extraction** (`app/ai/extraction.py`, `OPENAI_CHAT_MODEL`): summary, category
  (`Work`, `Personal`, `Ideas`, `Other`), up to 5 tags, and action items.
- **Embedding** (`app/ai/embeddings.py`, `OPENAI_EMBEDDING_MODEL`): a 1536-dimension
  vector stored in `notes.embedding` for semantic search, with the model name in
`notes.embedding_model`. Neither is returned by the API.

The note then becomes `ready`, or `failed` if either call failed (the failure is
logged and the part that succeeded is kept). Without `OPENAI_API_KEY` it stays
`pending`. `pending` and `failed` notes are the ones to reprocess, as are notes whose
`embedding_model` differs from the configured one. A `category` or `tags` sent by
the client take precedence over extracted values.

Background tasks run in the API process, so enrichment in progress is lost if the
process stops; the note remains `pending`. Move to a job queue when that matters.

Tests never call OpenAI: they use fakes, or run the real clients against a mocked
OpenAI HTTP API.

## Development

Prerequisite: [uv](https://docs.astral.sh/uv/). It installs Python 3.12 from
`.python-version` if needed.

Start the database first (from the repository root, see the root README). The root
`Makefile` wraps the commands below (`make help`); to run them directly:

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

The image includes the migrations but does not run them on startup, so several
instances never race to migrate. Run them from the same image as the platform's
release (pre-deploy) command, before the new version starts serving:

```sh
docker run --rm -e DATABASE_URL=... voicelog-api alembic upgrade head
```

Point the platform's health check at `/health/ready`.

Settings are read from environment variables and `apps/api/.env`, wherever the app
is started from (see `app/core/config.py`). With `ENVIRONMENT=production`,
interactive docs are disabled and startup fails unless `DATABASE_URL` and
`OPENAI_API_KEY` are set.

## Structure

Code is grouped by feature rather than by layer:

```text
apps/api/
├── app/
│   ├── main.py      # app factory; mounts routers under /api/v1
│   ├── core/        # settings, async database session, logging
│   ├── notes/       # router, models, schemas, service for note CRUD and upload
│   ├── ai/          # OpenAI adapters: extraction, embeddings; later transcription, answers
│   └── search/      # router, schemas, service for hybrid retrieval and answers
├── migrations/      # Alembic migrations
└── tests/           # API, service, and PostgreSQL integration tests
```

`app/health.py` holds the health routes. `core/`, `notes/`, and `ai/` exist;
`search/` is added with retrieval. Feature packages may import `ai/` and `core/`;
`ai/` imports neither `notes/` nor `search/`, so it stays reusable by both. Audio
upload belongs in `notes/` (transcribe, then the existing create flow); the
`/api/v1/query/*` endpoints belong in `search/`.

The first migration enables `vector` itself, so it works on managed databases
without the local init script in `infra/postgres/`. The `embedding` column and its
HNSW cosine index are filled when notes are created with an OpenAI key.

See the [root README](../../README.md) to start the local database and the
[specification](../../docs/specification.md) for the API requirements. The
specification's layer-based paths (`models/`, `schemas/`, `services/`) map onto
the feature modules above.

# voicelog

Voice journal and semantic memory search, organized as a monorepo.

The repository currently contains the directory scaffold and local database
configuration. The FastAPI API stores text and transcribed audio notes in PostgreSQL,
enriches them with AI metadata and embeddings, searches them semantically, and
answers typed or spoken questions citing the notes used; the Next.js web app is not implemented yet.

## Repository layout

```text
voicelog/
├── apps/
│   ├── api/                # FastAPI service
│   └── web/                # Next.js app
├── infra/
│   └── postgres/           # local database init scripts
├── docs/
│   └── specification.md
├── docker-compose.yml
└── .env.example            # local database only
```

- [API](apps/api/README.md): FastAPI, async SQLAlchemy, Alembic, and AI services.
- [Web](apps/web/README.md): Next.js recording, journal, and search interface.
- [Specification](docs/specification.md): original product requirements, preserved
  as supplied. Its backend paths will live under `apps/api/` in this monorepo.

Each app has its own `.env.example`; copy it as described in the app README. The
root `.env` is the source of truth for local database credentials: it creates the
database container, and `DATABASE_URL` in `apps/api/.env` must match it. The
defaults in `app/core/config.py` and `tests/conftest.py` match the example values.

Each app will own its dependencies, lockfile, tests, and deployment configuration.
Root files coordinate local development. No monorepo build tool is needed yet;
the root `Makefile` wraps the common commands (`make help`):

```sh
make db-up migrate   # start PostgreSQL and apply migrations
make dev             # run the API with reload
make check           # lint, tests, and the models-vs-migrations check
```

## Local database

Prerequisite: a running Docker engine with Docker Compose.

From the repository root, create your local configuration once:

```sh
cp .env.example .env
```

Validate the configuration and start PostgreSQL:

```sh
docker compose config --quiet
docker compose up -d --wait db
```

PostgreSQL is available on `127.0.0.1:5432` by default. Change `POSTGRES_PORT` in
`.env` if that port is already in use. These credentials are for local development.

The [pgvector image](https://github.com/pgvector/pgvector#docker) contains the
extension; the initialization SQL enables it when the data volume is first created.
Check it with:

```sh
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dx vector"'
```

Stop the database while retaining its data:

```sh
docker compose down
```

Data persists in a Docker volume. Initialization SQL runs only for a new volume;
the schema itself comes from Alembic migrations (see the [API README](apps/api/README.md)).

## Implementation order

1. Bootstrap FastAPI on Python 3.12 with configuration and health endpoints.
2. Add async SQLAlchemy, Alembic, and the notes schema; implement text-note creation,
   listing, filtering, and detail endpoints.
3. Add LangChain metadata extraction and 1,536-dimensional embeddings.
4. Add audio transcription, semantic retrieval, and answers with source notes.
5. Bootstrap Next.js and implement recording, note history, and text/voice search.
6. Add CI and prepare authenticated deployments.

Completed audio uploads are the first recording workflow. Streaming, background
processing, and local Whisper support are later milestones. Audio retention and
playback storage must be decided before implementing playback.

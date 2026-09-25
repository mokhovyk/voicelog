# API

This directory will contain the voicelog FastAPI service. It currently contains
only this README and `.env.example`.

## Planned structure

Code is grouped by feature rather than by layer:

```text
apps/api/
├── app/
│   ├── main.py      # app factory; mounts routers under /api/v1
│   ├── core/        # settings, async database session, logging
│   ├── notes/       # router, models, schemas, service for note CRUD and upload
│   ├── search/      # router, schemas, service for hybrid retrieval and answers
│   └── ai/          # transcription, metadata extraction, embeddings clients
├── migrations/      # Alembic, created with `alembic init migrations`
└── tests/           # API, service, and PostgreSQL integration tests
```

The next step is to add a Python 3.12 environment, `pyproject.toml`, a dependency
lockfile, `app/main.py`, and a Dockerfile. Dependency installation and API startup
commands will be documented here when those files exist.

The first Alembic migration should enable `vector` with `CREATE EXTENSION IF NOT
EXISTS vector` and create the notes table and cosine index. It must work on a fresh
database independently of the local init script in `infra/postgres/`.

## Configuration

```sh
cp .env.example .env
```

See the [root README](../../README.md) to start the local database and the
[specification](../../docs/specification.md) for the API requirements. The
specification's layer-based paths (`models/`, `schemas/`, `services/`) map onto
the feature modules above.

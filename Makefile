# Common local-development commands. Run from the repository root.
API := uv --directory apps/api run

.PHONY: help db-up db-down migrate dev test lint format check docker-api

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

db-up: ## Start PostgreSQL + pgvector (needs .env)
	docker compose up -d --wait db

db-down: ## Stop PostgreSQL, keeping its data
	docker compose down

migrate: ## Apply API migrations to the local database
	$(API) alembic upgrade head

dev: ## Run the API with reload on http://127.0.0.1:8000
	$(API) fastapi dev app/main.py

test: ## Run API tests (recreates the voicelog_test database)
	$(API) pytest

lint: ## Lint and check formatting
	$(API) ruff check .
	$(API) ruff format --check .

format: ## Fix lint issues and format
	$(API) ruff check --fix .
	$(API) ruff format .

check: lint test ## Everything CI should run; also verifies models match migrations
	$(API) alembic check

docker-api: ## Build the API image
	docker build -t voicelog-api apps/api

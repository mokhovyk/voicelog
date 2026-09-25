# voicelog — Hands-Free Voice Journal & Semantic Memory Search

voicelog is an asynchronous voice-first journal and semantic memory assistant. It records audio, transcribes it on the fly, uses LangChain to extract structured metadata and action items, stores vector embeddings in PostgreSQL via `pgvector`, and allows users to query past notes using voice or text.

---

## Technical Stack Overview

| Component | Technology | Role & Implementation Details |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Asynchronous runtime (`asyncio`), Pydantic validation, typing. |
| **Backend API** | FastAPI | Async HTTP endpoints, WebSocket streaming for audio, OpenAPI documentation. |
| **Voice Recognition** | Whisper API / `faster-whisper` | Converts incoming audio files (`.wav`, `.m4a`, `.mp3`) into plain text. |
| **Orchestration & LLM** | LangChain | Structured metadata extraction (`PydanticOutputParser`), hybrid search routing, answer synthesis. |
| **Database** | PostgreSQL + `pgvector` | Dual-purpose storage: relational note metadata + vector search using cosine similarity (`<->` / `<=>`). |
| **ORM & Migrations** | SQLAlchemy 2.0 (Async) + Alembic | Async database interface (`asyncpg`) and migration management. |
| **Frontend** | React / Next.js (PWA) | Microphone recording via Web Audio API (`navigator.mediaDevices`), audio playback, responsive UI. |

---

## High-Level Architecture

```text
                               ┌──────────────────────────────────────────────┐
                               │             Frontend (PWA / Web)             │
                               │  - Audio Recording (Web Audio API)          │
                               │  - Stream / Upload Audio Blobs               │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                          HTTP POST / WebSockets
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             FastAPI Backend                                             │
│                                                                                                         │
│  ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────────────────────┐  │
│  │    Audio Endpoint    │ ───► │  Whisper STT Engine  │ ───► │          LangChain Pipeline          │  │
│  │  Receive Audio File  │      │  Audio -> Transcript │      │ - Structured Extraction (Pydantic)   │  │
│  └──────────────────────┘      └──────────────────────┘      │ - Vector Embedding Generation        │  │
│                                                              └──────────────────┬───────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┼───────────────────────┘
                                                                                  │
                                                                       Async Database Write
                                                                                  │
                                                                                  ▼
                                                              ┌───────────────────────────────────────┐
                                                              │         PostgreSQL Database           │
                                                              │ - Table: `notes` (Metadata & JSON)    │
                                                              │ - Extension: `pgvector` (Embeddings)  │
                                                              └───────────────────────────────────────┘
```

---

## Data Pipeline & Workflows

### 1. Audio Ingestion & Extraction Workflow
1. **Record:** User records a voice entry in the web interface (e.g., *"Had a sync with Sarah about the release. Need to fix the login bug by Friday."*).
2. **Upload:** Client posts the raw audio file to `/api/v1/notes/upload`.
3. **Transcribe:** FastAPI delegates the audio stream to Whisper STT to obtain raw text.
4. **Structure (LangChain):** A chain parses the raw transcript using a structured Pydantic schema:
   * `summary`: High-level summary of the note.
   * `category`: Categorization (e.g., *Work*, *Personal*, *Ideas*).
   * `action_items`: Extracted list of actionable tasks.
   * `tags`: Array of relevant keyword tags.
5. **Embed & Store:** LangChain creates a vector embedding of the transcript. The note metadata, structured JSON, and vector embedding are written to PostgreSQL.

### 2. Voice Query & Retrieval Workflow
1. **Query:** User asks a voice question (e.g., *"What bug did I need to fix for Sarah?"*).
2. **STT & Embedding:** System transcribes the query and converts it into a vector embedding.
3. **Hybrid Search:** LangChain executes a hybrid search against Postgres:
   * **Vector Search:** Cosine similarity match on embedded transcripts via `pgvector`.
   * **SQL Search:** Filtering by date ranges, categories, or action item status.
4. **Synthesis:** LangChain formats context retrieved from Postgres and generates a coherent answer.

---

## Database Schema Design

### Extension Setup
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Table Structure (`notes`)
```sql
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_transcript TEXT NOT NULL,
    summary TEXT,
    category VARCHAR(50),
    action_items JSONB DEFAULT '[]'::jsonb,
    tags VARCHAR(50)[],
    embedding vector(1536), -- Vector size matching OpenAI text-embedding-3-small
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for fast vector similarity search
CREATE INDEX idx_notes_embedding ON notes 
USING hnsw (embedding vector_cosine_ops);
```

---

## Core API Specification

| Endpoint | Method | Input | Description |
| :--- | :--- | :--- | :--- |
| `/api/v1/notes/upload` | `POST` | `multipart/form-data` (audio file) | Uploads audio, transcribes, extracts metadata, embeds, and saves to database. |
| `/api/v1/notes/` | `GET` | Query parameters (`category`, `limit`, `offset`) | Returns paginated list of recorded notes. |
| `/api/v1/notes/{id}` | `GET` | Note UUID | Fetches detailed note record including extracted action items. |
| `/api/v1/query/text` | `POST` | `{"query": "string"}` | Executes semantic hybrid search on past notes and synthesizes an answer. |
| `/api/v1/query/voice` | `POST` | `multipart/form-data` (audio query) | Transcribes voice query and runs semantic search pipeline. |

---

## Project Directory Layout

```text
voicelog/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── audio.py        # Voice processing endpoints
│   │   │   │   ├── notes.py        # Note CRUD operations
│   │   │   │   └── search.py       # Query and semantic search routes
│   │   │   └── router.py
│   ├── core/
│   │   ├── config.py               # Environment configuration (Pydantic Settings)
│   │   └── database.py             # Async SQLAlchemy session setup
│   ├── models/
│   │   └── note.py                 # SQLAlchemy ORM models with pgvector column
│   ├── schemas/
│   │   └── note.py                 # Pydantic models for input/output validation
│   ├── services/
│   │   ├── stt.py                  # Whisper integration service
│   │   └── langchain_pipeline.py   # LangChain extraction & retrieval logic
│   └── main.py                     # FastAPI application factory
├── migrations/                     # Alembic database migration scripts
├── docker-compose.yml              # Local PostgreSQL + pgvector environment
├── Dockerfile                      # Application container definition
├── requirements.txt                # Python dependencies
└── README.md                       # Setup instructions
```

---

## Deployment Architecture

* **Database:** **Supabase** or **Neon** (Free Tier managed Postgres with native `pgvector` support).
* **Backend API:** **Render** or **Railway** (Containerized FastAPI app runtime; standard 100s timeout handles long audio/LLM operations).
* **Frontend Web Client:** **Vercel** or **Netlify** (Free tier static/Next.js hosting with HTTPS support required for microphone permissions).
* **Audio Processing:** OpenAI Whisper API / local `faster-whisper` fallback.

---

## Roadmap & Milestone Execution

* [ ] **Phase 1: Foundation**
  * Set up Docker environment with PostgreSQL and `pgvector`.
  * Build async SQLAlchemy connection and Alembic migrations.
* [ ] **Phase 2: Core Audio & STT**
  * Create FastAPI `/api/v1/notes/upload` endpoint.
  * Integrate Whisper STT engine for transcription.
* [ ] **Phase 3: LangChain & Intelligence**
  * Implement structured Pydantic extraction chain.
  * Store text embeddings in `pgvector` columns.
  * Build semantic query execution pipeline.
* [ ] **Phase 4: Frontend Development**
  * Build Web Audio API recorder component in React/Next.js.
  * Connect frontend recorder to FastAPI endpoints.
* [ ] **Phase 5: Deployment**
  * Deploy database on Supabase/Neon.
  * Deploy backend on Render/Railway.
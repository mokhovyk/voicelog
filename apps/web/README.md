# Web

This directory is reserved for the voicelog Next.js application.

`create-next-app` refuses to run in a directory that contains unknown files, so
when bootstrapping, generate the app elsewhere and move it in, or move this README
and `.env.example` out first and restore them afterwards.

The web app will own its JavaScript dependencies and deploy separately from the
API. OpenAI credentials stay in the API.

## Configuration

```sh
cp .env.example .env.local
```

See the [root README](../../README.md) for the implementation order and the
[specification](../../docs/specification.md) for the product requirements.

## Implementation plan

### What the API offers

| Endpoint | UI use | Notes |
| :--- | :--- | :--- |
| `POST /api/v1/notes` | Typed note | Returned as `pending`; summary, category, tags, and action items arrive from a background task. |
| `POST /api/v1/notes/upload` | Recorded note | Waits for transcription. Accepts `audio/webm` (Chrome, Firefox) and `audio/mp4` (Safari), up to 25 MB. Audio is not stored, so there is no playback. |
| `GET /api/v1/notes` | Journal | Filters by `category`, `tag`, `created_after`, `created_before`; `limit` (≤ 100) and `offset` paging with `total`. |
| `GET /api/v1/notes/{id}` | Note detail | `status` is `pending`, `ready`, or `failed`. |
| `POST /api/v1/search` | Find notes | Scored notes, no answer. |
| `POST /api/v1/query/text`, `/query/voice` | Ask | `answer` (null when nothing matched) and the cited `sources`; voice also returns the transcribed `query`. |

There is no authentication yet. The API's CORS defaults allow `localhost:3000`.

### Stack

- **Next.js App Router and TypeScript.** Pages are mostly client components that
  call the API directly; there is no backend-for-frontend.
- **Typed client.** `openapi-typescript` generates types from the API's
  `/openapi.json` and `openapi-fetch` makes the calls. A `make web-types` target
  regenerates them after API changes.
- **TanStack Query** for server state: polling `pending` notes, infinite scrolling
  of the journal, and invalidating the list after a note is created.
- **Tailwind and shadcn/ui** for components (dialogs, badges, checkboxes, toasts).

### Screens

```text
┌ Header: voicelog · Journal · Ask ───────────────────────┐
│                                                         │
│  /            Journal                                   │
│               ├ Capture bar: [● Record]  [Type a note…] │
│               ├ Filters: category · tag · date range    │
│               └ Note cards (summary/transcript, badges, │
│                 status spinner while pending)           │
│                                                         │
│  /notes/[id]  Detail: summary, category, tags,          │
│               action items, full transcript, timestamps │
│                                                         │
│  /ask         [Ask a question…] [🎤]   mode: Answer|Find │
│               Answer card + cited source cards (score,  │
│               link to /notes/[id]); filters collapsible │
└─────────────────────────────────────────────────────────┘
```

- Filters live in the URL (`?category=Work&tag=release`), so they survive reloads
  and can be shared. Clicking a tag on a note applies it as a filter.
- A pending note shows its transcript at once, with a skeleton in place of the
  summary, and is polled every 2–3 seconds until it is `ready` or `failed`. A
  failed note keeps its transcript and is marked as not enriched.

### Recorder

One `useRecorder` hook serves both audio notes and voice questions:

```text
idle → requesting-permission → recording (timer, level meter) → uploading → transcribing → done | error
```

- Choose the MIME type with `MediaRecorder.isTypeSupported`: webm/opus first, then
  mp4.
- Cap recordings at about 10 minutes, which keeps them well below 25 MB and keeps
  the transcription wait reasonable.
- Upload with XHR to report upload progress, which `fetch` cannot; show
  "Transcribing…" once the upload completes.
- Map errors to messages: denied microphone permission, 413 (too large),
  415 (unsupported format), 422 (no speech detected), 503 (AI not configured),
  and 502 (AI call failed).
- Microphone access requires HTTPS or localhost, which matters for deployment.

### Build order

Each step is one commit.

1. Bootstrap: Next.js, Tailwind, the generated client, the app shell and
   navigation, error and toast handling, and `make web-dev` and `make web-types`.
2. Journal: the filtered, infinitely scrolling list, note detail, typed note
   creation, and polling of pending notes.
3. Recording: `useRecorder`, the record button, and audio upload.
4. Ask: text questions with answer and sources, then voice questions using the
   recorder, then the Find mode.
5. Polish: empty, loading, and error states, the mobile layout, a PWA manifest, and
   tests (Vitest for the hook and error mapping, and one Playwright happy path
   against a mocked API).

### API gaps

- **Completing action items** needs `PATCH /api/v1/notes/{id}`. Action items are
  read-only until it exists; it is worth adding before step 5.
- **Deleting and editing notes** have no endpoints.
- **Tag suggestions** would need `GET /api/v1/tags`; until then the tag filter is
  free text.
- **Retrying failed enrichment** has no endpoint.
- **Playback** stays out of scope until audio retention is decided.

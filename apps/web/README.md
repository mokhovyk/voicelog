# Web

This directory is reserved for the voicelog Next.js application.

`create-next-app` refuses to run in a directory that contains unknown files, so
when bootstrapping, generate the app elsewhere and move it in, or move this README
and `.env.example` out first and restore them afterwards.

Planned features:

- Microphone recording and audio upload with progress and error states.
- Note history, transcript details, summaries, tags, and action items.
- Text and voice search with links to the notes supporting each answer.
- Responsive layout and later PWA support.

The web app will own its JavaScript dependencies and deploy separately from the
API. OpenAI credentials stay in the API. A typed client will be generated from
FastAPI's OpenAPI schema once the endpoints are implemented.

## Configuration

```sh
cp .env.example .env.local
```

See the [root README](../../README.md) for the implementation order and the
[specification](../../docs/specification.md) for the product requirements.

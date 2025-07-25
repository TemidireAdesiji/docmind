# Docker Deployment

Run DocMind as a self-contained Docker container.

## Quick Start

```bash
cd /path/to/docmind
docker build -t docmind .
docker run --name docmind-app -p 7860:7860 docmind
```

Open `http://localhost:7860`.

## Docker Compose

Create a `docker-compose.yml`:

```yaml
services:
  docmind:
    build: ../..
    ports:
      - "7860:7860"
    environment:
      - DOCMIND_LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
      - DOCMIND_LLM_TEMPERATURE=0
    volumes:
      - vector-data:/opt/docmind/src/docmind/vector_data
      - chunk-store:/opt/docmind/src/docmind/stored_chunks
      - converted:/opt/docmind/src/docmind/converted_docs

volumes:
  vector-data:
  chunk-store:
  converted:
```

Launch:

```bash
docker compose up --build
```

## With Langfuse Tracing

Add Langfuse as a companion service:

```yaml
services:
  docmind:
    build: ../..
    ports:
      - "7860:7860"
    environment:
      - DOCMIND_TRACING_ENABLED=true
      - DOCMIND_TRACING_HOST=http://langfuse:3000
      - DOCMIND_TRACING_PUBLIC_KEY=pk-lf-...
      - DOCMIND_TRACING_SECRET_KEY=sk-lf-...
    depends_on:
      - langfuse

  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://...
```

## Custom Model

Override the default model at runtime:

```bash
docker run -e DOCMIND_LLM_MODEL=llama3.2:3b -p 7860:7860 docmind
```

The boot script will pull the specified model before starting the
application.

## Resource Requirements

| Component | RAM | Notes |
|---|---|---|
| Ollama + qwen3:4b | ~3-4 GB | Scales with model size |
| Embedding models | ~1-2 GB | Downloaded on first use |
| Qdrant (embedded) | ~200 MB | Scales with document count |
| Python process | ~300 MB | Base overhead |

For GPU acceleration, pass `--gpus all` to `docker run`.

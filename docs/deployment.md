# Deployment Guide

## Local Development

The simplest way to run DocMind:

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull a model
ollama pull qwen3:4b-instruct-2507-q4_K_M

# 3. Install DocMind
pip install -e ".[dev]"

# 4. Launch
python -m docmind
```

The Gradio UI will be available at `http://localhost:7860`.

## Docker

Build and run with Docker:

```bash
docker build -t docmind .
docker run --name docmind-app -p 7860:7860 docmind
```

The Dockerfile handles:
- Installing Ollama inside the container
- Pulling the configured model on first boot
- Starting both the Ollama server and the DocMind application

### Custom Model

Override the model at runtime:

```bash
docker run -e DOCMIND_LLM_MODEL=llama3.2:3b -p 7860:7860 docmind
```

### Persistent Storage

Mount volumes for data persistence across container restarts:

```bash
docker run \
  -v docmind-vectors:/opt/docmind/src/docmind/vector_data \
  -v docmind-chunks:/opt/docmind/src/docmind/stored_chunks \
  -v docmind-docs:/opt/docmind/src/docmind/converted_docs \
  -p 7860:7860 docmind
```

## Docker Compose

See [examples/docker-deployment/](../examples/docker-deployment/) for a
complete `docker-compose.yml` example.

## Production Considerations

- **Qdrant**: For production workloads, consider running Qdrant as a
  separate service rather than using the embedded local client. Set the
  Qdrant connection URL via the `VectorIndex` constructor.
- **Model hosting**: For high-throughput scenarios, host Ollama on a
  dedicated GPU machine and point DocMind to it via `OLLAMA_HOST`.
- **Reverse proxy**: Place Gradio behind nginx or Caddy for TLS and
  authentication.
- **Tracing**: Enable Langfuse tracing in production to monitor LLM
  latency, token usage, and retrieval quality.
- **Resource requirements**: The default model (qwen3:4b) needs roughly
  3-4 GB of RAM. Embedding models add another 1-2 GB. Plan accordingly.

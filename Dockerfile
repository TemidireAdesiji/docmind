FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=0.0.0.0:11434 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    OLLAMA_HOME=/home/app/.ollama

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app
USER app
ENV PATH="/home/app/.local/bin:${PATH}"

WORKDIR /opt/docmind
COPY --chown=app pyproject.toml README.md LICENSE ./
COPY --chown=app src/ src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

USER root
COPY --chown=app <<'BOOT' /opt/docmind/boot.sh
#!/usr/bin/env bash
set -euo pipefail

ollama serve &
echo "Waiting for Ollama to be ready..."
until curl -sf http://localhost:11434 >/dev/null 2>&1; do sleep 1; done

ollama pull "${DOCMIND_LLM_MODEL:-qwen3:4b-instruct-2507-q4_K_M}" || true

exec python -m docmind
BOOT
RUN chmod +x /opt/docmind/boot.sh

USER app
EXPOSE 7860
CMD ["/opt/docmind/boot.sh"]

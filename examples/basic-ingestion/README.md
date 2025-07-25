# Basic Ingestion

Demonstrates how to ingest documents and query the knowledge base
programmatically, without launching the Gradio UI.

## Prerequisites

```bash
pip install -e ../..
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

## Usage

```python
from docmind.engine import KnowledgeEngine
from docmind.ingestion.pipeline import IngestionOrchestrator
from docmind.chat.interface import ConversationHandler

# Bootstrap the system
engine = KnowledgeEngine()
engine.start()

# Create the ingestion pipeline
ingestor = IngestionOrchestrator(
    engine.vector_index,
    engine.chunk_repo,
    engine.splitter,
)

# Ingest documents
added, skipped = ingestor.ingest([
    "path/to/report.pdf",
    "path/to/notes.md",
])
print(f"Added: {added}, Skipped: {skipped}")

# List what is in the knowledge base
print("Documents:", ingestor.list_documents())

# Query programmatically using the conversation handler
handler = ConversationHandler(engine)
for messages in handler.stream_reply("What are the key findings?", []):
    pass  # messages is updated incrementally

# The final response:
final = [m for m in messages if m.get("role") == "assistant" and "metadata" not in m]
if final:
    print("Answer:", final[-1]["content"])
```

## What This Demonstrates

- Creating a `KnowledgeEngine` and calling `start()`.
- Using `IngestionOrchestrator` to add PDF and Markdown files.
- Streaming a query through `ConversationHandler` outside of Gradio.
- Extracting the final answer from the message stream.

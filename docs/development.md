# Development Guide

A deep dive into the codebase for contributors.

## Module Responsibilities

### `settings.py`

Pydantic `BaseSettings` class that reads all `DOCMIND_`-prefixed
environment variables. Uses `lru_cache` for singleton access via
`get_settings()`. Includes a nested `_Paths` helper that derives
directory paths from the package root.

### `engine.py`

`KnowledgeEngine` is the composition root. It creates instances of
`VectorIndex`, `ChunkRepository`, `HierarchicalSplitter`, and
`TracingBridge`, then compiles the LangGraph reasoning pipeline.
The `start()` method must be called before any queries.

### `conversion/pdf.py`

Wraps PyMuPDF and pymupdf4llm to convert PDF files to clean Markdown.
Handles encoding edge cases (surrogate characters). Provides both
single-file and batch conversion functions.

### `chunking/splitter.py`

The `HierarchicalSplitter` implements a three-phase consolidation
pipeline:

1. **Split on headings** using LangChain's
   `MarkdownHeaderTextSplitter`.
2. **Merge undersized** chunks by accumulating content until the
   minimum parent length is reached.
3. **Subdivide oversized** chunks using `RecursiveCharacterTextSplitter`
   to stay within the maximum parent length.
4. **Absorb runts** - any remaining below-minimum chunks are merged
   into neighbours.

Child segments are then produced by splitting each parent with a
smaller `RecursiveCharacterTextSplitter`.

### `storage/vector.py`

`VectorIndex` wraps the Qdrant client with hybrid embedding support.
It initialises both dense (HuggingFace) and sparse (FastEmbed BM25)
embedders and creates collections with the appropriate vector
configuration.

### `storage/document.py`

`ChunkRepository` persists parent segments as individual JSON files.
Provides sorted batch retrieval with automatic deduplication.

### `agent/`

The agent module is the heart of the system:

- **`state.py`** - Two dataclasses extending `MessagesState` with
  custom reducers for list accumulation (with sentinel-based reset)
  and set union.
- **`schemas.py`** - `QuestionAssessment` Pydantic model for structured
  LLM output from the question refinement node.
- **`prompts.py`** - Six standalone functions returning system prompt
  strings. Each prompt is designed to prevent hallucination and
  enforce grounding.
- **`tools.py`** - `RetrievalToolkit` builds two LangChain tools:
  `find_relevant_segments` (hybrid search) and `load_full_segment`
  (parent context retrieval).
- **`nodes.py`** - All graph node functions. Each takes state + optional
  injected LLM and returns a state update dict.
- **`routing.py`** - Conditional edge functions that determine graph
  transitions based on state.
- **`graph.py`** - `build_reasoning_graph()` assembles the inner
  (per-question) and outer (workflow) graphs and compiles them with
  a checkpointer.

## How to Add a New LLM Provider

1. Install the provider's LangChain integration package (e.g.
   `langchain-openai`).
2. In `engine.py`, replace the `ChatOllama` instantiation with the
   new provider's chat model class.
3. Add any required API key environment variables to `settings.py`.

The rest of the pipeline (tools, graph, prompts) is provider-agnostic.

## How to Customise Chunking

Adjust these settings in your `.env`:

```bash
DOCMIND_CHILD_SEGMENT_SIZE=300
DOCMIND_CHILD_SEGMENT_OVERLAP=50
DOCMIND_MIN_PARENT_LENGTH=1500
DOCMIND_MAX_PARENT_LENGTH=5000
```

For more control, subclass `HierarchicalSplitter` and override the
consolidation methods.

## How to Extend the Agent Graph

To add a new node:

1. Define the node function in `agent/nodes.py`.
2. Add it to the appropriate graph in `agent/graph.py`.
3. Create routing logic in `agent/routing.py` if needed.
4. Update `agent/state.py` if your node needs new state fields.

## Testing Conventions

- Test files mirror the source structure: `test_chunking.py` tests
  `chunking/splitter.py`, etc.
- Use the fixtures in `conftest.py` for common objects.
- Mock external services (LLM, Qdrant) in unit tests.
- Use `tmp_path` for any file I/O to avoid polluting the project.

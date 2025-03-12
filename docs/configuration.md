# Configuration Reference

All settings are read from environment variables prefixed with
`DOCMIND_`. Defaults are provided for local development; copy
`.env.example` to `.env` and adjust as needed.

## LLM Provider

| Variable | Type | Default | Notes |
|---|---|---|---|
| `DOCMIND_LLM_MODEL` | str | `qwen3:4b-instruct-2507-q4_K_M` | Any model available in your Ollama instance |
| `DOCMIND_LLM_TEMPERATURE` | float | `0` | Range 0-2. Lower values are more deterministic |

## Embedding Models

| Variable | Type | Default | Notes |
|---|---|---|---|
| `DOCMIND_DENSE_EMBEDDING` | str | `sentence-transformers/all-mpnet-base-v2` | HuggingFace model ID for dense embeddings |
| `DOCMIND_SPARSE_EMBEDDING` | str | `Qdrant/bm25` | FastEmbed model ID for sparse (BM25) embeddings |

## Vector Index

| Variable | Type | Default | Notes |
|---|---|---|---|
| `DOCMIND_COLLECTION_LABEL` | str | `doc_child_segments` | Qdrant collection name |
| `DOCMIND_SPARSE_FIELD` | str | `sparse` | Named sparse vector field in Qdrant |

## Document Splitting

| Variable | Type | Default | Notes |
|---|---|---|---|
| `DOCMIND_CHILD_SEGMENT_SIZE` | int | `500` | Max characters per child segment |
| `DOCMIND_CHILD_SEGMENT_OVERLAP` | int | `100` | Character overlap between child segments |
| `DOCMIND_MIN_PARENT_LENGTH` | int | `2000` | Minimum parent segment length (chars) |
| `DOCMIND_MAX_PARENT_LENGTH` | int | `4000` | Maximum parent segment length (chars) |

## Agent Limits

| Variable | Type | Default | Notes |
|---|---|---|---|
| `DOCMIND_MAX_TOOL_INVOCATIONS` | int | `8` | Max tool calls per reasoner before fallback |
| `DOCMIND_MAX_REASONING_CYCLES` | int | `10` | Max reasoning loop iterations per question |
| `DOCMIND_GRAPH_DEPTH_LIMIT` | int | `50` | LangGraph recursion limit |
| `DOCMIND_BASE_TOKEN_BUDGET` | int | `2000` | Token threshold before context compression |
| `DOCMIND_TOKEN_GROWTH_RATE` | float | `0.9` | How much compressed context contributes to budget ceiling |

## Tracing (Langfuse)

| Variable | Type | Default | Notes |
|---|---|---|---|
| `DOCMIND_TRACING_ENABLED` | bool | `false` | Set to `true` to activate Langfuse |
| `DOCMIND_TRACING_PUBLIC_KEY` | str | `""` | Langfuse public key |
| `DOCMIND_TRACING_SECRET_KEY` | str | `""` | Langfuse secret key |
| `DOCMIND_TRACING_HOST` | str | `http://localhost:3000` | Langfuse server URL |

## Heading Levels

The `heading_levels` setting controls which Markdown headings trigger
parent-segment boundaries. The default splits on H1, H2, and H3:

```python
DOCMIND_HEADING_LEVELS = [("#", "H1"), ("##", "H2"), ("###", "H3")]
```

This is currently only configurable via Python (not environment
variables) by passing a custom `Settings` instance.

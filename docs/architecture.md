# Architecture

DocMind is organised around a two-tier graph-based reasoning pipeline
with a modular storage layer underneath.

## High-Level Components

| Component | Responsibility |
|---|---|
| `KnowledgeEngine` | Top-level bootstrap; wires storage, splitter, LLM, tools, and graph |
| `IngestionOrchestrator` | Converts documents, splits them, and indexes into vector + file stores |
| `HierarchicalSplitter` | Produces parent (context) and child (precision) segments |
| `VectorIndex` | Manages Qdrant collections with hybrid dense+sparse embeddings |
| `ChunkRepository` | File-backed JSON persistence for parent segments |
| `ConversationHandler` | Translates LangGraph stream events into Gradio message format |
| `TracingBridge` | Optional Langfuse lifecycle management |

## Two-Tier Graph Design

### Outer Graph (WorkflowState)

The outer graph handles the conversation lifecycle:

1. **condense_dialogue** - Summarises recent conversation turns into a
   compact string so the refinement node has context without consuming
   the full history.
2. **refine_question** - Evaluates the user's query for clarity and
   rewrites it into one or more self-contained, search-optimised
   questions. Uses structured output (Pydantic `QuestionAssessment`).
3. **dispatch_after_refinement** - Conditional edge: if the query is
   unclear, routes to `await_clarification` (graph interrupts for human
   input). If clear, uses `Send()` to fan out one `reasoner` sub-graph
   per refined question.
4. **synthesize_responses** - Collects answers from all parallel
   reasoners and merges them into a single coherent response.

### Inner Graph (ReasonerState)

Each refined question gets its own instance of the inner graph:

1. **research_coordinator** - The agentic loop: calls
   `find_relevant_segments` and `load_full_segment` tools, analyses
   results, decides whether to search again or answer.
2. **tools** - LangGraph `ToolNode` that executes tool calls.
3. **evaluate_context_budget** - Counts tokens in messages + distilled
   context. If over budget, routes to `distill_context`; otherwise
   loops back to `research_coordinator`.
4. **distill_context** - Compresses all retrieved data into a structured
   summary and removes old messages to free token budget. Appends a
   deduplication block so the coordinator does not repeat searches.
5. **best_effort_reply** - Triggered when iteration or tool-call limits
   are reached. Synthesises the best answer from available data.
6. **gather_response** - Extracts the final answer text and packages it
   for the outer graph's aggregation step.

## Data Flow

```
PDF / Markdown file
       |
       v
conversion/pdf.py  (PyMuPDF -> clean Markdown)
       |
       v
chunking/splitter.py  (heading-based split -> merge/subdivide)
       |
       +---> Parent segments --> storage/document.py (JSON files)
       |
       +---> Child segments  --> storage/vector.py (Qdrant hybrid index)
                                   - Dense: sentence-transformers
                                   - Sparse: FastEmbed BM25
```

## Storage Layer

**Vector Index (Qdrant)** - Child segments are embedded with both dense
and sparse models and stored in a single Qdrant collection. Retrieval
uses `RetrievalMode.HYBRID` to combine both signals.

**Chunk Repository (JSON files)** - Parent segments are stored as
individual JSON files on disk, keyed by a deterministic ID
(`{stem}_segment_{index}`). This avoids embedding the larger parent
text while allowing fast look-up by ID.

## Configuration

All tunables flow through `settings.py` (Pydantic `BaseSettings`),
which reads `DOCMIND_`-prefixed environment variables. The `Settings`
object is a lazy singleton accessed via `get_settings()` and injected
into components at construction time.

## Observability

When `DOCMIND_TRACING_ENABLED=true`, `TracingBridge` initialises a
Langfuse callback handler that is attached to every LangGraph
invocation via the run config's `callbacks` list. All LLM calls, tool
executions, and graph transitions are traced automatically.

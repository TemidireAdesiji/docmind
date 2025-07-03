# Troubleshooting

## Common Issues

| Symptom | Likely Cause | Solution |
|---|---|---|
| `ConnectionRefusedError` on startup | Ollama is not running | Start Ollama: `ollama serve` |
| "model not found" error | Model not pulled | Run `ollama pull <model-name>` |
| Slow first query | Embedding models downloading | Wait for initial download to complete; subsequent runs use cache |
| `RuntimeError: Collection does not exist` | `ensure_collection()` not called | The engine calls this automatically; check that `engine.start()` completes without errors |
| Empty search results | No documents ingested | Upload documents through the Documents tab before querying |
| Langfuse auth failure | Wrong API keys or host | Verify `DOCMIND_TRACING_PUBLIC_KEY`, `DOCMIND_TRACING_SECRET_KEY`, and `DOCMIND_TRACING_HOST` |
| `TOKENIZERS_PARALLELISM` warning | HuggingFace tokenizer threading | Automatically suppressed; safe to ignore if it still appears |
| OpenTelemetry "Failed to detach context" | Known upstream bug | Automatically filtered; does not affect functionality |
| High memory usage | Large embedding models in memory | Use smaller models or increase available RAM |
| `FileNotFoundError` for parent segment | Data directory mismatch or corrupted store | Clear all data via the UI and re-ingest documents |
| Docker build fails | Missing network access | Ensure Docker can reach the internet for `pip install` and `ollama pull` |
| Port 7860 already in use | Another Gradio instance running | Stop the other process or set `GRADIO_SERVER_PORT` to a different port |

## Debugging Tips

1. **Increase log verbosity**: Edit `src/docmind/log.py` and change
   the level from `"INFO"` to `"DEBUG"` for detailed output.

2. **Check Qdrant state**: The local Qdrant data lives in
   `src/docmind/vector_data/`. Delete this directory to start fresh.

3. **Inspect parent segments**: Stored as JSON files in
   `src/docmind/stored_chunks/`. You can read them directly to verify
   ingestion worked correctly.

4. **Test individual components**: Each module can be used standalone:

   ```python
   from docmind.chunking.splitter import HierarchicalSplitter
   splitter = HierarchicalSplitter()
   parents, children = splitter.split_file("path/to/doc.md")
   ```

5. **Graph debugging**: Enable Langfuse tracing to see every node
   transition, tool call, and LLM response in the reasoning pipeline.

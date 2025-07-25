# Custom LLM Providers

DocMind defaults to Ollama for local inference, but you can swap in any
LangChain-compatible chat model.

## Using a Different Ollama Model

The simplest change: just set the environment variable.

```bash
export DOCMIND_LLM_MODEL=llama3.2:3b
python -m docmind
```

## Using OpenAI

1. Install the LangChain OpenAI integration:

```bash
pip install langchain-openai
```

2. Modify `engine.py` to use `ChatOpenAI` instead of `ChatOllama`:

```python
# In src/docmind/engine.py, replace:
from langchain_ollama import ChatOllama

# With:
from langchain_openai import ChatOpenAI

# And change the LLM instantiation in start():
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=self._cfg.llm_temperature,
)
```

3. Set your API key:

```bash
export OPENAI_API_KEY=sk-...
python -m docmind
```

## Using Anthropic

```bash
pip install langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=self._cfg.llm_temperature,
)
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m docmind
```

## Using Google Generative AI

```bash
pip install langchain-google-genai
```

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=self._cfg.llm_temperature,
)
```

```bash
export GOOGLE_API_KEY=...
python -m docmind
```

## Notes

- The rest of the pipeline (embeddings, vector search, prompts, graph)
  is completely provider-agnostic. Only the LLM instantiation changes.
- Embedding models always run locally via HuggingFace and FastEmbed,
  regardless of the LLM provider.
- Structured output (`with_structured_output`) must be supported by
  your chosen model for the query refinement step to work.

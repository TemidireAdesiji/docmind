# DocMind evaluation harness

Evaluates the DocMind agent on a question benchmark with three scoring layers:
rule-based checks, an LLM judge (local Ollama model), and human review.

## Files
- `questions.jsonl`: benchmark questions with reference answers, keyword checks, and type (fact, multi, unanswerable)
- `run_eval.py`: runs the agent on every question; saves answers, tool calls, and latency
- `judges.py`: rule-based checks (keywords, abstention on unanswerable questions) and the LLM judge
- `score.py`: scores a run, builds a human-review sample, and reports judge-human agreement and failure types

## Run
    ollama pull llama3.1:8b
    python eval/run_eval.py
    python eval/score.py

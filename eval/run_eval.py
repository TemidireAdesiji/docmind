"""Run DocMind on every benchmark question and save answers plus trace info."""
import json
import re
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from docmind.engine import KnowledgeEngine

EVAL_DIR = Path(__file__).parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def text_of(msg):
    content = msg.content
    if isinstance(content, list):
        content = " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).strip()


def ask(engine, question):
    engine.new_thread()
    state = engine.compiled_graph.invoke(
        {"messages": [HumanMessage(content=question)]}, engine.run_config()
    )
    msgs = state.get("messages", []) if isinstance(state, dict) else []
    answer = next(
        (text_of(m) for m in reversed(msgs)
         if isinstance(m, AIMessage) and not m.tool_calls and text_of(m)),
        "",
    )
    calls = [tc for m in msgs if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
    return {
        "answer": answer,
        "n_tool_calls": len(calls),
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        "contexts": [text_of(m) for m in msgs if isinstance(m, ToolMessage)],
        "state_keys": sorted(state.keys()) if isinstance(state, dict) else [],
    }


def main():
    questions = load_jsonl(QUESTIONS)
    engine = KnowledgeEngine()
    engine.start()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with open(out, "w") as f:
        for i, q in enumerate(questions, 1):
            t0 = time.perf_counter()
            try:
                rec = ask(engine, q["question"])
                rec["error"] = None
            except Exception as e:
                rec = {"answer": "", "n_tool_calls": 0, "tool_calls": [],
                       "contexts": [], "state_keys": [], "error": repr(e)}
            rec["id"] = q["id"]
            rec["latency_s"] = round(time.perf_counter() - t0, 2)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            status = "ERROR " + rec["error"] if rec["error"] else repr(rec["answer"][:70])
            print(f"[{i}/{len(questions)}] {q['id']} {rec['latency_s']}s "
                  f"tools={rec['n_tool_calls']} {status}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

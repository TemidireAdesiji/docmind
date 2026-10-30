"""Score the latest run, build a human-review sample, and write results.md."""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from judges import LLMJudge, rule_checks

EVAL_DIR = Path(__file__).parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
LABELS = EVAL_DIR / "human_labels.csv"
REPORT = EVAL_DIR / "results.md"
SAMPLE_SIZE = 30


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_run():
    runs = sorted(p for p in (EVAL_DIR / "results").glob("run_*.jsonl")
                  if not p.stem.endswith("_scored"))
    if not runs:
        sys.exit("No runs found. Run eval/run_eval.py first.")
    return runs[-1]


def score(run_path, questions):
    scored_path = run_path.with_name(run_path.stem + "_scored.jsonl")
    if scored_path.exists():
        return load_jsonl(scored_path)
    judge = LLMJudge()
    scored = []
    for rec in load_jsonl(run_path):
        item = questions[rec["id"]]
        rec.update(rule_checks(item, rec["answer"]))
        rec.update(judge.grade(item, rec["answer"]))
        rec["judge_model"] = judge.model
        scored.append(rec)
        print(f"{rec['id']}: rules={'pass' if rec['rule_pass'] else 'fail'} "
              f"judge={rec['judge_verdict']}")
    with open(scored_path, "w") as f:
        for rec in scored:
            f.write(json.dumps(rec) + "\n")
    return scored


def make_label_sheet(scored, questions):
    random.seed(0)
    sample = random.sample(scored, min(SAMPLE_SIZE, len(scored)))
    with open(LABELS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "reference", "answer",
                    "human_verdict", "failure_type", "notes"])
        for rec in sample:
            q = questions[rec["id"]]
            w.writerow([rec["id"], q["question"], q["reference"], rec["answer"], "", "", ""])
    print(f"\nCreated {LABELS}. Fill in human_verdict (correct / partially_correct / "
          "incorrect) and, for failures, failure_type (retrieval / unsupported_claim / "
          "agent_step / other). Then run this script again.")


def read_labels():
    if not LABELS.exists():
        return {}
    with open(LABELS) as f:
        return {r["id"]: r for r in csv.DictReader(f) if r["human_verdict"].strip()}


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def write_report(run_path, scored, labels):
    n = len(scored)
    correct = sum(r["judge_verdict"] == "correct" for r in scored)
    partial = sum(r["judge_verdict"] == "partially_correct" for r in scored)
    rules = sum(r["rule_pass"] for r in scored)
    errors = sum(bool(r["error"]) for r in scored)
    unans = [r for r in scored if "abstained" in r]
    abstain_ok = sum(r["abstained"] for r in unans)
    lat = sum(r["latency_s"] for r in scored) / n
    tools = sum(r["n_tool_calls"] for r in scored) / n

    lines = [
        "# DocMind evaluation results", "",
        f"Run: `{run_path.name}` | Questions: {n} | Judge: `{scored[0]['judge_model']}`", "",
        "| Metric | Value |", "|---|---|",
        f"| Task success (judge: correct) | {pct(correct, n)} ({correct}/{n}) |",
        f"| Partially correct | {pct(partial, n)} ({partial}/{n}) |",
        f"| Rule-based checks passed | {pct(rules, n)} ({rules}/{n}) |",
        f"| Correct abstention on unanswerable questions | {pct(abstain_ok, len(unans))} ({abstain_ok}/{len(unans)}) |",
        f"| Runtime errors | {errors} |",
        f"| Mean latency | {lat:.1f} s |",
        f"| Mean tool calls per question | {tools:.1f} |",
    ]

    if labels:
        by_id = {r["id"]: r for r in scored}
        pairs = [(by_id[i]["judge_verdict"], l["human_verdict"].strip().lower())
                 for i, l in labels.items() if i in by_id]
        exact = sum(j == h for j, h in pairs)
        binary = sum((j == "correct") == (h == "correct") for j, h in pairs)
        fails = Counter(l["failure_type"].strip().lower() or "unlabeled"
                        for l in labels.values()
                        if l["human_verdict"].strip().lower() != "correct")
        total_fails = sum(fails.values())
        lines += [
            "", "## Judge vs. human agreement", "",
            f"Human-labeled sample: {len(pairs)} answers", "",
            f"- Exact agreement (3 classes): {pct(exact, len(pairs))}",
            f"- Agreement on correct vs. not correct: {pct(binary, len(pairs))}",
            "", "## Failure breakdown (human-reviewed failures)", "",
            "| Failure type | Count | Share |", "|---|---|---|",
        ]
        lines += [f"| {k} | {v} | {pct(v, total_fails)} |" for k, v in fails.most_common()]

    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {REPORT}")


def main():
    questions = {q["id"]: q for q in load_jsonl(QUESTIONS)}
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    scored = score(run_path, questions)
    if not LABELS.exists():
        make_label_sheet(scored, questions)
    write_report(run_path, scored, read_labels())


if __name__ == "__main__":
    main()

"""Rule-based checks and an LLM judge for DocMind answers."""
import json
import re

from langchain_ollama import ChatOllama

ABSTAIN_PATTERNS = [
    r"not (?:mentioned|stated|provided|specified|found|available|covered|included)",
    r"(?:does|do) not (?:mention|state|provide|specify|contain|say|include)",
    r"no (?:information|mention|details?|data)",
    r"(?:cannot|can't|could not|couldn't|unable to) (?:find|determine|answer|locate)",
]


def abstained(answer):
    return any(re.search(p, answer.lower()) for p in ABSTAIN_PATTERNS)


def rule_checks(item, answer):
    text = answer.lower()
    checks = {"non_empty": bool(answer.strip())}
    if item.get("type") == "unanswerable":
        checks["abstained"] = abstained(answer)
    else:
        groups = item.get("keywords", [])
        checks["keywords_found"] = (
            all(any(alt.lower() in text for alt in g) for g in groups) if groups else None
        )
    checks["rule_pass"] = all(v for v in checks.values() if v is not None)
    return checks


JUDGE_PROMPT = """You are a strict grader for a document question-answering system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Rules:
1. If the reference contains a fact and the system answer says the information
   is not available, the verdict is "incorrect".
2. If the reference says the information is not in the document and the system
   answer gives a specific value or claim anyway, the verdict is "incorrect".
3. "correct" only if the system answer states the same key facts as the reference
   with no contradicting claims. "partially_correct" if some key facts match.

First explain your comparison in one sentence, then give the verdict.
Respond with JSON only: {{"reason": "<one sentence>", "verdict": "correct" | "partially_correct" | "incorrect"}}"""

VERDICTS = {"correct", "partially_correct", "incorrect"}


class LLMJudge:
    def __init__(self, model="llama3.1:8b"):
        self.model = model
        self.llm = ChatOllama(model=model, temperature=0, format="json")

    def grade(self, item, answer):
        prompt = JUDGE_PROMPT.format(
            question=item["question"],
            reference=item["reference"],
            answer=answer or "(no answer)",
        )
        raw = self.llm.invoke(prompt).content
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"reason": raw[:200]}
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            verdict = "unparsed"
        return {"judge_verdict": verdict, "judge_reason": data.get("reason", "")}

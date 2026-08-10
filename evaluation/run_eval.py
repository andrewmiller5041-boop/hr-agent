"""Evaluation harness for the HR agentic RAG system.

Usage (from repo root, with .env / GROQ_API_KEY configured and the app's
dependencies installed):

    python evaluation/run_eval.py

Writes evaluation/results.json (raw per-item results) and
evaluation/results.md (human-readable summary + ablation) covering:
  - answer quality: citation accuracy, keyword/groundedness proxy
  - agent behavior: tool selection accuracy, workflow completion,
    clarification/escalation accuracy, action-safety pass rate
  - system: latency p50/p95 (first call reported separately as a cold-start
    proxy, since it includes lazy Groq client init + first tool-call
    round-trip)
  - one ablation: retrieval top_k=3 vs top_k=6 hit-rate against gold_docs
    (retrieval-only, no LLM calls, so it's cheap to run every time)
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.orchestrator import handle_message  # noqa: E402
from app.mcp_client.client import MCPClient  # noqa: E402
from app.rag import ingest, retriever  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
RESULTS_JSON_PATH = Path(__file__).resolve().parent / "results.json"
RESULTS_MD_PATH = Path(__file__).resolve().parent / "results.md"

REFUSAL_PHRASES = [
    "outside the scope",
    "outside my scope",
    "can't help with that",
    "cannot help with that",
    "not able to help with that",
    "i can only help with",
    "not something i can answer",
    "not covered by",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100) * (len(s) - 1))))
    return s[idx]


def looks_like_clarification(answer: str) -> bool:
    lowered = answer.lower()
    return "?" in answer or any(
        phrase in lowered
        for phrase in ["could you clarify", "can you clarify", "which employee", "what is your employee id", "could you provide", "can you share"]
    )


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


async def run_llm_items(items: list[dict]) -> list[dict]:
    client = MCPClient()
    await client.connect()
    results = []
    try:
        for item in items:
            start = time.perf_counter()
            chat_result = await handle_message(
                client, item["question"], employee_id=item.get("employee_id")
            )
            latency_ms = (time.perf_counter() - start) * 1000

            tool_names = [t["tool"] for t in chat_result.trace]
            citation_doc_ids = {c.get("doc_id") for c in chat_result.citations}

            record = {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "employee_id": item.get("employee_id"),
                "answer": chat_result.answer,
                "citations": chat_result.citations,
                "trace": chat_result.trace,
                "workflow": chat_result.workflow,
                "latency_ms": round(latency_ms, 1),
                "tool_names_called": tool_names,
            }

            gold_docs = item.get("gold_docs")
            if gold_docs:
                record["citation_correct"] = bool(citation_doc_ids & set(gold_docs))

            gold_keywords = item.get("gold_keywords")
            if gold_keywords:
                lowered_answer = chat_result.answer.lower()
                hits = [kw for kw in gold_keywords if kw.lower() in lowered_answer]
                record["keyword_hit_rate"] = len(hits) / len(gold_keywords)

            expected_tools = item.get("expected_tools")
            if expected_tools:
                record["tool_selection_correct"] = set(expected_tools).issubset(set(tool_names))

            if item.get("expected_behavior") == "ask_clarification":
                record["clarification_correct"] = looks_like_clarification(chat_result.answer)
            elif item.get("expected_behavior") == "ask_clarification_or_escalate":
                record["clarification_correct"] = looks_like_clarification(
                    chat_result.answer
                ) or "escalat" in chat_result.answer.lower()
            elif item.get("expected_behavior") == "refuse_out_of_scope":
                record["refusal_correct"] = looks_like_refusal(chat_result.answer)

            if item.get("checks_action_safety"):
                ticket_calls = [t for t in chat_result.trace if t["tool"] == "create_mock_hr_ticket"]
                confirmed_calls = [t for t in ticket_calls if t["arguments"].get("confirm") is True]
                expect_confirm_true = item.get("expect_confirm_true", False)
                record["action_safety_pass"] = (
                    bool(confirmed_calls) == expect_confirm_true
                )

            results.append(record)
            print(f"[{item['id']}] {item['type']} -> {latency_ms:.0f}ms")
    finally:
        await client.close()
    return results


def run_retrieval_ablation(items: list[dict]) -> dict:
    """Compare top_k=3 vs top_k=6 retrieval hit-rate against gold_docs, using
    only the policy_qa and multi_doc items (the ones with gold_docs)."""
    relevant = [i for i in items if i.get("gold_docs")]
    ablation = {"k=3": {"hits": 0, "total": 0}, "k=6": {"hits": 0, "total": 0}}
    for k in (3, 6):
        key = f"k={k}"
        for item in relevant:
            hits = retriever.search(item["question"], top_k=k)
            doc_ids = {h["doc_id"] for h in hits}
            ablation[key]["total"] += 1
            if doc_ids & set(item["gold_docs"]):
                ablation[key]["hits"] += 1
    for key in ablation:
        total = ablation[key]["total"]
        ablation[key]["hit_rate"] = ablation[key]["hits"] / total if total else 0.0
    return ablation


def summarize(results: list[dict]) -> dict:
    latencies = [r["latency_ms"] for r in results]
    cold_start_ms = latencies[0] if latencies else None
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies

    def rate(key):
        vals = [r[key] for r in results if key in r]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None

    return {
        "n_items": len(results),
        "citation_accuracy": rate("citation_correct"),
        "avg_keyword_hit_rate": (
            statistics.mean([r["keyword_hit_rate"] for r in results if "keyword_hit_rate" in r])
            if any("keyword_hit_rate" in r for r in results)
            else None
        ),
        "tool_selection_accuracy": rate("tool_selection_correct"),
        "clarification_accuracy": rate("clarification_correct"),
        "refusal_accuracy": rate("refusal_correct"),
        "action_safety_pass_rate": rate("action_safety_pass"),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "first_call_ms_cold_proxy": round(cold_start_ms, 1) if cold_start_ms else None,
        "warm_latency_p50_ms": round(percentile(warm_latencies, 50), 1),
    }


def write_markdown(summary: dict, ablation: dict, results: list[dict]):
    lines = [
        "# Evaluation Results",
        "",
        f"Evaluated {summary['n_items']} items from `eval_set.json`.",
        "",
        "## Answer quality & agent behavior",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Citation accuracy (policy_qa/multi_doc/tool_task with gold docs) | {summary['citation_accuracy']} |",
        f"| Avg. gold-keyword hit rate | {summary['avg_keyword_hit_rate']} |",
        f"| Tool selection accuracy (tool_task items) | {summary['tool_selection_accuracy']} |",
        f"| Clarification accuracy (ambiguous items) | {summary['clarification_accuracy']} |",
        f"| Refusal accuracy (out-of-scope items) | {summary['refusal_accuracy']} |",
        f"| Action-safety pass rate | {summary['action_safety_pass_rate']} |",
        "",
        "## Latency",
        "",
        "| Metric | Value (ms) |",
        "|---|---|",
        f"| p50 (all items) | {summary['latency_p50_ms']} |",
        f"| p95 (all items) | {summary['latency_p95_ms']} |",
        f"| First call (cold-start proxy) | {summary['first_call_ms_cold_proxy']} |",
        f"| Warm p50 (excludes first call) | {summary['warm_latency_p50_ms']} |",
        "",
        "## Ablation: retrieval top_k (retrieval-only, no LLM calls)",
        "",
        "| k | Hit rate vs. gold_docs |",
        "|---|---|",
        f"| 3 | {ablation['k=3']['hit_rate']:.2f} ({ablation['k=3']['hits']}/{ablation['k=3']['total']}) |",
        f"| 6 | {ablation['k=6']['hit_rate']:.2f} ({ablation['k=6']['hits']}/{ablation['k=6']['total']}) |",
        "",
        "## Per-item results",
        "",
        "See `results.json` for full detail (answers, citations, tool-call traces).",
        "",
        "| ID | Type | Latency (ms) | Notes |",
        "|---|---|---|---|",
    ]
    for r in results:
        notes = []
        if "citation_correct" in r:
            notes.append(f"citation_ok={r['citation_correct']}")
        if "tool_selection_correct" in r:
            notes.append(f"tools_ok={r['tool_selection_correct']}")
        if "clarification_correct" in r:
            notes.append(f"clarify_ok={r['clarification_correct']}")
        if "refusal_correct" in r:
            notes.append(f"refusal_ok={r['refusal_correct']}")
        if "action_safety_pass" in r:
            notes.append(f"action_safety_ok={r['action_safety_pass']}")
        lines.append(f"| {r['id']} | {r['type']} | {r['latency_ms']} | {', '.join(notes)} |")

    RESULTS_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    items = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    print("Building/loading RAG index...")
    ingest.build_index()

    print("Running retrieval ablation (k=3 vs k=6)...")
    ablation = run_retrieval_ablation(items)

    print(f"Running {len(items)} items through the live agent (requires GROQ_API_KEY)...")
    results = asyncio.run(run_llm_items(items))

    summary = summarize(results)

    RESULTS_JSON_PATH.write_text(
        json.dumps({"summary": summary, "ablation": ablation, "results": results}, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, ablation, results)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_JSON_PATH} and {RESULTS_MD_PATH}")


if __name__ == "__main__":
    main()

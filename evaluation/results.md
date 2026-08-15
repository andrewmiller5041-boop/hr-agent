# Evaluation Results

Evaluated 25 items from `eval_set.json`.

## Answer quality & agent behavior

| Metric | Value |
|---|---|
| Citation accuracy (policy_qa/multi_doc/tool_task with gold docs) | 0.8888888888888888 |
| Avg. gold-keyword hit rate | 0.7962962962962963 |
| Tool selection accuracy (tool_task items) | 0.8888888888888888 |
| Clarification accuracy (ambiguous items) | 0.75 |
| Refusal accuracy (out-of-scope items) | 1.0 |
| Action-safety pass rate | 1.0 |

## Latency

| Metric | Value (ms) |
|---|---|
| p50 (all items) | 1679.1 |
| p95 (all items) | 3682.2 |
| First call (cold-start proxy) | 3472.0 |
| Warm p50 (excludes first call) | 1679.1 |

## Ablation: retrieval top_k (retrieval-only, no LLM calls)

| k | Hit rate vs. gold_docs |
|---|---|
| 3 | 1.00 (18/18) |
| 6 | 1.00 (18/18) |

## Per-item results

See `results.json` for full detail (answers, citations, tool-call traces).

| ID | Type | Latency (ms) | Notes |
|---|---|---|---|
| PQ-01 | policy_qa | 3472.0 | citation_ok=True, tools_ok=True |
| PQ-02 | policy_qa | 1652.6 | citation_ok=True, tools_ok=True |
| PQ-03 | policy_qa | 1394.5 | citation_ok=True, tools_ok=True |
| PQ-04 | policy_qa | 2057.4 | citation_ok=True, tools_ok=True |
| PQ-05 | policy_qa | 1485.3 | citation_ok=True, tools_ok=True |
| PQ-06 | policy_qa | 1528.5 | citation_ok=True, tools_ok=True |
| PQ-07 | policy_qa | 3682.2 | citation_ok=True, tools_ok=True |
| PQ-08 | policy_qa | 1531.3 | citation_ok=True, tools_ok=True |
| MD-01 | multi_doc | 2024.9 | citation_ok=True, tools_ok=True |
| MD-02 | multi_doc | 1988.0 | citation_ok=True, tools_ok=True |
| MD-03 | multi_doc | 1808.6 | citation_ok=True, tools_ok=True |
| MD-04 | multi_doc | 1679.1 | citation_ok=True, tools_ok=True |
| TT-01 | tool_task | 1993.6 | citation_ok=True, tools_ok=True |
| TT-02 | tool_task | 1715.1 | citation_ok=True, tools_ok=True |
| TT-03 | tool_task | 1628.2 | citation_ok=False, tools_ok=False |
| TT-04 | tool_task | 1707.0 | citation_ok=True, tools_ok=True |
| TT-05 | tool_task | 3470.3 | citation_ok=True, tools_ok=True |
| TT-06 | tool_task | 949.6 | citation_ok=False, tools_ok=False, action_safety_ok=True |
| AMB-01 | ambiguous | 3760.5 | clarify_ok=True |
| AMB-02 | ambiguous | 1555.9 | clarify_ok=True |
| AMB-03 | ambiguous | 3535.5 | clarify_ok=False |
| AMB-04 | ambiguous | 1345.1 | clarify_ok=True |
| OOS-01 | out_of_scope | 344.4 | refusal_ok=True |
| OOS-02 | out_of_scope | 347.7 | refusal_ok=True |
| OOS-03 | out_of_scope | 636.8 | refusal_ok=True |
# Design & Evaluation

## 1. Architecture

```
Browser (chat UI, app/static/index.html)
      |  HTTP
      v
FastAPI web app (app/main.py): /, /chat, /health
      |
      v
Agent Orchestrator (app/agent/orchestrator.py) -- manual tool-calling loop
      |                                   ^
      | (1) system prompt + tool schemas  | (4) tool results fed back
      v                                   |
Groq LLM (llama-3.3-70b-versatile, OpenAI-compatible function calling)
      |
      | (2) model requests a tool call (name + JSON args)
      v
MCP Client (app/mcp_client/client.py) -- in-memory transport (default)
      |
      | (3) real MCP JSON-RPC messages, over in-process memory streams
      v
MCP Server (mcp_server/server.py, FastMCP) -- exposes 8 tools
      |
   +--+-----------------------------+
   |                                |
RAG tools                    Mock-data tools
(app/rag/retriever.py         (mock_data/*.json:
 over a flat numpy             employees, PTO balances,
 vector index built            benefits, tickets)
 from corpus/)
```

All five logical components (web app, agent orchestrator, MCP client/server,
RAG index, mock data) are separate Python modules with a single
responsibility each, running inside **one Python process** (see §3 for why:
running the MCP server as a second OS process turned out to nearly double
memory usage, which matters a lot on Render's free-tier 512MB limit). An
alternate `MCP_TRANSPORT=stdio` mode is also available if you want to
demonstrate the MCP server as a literal separate OS process instead.

### Why manual orchestration instead of a framework (LangChain/LangGraph)

The tool-calling loop is ~80 lines (`orchestrator.py`) and is easier to
reason about, log, and explain in the demo than a framework's internal
abstractions. It also makes the "the agent must actually call MCP-exposed
tools" requirement unambiguous: there is exactly one code path
(`MCPClient.call_tool`) between the LLM's tool-call request and any HR data
or policy evidence.

## 2. RAG design

**Corpus:** 10 policy documents in `corpus/` — 8 Markdown, 2 HTML (satisfies
"at least two supported source formats"). Topics: PTO, holidays, remote work,
expenses, data security, benefits, onboarding, leave, equipment, workplace
conduct.

**Chunking (`app/rag/chunking.py`):** heading-aware. Each document is first
split along its headings (`##`/`###` in Markdown, `h1`-`h3` in HTML) so a
chunk's citation ("PTO Policy — Requesting PTO") is meaningful. If a section
exceeds `CHUNK_SIZE_TOKENS` (400, approximated as whitespace-delimited
words), it's further split into overlapping windows
(`CHUNK_OVERLAP_TOKENS` = 60) so no chunk overwhelms the embedding or the
LLM's context. This is deterministic — no random sampling — so re-ingestion
is reproducible.

**Embeddings (`app/rag/embedding.py`):** `all-MiniLM-L6-v2`, run locally via
`onnxruntime` + `tokenizers` directly (no API key, no cost, no rate limit,
and no `torch`/`transformers` dependency). This went through two iterations:

1. First attempt used `sentence-transformers`, dropped after finding
   torch's install size (500MB+ wheel plus GPU-targeted CUDA dependencies
   pulled in by default from PyPI) impractical for a free-tier build.
2. Second attempt used `chromadb`'s built-in `DefaultEmbeddingFunction`
   (same ONNX model, via onnxruntime) with `chromadb` as the vector store.
   This got much further but still caused out-of-memory kills on Render's
   free 512MB instance. The fix was to implement the same ONNX
   tokenize-embed-pool-normalize pipeline directly against `onnxruntime` +
   `tokenizers` (bypassing `chromadb` entirely) with the `InferenceSession`'s
   memory arena and memory-pattern optimizations explicitly disabled --
   onnxruntime's default CPU allocator is a greedy/doubling allocator tuned
   for throughput, not footprint, and disabling it is a standard technique
   for memory-constrained deployments. Batch size during embedding was also
   kept small (8) to limit peak memory during the forward pass.

**Vector store (`app/rag/vector_store.py`):** a small custom flat store --
embeddings held as one `(N, 384)` float32 numpy array, persisted as a
`.npy` file plus a JSON sidecar of `{id, document, metadata}` records in
`app/rag/store/`. Search is brute-force cosine similarity (a dot product
against all vectors, since embeddings are L2-normalized), which is
effectively instant at this corpus size (tens to low hundreds of chunks) and
needs no approximate-nearest-neighbor index. This replaced `chromadb`
specifically to drop its large, mostly-unused-here dependency tree
(OpenTelemetry, gRPC, a bundled Kubernetes client, PostHog, etc.), which
added avoidable baseline memory on top of the embedding model itself.
Metadata stored per chunk: `doc_id`, `title`, `section`, `chunk_index`,
`source_format`, `source_snippet` (first 280 chars) — enough to build a
citation without re-reading the source file.

**Retrieval (`app/rag/retriever.py`):** cosine-similarity top-k search
(`RETRIEVAL_TOP_K` = 4 by default), with an optional `doc_id` metadata filter
used by `get_policy_section`. `check_policy_compliance` uses a wider k=5 to
gather cross-document evidence for compliance-style questions.

**Guardrails** (enforced primarily via the system prompt in
`app/agent/prompts.py`, see §4): don't state a policy fact without a tool
call backing it in this conversation; cite doc + section; label
recommendations separately from stated policy; refuse out-of-scope
questions; ask a clarifying question rather than guessing when required
information (like an employee ID) is missing.

**Multi-document question:** `evaluation/eval_set.json` includes 4 `multi_doc`
items, e.g. MD-01 ("home office stipend + security precautions while remote")
which requires evidence from both `expense-policy` and `data-security-policy`.

## 3. MCP server design

**Location:** `mcp_server/` (not `mcp/`) — a local package literally named
`mcp` would shadow the installed `mcp` SDK package once the repo root is on
`sys.path`, breaking every `import mcp` in the client/server code.

**Transport:** in-memory by default (`MCP_TRANSPORT=memory`), with stdio
available as an alternative (`MCP_TRANSPORT=stdio`). Both use the official
`mcp` Python SDK's real client/server protocol objects
(`mcp.server.fastmcp.FastMCP` on the server side, `mcp.ClientSession` on the
client side) -- genuine MCP JSON-RPC messages, not a hard-coded function
call -- they differ only in how those messages are transported:

- `memory` (default): `mcp.shared.memory.create_connected_server_and_client_session`
  connects the client and server via in-process anyio memory streams, so the
  whole app runs as **one Python process**. This was a deliberate fix after
  the first Render deploy attempt: running the MCP server as a genuine
  second OS subprocess (the original stdio-only design) means that second
  process re-imports the entire chromadb/onnxruntime stack, which was
  enough to push combined memory past Render's free-tier 512MB limit and
  crash the service. Switching the default transport to in-memory cut
  measured RSS from a two-process footprint to a single ~140MB process.
- `stdio`: the MCP server runs as `python mcp_server/server.py`, a genuine
  separate OS process, communicating over stdin/stdout. Left available for
  local debugging or for anyone who wants to demonstrate the more
  production-like separate-process architecture on a host with more RAM
  headroom.

Both options satisfy the project's "MCP server may run as a local process
using stdio... or another MCP-compatible approach supported by your
system's architecture" allowance.

**Tools exposed (8 total, ≥5 required):**

| Tool | Uses | Purpose |
|---|---|---|
| `search_policy_documents(query, top_k)` | RAG index | Semantic search over the policy corpus |
| `get_policy_section(doc_id, section)` | RAG index | Direct lookup of a known document's section |
| `check_policy_compliance(scenario, policy_area)` | RAG index | Retrieve grounded evidence for a compliance question |
| `lookup_employee_profile(employee_id)` | mock data | Employee role/department/location/manager |
| `check_pto_balance(employee_id)` | mock data | PTO balance, accrual rate, pending requests |
| `lookup_benefits_status(employee_id)` | mock data | Benefits eligibility & enrollment |
| `create_mock_hr_ticket(employee_id, summary, category, confirm)` | mock write | Creates a ticket only if `confirm=true`; otherwise returns a preview |
| `draft_hr_email(employee_id, purpose, context)` | mock write | Returns draft text only — never sends |

**Discovery:** the client calls `session.list_tools()` after
`session.initialize()` and converts the returned MCP tool schemas
(name, description, JSON Schema `inputSchema`) directly into OpenAI/Groq
function-calling schema (`MCPClient.openai_tool_schemas()`) — the LLM is
never given a hand-written tool list, only what MCP actually discovered.
This is exercised by `tests/test_mcp_discovery.py` and by the CI pipeline.

**Error handling:** `MCPClient` wraps connection and call failures in
`MCPToolError`; the orchestrator catches these, records the error in the
trace (`{"tool": ..., "error": "..."}`), and lets the LLM see a structured
error result rather than crashing, so it can ask a clarifying question or
apologize instead of failing the whole request. `/health` reports
`mcp_status: connected|disconnected` and `/chat` returns HTTP 503 with a
clear message if the MCP subprocess is unreachable.

## 4. Agent orchestration & safety guardrails

`app/agent/orchestrator.py` implements a standard tool-calling loop (max 6
iterations): send the system prompt + conversation + MCP tool schemas to
Groq; if the model requests tool call(s), execute each one through
`MCPClient.call_tool`, log `{tool, arguments, result_summary, error}` to a
`trace` list, extract citations from any RAG-tool result, feed the tool
result back as a `role: tool` message, and repeat; otherwise return the
model's final content as the answer.

**No hidden chain-of-thought** is exposed — the trace is purely the
architectural record of tool calls and results, not the model's internal
reasoning tokens.

**Irreversible-action guardrail (defense in depth):**
1. *Tool level* (`mcp_server/tools/hr_data_tools.py`): `create_mock_hr_ticket`
   defaults `confirm=False` and returns a preview-only response unless
   `confirm=True` is explicitly passed.
2. *Orchestrator level* (`app/agent/workflows.py`):
   `guard_tool_arguments()` checks the user's latest message for an
   affirmative confirmation phrase before ever letting a `confirm=true`
   argument through to the tool call — even if the model tries to set it.
3. *Prompt level*: the system prompt explicitly instructs the model to never
   pass `confirm=true` without the user's explicit go-ahead in this
   conversation.

`draft_hr_email` never needs confirmation because it only returns text — it
cannot send anything.

**Ambiguous requests:** the system prompt instructs the model to ask one
clear clarifying question when required information (most often
`employee_id`) is missing, instead of guessing.

**Sensitive HR issues:** the system prompt instructs the model to retrieve
the Workplace Conduct policy, tell the user it will be escalated to a human
HR representative, and only create a mock case/ticket after explicit
confirmation — never to resolve the issue itself.

## 5. The two required agentic demo tasks

### Task 1 — Remote work eligibility
User: *"Can I work remotely from Colorado for six weeks?"* (employee_id
`E1001`)

Expected tool sequence:
1. `lookup_employee_profile("E1001")` → home state (CA), remote-eligible
2. `search_policy_documents("remote work another state 6 weeks")` and/or
   `check_policy_compliance(...)` → Remote Work Policy's tiered approval
   rules (10-day / 30-day thresholds)
3. Final answer: 6 weeks (42 days) exceeds the 30-day threshold → requires
   formal HR/Legal/Payroll review, cites the Remote Work Policy section, and
   recommends next steps.

### Task 2 — PTO request guidance
User: *"Can I take 3 PTO days next week?"* (employee_id `E1002`)

Expected tool sequence:
1. `check_pto_balance("E1002")` → balance (4.0 days)
2. `search_policy_documents("PTO request approval lead time")` → PTO Policy's
   5-business-day lead time + manager approval requirement
3. Final answer: balance is sufficient, but manager approval is required and
   the request is being submitted with less lead time than ideal → cites PTO
   Policy, and only calls `create_mock_hr_ticket(..., confirm=true)` if the
   user explicitly confirms creating a ticket (see `TT-06` in the eval set).

## 6. Deployment

Single Render web service (`render.yaml`): `pip install -r requirements.txt`
as the build command, `python -m uvicorn app.main:app --host 0.0.0.0 --port
$PORT` as the start command. Python is pinned to 3.11.9 (via the
`PYTHON_VERSION` Render environment variable) since some pinned dependencies
don't yet ship wheels for very new Python releases. The flat vector index is
built into the container at first request (via FastAPI's `lifespan`) if it
doesn't already exist, so no external database or paid storage is required —
`corpus/`, `mock_data/`, and the built vector store are all local files.
`GROQ_API_KEY` is the only required secret, set via the Render dashboard
(`sync: false` in `render.yaml`).

Two production incidents shaped the final deployment config, both
documented here for transparency:
1. An initial deploy failed to build because Render's Python version
   default didn't have a compatible `onnxruntime` wheel -- fixed by pinning
   `PYTHON_VERSION=3.11.9`.
2. After that, the service repeatedly hit Render's 512MB memory limit
   (`Exited with status 137`, an OOM kill) while building the index on first
   request. Root-caused to two compounding issues: running the MCP server as
   a second OS process (fixed by defaulting `MCP_TRANSPORT=memory`, see §3),
   and `onnxruntime`'s default memory-arena allocator being far more
   aggressive than this small model needs (fixed by disabling the arena/
   memory-pattern optimizations in `app/rag/embedding.py`, see §2).

**Cold start:** Render's free tier spins the service down after ~15 minutes
of inactivity. The first request after a spin-down pays for: container boot,
downloading the MiniLM ONNX embedding model (cached on disk after first run,
but not persisted across a full redeploy), and rebuilding the flat vector
index from the small corpus (a few seconds of embedding time). See
`deployed.md` for the observed cold vs. warm latency.

## 7. Evaluation

Full question set: `evaluation/eval_set.json` (25 items — 8 straightforward
policy Q&A, 4 multi-document questions, 6 tool-requiring tasks, 4 ambiguous
requests, 3 out-of-scope requests, each with gold docs/keywords/expected
tools/expected behavior as applicable).

Harness: `evaluation/run_eval.py`. Metrics reported in
`evaluation/results.md` / `results.json`:

- **Citation accuracy** — does at least one cited `doc_id` match the gold
  document(s) for that question.
- **Keyword hit rate** — fraction of expected gold keywords present in the
  final answer (a partial-match proxy for correctness).
- **Tool selection accuracy** — for tool-requiring tasks, were all expected
  tools actually called.
- **Clarification / refusal accuracy** — for ambiguous and out-of-scope
  items, did the agent ask a clarifying question / correctly refuse, rather
  than guessing or answering something it shouldn't.
- **Action-safety pass rate** — did `create_mock_hr_ticket` only ever get
  called with `confirm=true` when the user had actually confirmed in that
  turn.
- **Latency p50/p95** across all items, plus the first call reported
  separately as a cold-start proxy (it includes Groq client initialization)
  versus the warm p50 for the remaining calls.

**Ablation:** retrieval `top_k` = 3 vs. 6, measured as hit-rate against
`gold_docs` for every item that has them. This is retrieval-only (no LLM
calls), so it runs in every invocation of `run_eval.py` regardless of
whether `GROQ_API_KEY` is set.

To reproduce: `pip install -r requirements.txt && python evaluation/run_eval.py`
(requires `GROQ_API_KEY` for the LLM-dependent metrics; the ablation runs
without it).

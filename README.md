# Northwind Analytics HR Agent

An agentic HR assistant for a hypothetical company, Northwind Analytics. It
combines Retrieval-Augmented Generation (RAG) over a corpus of internal HR
policy documents with an MCP (Model Context Protocol) tool layer over mock
employee/PTO/benefits data, so it can answer policy questions and carry out
multi-step HR workflows (remote work eligibility, PTO request guidance,
benefits triage, expense compliance, HR case triage) with cited, grounded
answers.

Built for the Quantic "AI Engineering Techniques and Architectures" project.

**Deployed app:** see `deployed.md` (fill in your Render URL there).

## Architecture (at a glance)

```
Browser (chat UI)
      |
      v
FastAPI web app  ---/chat, /health---> Agent Orchestrator (manual, no framework)
      |                                        |
      |                                        v
      |                                 Groq LLM (tool-calling)
      |                                        |
      |                                        v
      |                          MCP Client <---in-process (default)---> MCP Server
      |                            (real MCP protocol messages; MCP_TRANSPORT=stdio
      |                             also available to run the server as a subprocess)
      |                                                              |
      |                                                    +---------+---------+
      |                                                    |                   |
      |                                              RAG tools           Mock-data tools
      |                                            (search_policy_*,   (lookup_employee_*,
      |                                             get_policy_section, check_pto_balance,
      |                                             check_policy_compliance)  lookup_benefits_status,
      |                                                    |             create_mock_hr_ticket,
      |                                                    v             draft_hr_email)
      |                                          Flat vector index (numpy)     |
      |                                          (built from corpus/)   mock_data/*.json
```

Everything above runs as **one deployed service** (see `render.yaml`), which
keeps this free-tier compatible. See `design-and-evaluation.md` for the full
justification of each choice.

## Repo layout

- `app/` — FastAPI web app, agent orchestrator, MCP client, RAG ingestion/retrieval
- `mcp_server/` — the MCP server and its 8 tool definitions (named `mcp_server/`
  rather than `mcp/` so it can't collide with the installed `mcp` SDK package)
- `corpus/` — 10 policy documents (8 Markdown, 2 HTML)
- `mock_data/` — synthetic employees, PTO balances, benefits, tickets
- `evaluation/` — eval set, evaluation script, results
- `tests/` — pytest suite (app startup, MCP discovery/call, RAG retrieval)
- `.github/workflows/ci.yml` — CI (install, import check, tests, gated deploy trigger)

## Local setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `GROQ_API_KEY` (free at
   https://console.groq.com/keys). No embedding API key is needed — embeddings
   run locally via the ONNX MiniLM model (`onnxruntime`, no torch required),
   downloaded once on first use.

4. Build the RAG index (optional — it auto-builds on first app startup too):

   ```bash
   python -m app.rag.ingest
   ```

5. Run the app:

   ```bash
   python -m uvicorn app.main:app --reload
   ```

   (Using `python -m uvicorn` instead of bare `uvicorn` ensures the repo root
   is on `sys.path` so `app.*` imports resolve correctly.)

   Open http://localhost:8000 for the chat UI. `GET /health` reports app +
   MCP status. `POST /chat` accepts `{"message": "...", "employee_id": "E1001"}`.

   You can also run the MCP server standalone for manual testing:
   `python mcp_server/server.py` (it will just idle waiting for stdio input —
   use the test suite or the app itself to exercise it end-to-end).

## Running tests

```bash
pytest -v
```

Covers: app import/startup, `/health`, MCP tool discovery (asserts all 8
tools are exposed), a real MCP tool call against mock data, and RAG
ingestion/retrieval (including a multi-document retrieval case).

## Running the evaluation

```bash
python evaluation/run_eval.py
```

Requires `GROQ_API_KEY` to be set (it exercises the live agent for the 25
eval items). Writes `evaluation/results.json` and `evaluation/results.md`.
The retrieval top_k ablation runs even without a key (it's retrieval-only).

## Deployment (Render)

See `render.yaml` for the service definition and `deployed.md` for the live
URL, health check, and cold-start notes.

1. Push this repo to GitHub.
2. In Render, "New Web Service" → connect the repo → it will detect
   `render.yaml`.
3. Set the `GROQ_API_KEY` environment variable in the Render dashboard
   (marked `sync: false` in `render.yaml` so it isn't committed).
4. Deploy. First boot builds the flat vector index from `corpus/` (a few
   seconds given the corpus size) and downloads the MiniLM ONNX embedding
   model on first run.

## AI tooling

See `ai-tooling.md` for how AI coding tools were used to build this project.

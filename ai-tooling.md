# AI Tooling Use

I used Claude (via Anthropic's Cowork) as the primary AI coding assistant for
this project, driven from a detailed project plan I first had it produce from
the assignment's PDF brief and rubric.

## What I used it for

- **Planning:** Claude read the assignment PDF and rubric and produced a
  day-by-day build plan (architecture choices, repo layout, tool list,
  evaluation design) mapped against every rubric bullet before any code was
  written.
- **Scaffolding the full codebase in one pass:** repo structure, the FastAPI
  web app (`app/main.py`, chat UI), the RAG pipeline (`app/rag/parsing.py`,
  `chunking.py`, `ingest.py`, `retriever.py`), the MCP server and its 8 tools
  (`mcp_server/`), the MCP client (`app/mcp_client/client.py`), the manual
  agent orchestrator and safety guardrails (`app/agent/`), the pytest suite,
  the GitHub Actions CI workflow, the evaluation harness and 25-item eval
  set, and all required documentation files.
- **Authoring the synthetic content:** the 10-document policy corpus and the
  mock employee/PTO/benefits/ticket JSON data were drafted by Claude to be
  internally consistent (e.g., PTO lead times referenced consistently across
  the PTO policy and the demo scenarios) and clearly fictional.
- **Sandboxed verification:** Claude installed the dependencies and ran the
  test suite in an isolated environment to catch integration bugs before I
  ever ran anything myself. This surfaced and fixed several real issues:
  - A naming collision between a local `mcp/` package and the installed
    `mcp` SDK package -- an early scaffolding pass put the MCP server code
    in a folder named `mcp/`, which shadowed the installed `mcp` SDK package
    and broke every `import mcp` in the codebase once the repo root was on
    `sys.path`. Fixed by moving all MCP server code to `mcp_server/`.
  - `mcp==1.1.2` (Claude's first guess at a pinned version) turned out to
    predate the `mcp.server.fastmcp.FastMCP` API used here; bumped to
    `mcp==1.12.4` after verifying it against the live MCP discovery/call
    tests.
  - Originally scaffolded with `sentence-transformers` for embeddings;
    switched to Chroma's built-in ONNX `DefaultEmbeddingFunction`
    (`onnxruntime`-based) after finding the torch dependency chain
    (500MB+ wheel plus GPU-targeted CUDA packages pulled in by default from
    PyPI) impractical for a free-tier build and a poor match for "modest
    free-tier resources."
  - Claude's sandbox network policy blocks the host chromadb downloads its
    ONNX model weights from (and would equally block HuggingFace if
    sentence-transformers had been kept), so it could not fully exercise the
    live embedding path itself. It verified the RAG pipeline's *logic*
    (parsing both corpus formats, heading-aware chunking counts, Chroma
    add/query mechanics, metadata, doc_id filtering) using a stubbed
    embedding function, and verified MCP discovery/tool-call and app-startup
    tests against the real MCP server. **I ran the full test suite and
    `evaluation/run_eval.py` myself** in an environment with normal internet
    access (and a real `GROQ_API_KEY`) to confirm the live embedding
    download and end-to-end agent behavior before treating anything as
    final.

## What worked well

- Generating the full cross-cutting scaffold (RAG + MCP + agent + web app +
  CI + eval) consistently in one session, with the pieces already wired
  together correctly (e.g., the MCP client's tool discovery feeding directly
  into the Groq function-calling schema) saved what would have been days of
  boilerplate and glue code.
- Having Claude actually execute the test suite in a sandbox surfaced a real
  bug (the `mcp` package name collision) before I had to debug it myself.

## What needed manual follow-up / didn't just work out of the box

- The evaluation harness's LLM-dependent metrics (citation accuracy, tool
  selection accuracy, clarification/refusal accuracy) require a real
  `GROQ_API_KEY` and a live run — Claude could not execute or validate those
  results itself since no key was available in its sandbox. I ran
  `evaluation/run_eval.py` myself after adding my key and reviewed the
  output in `evaluation/results.md` before treating any number as final.
- [Fill in: note any prompt tuning, corpus expansion, or bug fixes you made
  after your own testing and after recording the demo.]
- [Fill in: note anything you changed about the deployment configuration
  once you actually deployed to Render.]

## Responsibility

I reviewed all generated code and configuration before submitting, ran the
test suite and evaluation myself, and take responsibility for the
correctness, security, and academic integrity of the submitted work, per the
course's AI tooling policy.

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
  - Originally scaffolded with `sentence-transformers` for embeddings, then
    Chroma's built-in ONNX `DefaultEmbeddingFunction`, and finally settled on
    a custom `onnxruntime` + `tokenizers` implementation with `chromadb`
    dropped entirely (see the Render deployment debugging notes below for
    why) -- each step trading a heavier dependency for a lighter one better
    suited to "modest free-tier resources."
  - Claude's own sandbox network policy blocked the host the ONNX model
    weights are downloaded from, so it could not fully exercise the live
    embedding path itself. It verified the RAG pipeline's *logic* (parsing
    both corpus formats, heading-aware chunking counts, vector-store
    add/query mechanics, metadata, doc_id filtering) using a stubbed
    embedding function, and verified MCP discovery/tool-call and app-startup
    tests against the real MCP server. **I ran the full test suite,
    `evaluation/run_eval.py`, and the actual deployment myself** with normal
    internet access and a real `GROQ_API_KEY`.

## Debugging the Render deployment (done live, with Claude, after handoff)

Three real production issues came up only once this was actually deployed,
each diagnosed from Render's dashboard/logs and fixed iteratively:

1. **Build failure on Render:** `onnxruntime==1.20.1` had no wheel for the
   Python version Render used by default. Fixed by pinning
   `PYTHON_VERSION=3.11.9` as a Render environment variable (a `runtime.txt`
   file was tried first, on the assumption Render honored the Heroku-style
   convention -- it does not; the env var is Render's actual mechanism).
2. **Out-of-memory kill (exit 137) at startup:** running the MCP server as a
   second OS process (the original stdio-only design) meant a second full
   Python process re-importing the RAG stack, pushing combined memory over
   Render's free-tier 512MB limit. Fixed by adding an in-memory MCP
   transport (`MCP_TRANSPORT=memory`, the new default) that connects a real
   MCP `ClientSession` to the `Server` in-process via the SDK's own
   in-memory stream helper, instead of spawning a subprocess.
3. **Still OOM-killing after fix #2:** measured that `chromadb`'s own import
   footprint plus `onnxruntime`'s default memory-arena allocator (a
   greedy/doubling allocator tuned for throughput, not footprint) were
   together still too much for one process. Fixed by replacing `chromadb`
   with a small custom flat vector store (`app/rag/vector_store.py`, plain
   numpy, brute-force cosine similarity -- entirely adequate at this corpus
   size) and re-implementing the embedding call directly against
   `onnxruntime` with the memory arena and memory-pattern optimizations
   explicitly disabled.

This is a good example of a limitation of AI-assisted development worth
being explicit about: the sandbox Claude built and verified this in could
not reproduce Render's specific memory ceiling or Python version, so these
three issues only surfaced during real deployment -- Claude helped diagnose
and fix each one from the actual error logs, but the deploy-test-fix loop
itself had to happen against the real platform.

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
- After the three deployment issues above were fixed, the free-tier instance
  still OOM-killed on the very first embedding call (model load succeeded,
  but the transformer forward pass over the first batch of policy chunks
  pushed memory over the limit). Diagnosed with Claude by adding explicit
  RSS checkpoint logging (`app/rag/embedding.py`, `_log_rss`) around each
  stage of the embedding path so the exact failure point was visible in
  Render's logs instead of guessed at. That pinpointed the forward pass
  itself as the spike, which was fixed by cutting the tokenizer's max
  sequence length from 256 to 128 tokens (attention memory scales with
  sequence length squared) and reducing the embedding batch size from 8 to
  2, with each batch's intermediate tensors explicitly deleted and
  garbage-collected before the next batch starts.
- Separately, the diagnostic/tuning code Claude wrote used Python's
  Unix-only `resource` module for memory logging, which crashed immediately
  on my Windows dev machine (`ModuleNotFoundError: No module named
  'resource'`) even though it worked fine on Render's Linux containers.
  Fixed with a `resource`/`psutil`/no-op fallback chain so the same file
  works cross-platform.

## Responsibility

I reviewed all generated code and configuration before submitting, ran the
test suite and evaluation myself, and take responsibility for the
correctness, security, and academic integrity of the submitted work, per the
course's AI tooling policy.

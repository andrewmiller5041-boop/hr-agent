# Deployment Info

**Deployed URL:** `https://hr-agent-1i6m.onrender.com`

**Health check:** `https://hr-agent-1i6m.onrender.com/health`

**Chat UI:** `https://hr-agent-1i6m.onrender.com/`

## Cold-start behavior

This app is deployed on Render's free tier, which spins the service down
after approximately 15 minutes of inactivity. The first request after a
spin-down will:

1. Boot the container (a few seconds).
2. Download/load the local `all-MiniLM-L6-v2` ONNX embedding model if it
   isn't already cached in the container.
3. Rebuild the flat vector index from `corpus/` if `app/rag/store/` isn't
   already populated (fast — the corpus is small, well under a minute).
4. Connect the in-process MCP client/server session and complete MCP
   `initialize()` + `list_tools()` (no separate process is spawned by
   default — see `design-and-evaluation.md` §3 for why).

**Observed:**
- First `/chat` request in a session: **~6-7 seconds**
- Second `/chat` request immediately after: **~5 seconds** (about 1-2
  seconds faster)
- Note: this comparison was taken while the service had already been
  actively used within the prior ~15 minutes, so it reflects normal
  first-request-vs-subsequent-request latency rather than a true
  from-spin-down cold start (which would show a much larger gap — tens of
  seconds for container boot alone, on top of the steps below). Both
  numbers are still well within a usable range for the demo. See
  `evaluation/results.md` for the full p50/p95 latency breakdown across all
  25 evaluation items, including the first-call cold-start proxy reported
  there.

## Environment variables required

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | Yes | Free tier at https://console.groq.com/keys |
| `GROQ_MODEL` | No | Defaults to `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL_NAME` | No | Defaults to local MiniLM, no key needed |
| `PYTHON_VERSION` | Yes, on Render | Set to `3.11.9`. Render's default Python version didn't have a compatible `onnxruntime` wheel; see below. |
| `MCP_TRANSPORT` | No | Defaults to `memory` (recommended for free-tier RAM limits); `stdio` runs the MCP server as a separate process. |

## Notes / known issues

Three deployment issues came up across the first several deploy attempts
and were all resolved (see `ai-tooling.md` and `design-and-evaluation.md`
§6 for the full debugging narrative):
- **Build failure:** Render defaulted to a Python version without an
  `onnxruntime==1.20.1` wheel — fixed by setting `PYTHON_VERSION=3.11.9`.
- **Out-of-memory kill (exit 137), attempt 1:** the free-tier 512MB limit
  was exceeded during first-request index building — fixed by (a) running
  the MCP server in-process instead of as a second OS process, and (b)
  disabling onnxruntime's memory-arena allocator and dropping `chromadb` in
  favor of a small custom vector store.
- **Out-of-memory kill (exit 137), attempt 2:** still failed after the
  above, this time during the very first embedding forward pass rather than
  model loading. Diagnosed with explicit RSS logging added at each stage of
  the embedding path (visible in Render's logs), which pinpointed the
  transformer forward pass itself as the spike. Fixed by cutting the
  tokenizer's max sequence length from 256 to 128 tokens and the embedding
  batch size from 8 to 2, with explicit tensor cleanup between batches.

_Fill in any platform outage or further deployment issue notes here, per the
assignment's exception clause, if applicable._

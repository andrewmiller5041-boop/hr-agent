# Deployment Info

**Deployed URL:** _fill in after deploying, e.g._ `https://hr-agent.onrender.com`

**Health check:** `https://<your-app>.onrender.com/health`

**Chat UI:** `https://<your-app>.onrender.com/`

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

_Fill in after your first deployed cold-start test:_
- Cold-start latency for `/health`: **__ seconds**
- Cold-start latency for a full `/chat` request: **__ seconds**
- Warm-request `/chat` latency: **__ seconds** (see `evaluation/results.md`
  for the full p50/p95 breakdown)

## Environment variables required

| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | Yes | Free tier at https://console.groq.com/keys |
| `GROQ_MODEL` | No | Defaults to `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL_NAME` | No | Defaults to local MiniLM, no key needed |
| `PYTHON_VERSION` | Yes, on Render | Set to `3.11.9`. Render's default Python version didn't have a compatible `onnxruntime` wheel; see below. |
| `MCP_TRANSPORT` | No | Defaults to `memory` (recommended for free-tier RAM limits); `stdio` runs the MCP server as a separate process. |

## Notes / known issues

Two deployment issues came up on the first couple of deploy attempts and
were resolved (see `ai-tooling.md` and `design-and-evaluation.md` §6 for the
full debugging narrative):
- **Build failure:** Render defaulted to a Python version without an
  `onnxruntime==1.20.1` wheel — fixed by setting `PYTHON_VERSION=3.11.9`.
- **Out-of-memory kill (exit 137):** the free-tier 512MB limit was exceeded
  during first-request index building — fixed by (a) running the MCP server
  in-process instead of as a second OS process, and (b) disabling
  onnxruntime's memory-arena allocator and dropping `chromadb` in favor of a
  small custom vector store.

_Fill in any platform outage or further deployment issue notes here, per the
assignment's exception clause, if applicable._

# Deployment Info

**Deployed URL:** _fill in after deploying, e.g._ `https://hr-agent.onrender.com`

**Health check:** `https://<your-app>.onrender.com/health`

**Chat UI:** `https://<your-app>.onrender.com/`

## Cold-start behavior

This app is deployed on Render's free tier, which spins the service down
after approximately 15 minutes of inactivity. The first request after a
spin-down will:

1. Boot the container (a few seconds).
2. Download/load the `sentence-transformers/all-MiniLM-L6-v2` embedding
   model if it isn't already cached in the container image.
3. Rebuild the Chroma vector index from `corpus/` if `app/rag/store/` isn't
   already populated (fast — the corpus is small, well under a minute).
4. Spawn the MCP server subprocess and complete MCP `initialize()` +
   `list_tools()`.

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

## Notes / known issues

_Fill in any platform outage or deployment issue notes here, per the
assignment's exception clause, if applicable._

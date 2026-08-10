"""FastAPI web app: chat UI, /chat, /health.

Wires together the RAG index, the MCP client (which spawns mcp_server/server.py
over stdio), and the agent orchestrator.
"""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import config
from app.agent.orchestrator import handle_message
from app.mcp_client.client import MCPClient, MCPToolError
from app.rag import ingest, retriever

mcp_client = MCPClient()

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the vector index if it doesn't exist yet (first boot / cold start
    # on a fresh Render deploy). No-op if already built.
    ingest.build_index()
    try:
        await mcp_client.connect()
    except MCPToolError as exc:
        # Don't crash the whole app if the MCP subprocess fails to start --
        # /health reports it and /chat returns a clear 503 instead of a 500.
        print(f"[startup] MCP connect failed: {exc}")
    yield
    await mcp_client.close()


app = FastAPI(title="Northwind HR Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    employee_id: str | None = None
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    trace: list[dict]
    workflow: str
    latency_ms: int


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    try:
        indexed_docs = retriever.list_indexed_doc_ids()
    except Exception:
        indexed_docs = []

    return {
        "status": "ok",
        "mcp_status": "connected" if mcp_client.connected else "disconnected",
        "indexed_doc_count": len(indexed_docs),
        "groq_model": config.GROQ_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    if not mcp_client.connected:
        try:
            await mcp_client.connect()
        except MCPToolError as exc:
            raise HTTPException(
                status_code=503, detail=f"MCP server unavailable: {exc}"
            ) from exc

    start = time.perf_counter()
    try:
        result = await handle_message(
            mcp_client,
            req.message,
            employee_id=req.employee_id,
            history=req.history,
        )
    except RuntimeError as exc:
        # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = int((time.perf_counter() - start) * 1000)

    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        trace=result.trace,
        workflow=result.workflow,
        latency_ms=latency_ms,
    )

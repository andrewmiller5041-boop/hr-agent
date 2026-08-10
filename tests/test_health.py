"""Smoke test: the app can start (lifespan runs -- RAG index build + MCP
connect) and /health responds. Does not require GROQ_API_KEY."""
from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "mcp_status" in body
        assert "indexed_doc_count" in body


def test_root_serves_chat_ui():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

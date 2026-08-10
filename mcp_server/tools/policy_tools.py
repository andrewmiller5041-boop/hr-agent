"""RAG-backed MCP tools: these are the tools that touch the policy index."""
from app.rag import retriever


def search_policy_documents(query: str, top_k: int = 4) -> dict:
    """Semantic search over the policy corpus. Returns citable chunks."""
    hits = retriever.search(query, top_k=top_k)
    return {
        "query": query,
        "results": [
            {
                "doc_id": h["doc_id"],
                "title": h["title"],
                "section": h["section"],
                "snippet": h["snippet"],
                "score": h["score"],
            }
            for h in hits
        ],
    }


def get_policy_section(doc_id: str, section: str | None = None) -> dict:
    """Direct lookup of a known document's section(s) by doc_id."""
    matches = retriever.get_section(doc_id, section)
    if not matches:
        return {
            "found": False,
            "error": f"No section found for doc_id={doc_id!r} section={section!r}",
        }
    return {"found": True, "doc_id": doc_id, "sections": matches}


def check_policy_compliance(scenario: str, policy_area: str | None = None) -> dict:
    """Retrieve grounded evidence relevant to a compliance scenario.

    This tool intentionally does NOT render a yes/no verdict itself -- it
    surfaces the most relevant policy passages so the calling agent's LLM can
    reason over real, citable text rather than an opaque tool-side judgment.
    """
    query = scenario if not policy_area else f"{policy_area}: {scenario}"
    hits = retriever.search(query, top_k=5)
    return {
        "scenario": scenario,
        "policy_area": policy_area,
        "evidence": [
            {
                "doc_id": h["doc_id"],
                "title": h["title"],
                "section": h["section"],
                "snippet": h["snippet"],
            }
            for h in hits
        ],
    }

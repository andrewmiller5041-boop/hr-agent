"""Top-k retrieval over the local flat vector index, plus direct section
lookup. Backed by app/rag/vector_store.py + app/rag/embedding.py (no
chromadb)."""
from app import config
from app.rag import embedding, ingest


def _ensure_index():
    store = ingest.get_store()
    if store.count() == 0:
        ingest.build_index()
        store = ingest.get_store()
    return store


def search(query: str, top_k: int | None = None, doc_id: str | None = None) -> list[dict]:
    """Return the top_k most similar policy chunks to `query`.

    Optionally filter to a single doc_id (used by get_policy_section and by
    check_policy_compliance when it wants to stay within one policy area).
    """
    top_k = top_k or config.RETRIEVAL_TOP_K
    store = _ensure_index()
    query_embedding = embedding.embed_one(query)
    matches = store.query(query_embedding, top_k=top_k, doc_id=doc_id)

    hits = []
    for record, score in matches:
        meta = record["metadata"]
        hits.append(
            {
                "chunk_id": record["id"],
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title"),
                "section": meta.get("section"),
                "snippet": meta.get("source_snippet"),
                "text": record["document"],
                "score": score,
            }
        )
    return hits


def get_section(doc_id: str, section: str | None = None) -> list[dict]:
    """Direct (non-semantic) lookup of a policy document's section(s)."""
    store = _ensure_index()
    records = store.get_by_doc_id(doc_id)

    matches = []
    for record in records:
        meta = record["metadata"]
        if section is None or section.lower() in meta.get("section", "").lower():
            matches.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "title": meta.get("title"),
                    "section": meta.get("section"),
                    "text": record["document"],
                }
            )
    matches.sort(key=lambda m: m["section"])
    return matches


def list_indexed_doc_ids() -> list[str]:
    store = _ensure_index()
    return store.all_doc_ids()

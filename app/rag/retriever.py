"""Top-k retrieval over the Chroma policy index, plus direct section lookup."""
from app import config
from app.rag import ingest


def _ensure_index():
    collection = ingest.get_collection()
    if collection.count() == 0:
        ingest.build_index()
        collection = ingest.get_collection()
    return collection


def search(query: str, top_k: int | None = None, doc_id: str | None = None) -> list[dict]:
    """Return the top_k most similar policy chunks to `query`.

    Optionally filter to a single doc_id (used by get_policy_section and by
    check_policy_compliance when it wants to stay within one policy area).
    Embedding of both the query and the stored documents is handled by
    Chroma's configured embedding_function (see app/rag/ingest.py), so the
    same model is always used on both sides.
    """
    top_k = top_k or config.RETRIEVAL_TOP_K
    collection = _ensure_index()

    where = {"doc_id": doc_id} if doc_id else None
    results = collection.query(query_texts=[query], n_results=top_k, where=where)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    hits = []
    for i in range(len(ids)):
        meta = metas[i]
        distance = dists[i] if i < len(dists) else None
        hits.append(
            {
                "chunk_id": ids[i],
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title"),
                "section": meta.get("section"),
                "snippet": meta.get("source_snippet"),
                "text": docs[i],
                "score": (1 - distance) if distance is not None else None,
            }
        )
    return hits


def get_section(doc_id: str, section: str | None = None) -> list[dict]:
    """Direct (non-semantic) lookup of a policy document's section(s)."""
    collection = _ensure_index()
    result = collection.get(where={"doc_id": doc_id})
    metas = result.get("metadatas", [])
    docs = result.get("documents", [])

    matches = []
    for meta, doc in zip(metas, docs):
        if section is None or section.lower() in meta.get("section", "").lower():
            matches.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "title": meta.get("title"),
                    "section": meta.get("section"),
                    "text": doc,
                }
            )
    matches.sort(key=lambda m: m["section"])
    return matches


def list_indexed_doc_ids() -> list[str]:
    collection = _ensure_index()
    result = collection.get()
    return sorted({m.get("doc_id") for m in result.get("metadatas", [])})

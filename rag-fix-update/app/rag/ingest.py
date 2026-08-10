"""Build (or rebuild) the local flat vector index from the policy corpus.

Run directly to force a rebuild:
    python -m app.rag.ingest
"""
from app import config
from app.rag import embedding
from app.rag.chunking import chunk_document
from app.rag.parsing import parse_document
from app.rag.vector_store import FlatVectorStore

_store: FlatVectorStore | None = None


def get_store() -> FlatVectorStore:
    global _store
    if _store is None:
        _store = FlatVectorStore(config.VECTOR_STORE_DIR)
    return _store


def build_index(force: bool = False) -> int:
    """Parse -> chunk -> embed -> store. Returns the number of chunks indexed.

    Idempotent: if the store already has chunks and force=False, this is a
    no-op (fast path for app startup / cold start).
    """
    store = get_store()
    if force:
        store.clear()
    if not force and store.count() > 0:
        return store.count()

    corpus_files = sorted(
        p
        for p in config.CORPUS_DIR.glob("*")
        if p.suffix.lower() in (".md", ".html", ".htm")
    )

    all_chunks = []
    for path in corpus_files:
        parsed = parse_document(path)
        all_chunks.extend(chunk_document(parsed))

    if not all_chunks:
        return 0

    documents = [c.text for c in all_chunks]
    ids = [f"{c.doc_id}::{c.chunk_index}" for c in all_chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "title": c.title,
            "section": c.section,
            "chunk_index": c.chunk_index,
            "source_format": c.source_format,
            "source_snippet": c.text[:280],
        }
        for c in all_chunks
    ]

    embeddings = embedding.embed(documents)
    store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return len(ids)


if __name__ == "__main__":
    count = build_index(force=True)
    print(f"Indexed {count} chunks from {config.CORPUS_DIR}")

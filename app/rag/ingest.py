"""Build (or rebuild) the Chroma vector index from the policy corpus.

Run directly to force a rebuild:
    python -m app.rag.ingest
"""
import chromadb
from chromadb.utils import embedding_functions

from app import config
from app.rag.chunking import chunk_document
from app.rag.parsing import parse_document

_client = None
_embedding_function = None

COLLECTION_NAME = "policies"


def get_embedding_function():
    """Chroma's built-in ONNX embedding function (all-MiniLM-L6-v2 via
    onnxruntime). No torch/transformers required -- much smaller install and
    lower RAM footprint, which matters for free-tier hosting. The model
    weights are downloaded once (cached under ~/.cache/chroma/onnx_models)
    on first use.
    """
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_function


def get_client():
    global _client
    if _client is None:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection(client=None):
    client = client or get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def build_index(force: bool = False) -> int:
    """Parse -> chunk -> embed -> store. Returns the number of chunks indexed.

    Idempotent: if the collection already has chunks and force=False, this is
    a no-op (fast path for app startup / cold start).
    """
    client = get_client()
    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = get_collection(client)
    if not force and collection.count() > 0:
        return collection.count()

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

    # No `embeddings=` argument -- Chroma calls our embedding_function
    # (all-MiniLM-L6-v2 via onnxruntime) internally for both add() and
    # query(), so query text and document text always use the same model.
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


if __name__ == "__main__":
    count = build_index(force=True)
    print(f"Indexed {count} chunks from {config.CORPUS_DIR}")

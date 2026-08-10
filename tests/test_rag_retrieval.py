"""Smoke tests for ingestion + retrieval quality basics."""
from app.rag import ingest, retriever


def test_build_index_indexes_all_corpus_docs():
    count = ingest.build_index(force=True)
    assert count > 0
    doc_ids = retriever.list_indexed_doc_ids()
    assert "pto-policy" in doc_ids
    assert "remote-work-policy" in doc_ids
    # confirms both supported source formats (markdown + html) were ingested
    assert "equipment-policy" in doc_ids


def test_search_returns_relevant_pto_chunk():
    hits = retriever.search("How many PTO days do I get per year?", top_k=4)
    assert len(hits) > 0
    assert any(h["doc_id"] == "pto-policy" for h in hits)
    for h in hits:
        assert h["snippet"]
        assert h["title"]


def test_search_multi_document_question_hits_multiple_docs():
    hits = retriever.search(
        "Can I expense a home office chair if I work remotely full time?",
        top_k=6,
    )
    doc_ids = {h["doc_id"] for h in hits}
    # This question spans expense policy and remote work policy.
    assert "expense-policy" in doc_ids


def test_get_section_direct_lookup():
    sections = retriever.get_section("pto-policy")
    assert len(sections) > 0
    assert all(s["doc_id"] == "pto-policy" for s in sections)

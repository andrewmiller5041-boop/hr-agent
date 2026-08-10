"""Heading-aware chunking with a token-window fallback for long sections.

Strategy (documented in design-and-evaluation.md):
1. Split each document into sections along its headings (##/### in Markdown,
   h1/h2/h3 in HTML) — this keeps each chunk topically coherent, which is
   important for citation quality ("Section: Requesting PTO" is a more useful
   citation than "chunk 7").
2. If a section is still longer than CHUNK_SIZE_TOKENS, further split it into
   overlapping windows so no single chunk overwhelms the LLM context or dilutes
   the embedding.

We approximate "tokens" as whitespace-delimited words (word_count * ~1.3 ~= GPT
token count for English policy prose). This avoids an extra tokenizer
dependency and is deterministic, which matters for reproducible ingestion.
"""
from dataclasses import dataclass

from app import config


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    text: str
    chunk_index: int
    source_format: str


def _split_long_section(words: list[str], size: int, overlap: int):
    if len(words) <= size:
        yield words
        return
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield words[start:end]
        if end == len(words):
            break
        start = end - overlap  # deterministic overlap, no randomness


def chunk_document(parsed: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for heading, text in parsed["sections"]:
        words = text.split()
        for window_words in _split_long_section(
            words, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
        ):
            chunk_text = " ".join(window_words)
            if not chunk_text.strip():
                continue
            chunks.append(
                Chunk(
                    doc_id=parsed["doc_id"],
                    title=parsed["title"],
                    section=heading,
                    text=chunk_text,
                    chunk_index=idx,
                    source_format=parsed["source_format"],
                )
            )
            idx += 1
    return chunks

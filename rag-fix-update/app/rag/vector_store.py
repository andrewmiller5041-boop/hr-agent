"""Minimal flat vector store: brute-force cosine similarity over a small
in-memory numpy array, persisted to disk as a .npy file + a JSON sidecar.

This replaces chromadb. For a corpus this size (tens to a couple hundred
chunks), brute-force search over a plain numpy array is effectively instant
and needs no approximate-nearest-neighbor index -- and it avoids chromadb's
large, mostly-unused-here dependency tree, which matters for fitting in
Render's free-tier 512MB RAM limit.
"""
import json
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 384


class FlatVectorStore:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_path = self.persist_dir / "embeddings.npy"
        self._records_path = self.persist_dir / "records.json"
        self._embeddings: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._embeddings_path.exists() and self._records_path.exists():
            try:
                self._embeddings = np.load(self._embeddings_path)
                self._records = json.loads(
                    self._records_path.read_text(encoding="utf-8")
                )
            except Exception:  # noqa: BLE001
                # Corrupt/partial state (e.g. from an interrupted build) --
                # treat as empty so build_index() rebuilds cleanly.
                self._embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
                self._records = []

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._records = []
        if self._embeddings_path.exists():
            self._embeddings_path.unlink()
        if self._records_path.exists():
            self._records_path.unlink()

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        new_records = [
            {"id": ids[i], "document": documents[i], "metadata": metadatas[i]}
            for i in range(len(ids))
        ]
        if self._embeddings.shape[0] == 0:
            self._embeddings = embeddings.astype(np.float32)
        else:
            self._embeddings = np.concatenate(
                [self._embeddings, embeddings.astype(np.float32)], axis=0
            )
        self._records.extend(new_records)
        self._persist()

    def _persist(self) -> None:
        np.save(self._embeddings_path, self._embeddings)
        self._records_path.write_text(json.dumps(self._records), encoding="utf-8")

    def query(
        self, query_embedding: np.ndarray, top_k: int, doc_id: str | None = None
    ) -> list[tuple[dict, float]]:
        if self.count() == 0:
            return []
        scores = self._embeddings @ query_embedding.astype(np.float32)
        order = np.argsort(-scores)
        results: list[tuple[dict, float]] = []
        for idx in order:
            record = self._records[idx]
            if doc_id and record["metadata"].get("doc_id") != doc_id:
                continue
            results.append((record, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def get_by_doc_id(self, doc_id: str) -> list[dict]:
        return [r for r in self._records if r["metadata"].get("doc_id") == doc_id]

    def all_doc_ids(self) -> list[str]:
        return sorted({r["metadata"].get("doc_id") for r in self._records})

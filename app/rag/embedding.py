"""Local ONNX embedding function (all-MiniLM-L6-v2), implemented directly
against `onnxruntime` + `tokenizers` instead of depending on `chromadb`'s
built-in DefaultEmbeddingFunction. This produces numerically equivalent
embeddings (same public model artifact chromadb uses) with two changes made
specifically to fit Render's free-tier 512MB RAM limit:

1. No `chromadb` dependency at all -- this file plus a small flat vector
   store (see vector_store.py) replace it. Removing chromadb's own large,
   mostly-unused-here dependency tree (opentelemetry, grpcio, a bundled
   Kubernetes client, posthog, etc.) trims baseline memory.
2. The onnxruntime InferenceSession is created with its memory arena and
   memory-pattern optimizations disabled, and pinned to a single thread.
   onnxruntime's default CPU arena allocator is a doubling/greedy allocator
   designed for throughput, not footprint -- it can reserve far more memory
   than a small model like this actually needs. Disabling it is a standard
   technique for memory-constrained deployments (e.g. AWS Lambda) and is
   what actually fixed repeated out-of-memory kills on the 512MB instance.
"""
import hashlib
import sys
import tarfile
import time
from pathlib import Path

import httpx
import numpy as np

try:
    # Unix-only (works on Render/Linux); not available on Windows, where
    # this is used for local dev. Diagnostics are best-effort only.
    import resource

    def _current_rss_mb() -> float | None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

except ImportError:  # Windows
    try:
        import psutil

        def _current_rss_mb() -> float | None:
            return psutil.Process().memory_info().rss / (1024 * 1024)

    except ImportError:

        def _current_rss_mb() -> float | None:
            return None


def _log_rss(label: str) -> None:
    """Diagnostic checkpoint: print current process RSS (MB) so that if this
    still OOMs on a memory-constrained host, the platform logs show exactly
    which stage the memory spike happened at, instead of guessing."""
    rss_mb = _current_rss_mb()
    if rss_mb is None:
        print(f"[embedding] {label}: (memory reporting unavailable on this platform)", flush=True, file=sys.stderr)
    else:
        print(f"[embedding] {label}: RSS so far = {rss_mb:.1f} MB", flush=True, file=sys.stderr)

MODEL_NAME = "all-MiniLM-L6-v2"
DOWNLOAD_PATH = Path.home() / ".cache" / "hr_agent" / "onnx_models" / MODEL_NAME
EXTRACTED_FOLDER_NAME = "onnx"
ARCHIVE_FILENAME = "onnx.tar.gz"
# Same public model artifact chromadb's DefaultEmbeddingFunction uses.
MODEL_DOWNLOAD_URL = (
    "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
)
MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"

EMBEDDING_DIM = 384

_tokenizer = None
_session = None


def _sha256_matches(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def _ensure_model_downloaded() -> Path:
    extracted = DOWNLOAD_PATH / EXTRACTED_FOLDER_NAME
    required = ["model.onnx", "tokenizer.json"]
    if all((extracted / f).exists() for f in required):
        return extracted

    _log_rss("before model download")
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    archive_path = DOWNLOAD_PATH / ARCHIVE_FILENAME

    if not (archive_path.exists() and _sha256_matches(archive_path, MODEL_SHA256)):
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                with httpx.stream("GET", MODEL_DOWNLOAD_URL, timeout=60) as resp:
                    resp.raise_for_status()
                    with open(archive_path, "wb") as f:
                        for data in resp.iter_bytes(chunk_size=65536):
                            f.write(data)
                if not _sha256_matches(archive_path, MODEL_SHA256):
                    raise ValueError(
                        "Downloaded embedding model failed checksum verification."
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(1)
        else:
            raise RuntimeError(f"Could not download embedding model: {last_exc}")

    _log_rss("after model download")
    with tarfile.open(archive_path, mode="r:gz") as tar:
        tar.extractall(path=DOWNLOAD_PATH)
    _log_rss("after model extraction")
    return extracted


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer

        extracted = _ensure_model_downloaded()
        _tokenizer = Tokenizer.from_file(str(extracted / "tokenizer.json"))
        _tokenizer.enable_truncation(max_length=256)
        _tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
        _log_rss("after tokenizer loaded")
    return _tokenizer


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        extracted = _ensure_model_downloaded()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # Memory-footprint tuning (see module docstring) -- these are what
        # actually keep peak RSS under Render's 512MB limit.
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        # Skip graph-optimization passes (constant folding, operator fusion,
        # etc.) -- these run once at load time and can transiently use more
        # memory than the base model needs; irrelevant for our latency needs
        # at this tiny scale.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _log_rss("before InferenceSession creation")
        _session = ort.InferenceSession(
            str(extracted / "model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=so,
        )
        _log_rss("after InferenceSession creation")
    return _session


def embed(texts: list[str], batch_size: int = 8) -> np.ndarray:
    """Return an (N, 384) float32 array of L2-normalized embeddings.

    Small batch_size keeps peak memory low during the forward pass -- with a
    corpus this size the extra time from smaller batches is negligible.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    tokenizer = _get_tokenizer()
    session = _get_session()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = [tokenizer.encode(t) for t in batch]
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        last_hidden_state = outputs[0]

        mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1), last_hidden_state.shape
        )
        summed = np.sum(last_hidden_state * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts

        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        norm[norm == 0] = 1e-12
        normalized = (pooled / norm).astype(np.float32)
        all_embeddings.append(normalized)

    _log_rss(f"after embedding {len(texts)} text(s)")
    return np.concatenate(all_embeddings, axis=0)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]

"""Central place for environment-driven configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Must be set before numpy/onnxruntime are imported anywhere (this module is
# imported first by app/main.py). Numpy's BLAS backend and onnxruntime both
# default to spawning a thread pool sized to the HOST machine's CPU count,
# not the container's actual allotment -- on shared hosting like Render's
# free tier this can allocate far more thread-local buffer memory than a
# tiny model like ours needs. Pinning everything to 1 thread is a standard
# fix for memory-constrained containers.
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

BASE_DIR = Path(__file__).resolve().parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Embeddings: all-MiniLM-L6-v2 run locally via onnxruntime + tokenizers
# directly (see app/rag/embedding.py) -- no torch/transformers, and no
# chromadb either (replaced with a small flat vector store, see
# app/rag/vector_store.py). This keeps install size and RAM usage small
# enough for Render's free-tier 512MB limit.
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2 (onnxruntime, local)"
)

CORPUS_DIR = BASE_DIR / os.getenv("CORPUS_DIR", "corpus")
MOCK_DATA_DIR = BASE_DIR / os.getenv("MOCK_DATA_DIR", "mock_data")
VECTOR_STORE_DIR = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "app/rag/store")

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "400"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))

# "memory" (default): MCP server runs in-process (one Python process total --
# important for fitting in Render's free-tier 512MB RAM limit, since a
# second subprocess would re-import the whole chromadb/onnxruntime stack).
# "stdio": spawns mcp_server/server.py as a genuine separate OS process.
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "memory")
MCP_SERVER_CMD = os.getenv("MCP_SERVER_CMD", "python")
MCP_SERVER_SCRIPT = str(BASE_DIR / os.getenv("MCP_SERVER_SCRIPT", "mcp_server/server.py"))

# Deterministic seed used for any sampling (e.g. evaluation subsampling).
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

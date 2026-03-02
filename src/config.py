from pathlib import Path

GROQ_REASONING_MODEL = "groq/llama-3.3-70b-versatile"
GROQ_FAST_MODEL = "groq/llama-3.1-8b-instant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
PDFS_DIR = DATA_DIR / "raw_pdfs"
TREES_DIR = DATA_DIR / "trees"
BENCHMARK_DIR = DATA_DIR / "benchmark"

TOP_K_NODES = 3
TOP_K_CHUNKS = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_TREE_CONTEXT_TOKENS = 8000
LLM_TIMEOUT_SECONDS = 60

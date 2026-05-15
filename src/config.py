"""Central configuration. Edit model names / paths here, not in pipeline code."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"  # change if you pulled a different Ollama model

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 4

# Collection name in Chroma
COLLECTION_NAME = "f1_wikipedia"

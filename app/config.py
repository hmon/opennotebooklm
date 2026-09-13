import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://onb:onb@localhost:5433/onb")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
# A local model can stall or run away. Both must fail the stage, not hang the request.
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "1500"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = float(os.getenv("CHUNK_OVERLAP", "0.15"))

LEXICAL_TOP_K = int(os.getenv("LEXICAL_TOP_K", "30"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "30"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
RRF_K = 60

MIN_ANSWERABLE_CONFIDENCE = float(os.getenv("MIN_ANSWERABLE_CONFIDENCE", "0.7"))

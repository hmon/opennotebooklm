import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://onb:onb@localhost:5433/onb")
# All model inference is remote. Three llama.cpp servers, each serving one
# model, reached over an SSH tunnel; nothing loads weights in this process.
# llama.cpp compiles a JSON schema into a decoding grammar, so structured
# stages cannot emit output that violates their contract.
LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8090")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "qwen3-14b-claude-opus-distill-q4km")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")
# A remote model can stall or run away. Both must fail the stage, not hang.
LLAMA_TIMEOUT = float(os.getenv("LLAMA_TIMEOUT", "300"))
LLAMA_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "1500"))

EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8091")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))

RERANK_URL = os.getenv("RERANK_URL", "http://localhost:8092")
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")

MODEL_TIMEOUT = float(os.getenv("MODEL_TIMEOUT", "180"))

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = float(os.getenv("CHUNK_OVERLAP", "0.15"))

LEXICAL_TOP_K = int(os.getenv("LEXICAL_TOP_K", "30"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "30"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
RRF_K = 60

MIN_ANSWERABLE_CONFIDENCE = float(os.getenv("MIN_ANSWERABLE_CONFIDENCE", "0.7"))

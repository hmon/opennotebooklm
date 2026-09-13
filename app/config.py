import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://onb:onb@localhost:5433/onb")
# llama.cpp's server, speaking the OpenAI chat-completions API. Its JSON-schema
# support compiles the schema into a decoding grammar, so malformed structured
# output is prevented rather than repaired after the fact.
LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "qwen3.5-9b")
# A local model can stall or run away. Both must fail the stage, not hang the request.
LLAMA_TIMEOUT = float(os.getenv("LLAMA_TIMEOUT", "300"))
LLAMA_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "1500"))

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

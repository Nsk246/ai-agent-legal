import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OMDB_API_KEY      = os.getenv("OMDB_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")

# ── Validate on import ────────────────────────────────────────────────────────
REQUIRED_KEYS = {
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "OMDB_API_KEY":      OMDB_API_KEY,
    "TAVILY_API_KEY":    TAVILY_API_KEY,
}

missing = [k for k, v in REQUIRED_KEYS.items() if not v or v.startswith("your_")]
if missing:
    raise EnvironmentError(
        f"\n[CONFIG ERROR] Missing or placeholder API keys: {missing}\n"
        f"Edit your .env file and replace the placeholder values.\n"
    )

# ── Claude models ─────────────────────────────────────────────────────────────
CLAUDE_FAST_MODEL      = "claude-haiku-4-5-20251001"
CLAUDE_REASONING_MODEL = "claude-sonnet-4-6"

# ── Embedding model (runs locally, no API cost) ───────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Agent timeouts (seconds) ──────────────────────────────────────────────────
FACT_AGENT_TIMEOUT      = 15
COMMUNITY_AGENT_TIMEOUT = 20
LLM_TIMEOUT             = 30

# ── Memory / Vector store ─────────────────────────────────────────────────────
FAISS_INDEX_PATH   = "memory/faiss_index"
MAX_MEMORY_RESULTS = 5

# ── Retry settings ────────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_WAIT  = 2

# ── Speed mode ────────────────────────────────────────────────────────────────
# FAST_MODE = True  → uses Haiku for ALL agents (~20s total, less thorough)
# FAST_MODE = False → uses Sonnet for detective+orchestrator (~50s, more thorough)
FAST_MODE = False

# ── Overload retry settings ───────────────────────────────────────────────────
# How many seconds to wait when API returns 529 overloaded
OVERLOAD_RETRY_WAIT = 3
OVERLOAD_MAX_RETRIES = 2

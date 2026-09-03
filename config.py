"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "companion_memory.db"
DEFAULT_SESSION_LOG_DIR = PROJECT_ROOT / "sessions"

# LLM
_requested_llm_model = os.getenv("LLM_MODEL") or os.getenv("COMPANION_LLM_MODEL")
if _requested_llm_model and _requested_llm_model.startswith(("gemini/", "groq/")):
    LLM_MODEL = _requested_llm_model
else:
    LLM_MODEL = "gemini/gemini-2.5-flash"
EMBEDDING_MODEL: str = os.getenv("COMPANION_EMBEDDING_MODEL", "gemini/text-embedding-004")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# LiteLLM reads these env vars directly; ensure Gemini alias is set when configured.
if GEMINI_API_KEY and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Memory retrieval
MEMORY_TOP_K: int = int(os.getenv("COMPANION_MEMORY_TOP_K", "8"))
MEMORY_TOKEN_BUDGET: int = int(os.getenv("COMPANION_MEMORY_TOKEN_BUDGET", "1200"))
RECONCILIATION_CANDIDATE_K: int = int(os.getenv("COMPANION_RECONCILIATION_K", "5"))
EMBEDDING_DIMENSION: int = int(os.getenv("COMPANION_EMBEDDING_DIM", "384"))

# Decay / relevance
MEMORY_DECAY_HALF_LIFE_DAYS: float = float(os.getenv("COMPANION_DECAY_HALF_LIFE_DAYS", "30.0"))
CONFIDENCE_BOOST_ON_REINFORCE: float = 0.05
MAX_CONFIDENCE: float = 1.0
MIN_CONFIDENCE: float = 0.1

# Generation
MAX_RESPONSE_TOKENS: int = int(os.getenv("COMPANION_MAX_RESPONSE_TOKENS", "1024"))
TEMPERATURE: float = float(os.getenv("COMPANION_TEMPERATURE", "0.75"))

# Database
DB_PATH: Path = Path(os.getenv("COMPANION_DB_PATH", str(DEFAULT_DB_PATH)))

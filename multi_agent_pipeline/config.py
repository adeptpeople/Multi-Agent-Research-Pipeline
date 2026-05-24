import os

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_NAME: str = "claude-sonnet-4-6"

# Agent execution limits
DEFAULT_TIMEOUT_MS: int = 30_000
MAX_RETRIES: int = 3
RETRY_DELAY_MS: int = 1_000

# Conflict detection
NUMERIC_CONFLICT_THRESHOLD: float = 0.05   # >5% relative diff flags as contested

# Credibility heuristics by domain suffix
CREDIBILITY_BY_DOMAIN: dict[str, float] = {
    ".gov": 0.95,
    ".edu": 0.88,
    ".org": 0.80,
    "journal": 0.85,
    "reuters": 0.82,
    "bbc": 0.80,
    "ap.org": 0.83,
    "default": 0.70,
}

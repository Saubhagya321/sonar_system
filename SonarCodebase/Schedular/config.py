import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")

# =============================================================================
# REQUIRED SETTINGS
# =============================================================================
API_URL = os.getenv("API_URL")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
PORT_DB_NAME = os.getenv("PORT_DB_NAME")

_missing = [
    name for name, value in (
        ("API_URL", API_URL),
        ("DB_CONNECTION_STRING", DB_CONNECTION_STRING),
        ("PORT_DB_NAME", PORT_DB_NAME),
    )
    if not value
]
if _missing:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        "Check Schedular/.env"
    )

# =============================================================================
# OPTIONAL SETTINGS
# =============================================================================
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
TEST_SPREAD_SECONDS = int(os.getenv("TEST_SPREAD_SECONDS", 15))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))
MISFIRE_GRACE_TIME = int(os.getenv("MISFIRE_GRACE_TIME", 2 * 3600))

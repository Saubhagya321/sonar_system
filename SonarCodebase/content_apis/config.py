import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
if not DB_CONNECTION_STRING:
    raise RuntimeError(
        "DB_CONNECTION_STRING environment variable is not set.\n"
        "Example: mssql+pyodbc://user:password@host/TransElite.Q3"
        "?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
    )

# =============================================================================
# DATA FILE PATHS
# =============================================================================
DATA_DIR = BASE_DIR.parent

USER_HISTORY_PATH = DATA_DIR / "centralized_data" / "userHistory.parquet"
CONTENT_PATH = DATA_DIR / "centralized_data" / "pair_charge_features_complete.csv"
FEATURE_MATRIX_PATH = BASE_DIR / "data" / "content_data" / "feature_matrix" / "feature_matrix.npy"

TOP_K = 10000

# =============================================================================
# API CONFIGURATION
# =============================================================================
TITLE = "Content-Based Recommendation System"
DESCRIPTION = "API for vehicle-chauffeur recommendations"
VERSION = "2.0.0"
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", 8003))

# =============================================================================
# RECOMMENDATION WINDOW DEFAULTS
# =============================================================================
DEFAULT_CANDIDATE_K = 1000
DEFAULT_FINAL_K = 100
DEFAULT_BATCH_FINAL_K = 100
MAX_CANDIDATE_WINDOW = 5000
WINDOW_EXPANSION_FACTOR = 2

SCORE_COLUMNS = ["similarity", "final_score"]

"""
Configuration settings for the recommendation system - UPDATED VERSION
Includes all fixes and new features discussed
"""
import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
if not DB_CONNECTION_STRING:
    raise RuntimeError(
        "DB_CONNECTION_STRING environment variable is not set.\n"
        "Example: mssql+pyodbc://user:password@host/Q3_Recommendation"
        "?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
    )

# Booking source database name (schema-qualified queries use this)
TRANSELITE_DB = os.environ.get("TRANSELITE_DB", "TransElite.Q3")

# Production environment check
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

DB_CONNECT_TIMEOUT = 30  # seconds
DB_POOL_RECYCLE = 3600  # seconds

# =============================================================================
# DATA PATHS
# =============================================================================
BASE_PATH = Path(__file__).resolve().parents[2] / "centralized_data"
DATA_DIR = BASE_PATH
BOOKING_DATA_PATH = BASE_PATH / "booking_data.parquet"
VEHICLE_DATA_PATH = BASE_PATH / "Vehicle.csv"
VEHICLE_TYPE_PATH = BASE_PATH / "VehicleType.csv"
PERSON_CHARGE_RATE_PATH = BASE_PATH / "PersonChargeRate.csv"
CHAUFFEUR_DATA_PATH = BASE_PATH / "Chauffeur.csv"
CLIENT_DATA_PATH = BASE_PATH / "Client.csv"
BOOKING_RATING_PATH = BASE_PATH / "BookingRating.csv"
CHAUFFEUR_STATUS_PATH = BASE_PATH / "ChauffeurStatusLog.csv"

# =============================================================================
# OUTPUT PATHS
# =============================================================================
OUTPUT_DIR = Path("output")
INTERACTION_MATRIX_PATH = OUTPUT_DIR / "interaction_matrix.pkl"
MODEL_READY_DATA_PATH = OUTPUT_DIR / "model_ready_ratings.pkl"
PAIR_FEATURES_PATH = OUTPUT_DIR / "pair_features.pkl"
BOOKING_CLEAN_PATH = OUTPUT_DIR / "booking_clean.pkl"
TRAINED_MODEL_PATH = OUTPUT_DIR / "trained_ensemble_model.pkl"
# Final recommendations CSV written after training.
# Default writes to centralized_data/ (live path used during normal standalone runs).
# The retraining orchestrator overrides this to staging/ so the live path is
# never touched during retraining — only at cutover.
RECOMMENDATIONS_CSV_FINAL_PATH = BASE_PATH / "collabrative_recommendations.csv"

# =============================================================================
# DATA CLEANING PARAMETERS
# =============================================================================
MIN_VALID_DATE = "2000-01-01"
MAX_VALID_DATE = "2040-12-31"

# =============================================================================
# FEATURE ENGINEERING PARAMETERS
# =============================================================================
# UPDATED: Changed weights based on testing (removed recency)
FREQUENCY_WEIGHT = 0.55
COMPLETION_WEIGHT = 0.35
PRICE_WEIGHT = 0.1

EXPLICIT_RATING_WEIGHT = 0.7
IMPLICIT_RATING_WEIGHT = 0.3

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
MIN_INTERACTIONS = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Ensemble weights
# UPDATED: Rebalanced based on performance testing
ENSEMBLE_WEIGHTS = {
    'item_based': 0.5,
    'user_based': 0.4,
    'popularity': 0.1
}

# Collaborative filtering parameters
N_NEIGHBORS = 50
MIN_SIMILARITY = 0.05

# Recommendation parameters
N_RECOMMENDATIONS = 100
BATCH_SIZE = 200
N_TOP_CANDIDATES = 500

# Matrix factorization parameters
N_COMPONENTS = 30
MAX_ITER = 100

# Diversity and bias control
# These help prevent high-ID bias and improve recommendation diversity
JITTER_STRENGTH = 0.01   # Small noise to break ties (prevents ID bias)
ENABLE_DIVERSITY = True  # Enable randomization for diversity

# Scoring weights for cold start recommendations
COLD_START_POPULARITY_WEIGHT = 0.4
COLD_START_CONTEXT_WEIGHT = 0.6
CONTEXT_SCORE_MULTIPLIER = 5.0

# Data quality thresholds
BOOKING_QUALITY_THRESHOLD = 50  # Max bookings for quality score normalization
RECENCY_DECAY_DAYS = 180  # Days for recency score decay
MIN_DATA_QUALITY_SCORE = 0.3  # Minimum data quality score

# =============================================================================
# API CONFIGURATION
# =============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = OUTPUT_DIR / "api.log"

# Model freshness thresholds (in days)
MODEL_WARNING_AGE_DAYS = 90   # Warn if model older than this
MODEL_CRITICAL_AGE_DAYS = 180  # Critical if model older than this

ALLOWED_ORIGINS: List[str] = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8080,http://localhost:8003"
).split(",")

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_PERIOD = os.getenv("RATE_LIMIT_PERIOD", "minute")


class Config:
    """
    Thin adapter over the module-level constants above.

    Exists as an instantiable class (rather than everyone importing the
    module constants directly) solely because retraining_pipeline's
    orchestrator does `config = Config()` and mutates output-path
    attributes on that instance to redirect a retraining run's outputs to
    a staging folder without touching the live paths. All values below are
    mirrors of the module-level constants, not an independent source of
    truth — change values up there, not here.
    """

    DB_CONNECTION_STRING = DB_CONNECTION_STRING
    TRANSELITE_DB = TRANSELITE_DB
    ENVIRONMENT = ENVIRONMENT
    DB_CONNECT_TIMEOUT = DB_CONNECT_TIMEOUT
    DB_POOL_RECYCLE = DB_POOL_RECYCLE

    BASE_PATH = BASE_PATH
    DATA_DIR = DATA_DIR
    BOOKING_DATA_PATH = BOOKING_DATA_PATH
    VEHICLE_DATA_PATH = VEHICLE_DATA_PATH
    VEHICLE_TYPE_PATH = VEHICLE_TYPE_PATH
    PERSON_CHARGE_RATE_PATH = PERSON_CHARGE_RATE_PATH
    CHAUFFEUR_DATA_PATH = CHAUFFEUR_DATA_PATH
    CLIENT_DATA_PATH = CLIENT_DATA_PATH
    BOOKING_RATING_PATH = BOOKING_RATING_PATH
    CHAUFFEUR_STATUS_PATH = CHAUFFEUR_STATUS_PATH

    OUTPUT_DIR = OUTPUT_DIR
    INTERACTION_MATRIX_PATH = INTERACTION_MATRIX_PATH
    MODEL_READY_DATA_PATH = MODEL_READY_DATA_PATH
    PAIR_FEATURES_PATH = PAIR_FEATURES_PATH
    BOOKING_CLEAN_PATH = BOOKING_CLEAN_PATH
    TRAINED_MODEL_PATH = TRAINED_MODEL_PATH
    RECOMMENDATIONS_CSV_FINAL_PATH = RECOMMENDATIONS_CSV_FINAL_PATH

    MIN_VALID_DATE = MIN_VALID_DATE
    MAX_VALID_DATE = MAX_VALID_DATE

    FREQUENCY_WEIGHT = FREQUENCY_WEIGHT
    COMPLETION_WEIGHT = COMPLETION_WEIGHT
    PRICE_WEIGHT = PRICE_WEIGHT

    EXPLICIT_RATING_WEIGHT = EXPLICIT_RATING_WEIGHT
    IMPLICIT_RATING_WEIGHT = IMPLICIT_RATING_WEIGHT

    MIN_INTERACTIONS = MIN_INTERACTIONS
    TEST_SIZE = TEST_SIZE
    RANDOM_STATE = RANDOM_STATE

    ENSEMBLE_WEIGHTS = ENSEMBLE_WEIGHTS

    N_NEIGHBORS = N_NEIGHBORS
    MIN_SIMILARITY = MIN_SIMILARITY

    N_RECOMMENDATIONS = N_RECOMMENDATIONS
    BATCH_SIZE = BATCH_SIZE
    N_TOP_CANDIDATES = N_TOP_CANDIDATES

    N_COMPONENTS = N_COMPONENTS
    MAX_ITER = MAX_ITER

    JITTER_STRENGTH = JITTER_STRENGTH
    ENABLE_DIVERSITY = ENABLE_DIVERSITY

    COLD_START_POPULARITY_WEIGHT = COLD_START_POPULARITY_WEIGHT
    COLD_START_CONTEXT_WEIGHT = COLD_START_CONTEXT_WEIGHT
    CONTEXT_SCORE_MULTIPLIER = CONTEXT_SCORE_MULTIPLIER

    BOOKING_QUALITY_THRESHOLD = BOOKING_QUALITY_THRESHOLD
    RECENCY_DECAY_DAYS = RECENCY_DECAY_DAYS
    MIN_DATA_QUALITY_SCORE = MIN_DATA_QUALITY_SCORE

    API_HOST = API_HOST
    API_PORT = API_PORT
    API_WORKERS = API_WORKERS

    LOG_LEVEL = LOG_LEVEL
    LOG_FILE = LOG_FILE

    MODEL_WARNING_AGE_DAYS = MODEL_WARNING_AGE_DAYS
    MODEL_CRITICAL_AGE_DAYS = MODEL_CRITICAL_AGE_DAYS

    ALLOWED_ORIGINS: List[str] = ALLOWED_ORIGINS

    RATE_LIMIT_REQUESTS = RATE_LIMIT_REQUESTS
    RATE_LIMIT_PERIOD = RATE_LIMIT_PERIOD

    # ------------------------------------------------------------------ #
    #  Class methods                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def validate_production_config(cls):
        """Validate required config for production"""
        if cls.ENVIRONMENT == "production":
            if not os.environ.get("DB_CONNECTION_STRING"):
                raise EnvironmentError("DB_CONNECTION_STRING must be set in production")
            if not os.environ.get("ALLOWED_ORIGINS"):
                raise EnvironmentError("ALLOWED_ORIGINS must be set in production")

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)

    @classmethod
    def validate(cls):
        """
        Validate configuration settings.
        Returns a tuple of (errors, warnings) lists.
        """
        errors = []
        warnings = []
        if cls.ENVIRONMENT == "production":
            if "localhost" in cls.DB_CONNECTION_STRING:
                warnings.append("Using localhost in production DB connection string")
            if "*" in cls.ALLOWED_ORIGINS:
                errors.append("CORS wildcard not allowed in production")

        # Check data directory exists
        if not cls.DATA_DIR.exists():
            warnings.append(f"Data directory does not exist: {cls.DATA_DIR}")

        # Check critical data files exist
        critical_files = [
            (cls.BOOKING_DATA_PATH, "booking_data.parquet"),
        ]
        for filepath, name in critical_files:
            if not filepath.exists():
                errors.append(f"Critical file missing: {name} at {filepath}")

        # Validate weights sum to 1.0
        ensemble_sum = sum(cls.ENSEMBLE_WEIGHTS.values())
        if not (0.99 <= ensemble_sum <= 1.01):
            warnings.append(f"Ensemble weights sum to {ensemble_sum:.3f} (should be 1.0)")

        feature_sum = (
            cls.FREQUENCY_WEIGHT +
            cls.COMPLETION_WEIGHT + cls.PRICE_WEIGHT
        )
        if not (0.99 <= feature_sum <= 1.01):
            warnings.append(f"Feature weights sum to {feature_sum:.3f} (should be 1.0)")

        rating_sum = cls.EXPLICIT_RATING_WEIGHT + cls.IMPLICIT_RATING_WEIGHT
        if not (0.99 <= rating_sum <= 1.01):
            warnings.append(f"Rating weights sum to {rating_sum:.3f} (should be 1.0)")

        return errors, warnings

    @classmethod
    def log_config(cls):
        """Log current configuration using the module logger."""
        logger.info("=" * 80)
        logger.info("CONFIGURATION SETTINGS")
        logger.info("=" * 80)
        logger.info(f"  Environment: {cls.ENVIRONMENT}")

        logger.info("  Data Paths:")
        logger.info(f"   Base path:        {cls.BASE_PATH}")
        logger.info(f"   Data directory:   {cls.DATA_DIR}")
        logger.info(f"   Output directory: {cls.OUTPUT_DIR}")

        logger.info("  Model Parameters:")
        logger.info(f"   Min interactions: {cls.MIN_INTERACTIONS}")
        logger.info(f"   N neighbors:      {cls.N_NEIGHBORS}")
        logger.info(f"   Min similarity:   {cls.MIN_SIMILARITY}")

        logger.info("  Ensemble Weights:")
        for name, weight in cls.ENSEMBLE_WEIGHTS.items():
            logger.info(f"   {name}: {weight:.2f}")

        logger.info("  Feature Weights:")
        logger.info(f"   Frequency:  {cls.FREQUENCY_WEIGHT:.2f}")
        logger.info(f"   Completion: {cls.COMPLETION_WEIGHT:.2f}")
        logger.info(f"   Price:      {cls.PRICE_WEIGHT:.2f}")

        logger.info("  API Configuration:")
        logger.info(f"   Host:      {cls.API_HOST}")
        logger.info(f"   Port:      {cls.API_PORT}")
        logger.info(f"   Workers:   {cls.API_WORKERS}")
        logger.info(f"   Log level: {cls.LOG_LEVEL}")

        logger.info("  Diversity Settings:")
        logger.info(f"   Enable diversity: {cls.ENABLE_DIVERSITY}")
        logger.info(f"   Jitter strength:  {cls.JITTER_STRENGTH}")

        logger.info("=" * 80)

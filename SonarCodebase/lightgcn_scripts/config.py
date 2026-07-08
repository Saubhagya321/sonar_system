
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")

# =============================================================================
# ENVIRONMENT
# =============================================================================
ENVIRONMENT = os.getenv('ENV', 'development')  # development, staging, production

#===========================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Set DEBUG_LOGGING=true in your environment to enable verbose debug output.
# When False (default), only INFO-level and above messages are shown.
DEBUG_LOGGING = os.getenv('DEBUG_LOGGING', 'false').lower() == 'true'

def setup_logging(debug: bool = None) -> None:
    """
    Configure root logger for the entire application.

    Call this once at your application entry point (e.g. main script or
    Flask app factory).  All module-level loggers created with
    ``logging.getLogger(__name__)`` will inherit this configuration.

    Args:
        debug: Override DEBUG_LOGGING from config if provided.
    """
    use_debug = debug if debug is not None else DEBUG_LOGGING
    level = logging.DEBUG if use_debug else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True   # re-configure even if basicConfig was called earlier
    )

# =============================================================================
# BASE DIRECTORIES
# =============================================================================
BASE_DIR = Path(__file__).parent

def _dir_from_env(env_var: str, default_subdir: str) -> Path:
    """
    Resolve a directory from the environment, anchored to BASE_DIR.

    A relative value (e.g. `.env`'s `DATA_DIR=./data`) must resolve against
    this module's own folder, not the process's current working directory —
    otherwise this breaks whenever lightgcn_scripts is imported from another
    cwd (e.g. retraining_pipeline/main_orchestrator.py loading it directly).
    """
    value = os.getenv(env_var)
    path = Path(value) if value else BASE_DIR / default_subdir
    return path if path.is_absolute() else BASE_DIR / path

DATA_DIR = _dir_from_env('DATA_DIR', 'data')
MODEL_DIR = _dir_from_env('MODEL_DIR', 'models')
OUTPUT_DIR = _dir_from_env('OUTPUT_DIR', 'output')
PIPELINE_DIR = _dir_from_env('PIPELINE_DIR', 'pipeline_data')

# =============================================================================
# DATE CLEANING BOUNDS
# =============================================================================
# Single source of truth for the valid booking-date range used across
# metadata_enrichment.py and data_preparation_and_loading.py. Dates outside
# this range are treated as invalid (masked/clipped to NaT).
MIN_VALID_DATE = os.getenv('MIN_VALID_DATE', '2015-01-01')
MAX_VALID_DATE = os.getenv('MAX_VALID_DATE', '2026-12-31')

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
if not DB_CONNECTION_STRING:
    raise RuntimeError(
        "DB_CONNECTION_STRING environment variable is not set.\n"
        "Example: mssql+pyodbc://user:password@host/TransElite.Q3"
        "?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
    )

# Name of the booking database, used to qualify cross-database table
# references in raw SQL (e.g. `FROM [{DB_NAME}].[dbo].[Port]`).
DB_NAME = os.getenv('DB_NAME', 'TransElite.Q3')


# =============================================================================
# DATA FILES - METADATA (CSV)
# =============================================================================
METADATA_FILES = {
    'vehicle': DATA_DIR / 'vehicle_final.csv',
    'vehicle_type': DATA_DIR / 'vehicle_type_final.csv',
    'person_charge_rate': DATA_DIR / 'personChargeRate_final.csv',
    'chauffeur': DATA_DIR / 'chauff_final.csv',
    'client': DATA_DIR / 'client_final.csv'
}

# =============================================================================
# DATA FILES - BOOKING DATA
# =============================================================================
BOOKING_DATA_PATH = DATA_DIR / 'booking_data.parquet'

# =============================================================================
# MODEL FILES
# =============================================================================
MODEL_PATHS = {
    'best_model': MODEL_DIR / 'best_model_complete_lightgcn.pt',
    'training_checkpoint': MODEL_DIR / 'best_lightgcn_model.pt',
    'training_history': MODEL_DIR / 'training_history.csv'
}

# =============================================================================
# PIPELINE DATA FILES
# =============================================================================
PIPELINE_PATHS = {
    'step2_results': PIPELINE_DIR / 'step2_results.pkl',
    'step3_results': PIPELINE_DIR / 'step3_results.pkl',
    'step4_results': PIPELINE_DIR / 'step4_results.pkl'
}

# =============================================================================
# GRAPH OUTPUT FILES
# =============================================================================
GRAPH_OUTPUT_PREFIX = 'chauffeur_vehicle_heterogeneous'
GRAPH_OUTPUT_DIR = OUTPUT_DIR / 'graph'

def get_graph_file_path(file_type):
    """Get graph output file path by type"""
    file_types = {
        'edges': f'{GRAPH_OUTPUT_PREFIX}_edges.csv',
        'adj_matrix': f'{GRAPH_OUTPUT_PREFIX}_adj_matrix.npz',
        'interaction_matrices': f'{GRAPH_OUTPUT_PREFIX}_interaction_matrices.npz',
        'pytorch': f'{GRAPH_OUTPUT_PREFIX}_pytorch.npz',
        'mappings': f'{GRAPH_OUTPUT_PREFIX}_mappings.npz',
        'analysis': f'{GRAPH_OUTPUT_PREFIX}_analysis.json'
    }
    return GRAPH_OUTPUT_DIR / file_types.get(file_type, file_type)

# =============================================================================
# RECOMMENDATION OUTPUT FILES
# =============================================================================
RECOMMENDATIONS_OUTPUT_DIR = OUTPUT_DIR / 'recommendations'

RECOMMENDATION_FILES = {
    'vehicle': RECOMMENDATIONS_OUTPUT_DIR / 'client_vehicle_recommendations_COMPLETE.csv',
    'chauffeur': RECOMMENDATIONS_OUTPUT_DIR / 'client_chauffeur_recommendations_COMPLETE.csv',
    'vc_pair': RECOMMENDATIONS_OUTPUT_DIR / 'client_vc_pair_recommendations_COMPLETE.csv'
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
BEST_CONFIG = {
    'embedding_dim': 192,
    'n_layers': 5,
    'learning_rate': 0.003090,
    'reg_weight': 0.000004,
    'batch_size': 1024,
    'n_epochs': 5,
    'patience': 20,
    'eval_every': 5,
    'use_scheduler': True,
    'scheduler_type': 'cosine',
    'dropout': 0.2022771474289386,
    'n_negatives': 4,
    'cv_weight': 0.8616826801241391,
    'cc_weight': 0.9343378488475521,
    'vc_weight': 1.048555781057173
}

# =============================================================================
# API CONFIGURATION
# =============================================================================
API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 5000)),
    'debug': os.getenv('API_DEBUG', 'false').lower() == 'true'
}

# =============================================================================
# DIRECTORY INITIALIZATION
# =============================================================================
def ensure_directories():
    """Create all required directories if they don't exist"""
    directories = [
        DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        PIPELINE_DIR,
        GRAPH_OUTPUT_DIR,
        RECOMMENDATIONS_OUTPUT_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


def validate_config():
    """
    Validate all required configuration at startup.
    Call this from your application entry point.
    """
    errors = []

    # Check that base directories exist or can be created
    for name, path in [('DATA_DIR', DATA_DIR), ('MODEL_DIR', MODEL_DIR)]:
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                errors.append(f"Cannot create directory {name}: {path}")
    
    if errors:
        raise ConfigurationError("Configuration errors:\n" + "\n".join(errors))
    
    return True
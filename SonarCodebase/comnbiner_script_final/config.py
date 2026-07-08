
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

def setup_logging():
    """Centralized logging configuration. Call this once at app startup."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "weights"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, WEIGHTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Grid Search Data Paths (CSV files)
GRID_SEARCH_PATHS = {
    'lightgcn': DATA_DIR / "grid_search" / "lightgcn_recommendations.csv",
    'collaborative': DATA_DIR / "grid_search" / "collaborative_recommendations.csv"
}

# Real-time Request Data Paths
REQUEST_PATHS = {
    'lightgcn_json': DATA_DIR / "requests" / "lightgcn_request_{client_id}.json",
    'collaborative_json': DATA_DIR / "requests" / "collaborative_request_{client_id}.json",
    'item_item_csv': DATA_DIR / "requests" / "item_item_similarity_{client_id}.csv",
    'user_item_csv': DATA_DIR / "requests" / "user_item_similarity_{client_id}.csv"
}

# Client Type Configuration
CLIENT_TYPE_FILE = os.environ.get("CLIENT_TYPE_FILE", DATA_DIR / "client_final.csv")

# Mapping from RefClientType values to system types
CLIENT_TYPE_MAPPING = {
    1: 'CORPORATE',
    2: 'GLOBAL CASH',
    7: 'GLOBAL CREDIT',
}

# Weight files per client type
WEIGHT_FILES = {
    'CORPORATE': WEIGHTS_DIR / "weights_CORPORATE.json",
    'GLOBAL_CASH': WEIGHTS_DIR / "weights_GLOBAL_CASH.json",
    'GLOBAL_CREDIT': WEIGHTS_DIR / "weights_GLOBAL_CREDIT.json",
    'GLOBAL CASH': WEIGHTS_DIR / "weights_GLOBAL_CASH.json",
    'GLOBAL CREDIT': WEIGHTS_DIR / "weights_GLOBAL_CREDIT.json",
    'UNKNOWN': WEIGHTS_DIR / "weights_UNKNOWN.json"
}

# RRF Configuration
RRF_K = 30  # Standard RRF constant

# Default weights - UPDATED: content-based now 0.15 instead of 0.1
DEFAULT_WEIGHTS = {
    'alpha1_rrf': 0.3,
    'alpha2_gcn': 0.3,
    'alpha3_item_sim': 0.15,  # Changed from 0.1 to 0.15
    'alpha4_user_sim': 0.15,  # Changed from 0.1 to 0.15
    'alpha5_confidence': 0.1
}

# Grid Search Configuration - OPTIMIZED FOR SPEED
GRID_SEARCH_CONFIG = {
    'alpha_ranges': {
        'alpha1_rrf': [0.2, 0.3, 0.4],
        'alpha2_gcn': [0.2, 0.3, 0.4],
        'alpha3_item_sim': [0.15, 0.2],
        'alpha4_user_sim': [0.15, 0.2],
        'alpha5_confidence': [0.1, 0.15]
    },
    'validation_k': 10,
    'min_interactions': 5,
    
    'max_clients_per_type': {
        'CORPORATE': 1200,
        'GLOBAL CASH': 400,
        'GLOBAL CREDIT': 200,
        'UNKNOWN': 100
    }
}

# Model-specific configurations
MODEL_CONFIGS = {
    'lightgcn': {
        'score_columns': ['cv_score', 'cc_score', 'vc_score'],
        'rank_column': 'rank',
        'pair_id_columns': ['original_chauffeur_id', 'original_vehicle_id']
    },
    'collaborative': {
        'score_column': 'predicted_rating',
        'confidence_column': 'confidence',
        'rank_column': 'rank',
        'pair_id_column': 'chauffeur_vehicle_pair'
    },
    'item_similarity': {
        'similarity_column': 'similarity',
        'ref_chauffeur_column': 'RefChauffeur',
        'ref_vehicle_column': 'RefVehicle'
    },
    'user_similarity': {
        'similarity_column': 'similarity',
        'ref_chauffeur_column': 'RefChauffeur',
        'ref_vehicle_column': 'RefVehicle'
    }
}

# Client type mapping
CLIENT_TYPES = {
    'CORPORATE': 'CORPORATE',
    'GLOBAL CASH': 'GLOBAL CASH',
    'GLOBAL CREDIT': 'GLOBAL CREDIT',
    'GLOBAL_CASH': 'GLOBAL CASH',
    'GLOBAL_CREDIT': 'GLOBAL CREDIT',
    'UNKNOWN': 'UNKNOWN'
}

# Business rules and filters
BUSINESS_RULES = {
    'max_recommendations': 100,
    'min_confidence_threshold': 0.0,
    'min_combined_score': 0.0
}
# Database Configuration

DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
TRANSELITE_DB = os.environ.get("TRANSELITE_DB", "TransElite.Q3")

if not DB_CONNECTION_STRING:
    raise RuntimeError("DB_CONNECTION_STRING environment variable must be set")

# Booking data configuration
BOOKING_COLUMNS = ['ID', 'RefChauffeur', 'RefVehicle', 'RefClient', 
                   'RefBookingStatus', 'AppointmentDate']

# Completed booking statuses (for ground truth)
COMPLETED_STATUSES = [9, 10, 11, 13]


API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "30"))
LIGHTGCN_API = os.environ.get("LIGHTGCN_API_URL", "http://localhost:5000/recommend")
COLLABORATIVE_API = os.environ.get("COLLABORATIVE_API_URL", "http://localhost:8000/api/v1/recommendations")
USER_CONTENT_API = os.environ.get("USER_CONTENT_API_URL", "http://localhost:8003/recommendations/user")
ITEM_CONTENT_API = os.environ.get("ITEM_CONTENT_API_URL", "http://localhost:8003/recommendations/item")
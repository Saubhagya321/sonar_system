

import os
from pathlib import Path

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DB_CONFIG = {
    'server': 'localhost',
    'database': os.environ.get("TRANSELITE_DB", "TransElite.Q3"),
    'driver': 'ODBC Driver 17 for SQL Server',
    'trusted_connection': 'yes'
}

TRANSELITE_DB = DB_CONFIG['database']

def get_db_connection_string() -> str:
    """Get SQL Server connection string."""
    return (
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"Trusted_Connection={DB_CONFIG['trusted_connection']};"
    )


DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")
if not DB_CONNECTION_STRING:
    raise RuntimeError(
        "DB_CONNECTION_STRING environment variable is not set.\n"
        "Example: mssql+pyodbc://user:password@host/Q3_Recommendation"
        "?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
    )

# =============================================================================
# LLM CONFIGURATION (Gemini)
# =============================================================================
LLM_CONFIG = {
    'API_KEY': os.environ.get('GEMINI_API_KEY', ''),
    'API_URL': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent',
    'TIMEOUT': 10,
    'TEMPERATURE': 0.1,
    'ENABLE_FALLBACK': True,
    'ENABLE_CACHE': True,
    'CACHE_TTL': 3600
}


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAMES = {
    'lightgcn': 'lightgcn',
    'collaborative': 'collaborative',
    'user_content': 'user_content',
    'item_content': 'item_content',
    'rrf': 'rrf'
}

MODEL_TO_WEIGHT_KEY = {
    'rrf': 'alpha1_rrf',
    'lightgcn': 'alpha2_gcn',
    'item_content': 'alpha3_item_sim',
    'user_content': 'alpha4_user_sim',
    'collaborative': 'alpha5_confidence'
}

WEIGHT_KEY_TO_MODEL = {v: k for k, v in MODEL_TO_WEIGHT_KEY.items()}

DEFAULT_BASE_WEIGHTS = {
    'alpha1_rrf': 0.30,
    'alpha2_gcn': 0.30,
    'alpha3_item_sim': 0.15,
    'alpha4_user_sim': 0.15,
    'alpha5_confidence': 0.10
}


# =============================================================================
# REJECTION CATEGORIES (Text-Based System)
# =============================================================================
REJECTION_CATEGORIES = {
    # LEARNABLE CATEGORIES (8)
    'client_preference': {
        'learnable': True,
        'description': "Client's personal taste or preference about driver/vehicle style",
        'examples': ['Client prefers female drivers', 'Client wants older chauffeur'],
        'model_weights': {
            'lightgcn': 0.15,
            'collaborative': 0.25,
            'user_content': 0.45,
            'item_content': 0.15
        }
    },
    
    'chauffeur_quality': {
        'learnable': True,
        'description': "Issues with driver behavior, professionalism, attitude, or skills",
        'examples': ['Driver was rude', 'Unprofessional behavior', 'Poor driving skills'],
        'model_weights': {
            'lightgcn': 0.20,
            'collaborative': 0.40,
            'user_content': 0.30,
            'item_content': 0.10
        }
    },
    
    'vehicle_mismatch': {
        'learnable': True,
        'description': "Vehicle type, size, or features don't match client needs",
        'examples': ['Car too small', 'Need SUV not sedan', 'Wrong vehicle class'],
        'model_weights': {
            'lightgcn': 0.20,
            'collaborative': 0.10,
            'user_content': 0.20,
            'item_content': 0.50
        }
    },
    
    'poor_match': {
        'learnable': True,
        'description': "General poor recommendation quality",
        'examples': ['Bad match overall', 'Doesn\'t fit', 'Wrong recommendation'],
        'model_weights': {
            'lightgcn': 0.25,
            'collaborative': 0.25,
            'user_content': 0.25,
            'item_content': 0.25
        }
    },
    
    'past_negative_experience': {
        'learnable': True,
        'description': "Client had bad history with this driver or vehicle before",
        'examples': ['Complained about this driver before', 'Had issues last time'],
        'model_weights': {
            'lightgcn': 0.30,
            'collaborative': 0.40,
            'user_content': 0.20,
            'item_content': 0.10
        }
    },
    
    'comfort_preference': {
        'learnable': True,
        'description': "Comfort or luxury level doesn't meet client expectations",
        'examples': ['Client expects premium', 'Not luxurious enough'],
        'model_weights': {
            'lightgcn': 0.15,
            'collaborative': 0.20,
            'user_content': 0.30,
            'item_content': 0.35
        }
    },
    
    'reliability_concern': {
        'learnable': True,
        'description': "Driver or vehicle reliability/punctuality issues",
        'examples': ['Driver is often late', 'Unreliable service'],
        'model_weights': {
            'lightgcn': 0.25,
            'collaborative': 0.45,
            'user_content': 0.15,
            'item_content': 0.15
        }
    },
    
    'service_expectation': {
        'learnable': True,
        'description': "Service level doesn't match client tier or expectations",
        'examples': ['VIP client needs better', 'Premium client wants top service'],
        'model_weights': {
            'lightgcn': 0.20,
            'collaborative': 0.30,
            'user_content': 0.35,
            'item_content': 0.15
        }
    },
    
    # NON-LEARNABLE CATEGORIES (3)
    'chauffeur_unavailable': {
        'learnable': False,
        'description': "Driver is busy, not available, or has scheduling conflict",
        'examples': ['Driver is booked', 'Not available'],
        'model_weights': {}
    },
    
    'vehicle_unavailable': {
        'learnable': False,
        'description': "Vehicle not available, in maintenance, or already booked",
        'examples': ['Car in maintenance', 'Vehicle booked'],
        'model_weights': {}
    },
    
    'external_factor': {
        'learnable': False,
        'description': "Weather, client cancellation, emergency, unclear reason",
        'examples': ['Weather issues', 'Client cancelled', 'Emergency'],
        'model_weights': {}
    }
}


# =============================================================================
# LEARNING CONFIGURATION (UPDATED: Separate Rewards & Penalties)
# =============================================================================
FEEDBACK_CONFIG = {
    # -------------------------------------------------------------------------
    # Observation Thresholds
    # -------------------------------------------------------------------------
    'MIN_CLIENT_OBSERVATIONS': 30,
    'MIN_SEGMENT_OBSERVATIONS': 50,
    'MIN_GLOBAL_OBSERVATIONS': 100,
    
    # -------------------------------------------------------------------------
    # Learning Parameters
    # -------------------------------------------------------------------------
    'LEARNING_RATE': 0.1,
    'MAX_LEARNED_BLEND': 0.65,
    'BLEND_SCALE_OBSERVATIONS': 15,
    
    # -------------------------------------------------------------------------
    # RRF Configuration
    # -------------------------------------------------------------------------
    'RRF_FIXED_PORTION': 0.15,
    'LEARNABLE_PORTION': 0.85,
    
    # -------------------------------------------------------------------------
    # Cache Configuration
    # -------------------------------------------------------------------------
    'WEIGHT_CACHE_TTL': 3600,
    'MAX_CACHE_SIZE': 10000,
    
   
    
    # Reward for models that scored the SELECTED pair
    'SELECTION_REWARD_FACTOR': 0.08,  # Small positive reward
    
    # Penalty factors for REJECTED pairs
    'SCORE_PENALTY_WEIGHT': 0.5,      # α: weight for score-based penalty
    'CATEGORY_PENALTY_WEIGHT': 0.5,   # β: weight for category-based penalty
    'REJECTION_PENALTY_FACTOR': 0.05, # Base penalty factor (smaller than before)
    
    # Maximum penalty per rejection (cap to prevent extreme values)
    'MAX_REJECTION_PENALTY': 0.15,
    
    # Legacy parameters (backward compatibility)
    'BASE_PENALTY_FACTOR': 0.15,  # Keep for backward compatibility
    'RANK_DECAY_FACTOR': 0.9,
    'REJECTION_PENALTY': 0.1,
    'REJECTION_PENALTY_TOP_K': 10,
    'REJECTION_SCORE_THRESHOLD': 0.5,
    
    # -------------------------------------------------------------------------
    # Hierarchical Learning
    # -------------------------------------------------------------------------
    'LEARNING_HIERARCHY': ['client', 'segment', 'global'],
    'ENABLE_SEGMENT_FALLBACK': True,
    'ENABLE_GLOBAL_FALLBACK': True,
}


# =============================================================================
# LEGACY REJECTION_REASONS (For backward compatibility)
# =============================================================================
REJECTION_REASONS = {
    'client_preference': {'learnable': True, 'maps_to': 'client_preference'},
    'poor_match': {'learnable': True, 'maps_to': 'poor_match'},
    'quality_concerns': {'learnable': True, 'maps_to': 'chauffeur_quality'},
    'vehicle_preference': {'learnable': True, 'maps_to': 'vehicle_mismatch'},
    'chauffeur_preference': {'learnable': True, 'maps_to': 'client_preference'},
    'past_bad_experience': {'learnable': True, 'maps_to': 'past_negative_experience'},
    'chauffeur_unavailable': {'learnable': False, 'maps_to': 'chauffeur_unavailable'},
    'vehicle_unavailable': {'learnable': False, 'maps_to': 'vehicle_unavailable'},
    'vehicle_breakdown': {'learnable': False, 'maps_to': 'vehicle_unavailable'},
    'chauffeur_sick': {'learnable': False, 'maps_to': 'chauffeur_unavailable'},
    'traffic_delay': {'learnable': False, 'maps_to': 'external_factor'},
    'weather_issue': {'learnable': False, 'maps_to': 'external_factor'},
    'booking_cancelled': {'learnable': False, 'maps_to': 'external_factor'},
    'dispatcher_override': {'learnable': False, 'maps_to': 'external_factor'},
    'scheduling_conflict': {'learnable': False, 'maps_to': 'chauffeur_unavailable'},
    'vehicle_maintenance': {'learnable': False, 'maps_to': 'vehicle_unavailable'},
    'system_error': {'learnable': False, 'maps_to': 'external_factor'},
    'other': {'learnable': False, 'maps_to': 'external_factor'},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def is_learnable_rejection(category_code: str) -> bool:
    """Check if a rejection category should influence model learning."""
    category = REJECTION_CATEGORIES.get(category_code, {})
    return category.get('learnable', False)


def get_category_model_weights(category_code: str) -> dict:
    """Get model penalty weights for a rejection category."""
    category = REJECTION_CATEGORIES.get(category_code, {})
    return category.get('model_weights', {})


def get_all_learnable_categories() -> list:
    """Get list of all learnable category codes."""
    return [code for code, config in REJECTION_CATEGORIES.items() if config['learnable']]


def get_all_nonlearnable_categories() -> list:
    """Get list of all non-learnable category codes."""
    return [code for code, config in REJECTION_CATEGORIES.items() if not config['learnable']]


def validate_config():
    """Validate configuration consistency."""
    total_weight = sum(DEFAULT_BASE_WEIGHTS.values())
    if not (0.99 <= total_weight <= 1.01):
        raise ValueError(f"DEFAULT_BASE_WEIGHTS must sum to 1.0, got {total_weight}")
    
    if not (0.0 <= FEEDBACK_CONFIG['LEARNING_RATE'] <= 1.0):
        raise ValueError("LEARNING_RATE must be between 0.0 and 1.0")
    
    alpha = FEEDBACK_CONFIG['SCORE_PENALTY_WEIGHT']
    beta = FEEDBACK_CONFIG['CATEGORY_PENALTY_WEIGHT']
    if not (0.99 <= alpha + beta <= 1.01):
        raise ValueError("SCORE_PENALTY_WEIGHT + CATEGORY_PENALTY_WEIGHT must sum to 1.0")
    
    for cat_code, cat_config in REJECTION_CATEGORIES.items():
        if cat_config['learnable']:
            weights = cat_config.get('model_weights', {})
            total = sum(weights.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Model weights for '{cat_code}' must sum to 1.0, got {total}")
    
    


__all__ = [
    'DB_CONFIG', 'TRANSELITE_DB', 'get_db_connection_string', 'LLM_CONFIG',
    'MODEL_NAMES', 'MODEL_TO_WEIGHT_KEY', 'WEIGHT_KEY_TO_MODEL',
    'DEFAULT_BASE_WEIGHTS', 'REJECTION_CATEGORIES', 'REJECTION_REASONS',
    'FEEDBACK_CONFIG', 'is_learnable_rejection', 'get_category_model_weights',
    'get_all_learnable_categories', 'get_all_nonlearnable_categories', 'validate_config',
]




##### Additional changes 
# =============================================================================

# FEEDBACK SYSTEM CONFIGURATION

# =============================================================================
 
FEEDBACK_CONFIG = {

    'LEARNING_RATE': 0.1,

    'SCORE_PENALTY_WEIGHT': 0.5,

    'CATEGORY_PENALTY_WEIGHT': 0.5,

    'SELECTION_REWARD_FACTOR': 0.08,

    'REJECTION_PENALTY_FACTOR': 0.05,

    'MAX_REJECTION_PENALTY': 0.15,

    'MIN_CLIENT_OBSERVATIONS': 10,

    'MIN_SEGMENT_OBSERVATIONS': 25,

    'MIN_GLOBAL_OBSERVATIONS': 50,

    'RRF_FIXED_PORTION': 0.15,

    'BLEND_SCALE_OBSERVATIONS': 15,

    'MAX_LEARNED_BLEND': 0.8,

}
 
MODEL_NAMES = ['lightgcn', 'collaborative', 'user_content', 'item_content']
 
DEFAULT_BASE_WEIGHTS = {

    'alpha1_rrf': 0.2,

    'alpha2_gcn': 0.35,

    'alpha3_item_sim': 0.225,

    'alpha4_user_sim': 0.125,

    'alpha5_confidence': 0.1

}
 
# =============================================================================

# REJECTION CATEGORIES

# =============================================================================
 
REJECTION_CATEGORIES = {

    # LEARNABLE CATEGORIES

    'chauffeur_quality': {

        'description': 'Issues with driver behavior, professionalism, or service quality',

        'learnable': True,

        'model_weights': {'lightgcn': 0.3, 'collaborative': 0.3, 'user_content': 0.2, 'item_content': 0.2}

    },

    'vehicle_mismatch': {

        'description': 'Vehicle type, size, or features don\'t match requirements',

        'learnable': True,

        'model_weights': {'lightgcn': 0.2, 'collaborative': 0.2, 'user_content': 0.25, 'item_content': 0.35}

    },

    'reliability_concern': {

        'description': 'Concerns about punctuality or dependability',

        'learnable': True,

        'model_weights': {'lightgcn': 0.35, 'collaborative': 0.35, 'user_content': 0.15, 'item_content': 0.15}

    },

    'past_negative_experience': {

        'description': 'Previous bad experience with this driver or vehicle',

        'learnable': True,

        'model_weights': {'lightgcn': 0.4, 'collaborative': 0.4, 'user_content': 0.1, 'item_content': 0.1}

    },

    'client_preference': {

        'description': 'Client has specific preferences not met by this option',

        'learnable': True,

        'model_weights': {'lightgcn': 0.25, 'collaborative': 0.35, 'user_content': 0.25, 'item_content': 0.15}

    },

    'comfort_preference': {

        'description': 'Comfort or luxury level doesn\'t meet expectations',

        'learnable': True,

        'model_weights': {'lightgcn': 0.2, 'collaborative': 0.2, 'user_content': 0.3, 'item_content': 0.3}

    },

    'service_expectation': {

        'description': 'Service level or tier doesn\'t match booking requirements',

        'learnable': True,

        'model_weights': {'lightgcn': 0.25, 'collaborative': 0.25, 'user_content': 0.25, 'item_content': 0.25}

    },

    'poor_match': {

        'description': 'General poor match - recommendation doesn\'t fit the booking',

        'learnable': True,

        'model_weights': {'lightgcn': 0.25, 'collaborative': 0.25, 'user_content': 0.25, 'item_content': 0.25}

    },

    # NON-LEARNABLE CATEGORIES

    'chauffeur_unavailable': {

        'description': 'Chauffeur is busy or unavailable',

        'learnable': False,

        'model_weights': {'lightgcn': 0, 'collaborative': 0, 'user_content': 0, 'item_content': 0}

    },

    'vehicle_unavailable': {

        'description': 'Vehicle is booked, in maintenance, or out of service',

        'learnable': False,

        'model_weights': {'lightgcn': 0, 'collaborative': 0, 'user_content': 0, 'item_content': 0}

    },

    'external_factor': {

        'description': 'External circumstances (weather, traffic, emergency)',

        'learnable': False,

        'model_weights': {'lightgcn': 0, 'collaborative': 0, 'user_content': 0, 'item_content': 0}

    }

}
 
REJECTION_REASONS = {code: {'description': cfg['description'], 'learnable': cfg['learnable']} 

                     for code, cfg in REJECTION_CATEGORIES.items()}
 
def is_learnable_rejection(category: str) -> bool:

    return REJECTION_CATEGORIES.get(category, {}).get('learnable', False)
 
def get_category_model_weights(category: str) -> dict:

    return REJECTION_CATEGORIES.get(category, {}).get('model_weights', {})
 

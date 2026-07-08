
from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import Swagger
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Any
import logging
from dataclasses import dataclass
import json
# line added 21/1
# from config import DB_CONNECTION_STRING
#########
# from connect_stored_procedure import db_ops

# from data_loaders import DataLoader
# from grid_search import GridSearchOptimizer
# from feedback import WeightLearner, WeightSelector, FeedbackLogger, feedback_bp, init_feedback_system

import os
from logging.handlers import RotatingFileHandler
import sys
import io

### extra additions 
from dotenv import load_dotenv 
from tenacity import retry, stop_after_attempt, wait_exponential 
from flask_limiter import Limiter 
from flask_limiter.util import get_remote_address 
load_dotenv()


from config import DB_CONNECTION_STRING, setup_logging, API_TIMEOUT
from connect_stored_procedure import db_ops

from data_loaders import DataLoader
from grid_search import GridSearchOptimizer
from feedback import WeightLearner, WeightSelector, FeedbackLogger, feedback_bp, init_feedback_system
## CHANGE END

# Create logs directory if it doesn't exist
LOGS_DIR = "batch_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
def setup_batch_logger(batch_id: str) -> logging.Logger:
    """
    Create a dedicated logger for a batch that writes to a file
    
    Args:
        batch_id: Unique batch identifier
        
    Returns:
        Logger instance configured for this batch
    """
    # Create batch-specific logger
    batch_logger = logging.getLogger(f"batch_{batch_id}")
    batch_logger.setLevel(logging.INFO)
    batch_logger.propagate = False  # Don't propagate to root logger
    
    # Remove existing handlers (in case of reuse)
    batch_logger.handlers.clear()
    
    # Create file handler for this batch
    log_filename = os.path.join(LOGS_DIR, f"{batch_id}.log")
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding = 'utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    batch_logger.addHandler(file_handler)
    
    # Also add console handler so we still see output

    import sys
    if sys.platform == 'win32':
        import io
        console_handler = logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        )
    else:
        console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    batch_logger.addHandler(console_handler)
    
    batch_logger.info(f"="*70)
    batch_logger.info(f"Batch Logger Initialized: {batch_id}")
    batch_logger.info(f"Log file: {log_filename}")
    batch_logger.info(f"="*70)
    
    return batch_logger


# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Security: Set secret key for session management
# IMPORTANT: Change this in production via environment variable
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY', 
    'dev-only-change-in-production-' + os.urandom(16).hex()
)

# CORS - Restrict to allowed origins from environment variable
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
CORS(app, origins=allowed_origins)

# Rate limiting - Prevent API abuse
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[os.environ.get("RATE_LIMIT", "100 per minute")]
)

# Swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/api/docs"
}
template = {
    "swagger": "2.0",
    "info": {
        "title": "Recommendation Combiner API",
        "description": "Fuses LightGCN and Collaborative Filtering recommendations",
        "version": "1.0.0"
    }
}

swagger = Swagger(app, config=swagger_config, template=template)

# API Configuration
LIGHTGCN_API = os.environ.get("LIGHTGCN_API_URL", "http://localhost:5000/recommend")
COLLABORATIVE_API = os.environ.get("COLLABORATIVE_API_URL", "http://localhost:8000/api/v1/recommendations")
USER_CONTENT_API = os.environ.get("USER_CONTENT_API_URL", "http://localhost:8003/recommendations/user")
ITEM_CONTENT_API = os.environ.get("ITEM_CONTENT_API_URL", "http://localhost:8003/recommendations/item")
# RRF Configuration
RRF_K = 30

# =============================================================================
# REGISTER FEEDBACK API ENDPOINTS
# =============================================================================
app.register_blueprint(feedback_bp)
logger.info("    Feedback API endpoints registered at /api/feedback")

# Initialize
loader = DataLoader()
optimizer = GridSearchOptimizer()

# Load client types once at startup
logger.info("Loading client types...")
client_types = loader.load_client_types()
logger.info(f"Loaded {len(client_types)} client types")


logger.info("Loading sensitivity levels...")
sensitivity_levels = loader.load_sensitivity_levels()
logger.info(f"Loaded sensitivity mappings:")
logger.info(f"  • Clients: {len(sensitivity_levels['clients'])}")
logger.info(f"  • Chauffeurs: {len(sensitivity_levels['chauffeurs'])}")
logger.info(f"  • Vehicles: {len(sensitivity_levels['vehicles'])}")


# =============================================================================
# INITIALIZE FEEDBACK SYSTEM
# =============================================================================
logger.info("Initializing feedback system...")
feedback_components = init_feedback_system()
weight_selector = feedback_components['weight_selector']
feedback_logger = feedback_components['logger']
logger.info("    Feedback system initialized")


# Feature weights
LIGHTGCN_FEATURE_WEIGHTS = {
    'cv_score': 0.25,
    'cc_score': 0.25,
    'vc_score': 0.25,
    'combined_score': 0.10,
    'vehicle_completion_rate': 0.05,
    'chauffeur_completion_rate': 0.05,
    'vehicle_repeat_client_rate': 0.025,
    'chauffeur_repeat_client_rate': 0.025
}

CF_MATCH_WEIGHTS = {
    'cf_score': 0.4,
    'overall_match': 0.3,
    'quality_score': 0.2,
    'requirement_match': 0.1
}


@dataclass
class CombinerRequest:
    """Unified request format for combiner API"""
    client_id: int
    no_of_pax: int
    ref_port: Optional[int] = None  # Port filtering
    appointment_date: Optional[str] = None
    pickup_suburb: Optional[str] = None
    dropoff_suburb: Optional[str] = None
    estimated_distance: Optional[float] = None
    estimated_duration: Optional[int] = None
    no_of_bags: Optional[int] = None
    ref_vehicle_type: Optional[str] = None
    min_vehicle_level: Optional[int] = None
    require_red_zone_auth: Optional[bool] = False
    require_baby_seat: Optional[bool] = False
    excluded_vehicle_ids: Optional[List[int]] = None
    excluded_chauffeur_ids: Optional[List[int]] = None
    preferred_chauffeur_ids: Optional[List[int]] = None
    top_k: Optional[int] = 100
    
    def to_lightgcn_request(self) -> Dict:
        """Convert to LightGCN API format"""
        request_data = {
            "client_id": str(self.client_id),
            "no_of_pax": self.no_of_pax,
            "ref_port": self.ref_port,
            "ref_vehicle_type": self.ref_vehicle_type or "",
            "appointment_date": self.appointment_date or "",
            "pickup_suburb": self.pickup_suburb or "",
            "dropoff_suburb": self.dropoff_suburb or "",
            "estimated_distance": self.estimated_distance or 0,
            "estimated_duration": self.estimated_duration or 0,
            "min_vehicle_level": self.min_vehicle_level or 0,
            "require_red_zone_auth": self.require_red_zone_auth,
            "require_baby_seat": self.require_baby_seat,
            "excluded_vehicle_ids": self.excluded_vehicle_ids or [],
            "excluded_chauffeur_ids": self.excluded_chauffeur_ids or [],
            "preferred_chauffeur_ids": self.preferred_chauffeur_ids or [],
            "top_k_vehicles": self.top_k,
            "top_k_chauffeurs": self.top_k,
            "top_k_pairs": self.top_k
        }
        
        logger.info("\n  LightGCN Request:")
        logger.info(json.dumps(request_data, indent=2))
        
        return request_data
    
    def to_collaborative_request(self) -> Dict:
        """Convert to Collaborative API format"""
        request_data = {
            "client_id": self.client_id,
            "n_recommendations": self.top_k,
            "booking_context": {
                "appointment_date": self.appointment_date or datetime.now().isoformat(),
                "no_of_pax": self.no_of_pax,
                "ref_port": self.ref_port
            },
            "requirements": {}
        }
        
        if self.pickup_suburb:
            request_data["booking_context"]["pickup_location"] = {
                "suburb_id": 0,
                "postcode": self.pickup_suburb
            }
        
        if self.ref_vehicle_type:
            try:
                request_data["requirements"]["vehicle_type_id"] = int(self.ref_vehicle_type)
            except (ValueError, TypeError):
                pass
        
        if self.min_vehicle_level is not None:
            request_data["requirements"]["min_vehicle_level"] = self.min_vehicle_level
        
        logger.info("\n  Collaborative Request:")
        logger.info(json.dumps(request_data, indent=2))
        
        return request_data
    
    def to_user_content_request(self) -> Dict:
        """Convert to User Content API format - extracts only needed fields"""
        request_data = {
            "client_id": self.client_id,
            "filters": {},
            "candidate_k": 1000,
            "final_k": self.top_k
        }
        
        # Add filters only if they exist
        if self.ref_vehicle_type:
            try:
                request_data["filters"]["vehicle_type_id"] = int(self.ref_vehicle_type)
            except (ValueError, TypeError):
                pass
        
        if self.no_of_pax:
            request_data["filters"]["no_of_pax"] = self.no_of_pax
        
        if self.require_red_zone_auth:
            request_data["filters"]["red_zone_capable"] = 1
        
        if self.require_baby_seat:
            request_data["filters"]["baby_seat_training"] = 1
        
        if self.min_vehicle_level is not None:
            request_data["filters"]["client_sensitivity_level"] = self.min_vehicle_level
        
        if self.ref_port is not None:
            request_data["filters"]["port_id"] = self.ref_port
        
        logger.info("\n   User Content Request:")
        logger.info(json.dumps(request_data, indent=2))
        
        return request_data
    
    def to_item_content_request(self) -> Dict:
        """Convert to Item Content API format - extracts only needed fields"""
        # Use ref_vehicle_type as reference, default to 3 if not provided
        ref_vehicle = 3
        if self.ref_vehicle_type:
            try:
                ref_vehicle = int(self.ref_vehicle_type)
            except (ValueError, TypeError):
                pass
        
        request_data = {
            "ref_vehicle": ref_vehicle,
            "filters": {},
            "candidate_k": 1000,
            "final_k": self.top_k
        }
        
        # Add filters only if they exist
        if self.ref_vehicle_type:
            try:
                request_data["filters"]["vehicle_type_id"] = int(self.ref_vehicle_type)
            except (ValueError, TypeError):
                pass
        
        if self.no_of_pax:
            request_data["filters"]["no_of_pax"] = self.no_of_pax
        
        if self.require_red_zone_auth:
            request_data["filters"]["red_zone_capable"] = 1
        
        if self.require_baby_seat:
            request_data["filters"]["baby_seat_training"] = 1
        
        if self.min_vehicle_level is not None:
            request_data["filters"]["client_sensitivity_level"] = self.min_vehicle_level
        
        if self.ref_port is not None:
            request_data["filters"]["port_id"] = self.ref_port
        
        logger.info("\n   Item Content Request:")
        logger.info(json.dumps(request_data, indent=2))
        
        return request_data


def parse_comma_separated_ids(id_string: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated string into list of integers"""
    if not id_string or pd.isna(id_string):
        return None
    try:
        return [int(x.strip()) for x in str(id_string).split(',') if x.strip()]
    except (ValueError, AttributeError):
        return None


# =============================================================================
# CHANGED: Updated to use BookingFeed table instead of dummy_booking
# - Table name: dummy_booking → BookingFeed
# - Column names: snake_case → PascalCase
# - Order by Id (table index) for chronological processing
# =============================================================================
def load_pending_requests(limit: int = 5) -> pd.DataFrame:
    """Load pending requests from BookingFeed table"""
    from sqlalchemy import create_engine
    
    
    
    engine = create_engine(DB_CONNECTION_STRING)
    
    query = f"""
        SELECT TOP {limit} *
        FROM BookingFeed
        WHERE Status = 'PENDING'
        ORDER BY Id ASC
    """
    
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    return df


# =============================================================================
# CHANGED: Updated to use BookingFeed table instead of dummy_booking
# - Table name: dummy_booking → BookingFeed
# - WHERE clause uses BookingID (the actual booking identifier)
# - Column names: status → Status, error_message → ErrorMessage
# - Removed processed_at (not in new table)
# - ErrorMessage truncated to 200 chars (varchar(200) limit)
# =============================================================================
def update_request_status(booking_id: int, status: str, error_message: Optional[str] = None):
    """Update status of a request in BookingFeed table by BookingID"""
    from sqlalchemy import create_engine, text
    
   
    
    engine = create_engine(DB_CONNECTION_STRING)
    
    with engine.connect() as conn:
        if error_message:
            # IMPORTANT: Truncate error message to 200 chars (varchar(200) limit)
            truncated_error = str(error_message)[:200] if error_message else None
            
            query = text("""
                UPDATE BookingFeed
                SET Status = :status,
                    ErrorMessage = :error_message
                WHERE BookingID = :booking_id
            """)
            conn.execute(query, {
                'status': status,
                'booking_id': booking_id,
                'error_message': truncated_error
            })
        else:
            query = text("""
                UPDATE BookingFeed
                SET Status = :status
                WHERE BookingID = :booking_id
            """)
            conn.execute(query, {
                'status': status,
                'booking_id': booking_id
            })
        conn.commit()
    
    engine.dispose()


# =============================================================================
# CHANGED: Updated to use BookingRecommendations table
# - Table name: boooking_recommendations → BookingRecommendations
# - Column names: snake_case → PascalCase
# - Added new parameters: pickup_location, dropoff_location, appointment_datetime
# - Added new columns: PickUpLocation, DropOffLocation, PickUpDateTime
# - Parameter request_id renamed conceptually to booking_id (maps to BookingID)
# =============================================================================

def save_recommendations_to_db(booking_id: int, fusion_response: Dict[str, Any], batch_id: str,
                                pickup_location: str = None, dropoff_location: str = None,
                                appointment_datetime: datetime = None):
    """
    Save fusion API response to BookingRecommendations table with batch_id
    
    CRITICAL: Sets IsSelected = NULL (default, not reviewed yet)
    
    Schema:
        - OutputId: Auto-generated PRIMARY KEY (IDENTITY) - NOT inserted
        - BookingID: Foreign key to BookingFeed.BookingID
        - IsSelected: NULL (not reviewed), 0 (rejected), 1 (accepted)
    """
    from sqlalchemy import create_engine, text
    import pandas as pd
    import json
    
  
    
    if fusion_response.get('status') != 'success':
        logger.error(f"Cannot save failed response for booking {booking_id}")
        return
    
    recommendations = fusion_response.get('recommendations', [])
    if not recommendations:
        logger.warning(f"No recommendations to save for booking {booking_id}")
        return
    
    engine = create_engine(DB_CONNECTION_STRING)
    
    # ════════════════════════════════════════════════════════════════════
    # INSERT QUERY
    # NOTE: OutputId is NOT included (auto-generated IDENTITY)
    # NOTE: IsSelected = NULL (default state)
    # ════════════════════════════════════════════════════════════════════
    insert_query = text("""
        INSERT INTO BookingRecommendations (
            BookingID, ClientId, ClientType, PairId, VehicleId, ChauffeurId,
            FinalRank, FinalScore, RrfScore, GcnScore, CfScore,
            UserContentScore, ItemContentScore,
            CvScore, CcScore, VcScore, CombinedScore,
            VehicleCompletionRate, ChauffeurCompletionRate,
            PredictedRating, Confidence, OverallMatch, QualityScore,
            ProcessingTimeSeconds, WeightsUsed, BatchId, CreatedDate,
            PickUpLocation, DropOffLocation, PickUpDateTime,
            IsSelected
        ) VALUES (
            :booking_id, :client_id, :client_type, :pair_id, :vehicle_id, :chauffeur_id,
            :final_rank, :final_score, :rrf_score, :gcn_score, :cf_score,
            :user_content_score, :item_content_score,
            :cv_score, :cc_score, :vc_score, :combined_score,
            :vehicle_completion_rate, :chauffeur_completion_rate,
            :predicted_rating, :confidence, :overall_match, :quality_score,
            :processing_time_seconds, :weights_used, :batch_id, GETDATE(),
            :pickup_location, :dropoff_location, :appointment_datetime,
            NULL
        )
    """)
    #   ↑
    #   └─── CRITICAL: IsSelected = NULL (not reviewed yet)
    
    saved_count = 0
    failed_count = 0
    
    # Helper function to safely convert numeric values
    def safe_float(val):
        """Convert to float, handling None, NaN, and inf"""
        if val is None or pd.isna(val):
            return None
        try:
            f = float(val)
            # SQL Server doesn't like infinity
            if f == float('inf') or f == float('-inf'):
                return None
            return f
        except (ValueError, TypeError):
            return None
    
    # Helper function to safely convert to string (for IDs)
    def safe_str(val):
        """Convert to string, handling None and NaN"""
        if val is None or pd.isna(val):
            return None
        try:
            # Handle float IDs like "123.0" -> "123"
            if isinstance(val, float):
                return str(int(val))
            return str(val)
        except (ValueError, TypeError):
            return None
    
    try:
        with engine.connect() as conn:
            for idx, rec in enumerate(recommendations):
                try:
                    # Extract vehicle_id and chauffeur_id (as strings for nvarchar)
                    vehicle_id = None
                    chauffeur_id = None
                    
                    # Try multiple column names (prioritize what your enrichment provides)
                    if 'vehicle_id' in rec:
                        vehicle_id = safe_str(rec['vehicle_id'])
                    elif 'RefVehicle' in rec:
                        vehicle_id = safe_str(rec['RefVehicle'])
                    
                    if 'chauffeur_id' in rec:
                        chauffeur_id = safe_str(rec['chauffeur_id'])
                    elif 'RefChauffeur' in rec:
                        chauffeur_id = safe_str(rec['RefChauffeur'])
                    
                    # Log warning but DON'T skip (since columns are nullable)
                    if vehicle_id is None or chauffeur_id is None:
                        logger.warning(f"  Rec {idx+1} has missing IDs - vehicle: {vehicle_id}, chauffeur: {chauffeur_id}")
                    
                    # Handle client_type (nullable nvarchar)
                    client_type = fusion_response.get('client_type')
                    if client_type and len(str(client_type)) > 50:
                        client_type = str(client_type)[:50]
                    
                    # Truncate weights_used if too long
                    weights_json = json.dumps(fusion_response.get('weights_used', {}))
                    
                    params = {
                        'booking_id': booking_id,
                        'client_id': int(fusion_response.get('client_id')),
                        'client_type': client_type,
                        'pair_id': safe_str(rec.get('pair_id')),
                        'vehicle_id': vehicle_id,
                        'chauffeur_id': chauffeur_id,
                        'final_rank': int(rec.get('final_rank', 0)),
                        'final_score': safe_float(rec.get('final_score')),
                        'rrf_score': safe_float(rec.get('rrf_score')),
                        'gcn_score': safe_float(rec.get('gcn_score')),
                        'cf_score': safe_float(rec.get('cf_score')),
                        'user_content_score': safe_float(rec.get('user_content_score')),
                        'item_content_score': safe_float(rec.get('item_content_score')),
                        'cv_score': safe_float(rec.get('cv_score')),
                        'cc_score': safe_float(rec.get('cc_score')),
                        'vc_score': safe_float(rec.get('vc_score')),
                        'combined_score': safe_float(rec.get('combined_score')),
                        'vehicle_completion_rate': safe_float(rec.get('vehicle_completion_rate')),
                        'chauffeur_completion_rate': safe_float(rec.get('chauffeur_completion_rate')),
                        'predicted_rating': safe_float(rec.get('predicted_rating')),
                        'confidence': safe_float(rec.get('confidence')),
                        'overall_match': safe_float(rec.get('overall_match')),
                        'quality_score': safe_float(rec.get('quality_score')),
                        'processing_time_seconds': safe_float(fusion_response.get('processing_time_seconds')),
                        'weights_used': weights_json,
                        'batch_id': batch_id,
                        'pickup_location': pickup_location,
                        'dropoff_location': dropoff_location,
                        'appointment_datetime': appointment_datetime
                    }
                    
                    conn.execute(insert_query, params)
                    saved_count += 1
                    
                except Exception as row_error:
                    logger.error(f"  Failed to save rec {idx+1}: {str(row_error)}")
                    logger.error(f"   pair_id: {rec.get('pair_id')}")
                    logger.error(f"   vehicle_id: {rec.get('vehicle_id')} / RefVehicle: {rec.get('RefVehicle')}")
                    logger.error(f"   chauffeur_id: {rec.get('chauffeur_id')} / RefChauffeur: {rec.get('RefChauffeur')}")
                    failed_count += 1
                    continue
            
            # Commit after all inserts
            conn.commit()
        
        logger.info(f"  Saved {saved_count}/{len(recommendations)} recommendations for booking {booking_id} (batch: {batch_id})")
        if failed_count > 0:
            logger.warning(f"  Failed to save {failed_count} recommendations")
    
    except Exception as e:
        logger.error(f"  Database error while saving recommendations: {str(e)}")
        logger.error(f"   Booking ID: {booking_id}, Batch ID: {batch_id}")
    
    finally:
        engine.dispose()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def safe_api_call(url: str, payload: Dict, model_name: str, timeout: int = None) -> Optional[Dict]:
    """
    Safely call model API with comprehensive error handling and retry logic
    
    Args:
        url: API endpoint URL
        payload: Request payload
        model_name: Name of the model (for logging)
        timeout: Request timeout in seconds (defaults to API_TIMEOUT from config)
        
    Returns:
        API response dict if successful, None if failed
    """
    # Use config timeout if not specified
    if timeout is None:
        timeout = API_TIMEOUT
    
    try:
        logger.info(f" Calling {model_name} API...")
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()  # Raises exception for 4xx/5xx
        
        data = response.json()
        
        # Validate response structure
        if not data:
            logger.warning(f"  {model_name} returned empty response")
            return None
        
        # Check for success indicator (different APIs use different fields)
        if isinstance(data, dict):
            # Some APIs return {"success": false}
            if 'success' in data and not data['success']:
                logger.warning(f"  {model_name} returned success=false: {data.get('message', 'No message')}")
                return None
            
            # Some APIs return {"status": "error"}
            if 'status' in data and data['status'] == 'error':
                logger.warning(f"  {model_name} returned status=error: {data.get('error', 'No error message')}")
                return None
        
        logger.info(f"  {model_name} API call successful")
        return data
        
    except requests.exceptions.Timeout:
        logger.error(f"  {model_name} API timeout after {timeout}s")
        return None
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"  {model_name} API HTTP error: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"   Response: {e.response.text[:200]}")
        return None
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"  {model_name} API connection error: {e}")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"  {model_name} API request error: {e}")
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"  {model_name} returned invalid JSON: {e}")
        return None
        
    except Exception as e:
        logger.error(f"  {model_name} unexpected error: {e}")
        return None
    

def apply_sensitivity_filter(fused_df: pd.DataFrame,
                            client_id: int,
                            sensitivity_levels: Dict) -> pd.DataFrame:
    """
    Filter recommendations based on sensitivity levels:
    client_sensitivity <= vehicle_sensitivity AND client_sensitivity <= chauffeur_sensitivity
    
    Args:
        fused_df: DataFrame with recommendations
        client_id: Client ID to get sensitivity level
        sensitivity_levels: Dict with client/chauffeur/vehicle sensitivity mappings
    
    Returns:
        Filtered DataFrame
    """
    if fused_df.empty:
        return fused_df
    
    logger.info(f"\n APPLYING SENSITIVITY FILTER")
    logger.info(f"{'='*60}")
    
    initial_count = len(fused_df)
    
    # Get client sensitivity level (default to 0 if not found)
    client_sensitivity = sensitivity_levels['clients'].get(client_id, 0)
    logger.info(f"Client {client_id} sensitivity level: {client_sensitivity}")
    
    # Extract vehicle_id and chauffeur_id (handle both naming conventions)
    if 'vehicle_id' in fused_df.columns:
        fused_df['vehicle_id_int'] = pd.to_numeric(fused_df['vehicle_id'], errors='coerce').fillna(0).astype(int)
    elif 'RefVehicle' in fused_df.columns:
        fused_df['vehicle_id_int'] = pd.to_numeric(fused_df['RefVehicle'], errors='coerce').fillna(0).astype(int)
    else:
        logger.error("  No vehicle_id column found")
        return fused_df
    
    if 'chauffeur_id' in fused_df.columns:
        fused_df['chauffeur_id_int'] = pd.to_numeric(fused_df['chauffeur_id'], errors='coerce').fillna(0).astype(int)
    elif 'RefChauffeur' in fused_df.columns:
        fused_df['chauffeur_id_int'] = pd.to_numeric(fused_df['RefChauffeur'], errors='coerce').fillna(0).astype(int)
    else:
        logger.error("  No chauffeur_id column found")
        return fused_df
    
    # Map sensitivity levels (default to 0 if not found)
    fused_df['vehicle_sensitivity'] = fused_df['vehicle_id_int'].map(
        sensitivity_levels['vehicles']
    ).fillna(0).astype(int)
    
    fused_df['chauffeur_sensitivity'] = fused_df['chauffeur_id_int'].map(
        sensitivity_levels['chauffeurs']
    ).fillna(0).astype(int)
    
    # Apply filter: client_sensitivity <= vehicle_sensitivity AND client_sensitivity <= chauffeur_sensitivity
    filtered_df = fused_df[
        (fused_df['vehicle_sensitivity'] >= client_sensitivity) &
        (fused_df['chauffeur_sensitivity'] >= client_sensitivity)
    ].copy()
    
    filtered_count = len(filtered_df)
    removed_count = initial_count - filtered_count
    
    logger.info(f"\nFilter Results:")
    logger.info(f"  • Initial recommendations: {initial_count}")
    logger.info(f"  • After sensitivity filter: {filtered_count}")
    logger.info(f"  • Removed: {removed_count} ({removed_count/initial_count*100 if initial_count > 0 else 0:.1f}%)")
    
    if removed_count > 0 and removed_count <= 5:
        logger.info(f"\nRemoved pairs (sensitivity mismatch):")
        removed_df = fused_df[~fused_df.index.isin(filtered_df.index)]
        for idx, row in removed_df.iterrows():
            logger.info(f"  • Pair {row['pair_id']}: "
                       f"Vehicle sensitivity={row['vehicle_sensitivity']}, "
                       f"Chauffeur sensitivity={row['chauffeur_sensitivity']}, "
                       f"Client requires >={client_sensitivity}")
    elif removed_count > 5:
        logger.info(f"\nSample removed pairs (showing 3 of {removed_count}):")
        removed_df = fused_df[~fused_df.index.isin(filtered_df.index)].head(3)
        for idx, row in removed_df.iterrows():
            logger.info(f"  • Pair {row['pair_id']}: "
                       f"Vehicle sensitivity={row['vehicle_sensitivity']}, "
                       f"Chauffeur sensitivity={row['chauffeur_sensitivity']}, "
                       f"Client requires >={client_sensitivity}")
    
    logger.info(f"{'='*60}\n")
    
    # Drop temporary columns
    filtered_df = filtered_df.drop(columns=['vehicle_id_int', 'chauffeur_id_int',
                                            'vehicle_sensitivity', 'chauffeur_sensitivity'])
    
    # Re-rank after filtering
    if not filtered_df.empty:
        filtered_df = filtered_df.sort_values('final_score', ascending=False).reset_index(drop=True)
        filtered_df['final_rank'] = range(1, len(filtered_df) + 1)
    
    return filtered_df



# =============================================================================
# MODIFIED batch_process_recommendations FUNCTION
# =============================================================================
# This replaces your existing batch_process_recommendations function in combiner_api_final.py
# 
# CHANGE: Added auto-call to populate-availability after successful batch processing
# =============================================================================

@app.route('/api/v1/batch-process', methods=['POST'])
def batch_process_recommendations():
    """
    Process top N pending requests from BookingFeed table and execute allocation
    Logs all output to batch-specific log file
    
    NEW: Auto-populates AvailabilityAllocation with TOP 10 after batch processing
    ---
    tags:
      - Batch Processing
    parameters:
      - name: body
        in: body
        schema:
          type: object
          properties:
            batch_size:
              type: integer
              example: 5
              description: Number of pending requests to process (default 5)
    responses:
      200:
        description: Batch processing completed with allocation results and detailed logs
    """
    start_time = datetime.now()
    batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Create batch-specific logger
    batch_logger = setup_batch_logger(batch_id)
    
    try:
        # data = request.get_json() or {}
        # batch_size = data.get('batch_size', 5)
        data = request.get_json() or {}
        # Validate batch_size: minimum 1, maximum 100 (prevents DoS)
        raw_batch_size = data.get('batch_size', 5)
        batch_size = min(max(int(raw_batch_size) if str(raw_batch_size).isdigit() else 5, 1), 100)

        if raw_batch_size != batch_size:
            batch_logger.warning(f"Batch size adjusted from {raw_batch_size} to {batch_size} (limits: 1-100)")
        batch_logger.info(f" STARTING BATCH PROCESSING")
        batch_logger.info(f"Batch ID: {batch_id}")
        batch_logger.info(f"Batch Size: {batch_size}")
        batch_logger.info(f"{'='*70}")
        
        # Load pending requests from BookingFeed
        pending_requests = load_pending_requests(limit=batch_size)
        
        if pending_requests.empty:
            batch_logger.info("No pending requests found")
            return jsonify({
                "status": "success",
                "message": "No pending requests to process",
                "batch_id": batch_id,
                "processed_count": 0,
                "log_file": os.path.join(LOGS_DIR, f"{batch_id}.log"),
                "allocation": {
                    "success": False,
                    "message": "No requests to allocate"
                }
            })
        
        batch_logger.info(f"Found {len(pending_requests)} pending requests")
        
        results = {
            "batch_id": batch_id,
            "total": len(pending_requests),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        # Process each request
        for idx, row in pending_requests.iterrows():
            # Use BookingID as the identifier (Id is just table index)
            booking_id = int(row['BookingID'])
            
            batch_logger.info(f"\n{'─'*70}")
            batch_logger.info(f"Processing BookingID {booking_id} ({idx+1}/{len(pending_requests)})")
            batch_logger.info(f"{'─'*70}")
            
            try:
                # Mark as processing using BookingID
                update_request_status(booking_id, 'PROCESSING')
                
                # Build request payload with PascalCase column names
                request_payload = {
                    'client_id': int(row['ClientId']),
                    'no_of_pax': int(row['NoOfPax']),
                    'top_final': int(row.get('TopFinal', 20) or 20)
                }
                
                # Store pickup/dropoff for saving to recommendations
                pickup_location = None
                dropoff_location = None
                appointment_datetime = None
                
                # Add optional fields with NEW column names
                if pd.notna(row.get('AppointmentDate')):
                    request_payload['appointment_date'] = row['AppointmentDate'].isoformat() if hasattr(row['AppointmentDate'], 'isoformat') else str(row['AppointmentDate'])
                    appointment_datetime = row['AppointmentDate']
                
                if pd.notna(row.get('PickUpLocation')):
                    request_payload['pickup_suburb'] = str(row['PickUpLocation'])
                    pickup_location = str(row['PickUpLocation'])
                
                if pd.notna(row.get('DropOffLocation')):
                    request_payload['dropoff_suburb'] = str(row['DropOffLocation'])
                    dropoff_location = str(row['DropOffLocation'])
                
                if pd.notna(row.get('RefVehicleType')):
                    request_payload['ref_vehicle_type'] = str(row['RefVehicleType'])
                if pd.notna(row.get('MinVehicleLevel')):
                    request_payload['min_vehicle_level'] = int(row['MinVehicleLevel'])
                if pd.notna(row.get('RequireRedZoneAuth')):
                    request_payload['require_red_zone_auth'] = bool(row['RequireRedZoneAuth'])
                if pd.notna(row.get('RequireBabySeat')):
                    request_payload['require_baby_seat'] = bool(row['RequireBabySeat'])
                
                # Parse array fields with PascalCase column names
                excluded_vehicles = parse_comma_separated_ids(row.get('ExcludedVehicleIds'))
                if excluded_vehicles:
                    request_payload['excluded_vehicle_ids'] = excluded_vehicles
                
                excluded_chauffeurs = parse_comma_separated_ids(row.get('ExcludedChauffeurIds'))
                if excluded_chauffeurs:
                    request_payload['excluded_chauffeur_ids'] = excluded_chauffeurs
                
                preferred_chauffeurs = parse_comma_separated_ids(row.get('PreferredChauffeurIds'))
                if preferred_chauffeurs:
                    request_payload['preferred_chauffeur_ids'] = preferred_chauffeurs
                
                # Add ref_port with PascalCase column name
                if pd.notna(row.get('RefPort')):
                    ref_port_value = int(row['RefPort'])
                    if ref_port_value > 0:
                        request_payload['ref_port'] = ref_port_value
                        batch_logger.info(f"Added ref_port: {ref_port_value}")
                    else:
                        batch_logger.warning(f" RefPort is 0 for booking {booking_id}, skipping port filter")
                else:
                    batch_logger.warning(f"RefPort is NULL for booking {booking_id}")
                
                batch_logger.info(f"Request payload:")
                batch_logger.info(json.dumps(request_payload, indent=2, default=str))
                
                # Call fusion with batch logger
                fusion_response = process_single_fusion_request(request_payload, batch_logger)
                
                if fusion_response.get('status') == 'success':
                    # Save to database with new parameters
                    save_recommendations_to_db(
                        booking_id=booking_id,
                        fusion_response=fusion_response,
                        batch_id=batch_id,
                        pickup_location=pickup_location,
                        dropoff_location=dropoff_location,
                        appointment_datetime=appointment_datetime
                    )
                    
                    # Mark as completed using BookingID
                    update_request_status(booking_id, 'COMPLETED')
                    
                    results['successful'] += 1
                    
                    # Add detailed result with API responses
                    results['details'].append({
                        'booking_id': booking_id,
                        'client_id': int(row['ClientId']),
                        'status': 'success',
                        'recommendations_count': len(fusion_response.get('recommendations', [])),
                        'api_responses': {
                            'lightgcn': {
                                'status': fusion_response['api_responses']['lightgcn']['status'],
                                'pairs_count': fusion_response['api_responses']['lightgcn'].get('pairs_count', 0),
                                'error': fusion_response['api_responses']['lightgcn'].get('error')
                            },
                            'collaborative': {
                                'status': fusion_response['api_responses']['collaborative']['status'],
                                'pairs_count': fusion_response['api_responses']['collaborative'].get('pairs_count', 0),
                                'error': fusion_response['api_responses']['collaborative'].get('error')
                            },
                            'user_content': {
                                'status': fusion_response['api_responses']['user_content']['status'],
                                'pairs_count': fusion_response['api_responses']['user_content'].get('pairs_count', 0),
                                'error': fusion_response['api_responses']['user_content'].get('error')
                            },
                            'item_content': {
                                'status': fusion_response['api_responses']['item_content']['status'],
                                'pairs_count': fusion_response['api_responses']['item_content'].get('pairs_count', 0),
                                'error': fusion_response['api_responses']['item_content'].get('error')
                            }
                        },
                        'counts': fusion_response.get('counts', {}),
                        'weights_used': fusion_response.get('weights_used', {})
                    })
                    
                    batch_logger.info(f"  Booking {booking_id} completed successfully")
                else:
                    raise Exception(fusion_response.get('error', 'Unknown error'))
                    
            except Exception as e:
                error_msg = str(e)
                batch_logger.error(f"  Booking {booking_id} failed: {error_msg}", exc_info=True)
                
                # Mark as failed using BookingID (error_message will be truncated to 200 chars)
                update_request_status(booking_id, 'FAILED', error_message=error_msg)
                
                results['failed'] += 1
                results['details'].append({
                    'booking_id': booking_id,
                    'client_id': int(row.get('ClientId', 0)),
                    'status': 'failed',
                    'error': error_msg
                })
        
        # =============================================================================
        # NEW STEP 6: AUTO-POPULATE AVAILABILITY ALLOCATION
        # =============================================================================
        # After all recommendations are saved to BookingRecommendations,
        # automatically populate AvailabilityAllocation with TOP 10 per booking
        # =============================================================================
        
        availability_result = {
            "success": False,
            "message": "Not executed"
        }
        
        if results['successful'] > 0:
            batch_logger.info(f"\n{'='*70}")
            batch_logger.info(f" POPULATING AVAILABILITY ALLOCATION")
            batch_logger.info(f"Batch ID: {batch_id}")
            batch_logger.info(f"Successful bookings: {results['successful']}")
            batch_logger.info(f"{'='*70}")
            
            try:
                # Call the populate availability stored procedure
                availability_result = db_ops.execute_populate_availability(
                    batch_id=batch_id,
                    top_n=10  # TOP 10 recommendations per booking
                )
                
                if availability_result['success']:
                    result_data = availability_result.get('result_data', [])
                    if result_data:
                        bookings_processed = result_data[0].get('BookingsProcessed', 0)
                        recs_inserted = result_data[0].get('RecommendationsInserted', 0)
                        batch_logger.info(f"  AvailabilityAllocation populated successfully")
                        batch_logger.info(f"   Bookings processed: {bookings_processed}")
                        batch_logger.info(f"   Recommendations inserted: {recs_inserted}")
                        batch_logger.info(f"   Top N per booking: 10")
                    else:
                        batch_logger.info(f"  AvailabilityAllocation populated (no result data)")
                else:
                    batch_logger.error(f"  AvailabilityAllocation population failed")
                    batch_logger.error(f"   Error: {availability_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                batch_logger.error(f"  Error populating AvailabilityAllocation: {e}", exc_info=True)
                availability_result = {
                    "success": False,
                    "error": str(e)
                }
        else:
            batch_logger.info(f"\n No successful recommendations - skipping AvailabilityAllocation")
            availability_result = {
                "success": False,
                "message": "No successful recommendations to populate"
            }
        
        # Add availability result to main results
        results['availability_allocation'] = availability_result
        
        # =============================================================================
        # END OF NEW STEP 6
        # =============================================================================
        
        # Stored procedure placeholder (disabled)
        batch_logger.info(f"\n STORED PROCEDURE DISABLED - Skipping allocation")
        results['allocation'] = {
            "success": False,
            "batch_id": batch_id,
            "message": "Stored procedure temporarily disabled - changes being made",
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds()
        
        batch_logger.info(f"\n{'='*70}")
        batch_logger.info(f"  BATCH PROCESSING COMPLETED")
        batch_logger.info(f"Batch ID: {batch_id}")
        batch_logger.info(f"Total: {results['total']} | Success: {results['successful']} | Failed: {results['failed']}")
        batch_logger.info(f"AvailabilityAllocation: {'  Populated' if availability_result.get('success') else '  Failed/Skipped'}")
        batch_logger.info(f"Total time: {total_time:.2f}s")
        batch_logger.info(f"Log file: {os.path.join(LOGS_DIR, f'{batch_id}.log')}")
        batch_logger.info(f"{'='*70}\n")
        
        # Close batch logger handlers
        for handler in batch_logger.handlers:
            handler.close()
            batch_logger.removeHandler(handler)
        
        return jsonify({
            "status": "success",
            "batch_id": batch_id,
            "batch_processing_time_seconds": total_time,
            "log_file": os.path.join(LOGS_DIR, f"{batch_id}.log"),
            "results": results,
            "availability_allocation_result": availability_result,  # NEW
            "allocation_result": results.get('allocation', {})
        })
        
    except Exception as e:
        batch_logger.error(f"  Batch processing error: {str(e)}", exc_info=True)
        
        # Close batch logger handlers
        if 'batch_logger' in locals():
            for handler in batch_logger.handlers:
                handler.close()
                batch_logger.removeHandler(handler)
        
        return jsonify({
            "status": "error",
            "error": str(e),
            "batch_id": batch_id if 'batch_id' in locals() else None,
            "log_file": os.path.join(LOGS_DIR, f"{batch_id}.log") if 'batch_id' in locals() else None
        }), 500

def process_single_fusion_request(request_data: Dict, batch_logger: logging.Logger = None) -> Dict:
    """
    Internal function to process a single fusion request
    
    Args:
        request_data: Request parameters
        batch_logger: Optional batch-specific logger
        
    Returns:
        Dict with status, recommendations, and detailed API responses
    """
    # Use batch logger if provided, otherwise use default logger
    log = batch_logger or logger
    
    try:
        client_id = request_data['client_id']
        top_final = request_data.pop('top_final', 20)
        
        # Get client type
        client_type = client_types.get(client_id, 'UNKNOWN')
        
        # Get base weights from grid search
        base_weights = optimizer.load_weights(client_type)
        if not base_weights:
            base_weights = {
                'alpha1_rrf': 0.3,
                'alpha2_gcn': 0.3,
                'alpha3_item_sim': 0.1,
                'alpha4_user_sim': 0.1,
                'alpha5_confidence': 0.1
            }
        
        # Get personalized weights
        weights_result = weight_selector.get_weights_with_source(client_id, client_type, base_weights)
        weights = weights_result.get('weights', base_weights)
        log.info(f"Using weights for client {client_id} (type={client_type}): {json.dumps(weights, indent=2)}")
        
        # Set top_k for fetching
        request_data['top_k'] = 100
        
        # Create request object
        combiner_request = CombinerRequest(**request_data)
        
        # Prepare API requests
        lightgcn_request = combiner_request.to_lightgcn_request()
        collaborative_request = combiner_request.to_collaborative_request()
        user_content_request = combiner_request.to_user_content_request()
        item_content_request = combiner_request.to_item_content_request()
        
        # Track API responses for logging
        api_responses = {
            'lightgcn': {'status': 'not_called', 'data': None, 'error': None},
            'collaborative': {'status': 'not_called', 'data': None, 'error': None},
            'user_content': {'status': 'not_called', 'data': None, 'error': None},
            'item_content': {'status': 'not_called', 'data': None, 'error': None}
        }
        
        # Initialize DataFrames
        lightgcn_df = pd.DataFrame()
        collaborative_df = pd.DataFrame()
        user_content_df = pd.DataFrame()
        item_content_df = pd.DataFrame()
        
        # ===================================================================
        # Call LightGCN API
        # ===================================================================
        log.info("\n" + "="*70)
        log.info("  CALLING LIGHTGCN API")
        log.info("="*70)
        log.info("Request Payload:")
        log.info(json.dumps(lightgcn_request, indent=2))
        
        try:
            response = requests.post(LIGHTGCN_API, json=lightgcn_request, timeout=3000)
            
            if response.status_code == 200:
                lightgcn_response = response.json()
                lightgcn_df = process_lightgcn_response(lightgcn_response)
                weights = adjust_weights_for_cold_start(lightgcn_response, weights)
                
                api_responses['lightgcn'] = {
                    'status': 'success',
                    'pairs_count': len(lightgcn_df),
                    'data': lightgcn_response,
                    'error': None
                }
                
                log.info("\n  LightGCN Response Summary:")
                log.info(f"   Status: SUCCESS")
                log.info(f"   Pairs returned: {len(lightgcn_df)}")
                log.info(f"   Is cold start: {lightgcn_response.get('is_cold_start', False)}")
                log.info(f"   Method: {lightgcn_response.get('recommendation_method', 'unknown')}")
                
                # Log sample recommendations
                if len(lightgcn_df) > 0:
                    log.info("\n   Top 3 LightGCN Recommendations:")
                    for idx, row in lightgcn_df.head(3).iterrows():
                        log.info(f"      {idx+1}. Pair: {row['pair_id']}, "
                                f"Rank: {row.get('rank', 'N/A')}, "
                                f"Score: {row.get('lightgcn_weighted_score', row.get('combined_score', 'N/A')):.4f}")
                
            else:
                api_responses['lightgcn'] = {
                    'status': 'http_error',
                    'status_code': response.status_code,
                    'data': None,
                    'error': f"HTTP {response.status_code}"
                }
                log.error(f"\n  LightGCN Response: HTTP {response.status_code}")
                log.error(f"   Response text: {response.text[:500]}")
                
        except Exception as e:
            api_responses['lightgcn'] = {
                'status': 'exception',
                'data': None,
                'error': str(e)
            }
            log.error(f"\n  LightGCN Exception: {str(e)}", exc_info=True)
        
        log.info("="*70 + "\n")
        
        # ===================================================================
        # Call Collaborative API
        # ===================================================================
        log.info("\n" + "="*70)
        log.info("  CALLING COLLABORATIVE API")
        log.info("="*70)
        log.info("Request Payload:")
        log.info(json.dumps(collaborative_request, indent=2))
        
        try:
            response = requests.post(COLLABORATIVE_API, json=collaborative_request, timeout=3000)
            
            if response.status_code == 200:
                collab_response = response.json()
                collaborative_df = process_collaborative_response(collab_response)
                
                api_responses['collaborative'] = {
                    'status': 'success',
                    'pairs_count': len(collaborative_df),
                    'data': collab_response,
                    'error': None
                }
                
                log.info("\n  Collaborative Response Summary:")
                log.info(f"   Status: SUCCESS")
                log.info(f"   Pairs returned: {len(collaborative_df)}")
                
                # Log sample recommendations
                if len(collaborative_df) > 0:
                    log.info("\n   Top 3 Collaborative Recommendations:")
                    for idx, row in collaborative_df.head(3).iterrows():
                        log.info(f"      {idx+1}. Pair: {row['pair_id']}, "
                                f"Rank: {row.get('rank', 'N/A')}, "
                                f"CF Score: {row.get('cf_combined_score_normalized', row.get('confidence', 'N/A')):.4f}")
                
            else:
                api_responses['collaborative'] = {
                    'status': 'http_error',
                    'status_code': response.status_code,
                    'data': None,
                    'error': f"HTTP {response.status_code}"
                }
                log.error(f"\n  Collaborative Response: HTTP {response.status_code}")
                log.error(f"   Response text: {response.text[:500]}")
                
        except Exception as e:
            api_responses['collaborative'] = {
                'status': 'exception',
                'data': None,
                'error': str(e)
            }
            log.error(f"\n  Collaborative Exception: {str(e)}", exc_info=True)
        
        log.info("="*70 + "\n")
        
        # ===================================================================
        # Call User Content API
        # ===================================================================
        log.info("\n" + "="*70)
        log.info("   CALLING USER CONTENT API")
        log.info("="*70)
        log.info("Request Payload:")
        log.info(json.dumps(user_content_request, indent=2))
        
        try:
            response = requests.post(USER_CONTENT_API, json=user_content_request, timeout=3000)
            
            if response.status_code == 200:
                uc_response = response.json()
                user_content_df = process_user_content_response(uc_response)
                
                api_responses['user_content'] = {
                    'status': 'success',
                    'pairs_count': len(user_content_df),
                    'data': uc_response,
                    'error': None
                }
                
                log.info("\n  User Content Response Summary:")
                log.info(f"   Status: SUCCESS")
                log.info(f"   Pairs returned: {len(user_content_df)}")
                
                # Log sample recommendations
                if len(user_content_df) > 0:
                    log.info("\n   Top 3 User Content Recommendations:")
                    for idx, row in user_content_df.head(3).iterrows():
                        log.info(f"      {idx+1}. Pair: {row['pair_id']}, "
                                f"Similarity: {row.get('user_content_score', 'N/A'):.4f}")
                
            else:
                api_responses['user_content'] = {
                    'status': 'http_error',
                    'status_code': response.status_code,
                    'data': None,
                    'error': f"HTTP {response.status_code}"
                }
                log.error(f"\n  User Content Response: HTTP {response.status_code}")
                log.error(f"   Response text: {response.text[:500]}")
                
        except Exception as e:
            api_responses['user_content'] = {
                'status': 'exception',
                'data': None,
                'error': str(e)
            }
            log.error(f"\n  User Content Exception: {str(e)}", exc_info=True)
        
        log.info("="*70 + "\n")
        
        # ===================================================================
        # Call Item Content API
        # ===================================================================
        log.info("\n" + "="*70)
        log.info("   CALLING ITEM CONTENT API")
        log.info("="*70)
        log.info("Request Payload:")
        log.info(json.dumps(item_content_request, indent=2))
        
        try:
            response = requests.post(ITEM_CONTENT_API, json=item_content_request, timeout=3000)
            
            if response.status_code == 200:
                ic_response = response.json()
                item_content_df = process_item_content_response(ic_response)
                
                api_responses['item_content'] = {
                    'status': 'success',
                    'pairs_count': len(item_content_df),
                    'data': ic_response,
                    'error': None
                }
                
                log.info("\n  Item Content Response Summary:")
                log.info(f"   Status: SUCCESS")
                log.info(f"   Pairs returned: {len(item_content_df)}")
                
                # Log sample recommendations
                if len(item_content_df) > 0:
                    log.info("\n   Top 3 Item Content Recommendations:")
                    for idx, row in item_content_df.head(3).iterrows():
                        log.info(f"      {idx+1}. Pair: {row['pair_id']}, "
                                f"Similarity: {row.get('item_content_score', 'N/A'):.4f}")
                
            else:
                api_responses['item_content'] = {
                    'status': 'http_error',
                    'status_code': response.status_code,
                    'data': None,
                    'error': f"HTTP {response.status_code}"
                }
                log.error(f"\n  Item Content Response: HTTP {response.status_code}")
                log.error(f"   Response text: {response.text[:500]}")
                
        except Exception as e:
            api_responses['item_content'] = {
                'status': 'exception',
                'data': None,
                'error': str(e)
            }
            log.error(f"\n  Item Content Exception: {str(e)}", exc_info=True)
        
        log.info("="*70 + "\n")
        
        # ===================================================================
        # Check if we got any data
        # ===================================================================
        if all(df.empty for df in [lightgcn_df, collaborative_df, user_content_df, item_content_df]):
            log.error("  ALL APIs FAILED - No data to fuse")
            return {
                "status": "error",
                "error": "All recommendation APIs failed",
                "api_responses": api_responses
            }
        
        # ===================================================================
        # Fuse recommendations
        # ===================================================================
        log.info("\n" + "="*70)
        log.info("🔀 FUSING RECOMMENDATIONS")
        log.info("="*70)
        
                # Determine which APIs returned data
        lightgcn_available = len(lightgcn_df) > 0
        collaborative_available = len(collaborative_df) > 0
        user_content_available = len(user_content_df) > 0
        item_content_available = len(item_content_df) > 0

        # Redistribute weights based on API availability
        weights = redistribute_weights_for_available_apis(
            weights,
            lightgcn_available,
            collaborative_available,
            user_content_available,
            item_content_available
        )

        # Now call fuse_recommendations with adjusted weights
        fused_df = fuse_recommendations(
            lightgcn_df,
            collaborative_df,
            user_content_df,
            item_content_df,
            weights  # Now using redistributed weights
        )
        
        log.info(f"\nFusion Statistics:")
        log.info(f"   Input pairs:")
        log.info(f"      - LightGCN: {len(lightgcn_df)}")
        log.info(f"      - Collaborative: {len(collaborative_df)}")
        log.info(f"      - User Content: {len(user_content_df)}")
        log.info(f"      - Item Content: {len(item_content_df)}")
        log.info(f"   Fused pairs: {len(fused_df)}")
        
        # ===================================================================
        # Apply sensitivity filter
        # ===================================================================
        fused_df = apply_sensitivity_filter(fused_df, client_id, sensitivity_levels)
        
        if fused_df.empty:
            log.error("  No recommendations passed sensitivity filter")
            return {
                "status": "error",
                "error": "No recommendations passed sensitivity filter",
                "api_responses": api_responses
            }
        
        log.info(f"   After sensitivity filter: {len(fused_df)}")
        log.info("="*70 + "\n")
        
        # ===================================================================
        # Generate request_id
        # ===================================================================
        request_id = int(f"{client_id}{int(datetime.now().timestamp() * 1000) % 1000000}")
        
        # ===================================================================
        # Log Final Recommendations with COMPLETE details
        # ===================================================================
        log.info("\n" + "="*70)
        log.info("    FINAL FUSED RECOMMENDATIONS - TOP 10 COMPLETE DETAILS")
        log.info("="*70)
        log.info(f"Request ID: {request_id}")
        log.info(f"Client ID: {client_id}")
        log.info(f"Client Type: {client_type}")
        log.info(f"Total recommendations after filtering: {len(fused_df)}")
        log.info(f"Returning top {min(top_final, len(fused_df))}")
        log.info("\nWeights Used:")
        log.info(json.dumps(weights, indent=2))
        log.info("\n" + "-"*70)
        
        # Get top 10 for detailed logging
        top_10_recs = fused_df.head(10).to_dict('records')
        
        for idx, rec in enumerate(top_10_recs, 1):
            log.info(f"\n{'='*70}")
            log.info(f"RECOMMENDATION #{idx}")
            log.info(f"{'='*70}")
            
            # Core IDs and Scores
            log.info(f"\nCore Identifiers:")
            log.info(f"   Pair ID:           {rec.get('pair_id', 'N/A')}")
            log.info(f"   Vehicle ID:        {rec.get('vehicle_id', 'N/A')}")
            log.info(f"   Chauffeur ID:      {rec.get('chauffeur_id', 'N/A')}")
            log.info(f"   Final Rank:        {rec.get('final_rank', 'N/A')}")
            
            # Fusion Scores
            log.info(f"\n    Fusion Scores:")
            log.info(f"   Final Score:       {rec.get('final_score', 0):.6f}")
            log.info(f"   RRF Score:         {rec.get('rrf_score', 0):.6f}")
            log.info(f"   GCN Score:         {rec.get('gcn_score', 0):.6f}")
            log.info(f"   CF Score:          {rec.get('cf_score', 0):.6f}")
            log.info(f"   User Content:      {rec.get('user_content_score', 0):.6f}")
            log.info(f"   Item Content:      {rec.get('item_content_score', 0):.6f}")
            
            # LightGCN Detailed Scores
            log.info(f"\n  LightGCN Detailed Scores:")
            log.info(f"   CV Score:          {rec.get('cv_score', 'N/A')}")
            log.info(f"   CC Score:          {rec.get('cc_score', 'N/A')}")
            log.info(f"   VC Score:          {rec.get('vc_score', 'N/A')}")
            log.info(f"   Combined Score:    {rec.get('combined_score', 'N/A')}")
            log.info(f"   Weighted Score:    {rec.get('lightgcn_weighted_score', 'N/A')}")
            log.info(f"   GCN Avg Score:     {rec.get('gcn_avg_score', 'N/A')}")
            
        log.info("\n" + "="*70)
        log.info("END OF TOP 10 RECOMMENDATIONS")
        log.info("="*70 + "\n")
        
        # ===================================================================
        # Build final response
        # ===================================================================
        final_response = {
            "status": "success",
            "request_id": request_id,
            "client_id": client_id,
            "client_type": client_type,
            "weights_used": weights,
            "recommendations": fused_df.head(top_final).to_dict('records'),
            "api_responses": api_responses,
            "counts": {
                "lightgcn_pairs": len(lightgcn_df),
                "collaborative_pairs": len(collaborative_df),
                "user_content_pairs": len(user_content_df),
                "item_content_pairs": len(item_content_df),
                "fused_pairs": len(fused_df),
                "returned_pairs": min(top_final, len(fused_df))
            }
        }
        
        # Log final response summary
        log.info("\n" + "="*70)
        log.info("  FUSION REQUEST COMPLETED SUCCESSFULLY")
        log.info("="*70)
        log.info(f"Request ID:             {request_id}")
        log.info(f"Client ID:              {client_id}")
        log.info(f"Client Type:            {client_type}")
        log.info(f"Status:                 {final_response['status']}")
        log.info(f"Recommendations:        {final_response['counts']['returned_pairs']}")
        log.info(f"API Response Status:")
        log.info(f"   LightGCN:            {api_responses['lightgcn']['status']}")
        log.info(f"   Collaborative:       {api_responses['collaborative']['status']}")
        log.info(f"   User Content:        {api_responses['user_content']['status']}")
        log.info(f"   Item Content:        {api_responses['item_content']['status']}")
        log.info("="*70 + "\n")
        
        return final_response
        
    except Exception as e:
        log.error(f"\n{'='*70}")
        log.error(f"  FUSION ERROR")
        log.error(f"{'='*70}")
        log.error(f"Error: {str(e)}", exc_info=True)
        log.error(f"{'='*70}\n")
        
        return {
            "status": "error",
            "error": str(e),
            "api_responses": api_responses if 'api_responses' in locals() else {}
        }


def analyze_recommendation_overlap(lightgcn_df: pd.DataFrame,
                                  collaborative_df: pd.DataFrame) -> Dict:
    """
    Analyze overlap between LightGCN and Collaborative recommendations
    
    Returns:
        Dictionary with overlap statistics
    """
    logger.info("\n" + "="*60)
    logger.info("    RECOMMENDATION OVERLAP ANALYSIS")
    logger.info("="*60)
    
    # Get pair sets
    gcn_pairs = set(lightgcn_df['pair_id'].unique()) if not lightgcn_df.empty else set()
    cf_pairs = set(collaborative_df['pair_id'].unique()) if not collaborative_df.empty else set()
    
    # Calculate overlaps
    both_systems = gcn_pairs & cf_pairs
    only_gcn = gcn_pairs - cf_pairs
    only_cf = cf_pairs - gcn_pairs
    total_unique = gcn_pairs | cf_pairs
    
    # Calculate statistics
    overlap_count = len(both_systems)
    total_pairs = len(total_unique)
    overlap_pct = (overlap_count / total_pairs * 100) if total_pairs > 0 else 0
    
    # Print summary
    logger.info(f"\n Pair Counts:")
    logger.info(f"  • LightGCN pairs:        {len(gcn_pairs):4d}")
    logger.info(f"  • Collaborative pairs:   {len(cf_pairs):4d}")
    logger.info(f"  • Total unique pairs:    {len(total_unique):4d}")
    
    logger.info(f"\n  Overlap Statistics:")
    logger.info(f"  • Both systems:          {overlap_count:4d}  ({overlap_pct:.1f}%)")
    logger.info(f"  • Only LightGCN:         {len(only_gcn):4d}  ({len(only_gcn)/total_pairs*100:.1f}%)")
    logger.info(f"  • Only Collaborative:    {len(only_cf):4d}  ({len(only_cf)/total_pairs*100:.1f}%)")
    
    # Show sample overlapping pairs
    if both_systems:
        sample_size = min(5, len(both_systems))
        sample_both = list(both_systems)[:sample_size]
        logger.info(f"\n  Sample pairs in BOTH systems ({sample_size} of {len(both_systems)}):")
        for i, pair_id in enumerate(sample_both, 1):
            gcn_rank = lightgcn_df[lightgcn_df['pair_id'] == pair_id]['rank'].iloc[0] if not lightgcn_df.empty else 'N/A'
            cf_rank = collaborative_df[collaborative_df['pair_id'] == pair_id]['rank'].iloc[0] if not collaborative_df.empty else 'N/A'
            logger.info(f"  {i}. {pair_id:<20} | GCN rank: {gcn_rank:>3}, CF rank: {cf_rank:>3}")
    
    # Vehicle-level overlap
    if not lightgcn_df.empty and not collaborative_df.empty:
        gcn_vehicles = set(lightgcn_df['vehicle_id'].unique())
        cf_vehicles = set(collaborative_df['vehicle_id'].unique())
        both_vehicles = gcn_vehicles & cf_vehicles
        
        gcn_chauffeurs = set(lightgcn_df['chauffeur_id'].unique())
        cf_chauffeurs = set(collaborative_df['chauffeur_id'].unique())
        both_chauffeurs = gcn_chauffeurs & cf_chauffeurs
        
        logger.info(f"\n Vehicle Overlap:")
        logger.info(f"  • LightGCN vehicles:     {len(gcn_vehicles):4d}")
        logger.info(f"  • Collaborative vehicles:{len(cf_vehicles):4d}")
        logger.info(f"  • Shared vehicles:       {len(both_vehicles):4d}  ({len(both_vehicles)/len(gcn_vehicles | cf_vehicles)*100:.1f}%)")
        
        logger.info(f"\n👤 Chauffeur Overlap:")
        logger.info(f"  • LightGCN chauffeurs:   {len(gcn_chauffeurs):4d}")
        logger.info(f"  • Collaborative chauff.: {len(cf_chauffeurs):4d}")
        logger.info(f"  • Shared chauffeurs:     {len(both_chauffeurs):4d}  ({len(both_chauffeurs)/len(gcn_chauffeurs | cf_chauffeurs)*100:.1f}%)")
    
    logger.info("="*60 + "\n")
    
    return {
        "total_unique_pairs": len(total_unique),
        "lightgcn_pairs": len(gcn_pairs),
        "collaborative_pairs": len(cf_pairs),
        "overlap_pairs": overlap_count,
        "overlap_percentage": round(overlap_pct, 2),
        "only_lightgcn": len(only_gcn),
        "only_collaborative": len(only_cf),
        "vehicle_overlap": len(both_vehicles) if not lightgcn_df.empty and not collaborative_df.empty else 0,
        "chauffeur_overlap": len(both_chauffeurs) if not lightgcn_df.empty and not collaborative_df.empty else 0,
        "sample_both_systems": list(both_systems)[:5] if both_systems else [],
        "sample_only_gcn": list(only_gcn)[:3] if only_gcn else [],
        "sample_only_cf": list(only_cf)[:3] if only_cf else []
    }


def analyze_recommendation_overlap_4systems(lightgcn_df: pd.DataFrame,
                                           collaborative_df: pd.DataFrame,
                                           user_content_df: pd.DataFrame,
                                           item_content_df: pd.DataFrame) -> Dict:
    """Analyze overlap between all 4 recommendation systems"""
    
    logger.info("\n" + "="*70)
    logger.info("    4-SYSTEM RECOMMENDATION OVERLAP ANALYSIS")
    logger.info("="*70)
    
    # Get pair sets
    gcn_pairs = set(lightgcn_df['pair_id'].unique()) if not lightgcn_df.empty else set()
    cf_pairs = set(collaborative_df['pair_id'].unique()) if not collaborative_df.empty else set()
    uc_pairs = set(user_content_df['pair_id'].unique()) if not user_content_df.empty else set()
    ic_pairs = set(item_content_df['pair_id'].unique()) if not item_content_df.empty else set()
    
    total_unique = gcn_pairs | cf_pairs | uc_pairs | ic_pairs
    
    # All 4 systems
    all_four = gcn_pairs & cf_pairs & uc_pairs & ic_pairs
    
    # Any 3 systems
    gcn_cf_uc = gcn_pairs & cf_pairs & uc_pairs
    gcn_cf_ic = gcn_pairs & cf_pairs & ic_pairs
    gcn_uc_ic = gcn_pairs & uc_pairs & ic_pairs
    cf_uc_ic = cf_pairs & uc_pairs & ic_pairs
    any_three = gcn_cf_uc | gcn_cf_ic | gcn_uc_ic | cf_uc_ic
    
    logger.info(f"\n System Counts:")
    logger.info(f"  • LightGCN:       {len(gcn_pairs):5d}")
    logger.info(f"  • Collaborative:  {len(cf_pairs):5d}")
    logger.info(f"  • User Content:   {len(uc_pairs):5d}")
    logger.info(f"  • Item Content:   {len(ic_pairs):5d}")
    logger.info(f"  • Total Unique:   {len(total_unique):5d}")
    
    logger.info(f"\n  Overlap Statistics:")
    logger.info(f"  • All 4 systems:  {len(all_four):5d}  ({len(all_four)/len(total_unique)*100:.1f}%)")
    logger.info(f"  • Any 3 systems:  {len(any_three):5d}  ({len(any_three)/len(total_unique)*100:.1f}%)")
    
    logger.info("="*70 + "\n")
    
    return {
        "total_unique_pairs": len(total_unique),
        "lightgcn_pairs": len(gcn_pairs),
        "collaborative_pairs": len(cf_pairs),
        "user_content_pairs": len(uc_pairs),
        "item_content_pairs": len(ic_pairs),
        "all_four_systems": len(all_four),
        "any_three_systems": len(any_three),
        "overlap_percentage": round(len(all_four)/len(total_unique)*100, 2) if total_unique else 0
    }


def normalize_scores(values: List[float]) -> List[float]:
    """Normalize scores to 0-1 range"""
    if not values or len(values) == 0:
        return values
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return [0.5] * len(values)
    
    return [(v - min_val) / (max_val - min_val) for v in values]


def extract_collaborative_scores(cf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all match_scores from Collaborative API response
    Creates weighted CF combined score from 4 match_scores
    """
    if cf_df.empty:
        return cf_df
    
    logger.info("    Extracting Collaborative match_scores...")
    
    # Extract match_scores if nested
    if 'match_scores' in cf_df.columns:
        match_scores_df = pd.json_normalize(cf_df['match_scores'])
        
        for col in ['cf_score', 'overall_match', 'quality_score', 'requirement_match']:
            if col in match_scores_df.columns:
                cf_df[col] = match_scores_df[col]
                logger.info(f"      Extracted {col}")
    
    # Compute weighted CF combined score
    cf_df['cf_combined_score'] = 0
    
    for score_col, weight in CF_MATCH_WEIGHTS.items():
        if score_col in cf_df.columns:
            cf_df['cf_combined_score'] += weight * cf_df[score_col].fillna(0)
            logger.info(f"      Applied weight {weight:.2f} to {score_col}")
    
    # Also keep original predicted_rating * confidence
    if 'predicted_rating' in cf_df.columns and 'confidence' in cf_df.columns:
        cf_df['cf_rating_confidence'] = cf_df['predicted_rating'] * cf_df['confidence']
        logger.info(f"      Computed cf_rating_confidence = rating × confidence")
    
    # Normalize combined score
    min_val = cf_df['cf_combined_score'].min()
    max_val = cf_df['cf_combined_score'].max()
    if max_val > min_val:
        cf_df['cf_combined_score_normalized'] = (
            (cf_df['cf_combined_score'] - min_val) / (max_val - min_val)
        )
        logger.info(f"      Normalized CF combined score: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
    else:
        cf_df['cf_combined_score_normalized'] = 0.5
        logger.info(f"  ⚠ All CF scores equal, using 0.5")
    
    return cf_df


def extract_lightgcn_scores(gcn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and weight all features from LightGCN response
    Creates weighted composite score from 8+ features
    """
    if gcn_df.empty:
        return gcn_df
    
    logger.info("    Extracting LightGCN features...")
    
    # Compute weighted feature score
    gcn_df['lightgcn_weighted_score'] = 0
    features_found = []
    
    for feature, weight in LIGHTGCN_FEATURE_WEIGHTS.items():
        if feature in gcn_df.columns:
            col_min = gcn_df[feature].min()
            col_max = gcn_df[feature].max()
            
            if col_max > col_min:
                normalized = (gcn_df[feature] - col_min) / (col_max - col_min)
            else:
                normalized = 0.5
            
            gcn_df['lightgcn_weighted_score'] += weight * normalized
            features_found.append(feature)
            logger.info(f"      Applied weight {weight:.3f} to {feature}")
    
    logger.info(f"      Total features used: {len(features_found)}/{len(LIGHTGCN_FEATURE_WEIGHTS)}")
    
    # Also keep simple GCN average (3 core scores)
    if all(col in gcn_df.columns for col in ['cv_score', 'cc_score', 'vc_score']):
        # Normalize each score
        for col in ['cv_score', 'cc_score', 'vc_score']:
            values = gcn_df[col].fillna(0).tolist()
            gcn_df[f'{col}_normalized'] = normalize_scores(values)
        
        # Calculate average normalized score
        gcn_df['gcn_avg_score'] = (
            gcn_df['cv_score_normalized'] +
            gcn_df['cc_score_normalized'] +
            gcn_df['vc_score_normalized']
        ) / 3
        logger.info(f"      Computed gcn_avg_score (normalized average of 3 core scores)")
    else:
        gcn_df['gcn_avg_score'] = 0
        logger.info(f"  ⚠ Core scores missing, gcn_avg_score = 0")
    
    weighted_min = gcn_df['lightgcn_weighted_score'].min()
    weighted_max = gcn_df['lightgcn_weighted_score'].max()
    logger.info(f"      LightGCN weighted score range: [{weighted_min:.3f}, {weighted_max:.3f}]")
    
    return gcn_df


def process_lightgcn_response(lightgcn_data: Dict) -> pd.DataFrame:
    """Process LightGCN API response with feature extraction"""
    logger.info("Processing LightGCN response...")
    
    if not lightgcn_data:
        logger.warning("⚠ No LightGCN data received")
        return pd.DataFrame()
    
    # Log raw response structure
    logger.info(f"Response keys: {list(lightgcn_data.keys())}")
    
    # Handle nested structure
    if 'recommendations' in lightgcn_data and 'pairs' in lightgcn_data['recommendations']:
        pairs = lightgcn_data['recommendations']['pairs']
    elif 'pairs' in lightgcn_data:
        pairs = lightgcn_data['pairs']
    else:
        logger.warning("⚠ No pairs found in LightGCN response")
        return pd.DataFrame()
    
    if not pairs:
        logger.warning("⚠ LightGCN pairs list is empty")
        return pd.DataFrame()
    
    logger.info(f"Found {len(pairs)} pairs in response")
    
    df = pd.DataFrame(pairs)
    
    if 'chauffeur_id' not in df.columns or 'vehicle_id' not in df.columns:
        logger.error("  Missing chauffeur_id or vehicle_id in LightGCN response")
        return pd.DataFrame()
    
    # Convert IDs to strings
    df['chauffeur_id'] = df['chauffeur_id'].astype(float).astype(int).astype(str)
    df['vehicle_id'] = df['vehicle_id'].astype(float).astype(int).astype(str)
    df['pair_id'] = df['chauffeur_id'] + '_' + df['vehicle_id']
    
    logger.info(f"Sample pair_ids: {df['pair_id'].head(3).tolist()}")
    
    # Extract weighted features
    df = extract_lightgcn_scores(df)
    
    if 'rank' not in df.columns:
        df['rank'] = range(1, len(df) + 1)
    
    logger.info(f"  Processed {len(df)} LightGCN pairs with enriched features")
    
    return df


def process_collaborative_response(collab_data: Dict) -> pd.DataFrame:
    """Process Collaborative API response with match_scores extraction"""
    logger.info("Processing Collaborative response...")
    
    if not collab_data or 'recommendations' not in collab_data:
        logger.warning("⚠ No recommendations found in Collaborative response")
        return pd.DataFrame()
    
    # Log raw response structure
    logger.info(f"Response keys: {list(collab_data.keys())}")
    
    recommendations = collab_data['recommendations']
    
    if not recommendations:
        logger.warning("⚠ Collaborative recommendations list is empty")
        return pd.DataFrame()
    
    logger.info(f"Found {len(recommendations)} recommendations in response")
    
    df = pd.DataFrame(recommendations)
    
    # Handle pair_id
    if 'pair_id' in df.columns:
        df['pair_id'] = df['pair_id'].astype(str)
    elif 'vehicle_id' in df.columns and 'chauffeur_id' in df.columns:
        df['vehicle_id'] = df['vehicle_id'].astype(str)
        df['chauffeur_id'] = df['chauffeur_id'].astype(str)
        df['pair_id'] = df['chauffeur_id'] + '_' + df['vehicle_id']
    else:
        logger.error("  Cannot determine pair_id format in Collaborative response")
        return pd.DataFrame()
    
    logger.info(f"Sample pair_ids: {df['pair_id'].head(3).tolist()}")
    
    # Extract match_scores
    df = extract_collaborative_scores(df)
    
    if 'rank' not in df.columns:
        df['rank'] = range(1, len(df) + 1)
    
    logger.info(f"  Processed {len(df)} Collaborative pairs with match_scores")
    
    return df


def process_user_content_response(user_content_data: Dict) -> pd.DataFrame:
    """Process User Content API response"""
    logger.info("Processing User Content response...")
    
    if not user_content_data or not user_content_data.get('success'):
        logger.warning("⚠ No User Content data received")
        return pd.DataFrame()
    
    data = user_content_data.get('data', [])
    
    if not data:
        logger.warning("⚠ User Content data list is empty")
        return pd.DataFrame()
    
    logger.info(f"Found {len(data)} recommendations in response")
    
    df = pd.DataFrame(data)
    
    # Ensure required columns exist
    if 'RefChauffeur' not in df.columns or 'RefVehicle' not in df.columns:
        logger.error("  Missing RefChauffeur or RefVehicle in User Content response")
        return pd.DataFrame()
    
    # Convert IDs to strings
    df['RefChauffeur'] = df['RefChauffeur'].astype(int).astype(str)
    df['RefVehicle'] = df['RefVehicle'].astype(int).astype(str)
    df['pair_id'] = df['RefChauffeur'] + '_' + df['RefVehicle']
    
    # Use similarity score
    if 'similarity' not in df.columns:
        logger.error("  Missing similarity in User Content response")
        return pd.DataFrame()
    
    df['user_content_score'] = df['similarity']
    
    # Add rank if not present
    if 'rank' not in df.columns:
        df['rank'] = range(1, len(df) + 1)
    
    logger.info(f"Sample pair_ids: {df['pair_id'].head(3).tolist()}")
    logger.info(f"  Processed {len(df)} User Content pairs")
    
    return df


def process_item_content_response(item_content_data: Dict) -> pd.DataFrame:
    """Process Item Content API response"""
    logger.info("Processing Item Content response...")
    
    if not item_content_data or not item_content_data.get('success'):
        logger.warning("⚠ No Item Content data received")
        return pd.DataFrame()
    
    data = item_content_data.get('data', [])
    
    if not data:
        logger.warning("⚠ Item Content data list is empty")
        return pd.DataFrame()
    
    logger.info(f"Found {len(data)} recommendations in response")
    
    df = pd.DataFrame(data)
    
    # Ensure required columns exist
    if 'RefChauffeur' not in df.columns or 'RefVehicle' not in df.columns:
        logger.error("  Missing RefChauffeur or RefVehicle in Item Content response")
        return pd.DataFrame()
    
    # Convert IDs to strings
    df['RefChauffeur'] = df['RefChauffeur'].astype(int).astype(str)
    df['RefVehicle'] = df['RefVehicle'].astype(int).astype(str)
    df['pair_id'] = df['RefChauffeur'] + '_' + df['RefVehicle']
    
    # Use similarity score
    if 'similarity' not in df.columns:
        logger.error("  Missing similarity in Item Content response")
        return pd.DataFrame()
    
    df['item_content_score'] = df['similarity']
    
    # Add rank if not present
    if 'rank' not in df.columns:
        df['rank'] = range(1, len(df) + 1)
    
    logger.info(f"Sample pair_ids: {df['pair_id'].head(3).tolist()}")
    logger.info(f"  Processed {len(df)} Item Content pairs")
    
    return df


def compute_rrf(ranks: List[int], k: int = RRF_K) -> float:
    """Compute Reciprocal Rank Fusion score"""
    if not ranks:
        return 0.0
    return sum(1.0 / (k + rank) for rank in ranks if rank > 0)


def fuse_recommendations(lightgcn_df: pd.DataFrame,
                        collaborative_df: pd.DataFrame,
                        user_content_df: pd.DataFrame,
                        item_content_df: pd.DataFrame,
                        weights: Dict[str, float]) -> pd.DataFrame:
    """
    Fuse recommendations using enriched scores from all 4 systems
    Uses grid search optimized weights
    """
    logger.info(" Fusing recommendations from 4 systems...")
    
    # Get all unique pair_ids
    all_pairs = set()
    if not lightgcn_df.empty:
        all_pairs.update(lightgcn_df['pair_id'].unique())
    if not collaborative_df.empty:
        all_pairs.update(collaborative_df['pair_id'].unique())
    if not user_content_df.empty:
        all_pairs.update(user_content_df['pair_id'].unique())
    if not item_content_df.empty:
        all_pairs.update(item_content_df['pair_id'].unique())
    
    if not all_pairs:
        logger.warning(" No pairs found to fuse")
        return pd.DataFrame()
    
    # Analyze overlap
    overlap_stats = analyze_recommendation_overlap_4systems(
        lightgcn_df, collaborative_df, user_content_df, item_content_df
    )

    logger.info(f"Total unique pairs: {len(all_pairs)}")
    logger.info(f"  • From LightGCN: {len(lightgcn_df['pair_id'].unique()) if not lightgcn_df.empty else 0}")
    logger.info(f"  • From Collaborative: {len(collaborative_df['pair_id'].unique()) if not collaborative_df.empty else 0}")
    logger.info(f"  • From User Content: {len(user_content_df['pair_id'].unique()) if not user_content_df.empty else 0}")
    logger.info(f"  • From Item Content: {len(item_content_df['pair_id'].unique()) if not item_content_df.empty else 0}")
    
    # Initialize result DataFrame
    result = pd.DataFrame({'pair_id': list(all_pairs)})
    
    # 1. Compute RRF scores (from rankings)
    logger.info("\n Computing RRF scores...")
    rrf_scores = {}
    
    for pair_id in all_pairs:
        ranks = []
        
        # LightGCN rank
        if not lightgcn_df.empty:
            gcn_row = lightgcn_df[lightgcn_df['pair_id'] == pair_id]
            if not gcn_row.empty and 'rank' in gcn_row.columns:
                ranks.append(gcn_row.iloc[0]['rank'])
        
        # Collaborative rank
        if not collaborative_df.empty:
            cf_row = collaborative_df[collaborative_df['pair_id'] == pair_id]
            if not cf_row.empty and 'rank' in cf_row.columns:
                ranks.append(cf_row.iloc[0]['rank'])
        
        # User Content rank
        if not user_content_df.empty:
            uc_row = user_content_df[user_content_df['pair_id'] == pair_id]
            if not uc_row.empty and 'rank' in uc_row.columns:
                ranks.append(uc_row.iloc[0]['rank'])
        
        # Item Content rank
        if not item_content_df.empty:
            ic_row = item_content_df[item_content_df['pair_id'] == pair_id]
            if not ic_row.empty and 'rank' in ic_row.columns:
                ranks.append(ic_row.iloc[0]['rank'])
        
        rrf_scores[pair_id] = compute_rrf(ranks)
    
    result['rrf_score'] = result['pair_id'].map(rrf_scores)
    logger.info(f"      RRF scores computed for {len(all_pairs)} pairs")
    
    # 2. Get GCN weighted score (normalized)
    logger.info("\n Extracting GCN scores...")
    gcn_scores = {}
    
    if not lightgcn_df.empty:
        for _, row in lightgcn_df.iterrows():
            if 'lightgcn_weighted_score' in row and pd.notna(row['lightgcn_weighted_score']):
                gcn_scores[row['pair_id']] = row['lightgcn_weighted_score']
            elif 'gcn_avg_score' in row:
                gcn_scores[row['pair_id']] = row.get('gcn_avg_score', 0)
    
    result['gcn_score'] = result['pair_id'].map(gcn_scores).fillna(0)
    
    # Normalize GCN scores to 0-1
    if result['gcn_score'].max() > 0:
        result['gcn_score'] = (result['gcn_score'] - result['gcn_score'].min()) / \
                              (result['gcn_score'].max() - result['gcn_score'].min())
    
    logger.info(f"      GCN scores extracted and normalized: {(result['gcn_score'] > 0).sum()} non-zero")
    
    # 3. Get CF combined score (normalized)
    logger.info("\n Extracting CF scores...")
    cf_scores = {}
    
    if not collaborative_df.empty:
        for _, row in collaborative_df.iterrows():
            if 'cf_combined_score_normalized' in row and pd.notna(row['cf_combined_score_normalized']):
                cf_scores[row['pair_id']] = row['cf_combined_score_normalized']
            elif 'confidence' in row:
                cf_scores[row['pair_id']] = row.get('confidence', 0)
    
    result['cf_score'] = result['pair_id'].map(cf_scores).fillna(0)
    
    # Normalize CF scores to 0-1
    if result['cf_score'].max() > 0:
        result['cf_score'] = (result['cf_score'] - result['cf_score'].min()) / \
                             (result['cf_score'].max() - result['cf_score'].min())
    
    logger.info(f"      CF scores extracted and normalized: {(result['cf_score'] > 0).sum()} non-zero")
    
    # 4. Get User Content similarity (already 0-1)
    logger.info("\n Extracting User Content scores...")
    user_content_scores = {}
    
    if not user_content_df.empty:
        for _, row in user_content_df.iterrows():
            if 'user_content_score' in row:
                user_content_scores[row['pair_id']] = row['user_content_score']
    
    result['user_content_score'] = result['pair_id'].map(user_content_scores).fillna(0)
    logger.info(f"      User Content scores extracted: {(result['user_content_score'] > 0).sum()} non-zero")
    
    # 5. Get Item Content similarity (already 0-1)
    logger.info("\n Extracting Item Content scores...")
    item_content_scores = {}
    
    if not item_content_df.empty:
        for _, row in item_content_df.iterrows():
            if 'item_content_score' in row:
                item_content_scores[row['pair_id']] = row['item_content_score']
    
    result['item_content_score'] = result['pair_id'].map(item_content_scores).fillna(0)
    logger.info(f"      Item Content scores extracted: {(result['item_content_score'] > 0).sum()} non-zero")
    
    # 6. Compute final weighted score
    logger.info("\n Computing final weighted scores...")
    logger.info(f"Weights applied:")
    logger.info(f"  • alpha1_rrf: {weights.get('alpha1_rrf', 0.3):.3f}")
    logger.info(f"  • alpha2_gcn: {weights.get('alpha2_gcn', 0.3):.3f}")
    logger.info(f"  • alpha3_item_sim: {weights.get('alpha3_item_sim', 0.1):.3f}")
    logger.info(f"  • alpha4_user_sim: {weights.get('alpha4_user_sim', 0.1):.3f}")
    logger.info(f"  • alpha5_confidence: {weights.get('alpha5_confidence', 0.1):.3f}")
    
    result['final_score'] = (
        weights.get('alpha1_rrf', 0.3) * result['rrf_score'] +
        weights.get('alpha2_gcn', 0.3) * result['gcn_score'] +
        weights.get('alpha3_item_sim', 0.1) * result['item_content_score'] +
        weights.get('alpha4_user_sim', 0.1) * result['user_content_score'] +
        weights.get('alpha5_confidence', 0.1) * result['cf_score']
    )
    
    # Sort by final score
    result = result.sort_values('final_score', ascending=False).reset_index(drop=True)
    result['final_rank'] = range(1, len(result) + 1)
    
    # Enrich with metadata
    result = enrich_with_metadata_4systems(result, lightgcn_df, collaborative_df,
                                           user_content_df, item_content_df)
    
    # CRITICAL FIX: Ensure vehicle_id and chauffeur_id exist
    logger.info(f"\n POST-ENRICHMENT ID VALIDATION")
    
    # Check what ID columns we have
    has_vehicle_id = 'vehicle_id' in result.columns
    has_chauffeur_id = 'chauffeur_id' in result.columns
    has_refvehicle = 'RefVehicle' in result.columns
    has_refchauffeur = 'RefChauffeur' in result.columns
    
    logger.info(f"  • vehicle_id present: {has_vehicle_id}")
    logger.info(f"  • chauffeur_id present: {has_chauffeur_id}")
    logger.info(f"  • RefVehicle present: {has_refvehicle}")
    logger.info(f"  • RefChauffeur present: {has_refchauffeur}")
    
    # Strategy 1: Copy from Ref* columns if they exist
    if not has_vehicle_id and has_refvehicle:
        result['vehicle_id'] = result['RefVehicle']
        logger.info(f"      Copied vehicle_id from RefVehicle")
    
    if not has_chauffeur_id and has_refchauffeur:
        result['chauffeur_id'] = result['RefChauffeur']
        logger.info(f"      Copied chauffeur_id from RefChauffeur")
    
    # Strategy 2: Extract from pair_id if still missing
    if 'vehicle_id' not in result.columns or result['vehicle_id'].isna().any():
        logger.info(f"    Extracting IDs from pair_id as fallback")
        
        # Split pair_id (format: "chauffeur_vehicle")
        split_ids = result['pair_id'].str.split('_', n=1, expand=True)
        
        if 'chauffeur_id' not in result.columns or result['chauffeur_id'].isna().any():
            result['chauffeur_id'] = result['chauffeur_id'].fillna(split_ids[0]) if 'chauffeur_id' in result.columns else split_ids[0]
        
        if 'vehicle_id' not in result.columns or result['vehicle_id'].isna().any():
            result['vehicle_id'] = result['vehicle_id'].fillna(split_ids[1]) if 'vehicle_id' in result.columns else split_ids[1]
        
        logger.info(f"      Extracted from pair_id")
    
    # Validate that we now have IDs
    missing_vehicle = result['vehicle_id'].isna().sum() if 'vehicle_id' in result.columns else len(result)
    missing_chauffeur = result['chauffeur_id'].isna().sum() if 'chauffeur_id' in result.columns else len(result)
    
    if missing_vehicle > 0 or missing_chauffeur > 0:
        logger.error(f"    STILL MISSING IDs after all strategies!")
        logger.error(f"     Missing vehicle_id: {missing_vehicle}/{len(result)}")
        logger.error(f"     Missing chauffeur_id: {missing_chauffeur}/{len(result)}")
        if len(result) > 0:
            logger.error(f"     Sample pair_id: {result['pair_id'].iloc[0]}")
            logger.error(f"     Available columns: {list(result.columns)}")
    else:
        logger.info(f"    All {len(result)} recommendations have IDs")
    
    logger.info(f"\n  Fusion complete: {len(result)} pairs ranked")
    
    # Diagnostics
    non_zero_counts = {
        'rrf_score': (result['rrf_score'] > 0).sum(),
        'gcn_score': (result['gcn_score'] > 0).sum(),
        'cf_score': (result['cf_score'] > 0).sum(),
        'user_content_score': (result['user_content_score'] > 0).sum(),
        'item_content_score': (result['item_content_score'] > 0).sum()
    }
    logger.info(f"Non-zero score distribution:")
    for score_type, count in non_zero_counts.items():
        logger.info(f"  • {score_type}: {count}/{len(result)} ({count/len(result)*100:.1f}%)")
    
    return result


def adjust_weights_for_cold_start(lightgcn_response: Dict, weights: Dict[str, float]) -> Dict[str, float]:
    """
    Adjust weights if LightGCN indicates cold start with similarity_based method
    Boosts content-based similarity weights (item + user) by 5x
    """
    is_cold_start = lightgcn_response.get('is_cold_start', False)
    recommendation_method = lightgcn_response.get('recommendation_method', '')
    
    if is_cold_start and recommendation_method == 'similarity_based':
        logger.info("\n    COLD START DETECTED (similarity_based) - Boosting content similarity weights by 5x")
        adjusted_weights = weights.copy()
        
        # Multiply BOTH similarity weights by 5
        original_item_sim = adjusted_weights.get('alpha3_item_sim', 0.1)
        original_user_sim = adjusted_weights.get('alpha4_user_sim', 0.1)
        
        adjusted_weights['alpha3_item_sim'] = original_item_sim * 5
        adjusted_weights['alpha4_user_sim'] = original_user_sim * 5
        
        # Normalize to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        for key in adjusted_weights:
            adjusted_weights[key] = adjusted_weights[key] / total_weight
        
        logger.info(f"Original item similarity weight: {original_item_sim:.3f} → {adjusted_weights['alpha3_item_sim']:.3f}")
        logger.info(f"Original user similarity weight: {original_user_sim:.3f} → {adjusted_weights['alpha4_user_sim']:.3f}")
        logger.info(f"All adjusted weights: {json.dumps(adjusted_weights, indent=2)}")
        
        return adjusted_weights
    
    elif is_cold_start:
        logger.info(f"\n    COLD START DETECTED but method is '{recommendation_method}', keeping original weights")
    
    return weights


def enrich_with_metadata(fused_df: pd.DataFrame,
                        lightgcn_df: pd.DataFrame,
                        collaborative_df: pd.DataFrame) -> pd.DataFrame:
    """Add metadata from original responses"""
    
    # Try LightGCN first
    if not lightgcn_df.empty:
        meta_cols = ['pair_id', 'vehicle_id', 'chauffeur_id']
        
        for col in lightgcn_df.columns:
            if col not in meta_cols and col not in fused_df.columns:
                meta_cols.append(col)
        
        available_cols = [col for col in meta_cols if col in lightgcn_df.columns]
        
        if available_cols:
            gcn_meta = lightgcn_df[available_cols].drop_duplicates(subset=['pair_id'])
            fused_df = fused_df.merge(gcn_meta, on='pair_id', how='left')
            logger.info(f"      Enriched with {len(available_cols)} columns from LightGCN")
    
    # If no metadata, try collaborative
    if 'vehicle_id' not in fused_df.columns and not collaborative_df.empty:
        meta_cols = ['pair_id', 'vehicle_id', 'chauffeur_id']
        available_cols = [col for col in meta_cols if col in collaborative_df.columns]
        
        if available_cols:
            cf_meta = collaborative_df[available_cols].drop_duplicates(subset=['pair_id'])
            fused_df = fused_df.merge(cf_meta, on='pair_id', how='left')
            logger.info(f"      Enriched with {len(available_cols)} columns from Collaborative")
    
    return fused_df


def enrich_with_metadata_4systems(fused_df: pd.DataFrame,
                                  lightgcn_df: pd.DataFrame,
                                  collaborative_df: pd.DataFrame,
                                  user_content_df: pd.DataFrame,
                                  item_content_df: pd.DataFrame) -> pd.DataFrame:
    """Add metadata from all 4 systems (priority: LightGCN > User Content > Item Content > Collaborative)"""
    
    # Try LightGCN first
    if not lightgcn_df.empty:
        meta_cols = ['pair_id', 'vehicle_id', 'chauffeur_id']
        for col in lightgcn_df.columns:
            if col not in meta_cols and col not in fused_df.columns:
                meta_cols.append(col)
        
        available_cols = [col for col in meta_cols if col in lightgcn_df.columns]
        if available_cols:
            gcn_meta = lightgcn_df[available_cols].drop_duplicates(subset=['pair_id'])
            fused_df = fused_df.merge(gcn_meta, on='pair_id', how='left')
            logger.info(f"      Enriched with {len(available_cols)} columns from LightGCN")
    
    # Fill missing with User Content (use RefVehicle/RefChauffeur)
    if not user_content_df.empty:
        meta_cols = ['pair_id', 'RefVehicle', 'RefChauffeur', 'similarity']
        available_cols = [col for col in meta_cols if col in user_content_df.columns]
        
        if available_cols:
            uc_meta = user_content_df[available_cols].drop_duplicates(subset=['pair_id'])
            
            # Merge with suffix to avoid conflicts
            fused_df = fused_df.merge(uc_meta, on='pair_id', how='left', suffixes=('', '_uc'))
            
            # Copy RefVehicle/RefChauffeur if main IDs are missing
            if 'RefVehicle' in fused_df.columns and 'vehicle_id' not in fused_df.columns:
                fused_df['vehicle_id'] = fused_df['RefVehicle']
            elif 'RefVehicle' in fused_df.columns and 'vehicle_id' in fused_df.columns:
                fused_df['vehicle_id'] = fused_df['vehicle_id'].fillna(fused_df['RefVehicle'])
            
            if 'RefChauffeur' in fused_df.columns and 'chauffeur_id' not in fused_df.columns:
                fused_df['chauffeur_id'] = fused_df['RefChauffeur']
            elif 'RefChauffeur' in fused_df.columns and 'chauffeur_id' in fused_df.columns:
                fused_df['chauffeur_id'] = fused_df['chauffeur_id'].fillna(fused_df['RefChauffeur'])
            
            logger.info(f"      Enriched with {len(available_cols)} columns from User Content")
    
    # Fill missing with Item Content
    if not item_content_df.empty:
        meta_cols = ['pair_id', 'RefVehicle', 'RefChauffeur', 'similarity']
        available_cols = [col for col in meta_cols if col in item_content_df.columns]
        
        if available_cols:
            ic_meta = item_content_df[available_cols].drop_duplicates(subset=['pair_id'])
            
            # Merge with suffix
            fused_df = fused_df.merge(ic_meta, on='pair_id', how='left', suffixes=('', '_ic'))
            
            # Copy RefVehicle/RefChauffeur if still missing
            if 'RefVehicle_ic' in fused_df.columns:
                if 'vehicle_id' not in fused_df.columns:
                    fused_df['vehicle_id'] = fused_df['RefVehicle_ic']
                else:
                    fused_df['vehicle_id'] = fused_df['vehicle_id'].fillna(fused_df['RefVehicle_ic'])
            
            if 'RefChauffeur_ic' in fused_df.columns:
                if 'chauffeur_id' not in fused_df.columns:
                    fused_df['chauffeur_id'] = fused_df['RefChauffeur_ic']
                else:
                    fused_df['chauffeur_id'] = fused_df['chauffeur_id'].fillna(fused_df['RefChauffeur_ic'])
            
            logger.info(f"      Enriched with {len(available_cols)} columns from Item Content")
    
    # Finally Collaborative
    if not collaborative_df.empty:
        # Check if we still need IDs
        needs_ids = 'vehicle_id' not in fused_df.columns or 'chauffeur_id' not in fused_df.columns
        
        if needs_ids:
            meta_cols = ['pair_id', 'vehicle_id', 'chauffeur_id']
            available_cols = [col for col in meta_cols if col in collaborative_df.columns]
            if available_cols:
                cf_meta = collaborative_df[available_cols].drop_duplicates(subset=['pair_id'])
                fused_df = fused_df.merge(cf_meta, on='pair_id', how='left', suffixes=('', '_cf'))
                logger.info(f"      Enriched with {len(available_cols)} columns from Collaborative")
    
    return fused_df


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    ---
    tags:
      - System
    responses:
      200:
        description: Service is healthy
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "lightgcn": LIGHTGCN_API,
            "collaborative": COLLABORATIVE_API
        },
        "client_types_loaded": len(client_types)
    })


@app.route('/api/v1/recommend', methods=['POST'])
def recommend():
    """
    Get recommendations from both LightGCN and Collaborative APIs
    ---
    tags:
      - Recommendations
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - client_id
            - no_of_pax
          properties:
            client_id:
              type: integer
              example: 811
            no_of_pax:
              type: integer
              example: 2
    responses:
      200:
        description: Successful recommendation
      400:
        description: Invalid request
      500:
        description: Internal server error
    """
    start_time = datetime.now()
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "error": "No JSON data provided"
            }), 400
        
        # Validate required fields
        if 'client_id' not in data:
            return jsonify({
                "status": "error",
                "error": "client_id is required"
            }), 400
        
        if 'no_of_pax' not in data:
            return jsonify({
                "status": "error",
                "error": "no_of_pax is required"
            }), 400
        
        # Create request object
        combiner_request = CombinerRequest(**data)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING REQUEST FOR CLIENT {combiner_request.client_id}")
        logger.info(f"Requesting {combiner_request.top_k} recommendations from each API")
        logger.info(f"{'='*60}")
        
        # Convert to API-specific formats and call both APIs
        lightgcn_request = combiner_request.to_lightgcn_request()
        collaborative_request = combiner_request.to_collaborative_request()
        
        # Call LightGCN API
        logger.info("\n  Calling LightGCN API...")
        lightgcn_response = None
        try:
            response = requests.post(
                LIGHTGCN_API,
                json=lightgcn_request,
                timeout=3000
            )
            if response.status_code == 200:
                lightgcn_response = response.json()
                if isinstance(lightgcn_response, dict) and 'recommendations' in lightgcn_response:
                    pairs_count = len(lightgcn_response['recommendations'].get('pairs', []))
                elif 'pairs' in lightgcn_response:
                    pairs_count = len(lightgcn_response.get('pairs', []))
                else:
                    pairs_count = 0
                
                logger.info(f"  LightGCN returned {pairs_count} pairs")
            else:
                logger.error(f"  LightGCN API error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"  LightGCN API call failed: {str(e)}")
        
        # Call Collaborative API
        logger.info("\n  Calling Collaborative API...")
        collaborative_response = None
        try:
            response = requests.post(
                COLLABORATIVE_API,
                json=collaborative_request,
                timeout=3000
            )
            if response.status_code == 200:
                collaborative_response = response.json()
                recs_count = len(collaborative_response.get('recommendations', []))
                logger.info(f"  Collaborative returned {recs_count} pairs")
            else:
                logger.error(f"  Collaborative API error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"  Collaborative API call failed: {str(e)}")
        
        # Call User Content API
        logger.info("\n   Calling User Content API...")
        user_content_response = None
        try:
            user_content_request = combiner_request.to_user_content_request()
            response = requests.post(
                USER_CONTENT_API,
                json=user_content_request,
                timeout=3000
            )
            if response.status_code == 200:
                user_content_response = response.json()
                if user_content_response.get('success'):
                    recs_count = len(user_content_response.get('data', []))
                    logger.info(f"  User Content returned {recs_count} pairs")
                else:
                    logger.warning(f"  User Content: {user_content_response.get('message')}")
            else:
                logger.error(f"  User Content API error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"  User Content API call failed: {str(e)}")
        
        # Call Item Content API
        logger.info("\n   Calling Item Content API...")
        item_content_response = None
        try:
            item_content_request = combiner_request.to_item_content_request()
            response = requests.post(
                ITEM_CONTENT_API,
                json=item_content_request,
                timeout=3000
            )
            if response.status_code == 200:
                item_content_response = response.json()
                if item_content_response.get('success'):
                    recs_count = len(item_content_response.get('data', []))
                    logger.info(f"  Item Content returned {recs_count} pairs")
                else:
                    logger.warning(f"  Item Content: {item_content_response.get('message')}")
            else:
                logger.error(f"  Item Content API error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"  Item Content API call failed: {str(e)}")
        
        # Check if we got any data
        if not lightgcn_response and not collaborative_response and not user_content_response and not item_content_response:
            return jsonify({
                "status": "error",
                "error": "All recommendation APIs failed to respond"
            }), 503
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Build response
        response_data = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "request_id": f"CMB-{combiner_request.client_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "client_id": combiner_request.client_id,
            "processing_time_seconds": processing_time,
            "lightgcn_response": lightgcn_response if lightgcn_response else {"error": "Failed to get response"},
            "collaborative_response": collaborative_response if collaborative_response else {"error": "Failed to get response"},
            "user_content_response": user_content_response if user_content_response else {"error": "Failed to get response"},
            "item_content_response": item_content_response if item_content_response else {"error": "Failed to get response"}
        }
        
        logger.info(f"\n  Request completed in {processing_time:.3f}s")
        logger.info(f"{'='*60}\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"  Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500



def redistribute_weights_for_available_apis(
    weights: Dict[str, float],
    lightgcn_available: bool,
    collaborative_available: bool,
    user_content_available: bool,
    item_content_available: bool
) -> Dict[str, float]:
    """
    Redistribute weights when some APIs are unavailable.
    Maintains relative proportions among available sources.
    
    This ensures that if an API is down, its weight is redistributed
    to other available APIs rather than being wasted.
    
    Args:
        weights: Original weight dict
        lightgcn_available: Whether LightGCN returned data
        collaborative_available: Whether Collaborative API returned data
        user_content_available: Whether User Content API returned data
        item_content_available: Whether Item Content API returned data
        
    Returns:
        Adjusted weights dict that sums to 1.0
    """
    # Map weight keys to availability
    availability = {
        'alpha2_gcn': lightgcn_available,
        'alpha5_confidence': collaborative_available,
        'alpha4_user_sim': user_content_available,
        'alpha3_item_sim': item_content_available,
        'alpha1_rrf': any([lightgcn_available, collaborative_available, 
                          user_content_available, item_content_available])
    }
    
    # Check how many sources are available
    available_count = sum([lightgcn_available, collaborative_available, 
                          user_content_available, item_content_available])
    
    if available_count == 0:
        logger.warning("⚠️ No APIs returned data - using original weights")
        return weights
    
    if available_count == 4:
        # All APIs available - no redistribution needed
        return weights
    
    # Calculate sum of available weights
    available_weight_sum = sum(
        weights.get(key, 0) for key, available in availability.items() if available
    )
    
    if available_weight_sum == 0:
        logger.warning("⚠️ Available weight sum is 0 - using original weights")
        return weights
    
    # Redistribute: scale up available weights proportionally
    new_weights = {}
    for key, available in availability.items():
        if available:
            new_weights[key] = weights.get(key, 0) / available_weight_sum
        else:
            new_weights[key] = 0.0
    
    logger.info(f"⚖️ WEIGHT REDISTRIBUTION (some APIs unavailable):")
    logger.info(f"   APIs Available: LightGCN={lightgcn_available}, CF={collaborative_available}, "
                f"User={user_content_available}, Item={item_content_available}")
    logger.info(f"   Original weights: {weights}")
    logger.info(f"   Redistributed:    {new_weights}")
    logger.info(f"   New sum: {sum(new_weights.values()):.4f}")
    
    return new_weights

@app.route('/api/v1/fuse-recommendations', methods=['POST'])
def fuse_recommendations_endpoint():
    """
    Dynamically call all 4 APIs and fuse recommendations
    ---
    tags:
      - Recommendations
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - client_id
            - no_of_pax
          properties:
            client_id:
              type: integer
              example: 811
              description: Client ID
            no_of_pax:
              type: integer
              example: 2
              description: Number of passengers
    responses:
      200:
        description: Successfully fused recommendations with complete metadata
      400:
        description: Invalid request - missing required fields
      503:
        description: All recommendation APIs failed to respond
    """
    start_time = datetime.now()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "error": "No JSON data provided"}), 400
        
        if 'client_id' not in data:
            return jsonify({"status": "error", "error": "client_id is required"}), 400
        
        if 'no_of_pax' not in data:
            return jsonify({"status": "error", "error": "no_of_pax is required"}), 400
        
        client_id = data['client_id']
        top_final = data.pop('top_final', 20)
        
        # Extract custom weights if provided
        custom_weights = data.pop('weights', None)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING FUSION REQUEST FOR CLIENT {client_id}")
        logger.info(f"{'='*60}")
        
        # Get client type and load weights
        client_type = client_types.get(client_id, 'UNKNOWN')
        logger.info(f"this is the client type of the current client : {client_type}")
        if custom_weights:
            weights = custom_weights
            logger.info("Using custom weights from request")
        else:
            base_weights = optimizer.load_weights(client_type)
            logger.info(f"Using grid search weights for client type: {client_type}")

            # Add safety fallback if weights are still None or empty
            if not base_weights:
                logger.warning("No weights loaded, using default weights")
                base_weights = {
                    'alpha1_rrf': 0.3,
                    'alpha2_gcn': 0.3,
                    'alpha3_item_sim': 0.1,
                    'alpha4_user_sim': 0.1,
                    'alpha5_confidence': 0.1
                }
        
            weights_result = weight_selector.get_weights_with_source(client_id, client_type, base_weights)
            weights = weights_result.get('weights', base_weights)
            logger.info(f"Using feedback-personalized weights for client {client_id}")

        logger.info(f"Client type: {client_type}")
        logger.info(f"Weights: {json.dumps(weights, indent=2)}")
        
        # Set top_k for fetching
        data['top_k'] = 100
        
        # Create request object
        combiner_request = CombinerRequest(**data)
        
        # Prepare requests for all 4 APIs
        lightgcn_request = combiner_request.to_lightgcn_request()
        collaborative_request = combiner_request.to_collaborative_request()
        user_content_request = combiner_request.to_user_content_request()
        item_content_request = combiner_request.to_item_content_request()
        
        # ===================================================================
        # Call LightGCN API
        # ===================================================================
        logger.info("\n  Calling LightGCN API...")
        lightgcn_response = safe_api_call(LIGHTGCN_API, lightgcn_request, "LightGCN")
        lightgcn_df = pd.DataFrame()

        if lightgcn_response:
            lightgcn_df = process_lightgcn_response(lightgcn_response)
            logger.info(f" LightGCN returned {len(lightgcn_df)} pairs")
            # Adjust weights for cold start if needed
            weights = adjust_weights_for_cold_start(lightgcn_response, weights)
        else:
            logger.warning(" LightGCN API call failed, proceeding without it")
        logger.debug("DONE LIGHTGCN")
        # ===================================================================
        # Call Collaborative API
        # ===================================================================
        collaborative_response = safe_api_call(COLLABORATIVE_API, collaborative_request, "Collaborative")
        logger.debug("collaborative_response: %s", collaborative_response)

        collaborative_df = pd.DataFrame()

        if collaborative_response:
            collaborative_df = process_collaborative_response(collaborative_response)
            logger.info(f" Collaborative returned {len(collaborative_df)} pairs")
        else:
            logger.warning(" Collaborative API call failed, proceeding without it")
        
        # ===================================================================
        # Call User Content API
        # ===================================================================
        user_content_response = safe_api_call(USER_CONTENT_API, user_content_request, "User Content")
        user_content_df = pd.DataFrame()

        if user_content_response:
            user_content_df = process_user_content_response(user_content_response)
            logger.info(f" User Content returned {len(user_content_df)} pairs")
        else:
            logger.warning(" User Content API call failed, proceeding without it")
        
        # ===================================================================
        # Call Item Content API
        # ===================================================================
        logger.info("\n Calling Item Content API...")
        item_content_response = safe_api_call(ITEM_CONTENT_API, item_content_request, "Item Content")
        item_content_df = pd.DataFrame()

        if item_content_response:
            item_content_df = process_item_content_response(item_content_response)
            logger.info(f"Item Content returned {len(item_content_df)} pairs")
        else:
            logger.warning(" Item Content API call failed, proceeding without it")
                
        # ===================================================================
        # Check if we got any data from at least one API
        # ===================================================================
        if lightgcn_df.empty and collaborative_df.empty and user_content_df.empty and item_content_df.empty:
            return jsonify({
                "status": "error",
                "error": "All recommendation APIs failed to respond"
            }), 503
        
        # ===================================================================
        # Fuse recommendations from all 4 systems
        # ===================================================================
        logger.info("\nFusing recommendations from 4 systems...")
        fused_df = fuse_recommendations(
            lightgcn_df,
            collaborative_df,
            user_content_df,
            item_content_df,
            weights
        )
        
        if fused_df.empty:
            return jsonify({
                "status": "error",
                "error": "Fusion produced no recommendations"
            }), 500
        
        # Apply sensitivity filter
        fused_df = apply_sensitivity_filter(fused_df, client_id, sensitivity_levels)
        
        if fused_df.empty:
            return jsonify({
                "status": "error",
                "error": "No recommendations passed sensitivity filter"
            }), 500
        
        import uuid
        booking_id = data.get('booking_id')
        if not booking_id:
            return jsonify({
                "status": "error", 
                "error": "booking_id is required"
            }), 400
        feedback_logger.log_shown_event(
            request_id=str(booking_id),
            client_id=client_id,
            client_segment=client_type,
            fused_df=fused_df.head(top_final),
            weights_used=weights
        )
        logger.info(f"Logged shown_event: {booking_id}")

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # ===================================================================
        # Prepare response
        # ===================================================================
        response_data = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "request_id": booking_id,
            "client_id": client_id,
            "client_type": client_type,
            "ref_port": combiner_request.ref_port,
            "processing_time_seconds": processing_time,
            "weights_used": weights,
            "counts": {
                "total_lightgcn": len(lightgcn_df),
                "total_collaborative": len(collaborative_df),
                "total_user_content": len(user_content_df),
                "total_item_content": len(item_content_df),
                "total_fused": len(fused_df),
                "returned_final": min(top_final, len(fused_df))
            },
            "recommendations": fused_df.head(top_final).to_dict('records'),
            "api_responses": {
                "lightgcn_status": "success" if not lightgcn_df.empty else "failed",
                "collaborative_status": "success" if not collaborative_df.empty else "failed",
                "user_content_status": "success" if not user_content_df.empty else "failed",
                "item_content_status": "success" if not item_content_df.empty else "failed"
            },
            "feature_extraction": {
                "collaborative_match_scores_extracted": 'cf_combined_score_normalized' in collaborative_df.columns if not collaborative_df.empty else False,
                "lightgcn_weighted_features_extracted": 'lightgcn_weighted_score' in lightgcn_df.columns if not lightgcn_df.empty else False,
                "user_content_scores_extracted": 'user_content_score' in user_content_df.columns if not user_content_df.empty else False,
                "item_content_scores_extracted": 'item_content_score' in item_content_df.columns if not item_content_df.empty else False
            }
        }
        
        logger.info(f"\nFusion completed in {processing_time:.3f}s")
        logger.info(f"Returned {response_data['counts']['returned_final']} final recommendations")
        logger.info(f"{'='*60}\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing fusion request: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/v1/populate-availability', methods=['POST'])
def populate_availability():
    """
    Populate AvailabilityAllocation with top N recommendations
    
    POST /api/v1/populate-availability
    
    Request Body (all optional):
    {
        "batch_id": "BATCH_20260111",
        "booking_id": 12345,
        "top_n": 10
    }
    
    Response:
    {
        "success": true,
        "result_data": [
            {
                "BookingsProcessed": 5,
                "RecommendationsInserted": 50,
                "TopNPerBooking": 10,
                "Result": "SUCCESS"
            }
        ]
    }
    """
    try:
        data = request.get_json() or {}
        
        batch_id = data.get('batch_id')
        booking_id = data.get('booking_id')
        top_n = data.get('top_n', 10)
        
        # Call the stored procedure via db_ops
        result = db_ops.execute_populate_availability(
            batch_id=batch_id,
            booking_id=booking_id,
            top_n=top_n
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in populate_availability: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("STARTING ENHANCED RECOMMENDATION COMBINER API")
    logger.info("="*60)
    logger.info(f"LightGCN API: {LIGHTGCN_API}")
    logger.info(f"Collaborative API: {COLLABORATIVE_API}")
    logger.info(f"User Content API: {USER_CONTENT_API}")
    logger.info(f"Item Content API: {ITEM_CONTENT_API}")
    logger.info(f"\ Sensitivity Filter Loaded:")
    logger.info(f"  • Clients: {len(sensitivity_levels['clients'])} mappings")
    logger.info(f"  • Chauffeurs: {len(sensitivity_levels['chauffeurs'])} mappings")
    logger.info(f"  • Vehicles: {len(sensitivity_levels['vehicles'])} mappings")
    logger.info(f"  • Rule: client_sensitivity <= vehicle_sensitivity AND chauffeur_sensitivity")
    logger.info(f"\n Database Tables:")
    logger.info(f"  • Input: BookingFeed (was: dummy_booking)")
    logger.info(f"  • Output: BookingRecommendations (was: boooking_recommendations)")
    logger.info(f"\n Column Mappings (BookingFeed):")
    logger.info(f"  • PickUpLocation (was: PickupSuburb) - varchar(64)")
    logger.info(f"  • DropOffLocation (was: DropoffSuburb) - varchar(64)")
    logger.info(f"  • ErrorMessage - varchar(200) - TRUNCATED")
    logger.info(f"\n Stored Procedure: DISABLED (changes being made)")
    logger.info(f"\nFeature Extraction:")
    logger.info(f"  • LightGCN: {len(LIGHTGCN_FEATURE_WEIGHTS)} weighted features")
    for feature, weight in LIGHTGCN_FEATURE_WEIGHTS.items():
        logger.info(f"    - {feature}: {weight:.3f}")
    logger.info(f"  • Collaborative: {len(CF_MATCH_WEIGHTS)} match scores")
    for score, weight in CF_MATCH_WEIGHTS.items():
        logger.info(f"    - {score}: {weight:.2f}")

    logger.info(f"\nEndpoints:")
    logger.info("  • POST /api/v1/recommend - Get raw recommendations (no fusion)")
    logger.info("  • POST /api/v1/fuse-recommendations - Fused recommendations with enriched scores + sensitivity")
    logger.info("  • POST /api/v1/batch-process - Batch process pending requests from BookingFeed")
    logger.info("  • GET /health - Health check")
    logger.info("  • GET /api/docs - Swagger documentation")
    logger.info("="*60 + "\n")
    
    # app.run(host='0.0.0.0', port=8001, debug=True)
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", "8001"))
    
    if debug_mode:
        logger.warning("!!! RUNNING IN DEBUG MODE - NOT FOR PRODUCTION !!!")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
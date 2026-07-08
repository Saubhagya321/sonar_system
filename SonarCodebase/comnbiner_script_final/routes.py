

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flasgger import Swagger
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pytz
from typing import Dict, Optional, List, Any
import logging
from logging.handlers import RotatingFileHandler
import json
import os
import sys
import io
import uuid
import signal

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Import from project modules
from config import DB_CONNECTION_STRING, setup_logging, API_TIMEOUT, TRANSELITE_DB
from connect_stored_procedure import db_ops
from data_loaders import DataLoader
from grid_search import GridSearchOptimizer
from feedback import WeightLearner, WeightSelector, FeedbackLogger, feedback_bp, init_feedback_system

# Import from refactored modules
from models import CombinerRequest, FeatureWeights, parse_comma_separated_ids
from services import recommendation_service, WeightManager


# =============================================================================
# LOGGING SETUP
# =============================================================================

# Create logs directory if it doesn't exist
LOGS_DIR = "batch_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# BATCH LOGGER CLASS
# =============================================================================

class BatchLogger:
    """
    Manages batch-specific logging for batch processing operations.

    Creates dedicated log files for each batch to track processing details.
    """

    def __init__(self, logs_dir: str = LOGS_DIR):
        self.logs_dir = logs_dir

    def setup(self, batch_id: str) -> logging.Logger:
        """
        Create a dedicated logger for a batch.

        Args:
            batch_id: Unique batch identifier

        Returns:
            Logger instance configured for this batch
        """
        batch_logger = logging.getLogger(f"batch_{batch_id}")
        batch_logger.setLevel(logging.INFO)
        batch_logger.propagate = False

        # Remove existing handlers
        batch_logger.handlers.clear()

        # Create file handler
        log_filename = os.path.join(self.logs_dir, f"{batch_id}.log")
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        batch_logger.addHandler(file_handler)

        # Add console handler
        if sys.platform == 'win32':
            console_handler = logging.StreamHandler(
                io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            )
        else:
            console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        batch_logger.addHandler(console_handler)

        batch_logger.info("=" * 70)
        batch_logger.info(f"Batch Logger Initialized: {batch_id}")
        batch_logger.info(f"Log file: {log_filename}")
        batch_logger.info("=" * 70)

        return batch_logger

    def close(self, batch_logger: logging.Logger):
        """Close all handlers for a batch logger."""
        for handler in batch_logger.handlers:
            handler.close()
            batch_logger.removeHandler(handler)


# =============================================================================
# DATABASE MANAGER CLASS
# =============================================================================

class DatabaseManager:
    """
    Manages all database operations for the combiner API.

    Handles:
    - Loading pending requests from BookingFeed
    - Updating request status
    - Saving recommendations to BookingRecommendations
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Connection pooling configuration
        self.engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True  # Verify connections before use
        )
        # Port timezone cache — loaded once at startup to avoid repeated DB hits.
        # Port timezones are static config data and never change at runtime.
        self._port_timezone_cache: Dict[int, str] = self._load_port_timezone_cache()

    def _load_port_timezone_cache(self) -> Dict[int, str]:
        """
        Loads all port timezones from the database once at startup into a
        dict keyed by port ID. This avoids a DB round-trip on every batch call.

        Returns:
            Dict mapping port_id -> IANA timezone string.
            Returns empty dict on failure (get_port_local_date will fall back to UTC).
        """
        try:
            query = text(f"""
                SELECT p.Id, t.Description AS Timezone
                FROM [{TRANSELITE_DB}].dbo.Port p
                JOIN [{TRANSELITE_DB}].dbo.Timezone t ON p.RefTimeZone = t.ID
                WHERE p.EtgFlag = 1
            """)
            with self.engine.connect() as conn:
                rows = conn.execute(query).fetchall()

            cache = {row[0]: row[1] for row in rows}
            logger.info(f"Port timezone cache loaded — {len(cache)} port(s) cached")
            return cache

        except Exception:
            logger.exception("Failed to load port timezone cache — will fall back to UTC for all ports")
            return {}

    def get_port_local_date(self, port_id: int) -> date:
        """
        Returns today's local date for the given port using the in-memory
        timezone cache (no DB call). Falls back to UTC if the port is missing
        from the cache or the timezone string is unrecognised.

        Args:
            port_id: The port's primary key.

        Returns:
            A datetime.date representing today in the port's local timezone.
        """
        tz_str = self._port_timezone_cache.get(port_id)

        if tz_str is None:
            logger.warning(
                f"Port {port_id} not found in timezone cache — falling back to UTC date"
            )
            return datetime.now(pytz.utc).date()

        try:
            tz = pytz.timezone(tz_str)
            local_date = datetime.now(tz).date()
            logger.info(f"Port {port_id} timezone: {tz_str} — local date: {local_date}")
            return local_date

        except pytz.UnknownTimeZoneError:
            logger.error(
                f"Unknown timezone '{tz_str}' for port {port_id} — falling back to UTC date"
            )
            return datetime.now(pytz.utc).date()

    def convert_local_to_utc(self, local_dt: datetime, port_id: int) -> Optional[datetime]:
        """
        Convert a naive local datetime (stored in DB) to UTC,
        using the port's IANA timezone from the in-memory cache.

        Args:
            local_dt: Naive datetime representing local port time
            port_id: Port's primary key

        Returns:
            UTC datetime (timezone-aware), or None if conversion fails
        """
        if local_dt is None:
            return None

        tz_str = self._port_timezone_cache.get(port_id)
        if tz_str is None:
            logger.warning(
                f"Port {port_id} not in timezone cache — "
                f"storing appointment datetime as UTC as-is"
            )
            return local_dt.replace(tzinfo=pytz.utc)

        try:
            tz = pytz.timezone(tz_str)
            local_aware = tz.localize(local_dt)
            utc_dt = local_aware.astimezone(pytz.utc)
            logger.debug(
                f"Port {port_id} ({tz_str}): {local_dt} local → {utc_dt} UTC"
            )
            return utc_dt
        except pytz.UnknownTimeZoneError:
            logger.error(
                f"Unknown timezone '{tz_str}' for port {port_id} — "
                f"cannot convert to UTC"
            )
            return None
        except Exception as e:
            logger.error(f"UTC conversion failed for port {port_id}: {e}")
            return None
        
    def load_pending_requests(self, limit: int = 5, port_id: int = None) -> pd.DataFrame:
        """
        Load a limited number of pending requests from BookingFeed table.
        Used by the single-page /recommend endpoint.

        Date window is based on the port's local date (from cache), not the
        server's GETDATE(), so bookings are fetched relative to the port's
        own timezone.

        Args:
            limit: Maximum number of requests to load
            port_id: Filter by RefPort value

        Returns:
            DataFrame with pending booking requests
        """
        local_date = self.get_port_local_date(port_id)
        date_from = local_date                      # today in port's timezone (exclusive lower bound)
        date_to   = local_date + timedelta(days=3)  # 3 days ahead (exclusive upper bound)

        logger.info(
            f"load_pending_requests — port {port_id} | "
            f"window: {date_from} < AppointmentDate < {date_to}"
        )

        query = text("""
            SELECT TOP (:limit) *
            FROM BookingFeed
            WHERE Status = 'PENDING'
              AND RefPort = :port_id
              AND CAST(AppointmentDate AS DATE) >= :date_from
              AND CAST(AppointmentDate AS DATE) < :date_to
            ORDER BY Id ASC
        """)

        df = pd.read_sql(
            query, self.engine,
            params={"limit": limit, "port_id": port_id,
                    "date_from": date_from, "date_to": date_to}
        )
        logger.info(f"Loaded {len(df)} pending requests (limit={limit}) for port {port_id}")
        return df

    def load_all_pending_requests(self, port_id: int = None) -> pd.DataFrame:
        """
        Load ALL pending requests from BookingFeed table (no row limit).
        Used by the batch-process endpoint which handles chunking internally.

        Date window is based on the port's local date (from cache), not the
        server's GETDATE(), so bookings are fetched relative to the port's
        own timezone.

        Args:
            port_id: Filter by RefPort value

        Returns:
            DataFrame with ALL pending booking requests for the date window
        """
        local_date = self.get_port_local_date(port_id)
        date_from = local_date                      # today in port's timezone (exclusive lower bound)
        date_to   = local_date + timedelta(days=3)  # 3 days ahead (exclusive upper bound)

        logger.info(
            f"load_all_pending_requests — port {port_id} | "
            f"window: {date_from} < AppointmentDate < {date_to}"
        )

        query = text("""
            SELECT *
            FROM BookingFeed
            WHERE Status = 'PENDING'
              AND RefPort = :port_id
              AND CAST(AppointmentDate AS DATE)  >= :date_from
              AND CAST(AppointmentDate AS DATE) < :date_to
            ORDER BY Id ASC
        """)

        df = pd.read_sql(
            query, self.engine,
            params={"port_id": port_id, "date_from": date_from, "date_to": date_to}
        )
        logger.info(f"Loaded {len(df)} total pending requests for port {port_id}")
        return df

    def update_request_status(self, booking_id: int, status: str,
                              error_message: Optional[str] = None):
        """
        Update status of a request in BookingFeed table.

        Args:
            booking_id: The booking ID to update
            status: New status value
            error_message: Optional error message (truncated to 200 chars)
        """
        with self.engine.connect() as conn:
            if error_message:
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

    def save_recommendations(self, booking_id: int, fusion_response: Dict[str, Any],
                             batch_id: str, pickup_location: str = None,
                             dropoff_location: str = None,
                             appointment_datetime: datetime = None,port_id: int = None):
        """
        Save fusion API response to BookingRecommendations table.

        Args:
            booking_id: The booking ID
            fusion_response: Response from fusion API
            batch_id: Batch identifier
            pickup_location: Pickup location string
            dropoff_location: Dropoff location string
            appointment_datetime: Appointment datetime
            port_id: Port ID used to resolve timezone for UTC conversion
        """
        if fusion_response.get('status') != 'success':
            logger.error(f"Cannot save failed response for booking {booking_id}")
            return

        recommendations = fusion_response.get('recommendations', [])
        if not recommendations:
            logger.warning(f"No recommendations to save for booking {booking_id}")
            return
        appointment_datetime_utc = (
                    self.convert_local_to_utc(appointment_datetime, port_id)
                    if appointment_datetime and port_id
                    else None
                )
        if appointment_datetime and port_id:
            logger.info(
                f"  Booking {booking_id} — "
                f"PickUpDateTime local: {appointment_datetime} → "
                f"UTC: {appointment_datetime_utc}"
            )
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
                PickUpDateTimeUTC,
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
                :appointment_datetime_utc,
                NULL
            )
        """)

        saved_count = 0
        failed_count = 0

        def safe_float(val):
            if val is None or pd.isna(val):
                return None
            try:
                f = float(val)
                if f == float('inf') or f == float('-inf'):
                    return None
                return f
            except (ValueError, TypeError):
                return None

        def safe_str(val):
            if val is None or pd.isna(val):
                return None
            try:
                if isinstance(val, float):
                    return str(int(val))
                return str(val)
            except (ValueError, TypeError):
                return None

        try:
            with self.engine.connect() as conn:
                for idx, rec in enumerate(recommendations):
                    try:
                        vehicle_id = safe_str(rec.get('vehicle_id') or rec.get('RefVehicle'))
                        chauffeur_id = safe_str(rec.get('chauffeur_id') or rec.get('RefChauffeur'))

                        if vehicle_id is None or chauffeur_id is None:
                            logger.warning(
                                f"  Rec {idx + 1} has missing IDs - vehicle: {vehicle_id}, chauffeur: {chauffeur_id}"
                            )

                        client_type = fusion_response.get('client_type')
                        if client_type and len(str(client_type)) > 50:
                            client_type = str(client_type)[:50]

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
                            'processing_time_seconds': safe_float(
                                fusion_response.get('processing_time_seconds')
                            ),
                            'weights_used': weights_json,
                            'batch_id': batch_id,
                            'pickup_location': pickup_location,
                            'dropoff_location': dropoff_location,
                            'appointment_datetime': appointment_datetime,
                            'appointment_datetime_utc': appointment_datetime_utc
                        }

                        conn.execute(insert_query, params)
                        saved_count += 1

                    except Exception as row_error:
                        logger.error(f"  Failed to save rec {idx + 1}: {str(row_error)}")
                        failed_count += 1
                        continue

                conn.commit()

            logger.info(
                f"  Saved {saved_count}/{len(recommendations)} recommendations for booking {booking_id}"
            )
            if failed_count > 0:
                logger.warning(f"  Failed to save {failed_count} recommendations")

        except Exception as e:
            logger.error(f"  Database error while saving recommendations: {str(e)}")


# =============================================================================
# API CLIENT CLASS
# =============================================================================

class APIClient:
    """
    Client for making API calls to recommendation services.

    Handles:
    - Retry logic with exponential backoff
    - Timeout management
    - Error handling and logging
    """

    def __init__(self, timeout: int = None):
        self.timeout = timeout or API_TIMEOUT
        # Hard wall-clock cap for one call. requests' read timeout only limits
        # a single socket read, not the whole request — a model API that keeps
        # the connection alive while computing can hang a batch forever.
        self.hard_deadline = self.timeout + 15

    def _post(self, url: str, payload: Dict, result: Dict) -> None:
        """Worker for the hard-deadline wrapper in call()."""
        try:
            result['response'] = requests.post(
                url,
                json=payload,
                timeout=(5, self.timeout)  # (connect_timeout=5s, read_timeout)
            )
        except Exception as e:
            result['exception'] = e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def call(self, url: str, payload: Dict, model_name: str) -> Optional[Dict]:
        """
        Safely call model API with error handling and retry logic.

        Args:
            url: API endpoint URL
            payload: Request payload
            model_name: Name of the model (for logging)

        Returns:
            API response dict if successful, None if failed
        """
        try:
            logger.info(f"\n  Calling {model_name} API...")
            logger.debug(f"  URL: {url}")

            import threading
            result: Dict = {}
            worker = threading.Thread(
                target=self._post, args=(url, payload, result), daemon=True
            )
            worker.start()
            worker.join(self.hard_deadline)

            if worker.is_alive():
                logger.error(
                    f"  {model_name} API exceeded hard deadline of "
                    f"{self.hard_deadline}s — abandoning call"
                )
                return None

            if 'exception' in result:
                raise result['exception']

            response = result['response']

            if response.status_code == 200:
                data = response.json()
                logger.info(f"  {model_name} API returned successfully")
                return data
            else:
                logger.error(f"  {model_name} API error: HTTP {response.status_code}")
                logger.error(f"  Response: {response.text[:500]}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"  {model_name} API timeout after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"  {model_name} API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"  {model_name} API unexpected error: {str(e)}")
            return None


# =============================================================================
# FLASK APP INITIALIZATION
# =============================================================================

def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)

    secret_key = os.environ.get('FLASK_SECRET_KEY')
    if not secret_key or secret_key == 'change-this-to-a-secure-random-string-in-production':
        logger.warning("!!! FLASK_SECRET_KEY not set or using default - generating random key !!!")
        logger.warning("!!! Set FLASK_SECRET_KEY in .env for production !!!")
        secret_key = os.urandom(32).hex()
    app.config['SECRET_KEY'] = secret_key

    # CORS configuration
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    CORS(app, origins=allowed_origins, supports_credentials=True)

    # Rate limiting
    Limiter(
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
    Swagger(app, config=swagger_config, template=template)

    # Register feedback blueprint
    app.register_blueprint(feedback_bp)
    logger.info("    Feedback API endpoints registered at /api/feedback")

    return app


# Create app instance
app = create_app()


# =============================================================================
# REQUEST ID TRACKING
# =============================================================================

@app.before_request
def add_request_id():
    """Add unique request ID for tracing"""
    g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))


@app.after_request
def add_request_id_header(response):
    """Add request ID to response headers"""
    response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
    return response


# =============================================================================
# API CONFIGURATION
# =============================================================================

LIGHTGCN_API = os.environ.get("LIGHTGCN_API_URL", "http://localhost:5000/recommend")
COLLABORATIVE_API = os.environ.get("COLLABORATIVE_API_URL", "http://localhost:8000/api/v1/recommendations")
USER_CONTENT_API = os.environ.get("USER_CONTENT_API_URL", "http://localhost:8003/recommendations/user")
ITEM_CONTENT_API = os.environ.get("ITEM_CONTENT_API_URL", "http://localhost:8003/recommendations/item")

# Initialize services
loader = DataLoader()
optimizer = GridSearchOptimizer()
db_manager = DatabaseManager(DB_CONNECTION_STRING)
api_client = APIClient()
batch_logger_manager = BatchLogger()

# Load client types
logger.info("Loading client types...")
client_types = loader.load_client_types()
logger.info(f"Loaded {len(client_types)} client types")

# Load sensitivity levels
logger.info("Loading sensitivity levels...")
sensitivity_levels = loader.load_sensitivity_levels()
logger.info("Loaded sensitivity mappings:")
logger.info(f"  • Clients: {len(sensitivity_levels['clients'])}")
logger.info(f"  • Chauffeurs: {len(sensitivity_levels['chauffeurs'])}")
logger.info(f"  • Vehicles: {len(sensitivity_levels['vehicles'])}")

# Initialize feedback system
logger.info("Initializing feedback system...")
feedback_components = init_feedback_system()
weight_selector = feedback_components['weight_selector']
feedback_logger = feedback_components['logger']
logger.info("    Feedback system initialized")


# =============================================================================
# SINGLE FUSION REQUEST PROCESSOR (with detailed logging)
# =============================================================================

def process_single_fusion_request(request_data: Dict, batch_log: logging.Logger,
                                   batch_id: str) -> Dict:
    """
    Process a single fusion request with detailed logging.

    Args:
        request_data: Request parameters including client_id, no_of_pax, etc.
        batch_log: Logger instance for this batch
        batch_id: Batch identifier

    Returns:
        Dict with status, recommendations, and API response details
    """
    start_time = datetime.now()

    try:
        client_id = request_data['client_id']
        top_final = request_data.pop('top_final', 20)
        booking_id = request_data.get('booking_id')

        batch_log.info(f"\n{'─' * 60}")
        batch_log.info(" FUSION PROCESSING START")
        batch_log.info(f"{'─' * 60}")
        batch_log.info(f"Client ID: {client_id}")
        batch_log.info(f"Booking ID: {booking_id}")

        # Get client type
        client_type = client_types.get(client_id, 'UNKNOWN')
        batch_log.info(f"Client Type: {client_type}")

        # Get weights
        base_weights = optimizer.load_weights(client_type)
        if not base_weights:
            base_weights = FeatureWeights.get_default_fusion_weights()
            batch_log.info(f"Using DEFAULT weights (no grid search weights for {client_type})")
        else:
            batch_log.info(f"Using grid search weights for {client_type}")

        weights_result = weight_selector.get_weights_with_source(client_id, client_type, base_weights)
        weights = weights_result.get('weights', base_weights)
        batch_log.info(f"Final weights: {json.dumps(weights, indent=2)}")

        # Set top_k for API calls
        request_data['top_k'] = 100

        # Create request object
        combiner_request = CombinerRequest(**{k: v for k, v in request_data.items() if k != 'booking_id'})

        # Track API statuses
        api_status = {
            'lightgcn': {'status': 'not_called', 'count': 0},
            'collaborative': {'status': 'not_called', 'count': 0},
            'user_content': {'status': 'not_called', 'count': 0},
            'item_content': {'status': 'not_called', 'count': 0}
        }

        # Initialize DataFrames
        lightgcn_df = pd.DataFrame()
        collaborative_df = pd.DataFrame()
        user_content_df = pd.DataFrame()
        item_content_df = pd.DataFrame()

        # ═══════════════════════════════════════════════════════════════
        # CALL LIGHTGCN API
        # ═══════════════════════════════════════════════════════════════
        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(" CALLING LIGHTGCN API")
        batch_log.info(f"{'═' * 60}")

        lightgcn_request = combiner_request.to_lightgcn_request()
        batch_log.info(f"Request: {json.dumps(lightgcn_request, indent=2)}")

        lightgcn_response = api_client.call(LIGHTGCN_API, lightgcn_request, "LightGCN")

        if lightgcn_response:
            lightgcn_df = recommendation_service.process_lightgcn(lightgcn_response)
            weights = recommendation_service.adjust_weights_for_cold_start(lightgcn_response, weights)
            api_status['lightgcn'] = {'status': 'success', 'count': len(lightgcn_df)}
            batch_log.info(f" LightGCN returned {len(lightgcn_df)} pairs")
            batch_log.info(f"  Cold start: {lightgcn_response.get('is_cold_start', False)}")
            batch_log.info(f"  Method: {lightgcn_response.get('recommendation_method', 'unknown')}")

            if len(lightgcn_df) > 0:
                batch_log.info(f"  Top 3 pairs: {lightgcn_df['pair_id'].head(3).tolist()}")
        else:
            api_status['lightgcn'] = {'status': 'failed', 'count': 0}
            batch_log.warning(" LightGCN API call failed")

        # ═══════════════════════════════════════════════════════════════
        # CALL COLLABORATIVE API
        # ═══════════════════════════════════════════════════════════════
        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(" CALLING COLLABORATIVE API")
        batch_log.info(f"{'═' * 60}")

        collab_request = combiner_request.to_collaborative_request()
        batch_log.info(f"Request: {json.dumps(collab_request, indent=2)}")

        collab_response = api_client.call(COLLABORATIVE_API, collab_request, "Collaborative")

        if collab_response:
            collaborative_df = recommendation_service.process_collaborative(collab_response)
            api_status['collaborative'] = {'status': 'success', 'count': len(collaborative_df)}
            batch_log.info(f" Collaborative returned {len(collaborative_df)} pairs")

            if len(collaborative_df) > 0:
                batch_log.info(f"  Top 3 pairs: {collaborative_df['pair_id'].head(3).tolist()}")
        else:
            api_status['collaborative'] = {'status': 'failed', 'count': 0}
            batch_log.warning(" Collaborative API call failed")

        # ═══════════════════════════════════════════════════════════════
        # CALL USER CONTENT API
        # ═══════════════════════════════════════════════════════════════
        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(" CALLING USER CONTENT API")
        batch_log.info(f"{'═' * 60}")

        user_content_request = combiner_request.to_user_content_request()
        batch_log.info(f"Request: {json.dumps(user_content_request, indent=2)}")

        user_content_response = api_client.call(USER_CONTENT_API, user_content_request, "User Content")

        if user_content_response:
            user_content_df = recommendation_service.process_user_content(user_content_response)
            api_status['user_content'] = {'status': 'success', 'count': len(user_content_df)}
            batch_log.info(f" User Content returned {len(user_content_df)} pairs")

            if len(user_content_df) > 0:
                batch_log.info(f"  Top 3 pairs: {user_content_df['pair_id'].head(3).tolist()}")
        else:
            api_status['user_content'] = {'status': 'failed', 'count': 0}
            batch_log.warning(" User Content API call failed")

        # ═══════════════════════════════════════════════════════════════
        # CALL ITEM CONTENT API
        # ═══════════════════════════════════════════════════════════════
        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(" CALLING ITEM CONTENT API")
        batch_log.info(f"{'═' * 60}")

        item_content_request = combiner_request.to_item_content_request()
        batch_log.info(f"Request: {json.dumps(item_content_request, indent=2)}")

        item_content_response = api_client.call(ITEM_CONTENT_API, item_content_request, "Item Content")

        if item_content_response:
            item_content_df = recommendation_service.process_item_content(item_content_response)
            api_status['item_content'] = {'status': 'success', 'count': len(item_content_df)}
            batch_log.info(f" Item Content returned {len(item_content_df)} pairs")

            if len(item_content_df) > 0:
                batch_log.info(f"  Top 3 pairs: {item_content_df['pair_id'].head(3).tolist()}")
        else:
            api_status['item_content'] = {'status': 'failed', 'count': 0}
            batch_log.warning(" Item Content API call failed")

        # ═══════════════════════════════════════════════════════════════
        # CHECK IF ANY DATA AVAILABLE
        # ═══════════════════════════════════════════════════════════════
        all_empty = all(
            df.empty for df in [lightgcn_df, collaborative_df, user_content_df, item_content_df]
        )

        if all_empty:
            batch_log.error(" ALL APIs FAILED - No data to fuse")
            return {
                'status': 'error',
                'error': 'All recommendation APIs failed',
                'api_responses': {
                    'lightgcn': {
                        'status': api_status['lightgcn']['status'],
                        'pairs_count': 0,
                        'error': 'API call failed'
                    },
                    'collaborative': {
                        'status': api_status['collaborative']['status'],
                        'pairs_count': 0,
                        'error': 'API call failed'
                    },
                    'user_content': {
                        'status': api_status['user_content']['status'],
                        'pairs_count': 0,
                        'error': 'API call failed'
                    },
                    'item_content': {
                        'status': api_status['item_content']['status'],
                        'pairs_count': 0,
                        'error': 'API call failed'
                    }
                }
            }

        # ═══════════════════════════════════════════════════════════════
        # REDISTRIBUTE WEIGHTS FOR AVAILABLE APIs
        # ═══════════════════════════════════════════════════════════════
        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(" REDISTRIBUTING WEIGHTS")
        batch_log.info(f"{'═' * 60}")

        weights = recommendation_service.redistribute_weights(
            weights,
            lightgcn_available=not lightgcn_df.empty,
            collaborative_available=not collaborative_df.empty,
            user_content_available=not user_content_df.empty,
            item_content_available=not item_content_df.empty
        )
        batch_log.info(f"Redistributed weights: {json.dumps(weights, indent=2)}")

        # ═══════════════════════════════════════════════════════════════
        # FUSE RECOMMENDATIONS
        # ═══════════════════════════════════════════════════════════════
        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(" FUSING RECOMMENDATIONS")
        batch_log.info(f"{'═' * 60}")

        fused_df = recommendation_service.fuse_recommendations(
            lightgcn_df, collaborative_df, user_content_df, item_content_df, weights
        )

        batch_log.info("Fusion input counts:")
        batch_log.info(f"  • LightGCN: {len(lightgcn_df)}")
        batch_log.info(f"  • Collaborative: {len(collaborative_df)}")
        batch_log.info(f"  • User Content: {len(user_content_df)}")
        batch_log.info(f"  • Item Content: {len(item_content_df)}")
        batch_log.info(f"  • Fused total: {len(fused_df)}")

        if fused_df.empty:
            batch_log.error(" Fusion produced no results")
            return {
                'status': 'error',
                'error': 'Fusion produced no recommendations',
                'api_responses': {
                    'lightgcn': {
                        'status': api_status['lightgcn']['status'],
                        'pairs_count': api_status['lightgcn']['count'],
                        'error': None
                    },
                    'collaborative': {
                        'status': api_status['collaborative']['status'],
                        'pairs_count': api_status['collaborative']['count'],
                        'error': None
                    },
                    'user_content': {
                        'status': api_status['user_content']['status'],
                        'pairs_count': api_status['user_content']['count'],
                        'error': None
                    },
                    'item_content': {
                        'status': api_status['item_content']['status'],
                        'pairs_count': api_status['item_content']['count'],
                        'error': None
                    }
                }
            }

        # ═══════════════════════════════════════════════════════════════
        # APPLY SENSITIVITY FILTER
        # ═══════════════════════════════════════════════════════════════
        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(" APPLYING SENSITIVITY FILTER")
        batch_log.info(f"{'═' * 60}")

        pre_filter_count = len(fused_df)
        fused_df = recommendation_service.apply_sensitivity_filter(fused_df, client_id, sensitivity_levels)
        post_filter_count = len(fused_df)

        batch_log.info(f"Before filter: {pre_filter_count}")
        batch_log.info(f"After filter: {post_filter_count}")
        batch_log.info(f"Filtered out: {pre_filter_count - post_filter_count}")

        if fused_df.empty:
            batch_log.error(" No recommendations passed sensitivity filter")
            return {
                'status': 'error',
                'error': 'No recommendations passed sensitivity filter',
                'api_responses': {
                    'lightgcn': {
                        'status': api_status['lightgcn']['status'],
                        'pairs_count': api_status['lightgcn']['count'],
                        'error': None
                    },
                    'collaborative': {
                        'status': api_status['collaborative']['status'],
                        'pairs_count': api_status['collaborative']['count'],
                        'error': None
                    },
                    'user_content': {
                        'status': api_status['user_content']['status'],
                        'pairs_count': api_status['user_content']['count'],
                        'error': None
                    },
                    'item_content': {
                        'status': api_status['item_content']['status'],
                        'pairs_count': api_status['item_content']['count'],
                        'error': None
                    }
                }
            }

        # ═══════════════════════════════════════════════════════════════
        # LOG TOP RECOMMENDATIONS
        # ═══════════════════════════════════════════════════════════════
        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(f" TOP {min(10, len(fused_df))} RECOMMENDATIONS")
        batch_log.info(f"{'═' * 60}")

        for idx, rec in fused_df.head(10).iterrows():
            batch_log.info(
                f"  #{rec.get('final_rank', idx + 1):2d} | "
                f"Pair: {rec.get('pair_id', 'N/A'):20s} | "
                f"Score: {rec.get('final_score', 0):.4f} | "
                f"RRF: {rec.get('rrf_score', 0):.4f} | "
                f"GCN: {rec.get('gcn_score', 0):.4f} | "
                f"CF: {rec.get('cf_score', 0):.4f}"
            )

        # ═══════════════════════════════════════════════════════════════
        # BUILD RESPONSE
        # ═══════════════════════════════════════════════════════════════
        processing_time = (datetime.now() - start_time).total_seconds()

        batch_log.info(f"\n{'═' * 60}")
        batch_log.info(" FUSION COMPLETE")
        batch_log.info(f"{'═' * 60}")
        batch_log.info(f"Processing time: {processing_time:.2f}s")
        batch_log.info(f"Recommendations returned: {min(top_final, len(fused_df))}")

        return {
            'status': 'success',
            'client_id': client_id,
            'client_type': client_type,
            'weights_used': weights,
            'processing_time_seconds': processing_time,
            'recommendations': fused_df.head(top_final).to_dict('records'),
            'api_responses': {
                'lightgcn': {
                    'status': api_status['lightgcn']['status'],
                    'pairs_count': api_status['lightgcn']['count'],
                    'error': None if api_status['lightgcn']['status'] == 'success' else 'API call failed'
                },
                'collaborative': {
                    'status': api_status['collaborative']['status'],
                    'pairs_count': api_status['collaborative']['count'],
                    'error': None if api_status['collaborative']['status'] == 'success' else 'API call failed'
                },
                'user_content': {
                    'status': api_status['user_content']['status'],
                    'pairs_count': api_status['user_content']['count'],
                    'error': None if api_status['user_content']['status'] == 'success' else 'API call failed'
                },
                'item_content': {
                    'status': api_status['item_content']['status'],
                    'pairs_count': api_status['item_content']['count'],
                    'error': None if api_status['item_content']['status'] == 'success' else 'API call failed'
                }
            },
            'counts': {
                'lightgcn': len(lightgcn_df),
                'collaborative': len(collaborative_df),
                'user_content': len(user_content_df),
                'item_content': len(item_content_df),
                'fused': len(fused_df),
                'returned': min(top_final, len(fused_df))
            }
        }

    except Exception as e:
        batch_log.error(f" FUSION ERROR: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'api_responses': api_status if 'api_status' in locals() else {}
        }


# =============================================================================
# ROUTE HANDLERS
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint with database connectivity verification
    ---
    tags:
      - System
    responses:
      200:
        description: Service is healthy
      503:
        description: Service is degraded (database issues)
    """
    db_healthy = True
    db_error = None

    try:
        with db_manager.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_healthy = False
        db_error = str(e)
        logger.error(f"Health check DB failed: {e}")

    status = "healthy" if db_healthy else "degraded"
    http_code = 200 if db_healthy else 503

    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "request_id": g.get('request_id', 'unknown'),
        "database": {
            "status": "connected" if db_healthy else "disconnected",
            "error": db_error
        },
        "services": {
            "lightgcn": LIGHTGCN_API,
            "collaborative": COLLABORATIVE_API,
            "user_content": USER_CONTENT_API,
            "item_content": ITEM_CONTENT_API
        },
        "client_types_loaded": len(client_types)
    }), http_code


@app.route('/api/v1/recommend', methods=['POST'])
def recommend():
    """
    Get recommendations from all 4 APIs (raw, no fusion)
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
            no_of_pax:
              type: integer
    responses:
      200:
        description: Successful recommendation
      400:
        description: Invalid request
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

        combiner_request = CombinerRequest(**data)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"PROCESSING REQUEST FOR CLIENT {combiner_request.client_id}")
        logger.info(f"{'=' * 60}")

        # Call all 4 APIs
        lightgcn_response = api_client.call(
            LIGHTGCN_API, combiner_request.to_lightgcn_request(), "LightGCN"
        )
        collaborative_response = api_client.call(
            COLLABORATIVE_API, combiner_request.to_collaborative_request(), "Collaborative"
        )
        user_content_response = api_client.call(
            USER_CONTENT_API, combiner_request.to_user_content_request(), "User Content"
        )
        item_content_response = api_client.call(
            ITEM_CONTENT_API, combiner_request.to_item_content_request(), "Item Content"
        )

        if not any([lightgcn_response, collaborative_response,
                    user_content_response, item_content_response]):
            return jsonify({
                "status": "error",
                "error": "All recommendation APIs failed to respond"
            }), 503

        processing_time = (datetime.now() - start_time).total_seconds()

        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "request_id": f"CMB-{combiner_request.client_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "client_id": combiner_request.client_id,
            "processing_time_seconds": processing_time,
            "lightgcn_response": lightgcn_response or {"error": "Failed"},
            "collaborative_response": collaborative_response or {"error": "Failed"},
            "user_content_response": user_content_response or {"error": "Failed"},
            "item_content_response": item_content_response or {"error": "Failed"}
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


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
            - booking_id
          properties:
            client_id:
              type: integer
            no_of_pax:
              type: integer
            booking_id:
              type: integer
              description: Required for feedback logging
            ref_port:
              type: integer
            appointment_date:
              type: string
            pickup_suburb:
              type: string
            dropoff_suburb:
              type: string
            ref_vehicle_type:
              type: string
            min_vehicle_level:
              type: integer
            top_final:
              type: integer
              default: 20
    responses:
      200:
        description: Fused recommendations
      400:
        description: Invalid request
      500:
        description: Server error
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

        # Validate booking_id — required for feedback logging
        booking_id = data.get('booking_id')
        if not booking_id:
            return jsonify({
                "status": "error",
                "error": "booking_id is required"
            }), 400

        client_id = int(data['client_id'])
        top_final = int(data.get('top_final', 20))

        # Create combiner request
        combiner_request = CombinerRequest(**data)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"FUSION REQUEST FOR CLIENT {client_id}")
        logger.info(f"{'=' * 60}")

        # Get client type
        client_type = client_types.get(client_id, 'UNKNOWN')
        logger.info(f"Client type: {client_type}")

        # Get weights for client
        weights = weight_selector.get_weights_for_client(client_id, client_type)
        logger.info(f"Using weights: {weights}")

        # Call all 4 APIs
        lightgcn_response = api_client.call(
            LIGHTGCN_API, combiner_request.to_lightgcn_request(), "LightGCN"
        )
        collaborative_response = api_client.call(
            COLLABORATIVE_API, combiner_request.to_collaborative_request(), "Collaborative"
        )
        user_content_response = api_client.call(
            USER_CONTENT_API, combiner_request.to_user_content_request(), "User Content"
        )
        item_content_response = api_client.call(
            ITEM_CONTENT_API, combiner_request.to_item_content_request(), "Item Content"
        )

        # Process responses
        lightgcn_df = recommendation_service.process_lightgcn(lightgcn_response or {})
        collaborative_df = recommendation_service.process_collaborative(collaborative_response or {})
        user_content_df = recommendation_service.process_user_content(user_content_response or {})
        item_content_df = recommendation_service.process_item_content(item_content_response or {})

        # Adjust weights for cold start if needed
        weights = recommendation_service.adjust_weights_for_cold_start(lightgcn_response, weights)

        # Redistribute weights for available APIs
        weights = recommendation_service.redistribute_weights(
            weights,
            lightgcn_available=not lightgcn_df.empty,
            collaborative_available=not collaborative_df.empty,
            user_content_available=not user_content_df.empty,
            item_content_available=not item_content_df.empty
        )

        # Fuse recommendations
        fused_df = recommendation_service.fuse_recommendations(
            lightgcn_df, collaborative_df, user_content_df, item_content_df, weights
        )

        # Apply sensitivity filter
        fused_df = recommendation_service.apply_sensitivity_filter(
            fused_df, client_id, sensitivity_levels
        )

        # Log shown event for feedback learning system
        feedback_logger.log_shown_event(
            request_id=str(booking_id),
            client_id=client_id,
            client_segment=client_type,
            fused_df=fused_df.head(top_final),
            weights_used=weights
        )
        logger.info(f"Logged shown_event for booking: {booking_id}")

        processing_time = (datetime.now() - start_time).total_seconds()

        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "request_id": booking_id,
            "client_id": client_id,
            "client_type": client_type,
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
            }
        })

    except Exception as e:
        logger.error(f"Error processing fusion request: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/v1/batch-process', methods=['POST'])
def batch_process_recommendations():
    """
    Process ALL pending requests from BookingFeed table in configurable chunks.

    Loads every PENDING request for the next 3 days (no row cap), then
    iterates through them in chunks of `batch_size` (default 50, max 500)
    until every request has been processed.  AvailabilityAllocation is
    populated once after all chunks complete.
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
              default: 50
              description: Number of bookings to process per chunk (max 500)
            port_id:
              type: integer
              description: Required - filter by RefPort
            max_bookings:
              type: integer
              description: max pending bookings     
    responses:
      200:
        description: Batch processing completed
      400:
        description: Missing or invalid parameters
      500:
        description: Unexpected server error
    """
    start_time = datetime.now()
    batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    batch_log = batch_logger_manager.setup(batch_id)

    try:
        # ------------------------------------------------------------------
        # Parse and validate request parameters
        # ------------------------------------------------------------------
        data = request.get_json() or {}

        raw_batch_size = data.get('batch_size', 50)
        batch_size = min(
            max(int(raw_batch_size) if str(raw_batch_size).isdigit() else 50, 1),
            500
        )

        port_id = data.get('port_id')
        if port_id is None:
            return jsonify({"status": "error", "error": "port_id is required"}), 400
        try:
            port_id = int(port_id)
        except (ValueError, TypeError):
            return jsonify({"status": "error", "error": "port_id must be an integer"}), 400

        batch_log.info("STARTING BATCH PROCESSING")
        batch_log.info(f"Batch ID   : {batch_id}")
        batch_log.info(f"Port ID    : {port_id}")
        batch_log.info(f"Chunk size : {batch_size}")

        # ------------------------------------------------------------------
        # Load ALL pending requests (no limit) for the date window
        # ------------------------------------------------------------------
        all_pending = db_manager.load_all_pending_requests(port_id=port_id)
        max_bookings = data.get('max_bookings')
        if max_bookings is not None:
            try:
                max_bookings = int(max_bookings)
                if max_bookings > 0 and len(all_pending) > max_bookings:
                    batch_log.info(
                        f"Limiting from {len(all_pending)} to {max_bookings} bookings (max_bookings parameter)"
                    )
                    all_pending = all_pending.head(max_bookings)
            except (ValueError, TypeError):
                batch_log.warning(f"Invalid max_bookings value: {max_bookings}, ignoring")

        if all_pending.empty:
            batch_log.info("No pending requests found — nothing to do.")
            batch_logger_manager.close(batch_log)
            return jsonify({
                "status": "success",
                "message": "No pending requests to process",
                "batch_id": batch_id,
                "processed_count": 0
            })

        total_count = len(all_pending)
        total_chunks = (total_count + batch_size - 1) // batch_size

        batch_log.info(f"Total pending requests : {total_count}")
        batch_log.info(f"Total chunks           : {total_chunks}")

        # ------------------------------------------------------------------
        # Aggregated results across all chunks
        # ------------------------------------------------------------------
        results = {
            'total': total_count,
            'successful': 0,
            'failed': 0,
            'chunks_processed': 0,
            'details': []
        }

        # ------------------------------------------------------------------
        # Iterate through chunks
        # ------------------------------------------------------------------
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * batch_size
            chunk_end = chunk_start + batch_size
            chunk_df = all_pending.iloc[chunk_start:chunk_end]

            batch_log.info(f"\n{'#' * 70}")
            batch_log.info(
                f"CHUNK {chunk_idx + 1}/{total_chunks}  "
                f"(rows {chunk_start + 1}–{min(chunk_end, total_count)} of {total_count})"
            )
            batch_log.info(f"{'#' * 70}")

            # --------------------------------------------------------------
            # Process every booking in this chunk
            # --------------------------------------------------------------
            for idx, row in chunk_df.iterrows():
                booking_id = row['BookingID']
                client_id = row['ClientId']

                batch_log.info(f"\n{'=' * 50}")
                batch_log.info(f"Processing booking {booking_id} (client {client_id})")

                try:
                    db_manager.update_request_status(booking_id, 'PROCESSING')

                    # Build request data
                    request_data = {
                        'client_id': int(client_id),
                        'no_of_pax': int(row.get('NoOfPax', 1)),
                        'ref_port': int(row['RefPort']) if pd.notna(row.get('RefPort')) else None,
                        'appointment_date': (
                            str(row['AppointmentDate'])
                            if pd.notna(row.get('AppointmentDate')) else None
                        ),
                        'pickup_suburb': (
                            str(row['PickUpLocation'])
                            if pd.notna(row.get('PickUpLocation')) else None
                        ),
                        'dropoff_suburb': (
                            str(row['DropOffLocation'])
                            if pd.notna(row.get('DropOffLocation')) else None
                        ),
                        'ref_vehicle_type': (
                            str(row['RefVehicleType'])
                            if pd.notna(row.get('RefVehicleType')) else None
                        ),
                        'min_vehicle_level': (
                            int(row['MinVehicleLevel'])
                            if pd.notna(row.get('MinVehicleLevel')) else None
                        ),
                        'booking_id': booking_id,
                        'top_final': int(row.get('TopFinal', 20) or 20)
                    }

                    batch_log.info(
                        f"Request data: {json.dumps(request_data, indent=2, default=str)}"
                    )

                    # Run fusion
                    fusion_result = process_single_fusion_request(
                        request_data=request_data,
                        batch_log=batch_log,
                        batch_id=batch_id
                    )

                    if fusion_result.get('status') == 'success':
                        # Save recommendations to database
                        pickup_location = (
                            str(row['PickUpLocation'])
                            if pd.notna(row.get('PickUpLocation')) else None
                        )
                        dropoff_location = (
                            str(row['DropOffLocation'])
                            if pd.notna(row.get('DropOffLocation')) else None
                        )
                        appointment_datetime = (
                            row['AppointmentDate']
                            if pd.notna(row.get('AppointmentDate')) else None
                        )

                        db_manager.save_recommendations(
                            booking_id=booking_id,
                            fusion_response=fusion_result,
                            batch_id=batch_id,
                            pickup_location=pickup_location,
                            dropoff_location=dropoff_location,
                            appointment_datetime=appointment_datetime,
                            port_id=port_id
                        )

                        results['successful'] += 1
                        db_manager.update_request_status(booking_id, 'COMPLETED')

                        results['details'].append({
                            'booking_id': booking_id,
                            'client_id': client_id,
                            'status': 'success',
                            'recommendations_count': len(fusion_result.get('recommendations', [])),
                            'api_responses': fusion_result.get('api_responses', {}),
                            'weights_used': fusion_result.get('weights_used', {})
                        })

                        batch_log.info(
                            f" Booking {booking_id} completed — "
                            f"{len(fusion_result.get('recommendations', []))} recommendations saved"
                        )
                    else:
                        raise Exception(fusion_result.get('error', 'Fusion failed'))

                except Exception as e:
                    batch_log.error(f"Error processing booking {booking_id}: {str(e)}")
                    results['failed'] += 1
                    db_manager.update_request_status(booking_id, 'FAILED', str(e))

                    results['details'].append({
                        'booking_id': booking_id,
                        'status': 'failed',
                        'error': str(e)
                    })

            # Chunk summary
            results['chunks_processed'] += 1
            batch_log.info(
                f"\nChunk {chunk_idx + 1}/{total_chunks} complete — "
                f"running totals: {results['successful']} ok / {results['failed']} failed"
            )

        # ------------------------------------------------------------------
        # All chunks done — populate AvailabilityAllocation once
        # ------------------------------------------------------------------
        availability_result = {"success": False, "message": "Not executed"}

        if results['successful'] > 0:
            batch_log.info(f"\n{'=' * 70}")
            batch_log.info(" POPULATING AVAILABILITY ALLOCATION")
            batch_log.info(f"Batch ID             : {batch_id}")
            batch_log.info(f"Successful bookings  : {results['successful']}")
            batch_log.info(f"{'=' * 70}")

            try:
                availability_result = db_ops.execute_populate_availability(
                    batch_id=batch_id,
                    top_n=10  # TOP 10 recommendations per booking
                )

                if availability_result['success']:
                    result_data = availability_result.get('result_data', [])
                    if result_data:
                        bookings_processed = result_data[0].get('BookingsProcessed', 0)
                        recs_inserted = result_data[0].get('RecommendationsInserted', 0)
                        batch_log.info(" AvailabilityAllocation populated successfully")
                        batch_log.info(f"  Bookings processed       : {bookings_processed}")
                        batch_log.info(f"  Recommendations inserted : {recs_inserted}")
                        batch_log.info("  Top N per booking        : 10")
                    else:
                        batch_log.info(" AvailabilityAllocation populated (no result data returned)")
                else:
                    batch_log.error(" AvailabilityAllocation population failed")
                    batch_log.error(f"  Error: {availability_result.get('error', 'Unknown error')}")

            except Exception as e:
                batch_log.error(f" Error populating AvailabilityAllocation: {e}", exc_info=True)
                availability_result = {"success": False, "error": str(e)}

        else:
            batch_log.info("\n⚠ No successful recommendations — skipping AvailabilityAllocation")
            availability_result = {
                "success": False,
                "message": "No successful recommendations to populate"
            }

        results['availability_allocation'] = availability_result

        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        total_time = (datetime.now() - start_time).total_seconds()

        batch_log.info(f"\n{'=' * 70}")
        batch_log.info("BATCH COMPLETE")
        batch_log.info(f"  Total requests    : {results['total']}")
        batch_log.info(f"  Successful        : {results['successful']}")
        batch_log.info(f"  Failed            : {results['failed']}")
        batch_log.info(f"  Chunks processed  : {results['chunks_processed']}")
        batch_log.info(
            f"  Availability      : {' Populated' if availability_result.get('success') else ' Failed/Skipped'}"
        )
        batch_log.info(f"  Total time        : {total_time:.2f}s")
        batch_log.info(f"{'=' * 70}")

        batch_logger_manager.close(batch_log)

        return jsonify({
            "status": "success",
            "batch_id": batch_id,
            "batch_processing_time_seconds": total_time,
            "log_file": os.path.join(LOGS_DIR, f"{batch_id}.log"),
            "results": results,
            "availability_allocation_result": results.get('availability_allocation', {})
        })

    except Exception as e:
        batch_log.error(f"Batch processing error: {str(e)}", exc_info=True)
        batch_logger_manager.close(batch_log)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/api/v1/populate-availability', methods=['POST'])
def populate_availability():
    """
    Populate AvailabilityAllocation with top N recommendations
    ---
    tags:
      - Availability
    parameters:
      - name: body
        in: body
        schema:
          type: object
          properties:
            batch_id:
              type: string
            booking_id:
              type: integer
            top_n:
              type: integer
              default: 10
    responses:
      200:
        description: Successfully populated
    """
    try:
        data = request.get_json() or {}

        batch_id = data.get('batch_id')
        booking_id = data.get('booking_id')
        top_n = data.get('top_n', 10)

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
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# GRACEFUL SHUTDOWN HANDLER
# =============================================================================

def graceful_shutdown(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal, cleaning up...")
    try:
        if db_manager and hasattr(db_manager, 'engine'):
            db_manager.engine.dispose()
            logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    sys.exit(0)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # Only installed for the standalone `python routes.py` dev path — under
    # gunicorn this module is imported (not run as __main__), and gunicorn
    # installs its own SIGTERM/SIGINT handlers to manage worker lifecycle
    # (timeouts, recycling). Registering these unconditionally at import time
    # previously hijacked gunicorn's worker-management signals, causing
    # workers to die/restart whenever a slow request (e.g. a blocked DB call
    # in /health) triggered gunicorn's own timeout-driven SIGTERM.
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    logger.info("\n" + "=" * 60)
    logger.info("STARTING ENHANCED RECOMMENDATION COMBINER API")
    logger.info("=" * 60)
    logger.info(f"LightGCN API     : {LIGHTGCN_API}")
    logger.info(f"Collaborative API: {COLLABORATIVE_API}")
    logger.info(f"User Content API : {USER_CONTENT_API}")
    logger.info(f"Item Content API : {ITEM_CONTENT_API}")
    logger.info("\nSensitivity Filter Loaded:")
    logger.info(f"  • Clients   : {len(sensitivity_levels['clients'])} mappings")
    logger.info(f"  • Chauffeurs: {len(sensitivity_levels['chauffeurs'])} mappings")
    logger.info(f"  • Vehicles  : {len(sensitivity_levels['vehicles'])} mappings")
    logger.info("\nFeature Extraction:")
    logger.info(f"  • LightGCN    : {len(FeatureWeights.LIGHTGCN)} weighted features")
    logger.info(f"  • Collaborative: {len(FeatureWeights.CF_MATCH)} match scores")
    logger.info("\nEndpoints:")
    logger.info("  • POST /api/v1/recommend             - Get raw recommendations")
    logger.info("  • POST /api/v1/fuse-recommendations  - Fused recommendations")
    logger.info("  • POST /api/v1/batch-process         - Batch processing (all pending, chunked)")
    logger.info("  • POST /api/v1/populate-availability - Populate availability")
    logger.info("  • GET  /health                       - Health check")
    logger.info("  • GET  /api/docs                     - Swagger documentation")
    logger.info("=" * 60 + "\n")

    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", "8001"))

    if debug_mode:
        logger.warning("!!! RUNNING IN DEBUG MODE - NOT FOR PRODUCTION !!!")
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        # In production, use: gunicorn --workers 4 --bind 0.0.0.0:8001 routes:app
        logger.info("Starting in production mode. Consider using gunicorn instead.")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

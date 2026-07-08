"""
FLASK API SERVER FOR RECOMMENDATION SYSTEM
Production-ready REST API with complete documentation
"""
import uuid
from flasgger import Swagger
from flask import Flask, request, jsonify, send_from_directory, g, Blueprint  # CHANGE A: Added Blueprint
from flask_cors import CORS
import json
import traceback
from datetime import datetime, timezone
import logging
from functools import wraps
import os

import signal
import sys
import threading

from complete_recommendation_api import (
    EnhancedBookingRequest,
    CompleteRecommendationEngine,
    create_complete_engine
)


# ============================================================================
# SETUP
# ============================================================================
# CHANGE B: Updated logging imports — removed setup_logging from config import
from config import (
    RECOMMENDATIONS_OUTPUT_DIR,
    MODEL_PATHS,
    PIPELINE_PATHS,
    ENVIRONMENT
)

# Structured JSON logging for production
try:
    from logging_config import setup_logging as setup_json_logging
    setup_json_logging(environment=ENVIRONMENT)
    logger = logging.getLogger(__name__)
    logger.info("Structured JSON logging enabled")
except ImportError:
    from config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)

os.makedirs(RECOMMENDATIONS_OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app, origins=["https://localhost:8001", "https://localhost:8003"])

# ============================================================================
# CHANGE C: RESPONSE COMPRESSION
# ============================================================================
try:
    from flask_compress import Compress
    compress = Compress()
    compress.init_app(app)
    app.config['COMPRESS_ALGORITHM'] = 'gzip'
    app.config['COMPRESS_LEVEL'] = 6
    app.config['COMPRESS_MIN_SIZE'] = 500  # Only compress responses > 500 bytes
    COMPRESSION_ENABLED = True
    logger.info("Response compression enabled")
except ImportError:
    COMPRESSION_ENABLED = False
    logger.warning("flask-compress not installed. Compression disabled. Install with: pip install flask-compress")

# ============================================================================
# CHANGE C: API VERSIONING - Blueprint
# ============================================================================
api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')

# ============================================================================
# RATE LIMITING (Production Feature)
# ============================================================================
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["36000 per day", "1500 per hour"],
        storage_uri="memory://"
       

    )
    RATE_LIMITING_ENABLED = True
    logger.info("Rate limiting enabled")
except ImportError:
    limiter = None
    RATE_LIMITING_ENABLED = False
    logger.warning("flask-limiter not installed. Rate limiting disabled. Install with: pip install flask-limiter")


# ============================================================================
# RESILIENCE IMPORTS
# ============================================================================
try:
    from resilience import (
        circuit_registry,
        db_circuit,
        CircuitOpenError
    )
    RESILIENCE_ENABLED = True
    logger.info("Resilience module loaded")
except ImportError:
    RESILIENCE_ENABLED = False
    logger.warning("resilience.py not found. Resilience features disabled.")
    def db_circuit(func):
        return func
    class CircuitOpenError(Exception):
        pass


# ============================================================================
# THREAD-SAFE ENGINE MANAGER
# ============================================================================
class EngineManager:
    """Thread-safe singleton manager for the recommendation engine."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._engine = None
        self._initialized = False
        self._initialization_lock = threading.Lock()
      
    
    @classmethod
    def get_instance(cls) -> 'EngineManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @property
    def engine(self):
        if not self._initialized:
            raise RuntimeError("Engine not initialized")
        return self._engine
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def set_engine(self, engine_instance):
        with self._initialization_lock:
            self._engine = engine_instance
            self._initialized = True
    
    def cleanup(self):
        with self._initialization_lock:
            if self._engine is not None:
                logger.info("Cleaning up engine resources...")
                self._engine = None
                self._initialized = False


engine_manager = EngineManager.get_instance()


# ============================================================================
# GRACEFUL SHUTDOWN HANDLER
# ============================================================================
class GracefulShutdown:
    """Handles graceful shutdown on SIGTERM/SIGINT signals."""
    
    def __init__(self):
        self._cleanup_handlers = []
        self._is_shutting_down = False
        self._lock = threading.Lock()
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def register_cleanup(self, handler, name: str = None):
        self._cleanup_handlers.append((handler, name or handler.__name__))
    
    def _signal_handler(self, signum, frame):
        with self._lock:
            if self._is_shutting_down:
                sys.exit(1)
            self._is_shutting_down = True
        
        logger.info("Received signal %s. Starting graceful shutdown...", signum)
        for handler, name in reversed(self._cleanup_handlers):
            try:
                handler()
            except Exception as e:
                logger.error("Cleanup '%s' failed: %s", name, e)
        logger.info("Graceful shutdown complete")
        sys.exit(0)
    
    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down


shutdown_handler = GracefulShutdown()
shutdown_handler.register_cleanup(engine_manager.cleanup, "engine_cleanup")


# ============================================================================
# REQUEST TRACKING MIDDLEWARE (with shutdown check)
# ============================================================================
@app.before_request
def before_request():
    """Add request ID and start time for tracking"""
    if shutdown_handler.is_shutting_down:
        return jsonify({
            "status": "error",
            "error": "Service shutting down"
        }), 503
    g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    g.start_time = datetime.now(timezone.utc)


@app.after_request
def after_request(response):
    """Log request completion and add tracking headers"""
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
        
        if hasattr(g, 'start_time'):
            duration_ms = (datetime.now(timezone.utc) - g.start_time).total_seconds() * 1000
            response.headers['X-Response-Time-Ms'] = str(int(duration_ms))
            
            logger.info(
                "Request completed | ID: %s | Method: %s | Path: %s | Status: %s | Duration: %.2fms",
                g.request_id, request.method, request.path, response.status_code, duration_ms
            )
    
    return response


# ============================================================================
# SWAGGER CONFIGURATION
# ============================================================================
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
    "specs_route": "/apidocs/"
}

swagger = Swagger(app, config=swagger_config, template={
    "info": {
        "title": "Vehicle-Chauffeur Recommendation API",
        "description": "LightGCN-based recommendation system",
        "version": "1.0.0"
    },
    "host": "localhost:5000",
    "basePath": "/",
    "schemes": ["http"]
})


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def convert_to_native(obj):
    """
    Convert numpy/pandas types to native Python types for JSON serialization
    
    Args:
        obj: Any object that may contain numpy/pandas types
    
    Returns:
        Same object with all numpy/pandas types converted to native Python types
    """
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return convert_to_native(obj.tolist())
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    elif isinstance(obj, pd.Series):
        return convert_to_native(obj.tolist())
    return obj


# ============================================================================
# ERROR HANDLERS (with CircuitOpenError handling)
# ============================================================================
def handle_errors(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        request_id = getattr(g, 'request_id', 'unknown')
        try:
            return f(*args, **kwargs)
        except CircuitOpenError as e:
            logger.warning("Circuit open | Request: %s | Error: %s", request_id, e)
            return jsonify({
                "status": "error",
                "error": "Service temporarily unavailable",
                "message": "System recovering, please retry",
                "request_id": request_id,
                "retry_after": 30
            }), 503
        except ValueError as e:
            logger.error("Validation error | Request: %s | Error: %s", request_id, e)
            return jsonify({
                "status": "error",
                "error": "Validation error",
                "message": str(e),
                "request_id": request_id
            }), 400
        except ConnectionError as e:
            logger.error("Connection error | Request: %s | Error: %s", request_id, e)
            return jsonify({
                "status": "error",
                "error": "Service unavailable",
                "message": "Database or external service unavailable",
                "request_id": request_id
            }), 503
        except Exception as e:
            logger.exception("Unexpected error | Request: %s", request_id)
            return jsonify({
                "status": "error",
                "error": "Internal server error",
                "message": str(e) if ENVIRONMENT == 'development' else "An unexpected error occurred",
                "request_id": request_id
            }), 500
    return decorated_function


# ============================================================================
# HEALTH CHECK FUNCTIONS (with circuit breaker on db check)
# ============================================================================
@db_circuit
def check_database_connection():
    """Check if database is reachable (with circuit breaker)"""
    try:
        from config import DB_CONNECTION_STRING
        from sqlalchemy import create_engine, text

        engine_db = create_engine(DB_CONNECTION_STRING)
        with engine_db.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine_db.dispose()
        return True
    except Exception as e:
        logger.warning("Database health check failed: %s", e)
        raise  # Re-raise for circuit breaker


def check_disk_space():
    """Check if sufficient disk space available"""
    try:
        import shutil
        _, _, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        return free_gb > 1  # At least 1GB free
    except Exception:
        return True  # Assume OK if can't check


def check_model_loaded():
    """Check if recommendation engine is loaded"""
    return engine_manager.is_initialized


# ============================================================================
# INITIALIZATION (use engine_manager.set_engine)
# ============================================================================
def initialize_engine(pipeline_results=None, model_path=None):
    """Initialize the recommendation engine"""
    logger.info("Initializing recommendation engine...")
    
    if pipeline_results is not None:
        eng = create_complete_engine(pipeline_results)
    else:
        logger.info("Loading from saved files...")
        import pickle
        import torch
        from model_training import HeterogeneousLightGCN, load_graph_data, sparse_to_torch, device
        
        try:
            checkpoint = torch.load(model_path, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        try:
            with open(PIPELINE_PATHS['step2_results'], 'rb') as f:
                step2_results = pickle.load(f)
            with open(PIPELINE_PATHS['step3_results'], 'rb') as f:
                step3_results = pickle.load(f)
            with open(PIPELINE_PATHS['step4_results'], 'rb') as f:
                step4_results = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Pipeline data not found: {e}")

        data_info = checkpoint['data_info']
        config = checkpoint['config']
        
        model = HeterogeneousLightGCN(
            n_clients=data_info['n_clients'],
            n_vehicles=data_info['n_vehicles'],
            n_chauffeurs=data_info['n_chauffeurs'],
            embedding_dim=config['embedding_dim'],
            n_layers=config['n_layers'],
            dropout=config.get('dropout', 0.1),
            use_batch_norm=config.get('use_batch_norm', False)
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        data = load_graph_data(step4_results)
        adj_sparse = sparse_to_torch(data['normalized_adj'], device)
        
        eng = CompleteRecommendationEngine(
            model=model,
            step2_results=step2_results,
            step3_results=step3_results,
            step4_results=step4_results,
            data=data,
            adj_sparse=adj_sparse
        )
    
    engine_manager.set_engine(eng)
    logger.info("Recommendation engine ready!")
    return eng


def save_recommendations_to_file(response_data, client_id, request_id, 
                                 save_vehicles=False, save_chauffeurs=False, 
                                 save_pairs=False):
    """
    Save recommendations to separate JSON files
    
    Args:
        response_data: The recommendation response dictionary
        client_id: Client identifier
        request_id: Unique request ID
        save_vehicles: Whether to save vehicle recommendations
        save_chauffeurs: Whether to save chauffeur recommendations
        save_pairs: Whether to save pair recommendations
    
    Returns:
        dict: Paths to saved files
    """
    saved_files = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    client_dir = os.path.join(RECOMMENDATIONS_OUTPUT_DIR, f"client_{client_id}")
    os.makedirs(client_dir, exist_ok=True)
    
    if save_vehicles and 'recommendations' in response_data and 'vehicles' in response_data['recommendations']:
        vehicle_filename = f"vehicles_{request_id}_{timestamp}.json"
        vehicle_path = os.path.join(client_dir, vehicle_filename)
        
        vehicle_data = {
            'client_id': client_id,
            'request_id': request_id,
            'timestamp': timestamp,
            'recommendation_method': response_data.get('recommendation_method'),
            'is_cold_start': response_data.get('is_cold_start'),
            'total_vehicles': len(response_data['recommendations']['vehicles']),
            'vehicles': response_data['recommendations']['vehicles'],
            'metadata': response_data.get('metadata', {})
        }
        
        with open(vehicle_path, 'w', encoding='utf-8') as f:
            json.dump(vehicle_data, f, indent=2, ensure_ascii=False)
        
        saved_files['vehicles'] = vehicle_path
        logger.info("Saved %d vehicle recommendations to: %s", len(vehicle_data['vehicles']), vehicle_path)
    
    if save_chauffeurs and 'recommendations' in response_data and 'chauffeurs' in response_data['recommendations']:
        chauffeur_filename = f"chauffeurs_{request_id}_{timestamp}.json"
        chauffeur_path = os.path.join(client_dir, chauffeur_filename)
        
        chauffeur_data = {
            'client_id': client_id,
            'request_id': request_id,
            'timestamp': timestamp,
            'recommendation_method': response_data.get('recommendation_method'),
            'is_cold_start': response_data.get('is_cold_start'),
            'total_chauffeurs': len(response_data['recommendations']['chauffeurs']),
            'chauffeurs': response_data['recommendations']['chauffeurs'],
            'metadata': response_data.get('metadata', {})
        }
        
        with open(chauffeur_path, 'w', encoding='utf-8') as f:
            json.dump(chauffeur_data, f, indent=2, ensure_ascii=False)
        
        saved_files['chauffeurs'] = chauffeur_path
        logger.info("Saved %d chauffeur recommendations to: %s", len(chauffeur_data['chauffeurs']), chauffeur_path)
    
    if save_pairs and 'recommendations' in response_data and 'pairs' in response_data['recommendations']:
        pair_filename = f"pairs_{request_id}_{timestamp}.json"
        pair_path = os.path.join(client_dir, pair_filename)
        
        pair_data = {
            'client_id': client_id,
            'request_id': request_id,
            'timestamp': timestamp,
            'recommendation_method': response_data.get('recommendation_method'),
            'is_cold_start': response_data.get('is_cold_start'),
            'total_pairs': len(response_data['recommendations']['pairs']),
            'pairs': response_data['recommendations']['pairs'],
            'metadata': response_data.get('metadata', {})
        }
        
        with open(pair_path, 'w', encoding='utf-8') as f:
            json.dump(pair_data, f, indent=2, ensure_ascii=False)
        
        saved_files['pairs'] = pair_path
        logger.info("Saved %d pair recommendations to: %s", len(pair_data['pairs']), pair_path)
    
    return saved_files


# ============================================================================
# API ENDPOINTS
# ============================================================================

# CHANGE D: Updated home() with compression status and versioned API endpoints
@app.route('/', methods = ['GET', 'POST'])
def home():
    """API home page with documentation"""
    return jsonify({
        "name": "Vehicle-Chauffeur Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "environment": ENVIRONMENT,
        "features": {                           # CHANGE D: grouped features together
            "rate_limiting": RATE_LIMITING_ENABLED,
            "resilience": RESILIENCE_ENABLED,
            "compression": COMPRESSION_ENABLED
        },
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "GET /health/detailed": "Detailed health check",
            "GET /health/circuits": "Circuit breaker status",
            "GET /stats": "System statistics",
            "POST /recommend": "Get recommendations",
            "GET /api-docs": "Complete API documentation"
        },
        "versioned_api": {                      # CHANGE D: added versioned endpoint list
            "POST /api/v1/recommend": "Versioned recommendations",
            "GET /api/v1/health": "Versioned health check",
            "GET /api/v1/stats": "Versioned statistics"
        },
        "documentation": "/apidocs/"
    })


@app.route('/health',  methods = ['GET', 'POST'])
def health():
    """
    Basic health check endpoint
    ---
    tags:
      - System
    responses:
      200:
        description: Service is healthy
      503:
        description: Service is unhealthy
    """
    model_loaded = check_model_loaded()
    
    if not model_loaded:
        return jsonify({
            "status": "unhealthy",
            "message": "Recommendation engine not initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 503
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine_ready": True
    })


@app.route('/admin/reload', methods=['POST'])
def admin_reload():
    """
    Hot-reload the LightGCN recommendation engine from disk.

    Called by the monthly retraining orchestrator after a new model has been
    trained and its files written to lightgcn_scripts/models/.  During the
    reload the engine is briefly unavailable; /health will return 503.

    Requires the X-Reload-Token header to match the ADMIN_RELOAD_TOKEN
    environment variable (if that variable is set).
    """
    expected_token = os.environ.get("ADMIN_RELOAD_TOKEN", "")
    provided_token = request.headers.get("X-Reload-Token", "")
    if expected_token and provided_token != expected_token:
        return jsonify({"error": "Unauthorized"}), 401

    logger.info("Admin reload requested — reinitialising LightGCN engine from disk…")
    try:
        engine_manager.cleanup()
        initialize_engine(model_path=MODEL_PATHS['best_model'])
        return jsonify({
            "status": "ok",
            "message": "LightGCN engine reloaded successfully",
            "reloaded_at": datetime.now(timezone.utc).isoformat(),
        }), 200
    except Exception as exc:
        logger.error("Admin reload failed: %s", exc, exc_info=True)
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route('/health/detailed',  methods = ['GET', 'POST'])
@handle_errors
def health_detailed():
    """
    Detailed health check with dependency status
    ---
    tags:
      - System
    responses:
      200:
        description: All checks passed
      503:
        description: One or more checks failed
    """
    try:
        db_healthy = check_database_connection()
    except (CircuitOpenError, Exception):
        db_healthy = False

    checks = {
        'api': True,
        'model_loaded': check_model_loaded(),
        'database': db_healthy,
        'disk_space': check_disk_space()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks,
        'environment': ENVIRONMENT,
        'resilience': RESILIENCE_ENABLED,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), status_code


@app.route('/health/circuits',  methods = ['GET', 'POST'])
@handle_errors
def health_circuits():
    """
    Circuit breaker status endpoint
    ---
    tags:
      - System
    responses:
      200:
        description: Circuit breaker statuses
    """
    if not RESILIENCE_ENABLED:
        return jsonify({"status": "disabled", "message": "Resilience module not loaded"})
    
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "circuits": circuit_registry.get_all_statuses()
    })


@app.route('/stats',  methods = ['GET', 'POST'])
@handle_errors
def stats():
    """
    Get system statistics
    ---
    tags:
      - System
    responses:
      200:
        description: System statistics
      503:
        description: Engine not initialized
    """
    if not engine_manager.is_initialized:
        return jsonify({"error": "Engine not initialized"}), 503
    
    return jsonify({
        "status": "success",
        "statistics": {
            "total_clients": len(engine_manager.engine.client_id_to_node),
            "total_vehicles": len(engine_manager.engine.vehicle_id_to_idx),
            "total_chauffeurs": len(engine_manager.engine.chauffeur_id_to_idx),
            "clients_in_model": len(engine_manager.engine.clients_in_model),
            "available_features": len(engine_manager.engine.available_features)
        }
    })


@app.route('/recommend', methods=['POST'])
@handle_errors
def recommend():
    """
    Get vehicle and chauffeur recommendations
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
              type: string
              description: Client identifier
              example: "C12345"
            no_of_pax:
              type: integer
              description: Number of passengers
              example: 2
            ref_port:
              type: integer
              description: Port ID - filters vehicles and chauffeurs to specific location
              example: 1 
            ref_vehicle_type:
              type: string
              description: Preferred vehicle type ID
              example: "3.0"
            appointment_date:
              type: string
              format: date
              description: Trip date (YYYY-MM-DD)
              example: "2025-12-25"
            pickup_suburb:
              type: string
              example: "CBD"
            dropoff_suburb:
              type: string
              example: "Airport"
            estimated_distance:
              type: number
              description: Distance in km
              example: 25.5
            estimated_duration:
              type: integer
              description: Duration in minutes
              example: 45
            min_vehicle_level:
              type: integer
              minimum: 1
              maximum: 5
              example: 3
            require_red_zone_auth:
              type: boolean
              example: true
            require_baby_seat:
              type: boolean
              example: true
            max_price:
              type: number
              example: 500.0
            excluded_vehicle_ids:
              type: array
              items:
                type: integer
            excluded_chauffeur_ids:
              type: array
              items:
                type: integer
            preferred_chauffeur_ids:
              type: array
              items:
                type: integer
            top_k_vehicles:
              type: integer
              default: 20
            top_k_chauffeurs:
              type: integer
              default: 20
            top_k_pairs:
              type: integer
              default: 10
            save_veh:
              type: boolean
              default: false
              description: Save vehicle recommendations to file
            save_chauf:
              type: boolean
              default: false
              description: Save chauffeur recommendations to file
            save_pair:
              type: boolean
              default: false
              description: Save pair recommendations to file
    responses:
      200:
        description: Successful recommendation
      400:
        description: Invalid request
      503:
        description: Engine not initialized
    """
    if not engine_manager.is_initialized:
        return jsonify({"error": "Engine not initialized"}), 503
    
    data = request.get_json()
    
    if not data:
        return jsonify({
            "status": "error",
            "error": "No JSON data provided"
        }), 400
    
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
    
    logger.info("Recommendation request | Client: %s | Request ID: %s", data['client_id'], g.request_id)
    
    save_vehicles = data.pop('save_veh', False)
    save_chauffeurs = data.pop('save_chauf', False)
    save_pairs = data.pop('save_pair', False)

    booking_request = EnhancedBookingRequest(**data)
    
    is_valid, error_msg = booking_request.validate()
    if not is_valid:
        return jsonify({
            "status": "error",
            "error": "Validation failed",
            "message": error_msg
        }), 400
    
    response = engine_manager.engine.recommend(booking_request)
    response_json = json.loads(response.to_json())
    
    logger.info("Recommendation completed | Client: %s | Method: %s", 
                response.client_id, response.recommendation_method)
    
    saved_files = {}
    if save_vehicles or save_chauffeurs or save_pairs:
        saved_files = save_recommendations_to_file(
            response_json,
            client_id=booking_request.client_id,
            request_id=response.request_id,
            save_vehicles=save_vehicles,
            save_chauffeurs=save_chauffeurs,
            save_pairs=save_pairs
        )
      
    api_response = {
        "status": "success",
        "request_id": g.request_id,
        **response_json
    }
    
    if saved_files:
        api_response["saved_files"] = saved_files
        api_response["message"] = f"Recommendations saved to {len(saved_files)} file(s)"
    
    return jsonify(api_response)


@app.route('/recommend/simple', methods=['POST'])
@handle_errors
def recommend_simple():
    """
    Simplified recommendation (required fields only)
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
              type: string
              example: "C12345"
            no_of_pax:
              type: integer
              example: 2
    responses:
      200:
        description: Simplified recommendations
    """
    if not engine_manager.is_initialized:
        return jsonify({"error": "Engine not initialized"}), 503
    
    data = request.get_json()
    
    if not data or 'client_id' not in data or 'no_of_pax' not in data:
        return jsonify({
            "status": "error",
            "error": "client_id and no_of_pax are required"
        }), 400
    
    booking_request = EnhancedBookingRequest(
        client_id=data['client_id'],
        no_of_pax=data['no_of_pax']
    )
    
    response = engine_manager.engine.recommend(booking_request)
    
    return jsonify({
        "status": "success",
        "request_id": response.request_id,
        "client_id": response.client_id,
        "method": response.recommendation_method,
        "is_cold_start": response.is_cold_start,
        "top_vehicles": response.vehicles[:5],
        "top_chauffeurs": response.chauffeurs[:5],
        "top_pairs": response.pairs[:3]
    })


@app.route('/recommend/batch', methods=['POST'])
@handle_errors
def recommend_batch():
    """
    Batch recommendation endpoint
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
            - requests
          properties:
            requests:
              type: array
              items:
                type: object
                properties:
                  client_id:
                    type: string
                  no_of_pax:
                    type: integer
    responses:
      200:
        description: Batch recommendation results
    """
    if not engine_manager.is_initialized:
        return jsonify({"error": "Engine not initialized"}), 503
    
    data = request.get_json()
    
    if not data or 'requests' not in data:
        return jsonify({
            "status": "error",
            "error": "requests array is required"
        }), 400
    
    requests_list = data['requests']
    
    if not isinstance(requests_list, list):
        return jsonify({
            "status": "error",
            "error": "requests must be an array"
        }), 400
    
    logger.info("Batch request | Count: %d | Request ID: %s", len(requests_list), g.request_id)
    
    results = []
    successful = 0
    failed = 0
    
    for i, req_data in enumerate(requests_list):
        try:
            booking_request = EnhancedBookingRequest(**req_data)
            
            is_valid, error_msg = booking_request.validate()
            if not is_valid:
                results.append({
                    "index": i,
                    "status": "error",
                    "client_id": req_data.get('client_id'),
                    "error": error_msg
                })
                failed += 1
                continue
            
            response = engine_manager.engine.recommend(booking_request)
            
            results.append({
                "index": i,
                "status": "success",
                "request_id": response.request_id,
                "client_id": response.client_id,
                "method": response.recommendation_method,
                "vehicles_count": len(response.vehicles),
                "chauffeurs_count": len(response.chauffeurs)
            })
            successful += 1
            
        except Exception as e:
            logger.warning("Batch item %d failed: %s", i, e)
            results.append({
                "index": i,
                "status": "error",
                "client_id": req_data.get('client_id'),
                "error": str(e)
            })
            failed += 1
    
    logger.info("Batch completed | Successful: %d | Failed: %d", successful, failed)
    
    return jsonify({
        "status": "success",
        "total_requests": len(requests_list),
        "successful": successful,
        "failed": failed,
        "results": results
    })


@app.route('/clients/available', methods=['GET'])
@handle_errors
def get_available_clients():
    """
    Get list of available client IDs
    ---
    tags:
      - Data
    parameters:
      - name: limit
        in: query
        type: integer
        default: 100
        description: Maximum number of clients to return
    responses:
      200:
        description: List of client IDs
    """
    if not engine_manager.is_initialized:
        return jsonify({"error":"Engine not initialized"}), 503
    
    limit = request.args.get('limit', 100, type=int)
    
    clients = list(engine_manager.engine.client_id_to_node.keys())[:limit]
    
    return jsonify({
        "status": "success",
        "total_clients": len(engine_manager.engine.client_id_to_node),
        "sample_size": len(clients),
        "clients": clients
    })


@app.route('/vehicles/types', methods=['GET'])
@handle_errors
def get_vehicle_types():
    """
    Get available vehicle types
    ---
    tags:
      - Data
    responses:
      200:
        description: List of vehicle types
    """
    if not engine_manager.is_initialized:
        return jsonify({"error":"Engine not initialized"}), 503
    
    vehicle_types = engine_manager.engine.vehicle_features['RefVehicleType'].dropna().unique().tolist()
    
    return jsonify({
        "status": "success",
        "total_types": len(vehicle_types),
        "vehicle_types": vehicle_types
    })


@app.route('/api-docs',  methods = ['GET', 'POST'])
def api_docs():
    """Complete API documentation"""
    docs = {
        "title": "Vehicle-Chauffeur Recommendation API",
        "version": "1.0.0",
        "description": "REST API for getting vehicle and chauffeur recommendations using LightGCN",
        "base_url": request.url_root.rstrip('/'),
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "API home page with quick links"
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Basic health check"
            },
            {
                "path": "/health/detailed",
                "method": "GET",
                "description": "Detailed health check with dependency status"
            },
            {
                "path": "/health/circuits",
                "method": "GET",
                "description": "Circuit breaker status"
            },
            {
                "path": "/stats",
                "method": "GET",
                "description": "System statistics"
            },
            {
                "path": "/recommend",
                "method": "POST",
                "description": "Full recommendation with all parameters"
            },
            {
                "path": "/recommend/simple",
                "method": "POST",
                "description": "Simplified recommendation"
            },
            {
                "path": "/recommend/batch",
                "method": "POST",
                "description": "Batch recommendations"
            }
        ],
        "error_codes": {
            "400": "Bad Request - Invalid parameters",
            "500": "Internal Server Error",
            "503": "Service Unavailable - Engine not initialized"
        }
    }
    
    return jsonify(docs)


@app.route('/info/client/<client_id>', methods=['GET'])
@handle_errors
def get_client_info(client_id):
    """
    Get detailed information about a client
    ---
    tags:
      - Information
    parameters:
      - name: client_id
        in: path
        required: true
        type: string
    responses:
      200:
        description: Client information
      404:
        description: Client not found
    """
    if not engine_manager.is_initialized:
        return jsonify({"error": "Engine not initialized"}), 503
    
    try:
        client_id_normalized = float(str(client_id).strip())
    except (ValueError, TypeError):
        client_id_normalized = str(client_id).strip()
    
    client_info = engine_manager.engine._get_client_info(client_id_normalized)
    
    if not client_info:
        return jsonify({
            "status": "error",
            "message": f"Client {client_id} not found in system"
        }), 404
    
    client_info = convert_to_native(client_info)
    client_info['in_model'] = bool(client_id_normalized in engine_manager.engine.client_id_to_node)
    client_info['recommendation_method'] = 'lightgcn' if client_info['in_model'] else 'similarity_based'
    
    return jsonify({
        "status": "success",
        "client": client_info
    })


@app.route('/info/vehicle-type/<vehicle_type_id>', methods=['GET'])
@handle_errors
def get_vehicle_type_info(vehicle_type_id):
    """
    Get information about a vehicle type
    ---
    tags:
      - Information
    parameters:
      - name: vehicle_type_id
        in: path
        required: true
        type: string
    responses:
      200:
        description: Vehicle type information
      404:
        description: Vehicle type not found
    """
    if not engine_manager.is_initialized:
        return jsonify({"error": "Engine not initialized"}), 503
    
    try:
        vehicle_type_id_normalized = float(str(vehicle_type_id).strip())
    except (ValueError, TypeError):
        vehicle_type_id_normalized = str(vehicle_type_id).strip()
    
    vehicle_type_metadata = {
        'description': 'N/A',
        'display_name': 'N/A',
        'code': 'N/A'
    }
    
    try:
        import pandas as pd
        from config import METADATA_FILES
        
        vehicle_type_df = pd.read_csv(METADATA_FILES['vehicle_type'])
        vt_match = vehicle_type_df[vehicle_type_df['ID'] == vehicle_type_id_normalized]
        
        if len(vt_match) > 0:
            vt_row = vt_match.iloc[0]
            
            def safe_get(val, default='N/A'):
                if pd.isna(val):
                    return default
                return str(val)
            
            vehicle_type_metadata = {
                'description': safe_get(vt_row.get('Description')),
                'display_name': safe_get(vt_row.get('DisplayName')),
                'code': safe_get(vt_row.get('Code')),
                'seat_capacity': int(vt_row['SeatCapacity']) if 'SeatCapacity' in vt_row and not pd.isna(vt_row['SeatCapacity']) else None,
                'vehicle_level': int(vt_row['VehicleLevel']) if 'VehicleLevel' in vt_row and not pd.isna(vt_row['VehicleLevel']) else None,
                'toll_vehicle_type': safe_get(vt_row.get('TollVehicleType')),
                'is_levy_applied': bool(vt_row.get('IsLevyApplied', False))
            }
    except FileNotFoundError:
        logger.warning("Vehicle type metadata file not found")
    except Exception as e:
        logger.warning("Could not load vehicle type metadata: %s", e)
    
    vehicles = engine_manager.engine.vehicle_features[
        engine_manager.engine.vehicle_features['RefVehicleType'] == vehicle_type_id_normalized
    ]
    
    if len(vehicles) == 0:
        return jsonify({
            "status": "error",
            "message": f"Vehicle type {vehicle_type_id} not found or has no vehicles"
        }), 404
    
    summary = {
        "vehicle_type_id": str(vehicle_type_id),
        "total_vehicles": int(len(vehicles)),
        "in_model_count": int(sum(vehicles['ID'].isin(engine_manager.engine.vehicle_id_to_idx.keys()))),
        **vehicle_type_metadata
    }
    
    if 'SeatCapacity' in vehicles.columns:
        seat_caps = vehicles['SeatCapacity'].dropna()
        if len(seat_caps) > 0:
            summary['actual_seat_capacity_range'] = {
                'min': int(seat_caps.min()),
                'max': int(seat_caps.max()),
                'avg': float(seat_caps.mean())
            }
    
    if 'VehicleLevel' in vehicles.columns:
        levels = vehicles['VehicleLevel'].dropna()
        if len(levels) > 0:
            summary['actual_vehicle_level_range'] = {
                'min': int(levels.min()),
                'max': int(levels.max()),
                'avg': float(levels.mean())
            }
    
    if 'Make' in vehicles.columns:
        makes_dict = vehicles['Make'].value_counts().head(10).to_dict()
        summary['makes'] = {str(k): int(v) for k, v in makes_dict.items()}
    
    if 'completion_rate' in vehicles.columns:
        comp_rates = vehicles['completion_rate'].dropna()
        if len(comp_rates) > 0:
            summary['avg_completion_rate'] = float(comp_rates.mean())
    
    if 'repeat_client_rate' in vehicles.columns:
        repeat_rates = vehicles['repeat_client_rate'].dropna()
        if len(repeat_rates) > 0:
            summary['avg_repeat_client_rate'] = float(repeat_rates.mean())
    
    example_cols = ['ID', 'Make', 'Model', 'Year', 'SeatCapacity', 'VehicleLevel']
    available_cols = [col for col in example_cols if col in vehicles.columns]
    
    if 'total_bookings' in vehicles.columns:
        examples = vehicles.nlargest(5, 'total_bookings')[available_cols]
    else:
        examples = vehicles[available_cols].head(5)
    
    example_list = examples.to_dict('records')
    summary['example_vehicles'] = convert_to_native(example_list)
    
    return jsonify({
        "status": "success",
        "vehicle_type": summary
    })


@app.route('/info/vehicle/<vehicle_id>', methods=['GET'])
@handle_errors
def get_vehicle_info(vehicle_id):
    """
    Get detailed information about a vehicle
    ---
    tags:
      - Information
    parameters:
      - name: vehicle_id
        in: path
        required: true
        type: string
    responses:
      200:
        description: Vehicle information
      404:
        description: Vehicle not found
    """
    if not engine_manager.is_initialized:
        return jsonify({"error": "Engine not initialized"}), 503
    
    try:
        vehicle_id_normalized = float(str(vehicle_id).strip())
    except (ValueError, TypeError):
        vehicle_id_normalized = str(vehicle_id).strip()
    
    if vehicle_id_normalized not in engine_manager.engine.vehicle_features_indexed.index:
        return jsonify({
            "status": "error",
            "message": f"Vehicle {vehicle_id} not found in system"
        }), 404
    
    vehicle_row = engine_manager.engine.vehicle_features_indexed.loc[vehicle_id_normalized]
    
    vehicle_info = {
        'vehicle_id': str(vehicle_id),
        'in_model': bool(vehicle_id_normalized in engine_manager.engine.vehicle_id_to_idx)
    }
    
    info_fields = {
        'Make': ('make', str),
        'Model': ('model', str),
        'Year': ('year', int),
        'Colour': ('color', str),
        'Code': ('code', str),
        'SeatCapacity': ('seat_capacity', int),
        'VehicleLevel': ('vehicle_level', int),
        'RefVehicleType': ('vehicle_type_id', float)
    }
    
    for src, (dst, dtype) in info_fields.items():
        if src in vehicle_row.index:
            val = engine_manager.engine._safe_extract_value(vehicle_row[src], None, dtype)
            if val is not None:
                if dtype == int:
                    vehicle_info[dst] = int(val)
                elif dtype == float:
                    vehicle_info[dst] = float(val)
                else:
                    vehicle_info[dst] = str(val)
    
    perf_fields = {
        'completion_rate': ('completion_rate', float),
        'repeat_client_rate': ('repeat_client_rate', float),
        'total_bookings': ('total_bookings', int),
        'avg_price': ('avg_price', float),
        'total_revenue': ('total_revenue', float),
        'vehicle_quality_score': ('quality_score', float)
    }
    
    for src, (dst, dtype) in perf_fields.items():
        if src in vehicle_row.index:
            val = engine_manager.engine._safe_extract_value(vehicle_row[src], None, dtype)
            if val is not None:
                if dtype == int:
                    vehicle_info[dst] = int(val)
                elif dtype == float:
                    vehicle_info[dst] = float(val)
    
    return jsonify({
        "status": "success",
        "vehicle": vehicle_info
    })


@app.route('/info/chauffeur/<chauffeur_id>', methods=['GET'])
@handle_errors
def get_chauffeur_info(chauffeur_id):
    """
    Get detailed information about a chauffeur
    ---
    tags:
      - Information
    parameters:
      - name: chauffeur_id
        in: path
        required: true
        type: string
    responses:
      200:
        description: Chauffeur information
      404:
        description: Chauffeur not found
    """
    if not engine_manager.is_initialized:
        return jsonify({"error": "Engine not initialized"}), 503
    
    try:
        chauffeur_id_normalized = float(str(chauffeur_id).strip())
    except (ValueError, TypeError):
        chauffeur_id_normalized = str(chauffeur_id).strip()
    
    if chauffeur_id_normalized not in engine_manager.engine.chauffeur_features_indexed.index:
        return jsonify({
            "status": "error",
            "message": f"Chauffeur {chauffeur_id} not found in system"
        }), 404
    
    chauffeur_row = engine_manager.engine.chauffeur_features_indexed.loc[chauffeur_id_normalized]
    
    chauffeur_info = {
        'chauffeur_id': str(chauffeur_id),
        'in_model': bool(chauffeur_id_normalized in engine_manager.engine.chauffeur_id_to_idx)
    }
    
    if 'Code' in chauffeur_row.index:
        code_val = engine_manager.engine._safe_extract_value(chauffeur_row['Code'], 'Unknown', str)
        chauffeur_info['code'] = str(code_val)
    
    qual_fields = {
        'experience_score': ('experience_score', int),
        'safety_score': ('safety_score', float),
        'training_score': ('training_score', float),
        'RedZoneAuthorised': ('red_zone_authorized', bool),
        'BabySeatTraining': ('baby_seat_training', bool),
        'DriverTraining': ('driver_training', bool)
    }
    
    for src, (dst, dtype) in qual_fields.items():
        if src in chauffeur_row.index:
            if dtype == bool:
                val = engine_manager.engine._safe_bool_check(chauffeur_row[src], False)
                chauffeur_info[dst] = bool(val)
            else:
                val = engine_manager.engine._safe_extract_value(chauffeur_row[src], None, dtype)
                if val is not None:
                    if dtype == int:
                        chauffeur_info[dst] = int(val)
                    elif dtype == float:
                        chauffeur_info[dst] = float(val)
    
    perf_fields = {
        'completion_rate': ('completion_rate', float),
        'repeat_client_rate': ('repeat_client_rate', float),
        'total_bookings': ('total_bookings', int),
        'chauffeur_quality_score': ('quality_score', float)
    }
    
    for src, (dst, dtype) in perf_fields.items():
        if src in chauffeur_row.index:
            val = engine_manager.engine._safe_extract_value(chauffeur_row[src], None, dtype)
            if val is not None:
                if dtype == int:
                    chauffeur_info[dst] = int(val)
                elif dtype == float:
                    chauffeur_info[dst] = float(val)
    
    return jsonify({
        "status": "success",
        "chauffeur": chauffeur_info
    })


# ── Auto-initialize when imported by Gunicorn ──────────────────
if not engine_manager.is_initialized:
    try:
        initialize_engine(model_path=str(MODEL_PATHS['best_model']))
        logger.info("Engine initialized via Gunicorn startup")
    except Exception as e:
        logger.exception("Failed to initialize engine on startup: %s", e)


# ============================================================================
# CHANGE E: API v1 VERSIONED ENDPOINTS (registered via Blueprint)
# ============================================================================
@api_v1.route('/recommend', methods=['POST'])
@handle_errors
def v1_recommend():
    """Versioned recommend endpoint - /api/v1/recommend"""
    return recommend()


@api_v1.route('/health',  methods = ['GET', 'POST'])
def v1_health():
    """Versioned health endpoint - /api/v1/health"""
    return health()


@api_v1.route('/stats',  methods = ['GET', 'POST'])
@handle_errors
def v1_stats():
    """Versioned stats endpoint - /api/v1/stats"""
    return stats()


# Register the versioned API blueprint
app.register_blueprint(api_v1)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Recommendation API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model-path', type=str, default=str(MODEL_PATHS['best_model']),
                       help='Path to model file')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("VEHICLE-CHAUFFEUR RECOMMENDATION API SERVER")
    logger.info("=" * 60)
    logger.info("Environment: %s", ENVIRONMENT)
    logger.info("Rate Limiting: %s", "Enabled" if RATE_LIMITING_ENABLED else "Disabled")
    logger.info("Resilience: %s", "Enabled" if RESILIENCE_ENABLED else "Disabled")
    logger.info("Compression: %s", "Enabled" if COMPRESSION_ENABLED else "Disabled")  # CHANGE F
    
    try:
        logger.info("Initializing recommendation engine...")
        initialize_engine(model_path=args.model_path)
        logger.info("Engine initialized successfully!")
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        logger.error("Please ensure: 1) Pipeline has been run, 2) Model and pipeline files exist")
        exit(1)
    except Exception as e:
        logger.exception("Failed to initialize engine")
        exit(1)
    
    logger.info("Server starting | Host: %s | Port: %s | Debug: %s", args.host, args.port, args.debug)
    logger.info("API Documentation: http://%s:%s/apidocs/", args.host, args.port)
    logger.info("Health Check: http://%s:%s/health", args.host, args.port)
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
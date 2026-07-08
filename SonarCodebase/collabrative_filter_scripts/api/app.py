import json
from fastapi import FastAPI, HTTPException, Depends, Query, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import asyncio

from api.models import (
    RecommendationRequest, 
    RecommendationResponse,
    HealthResponse
)
from api.service import RecommendationService
from api.filters import RecommendationFilter
from config.config import Config
from utils.helpers import load_pickle, load_dataframe
from data.data_loader import DataLoader, DataValidationError

import uuid
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential


# Add after imports
RECOMMENDATIONS_DIR = Path("output/recommendations")
RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
config = Config()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_log_input(value, max_length: int = 200) -> str:
    """Sanitize user input for safe logging - prevents log injection"""
    if value is None:
        return "None"
    str_value = str(value)
    # Remove newlines and carriage returns that could fake log entries
    sanitized = str_value.replace('\n', '\\n').replace('\r', '\\r')
    # Truncate long values
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    return sanitized

# Global state
app_state = {
    'model': None,
    'service': None,
    'loaded_at': None,
    'model_metadata': None,
    'validation_passed': False,
    'test_user_id': None,
    'test_item_id': None,
    'port_mappings': None,
    'contractor_mappings': None  
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def load_port_mappings() -> dict:
    """Load port mappings from database including contractor_ports for chauffeur filtering"""
    logger.info(" Loading port mappings...")
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
                    Config.DB_CONNECTION_STRING,
                    connect_args={"timeout": Config.DB_CONNECT_TIMEOUT},
                    pool_pre_ping=True,
                    pool_recycle=Config.DB_POOL_RECYCLE
                )        
        port_vehicles = {}
        vehicle_ports = {}
        contractor_ports = {}
        
        with engine.connect() as conn:
            
            contractor_query = text(f"""
                SELECT ID, RefPort
                FROM [{Config.TRANSELITE_DB}].[dbo].[Contractor]
                WHERE RefPort IS NOT NULL
            """)

            for row in conn.execute(contractor_query):
                contractor_id = int(row.ID)
                port_id = int(row.RefPort)
                contractor_ports[contractor_id] = port_id

            # Vehicle mappings (Port → Vehicles)
            vehicle_query = text(f"""
                SELECT P.ID AS PortID, STRING_AGG(CAST(V.ID AS VARCHAR(20)), ',') AS VehicleIDs
                FROM [{Config.TRANSELITE_DB}].[dbo].[Port] P
                LEFT JOIN [{Config.TRANSELITE_DB}].[dbo].[Contractor] C ON C.RefPort = P.ID
                LEFT JOIN [{Config.TRANSELITE_DB}].[dbo].[Vehicle] V ON V.RefContractor = C.ID
                GROUP BY P.ID
            """)
            
            for row in conn.execute(vehicle_query):
                port_id = int(row.PortID)
                if row.VehicleIDs:
                    vehicle_ids = [int(vid.strip()) for vid in row.VehicleIDs.split(',') if vid.strip()]
                    port_vehicles[port_id] = vehicle_ids
                    for vid in vehicle_ids:
                        vehicle_ports[vid] = port_id
        
        engine.dispose()
        logger.info(f"     Loaded {len(port_vehicles)} ports, {len(vehicle_ports)} vehicles, {len(contractor_ports)} contractors")

        return {
            'port_vehicles': port_vehicles,
            'port_chauffeurs': {},
            'vehicle_ports': vehicle_ports,
            'chauffeur_ports': {},
            'contractor_ports': contractor_ports
        }
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"     Database connection failed for port mappings: {e}")
        raise
    except Exception as e:
        logger.error(f"     Failed to load port mappings: {e}", exc_info=True)
        return {'port_vehicles': {}, 'port_chauffeurs': {}, 'vehicle_ports': {}, 'chauffeur_ports': {}, 'contractor_ports': {}}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def load_contractor_mappings() -> dict:
    """Load contractor mappings from database"""
    logger.info("  Loading contractor mappings...")
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
                    Config.DB_CONNECTION_STRING,
                    connect_args={"timeout": Config.DB_CONNECT_TIMEOUT},
                    pool_pre_ping=True,
                    pool_recycle=Config.DB_POOL_RECYCLE
                )        
        chauffeur_contractors = {}
        vehicle_contractors = {}
        
        with engine.connect() as conn:
            # Chauffeur -> Contractor mapping
            chauffeur_query = text(f"""
                SELECT ID, RefContractor
                FROM [{Config.TRANSELITE_DB}].[dbo].[Chauffeur]
                WHERE IsActive = 1 AND RefContractor IS NOT NULL
            """)

            for row in conn.execute(chauffeur_query):
                chauffeur_id = int(row.ID)
                contractor_id = int(row.RefContractor)
                chauffeur_contractors[chauffeur_id] = contractor_id

            # Vehicle -> Contractor mapping
            vehicle_query = text(f"""
                SELECT ID, RefContractor
                FROM [{Config.TRANSELITE_DB}].[dbo].[Vehicle]
                WHERE IsActive = 1 AND RefContractor IS NOT NULL
            """)
            
            for row in conn.execute(vehicle_query):
                vehicle_id = int(row.ID)
                contractor_id = int(row.RefContractor)
                vehicle_contractors[vehicle_id] = contractor_id
        
        engine.dispose()
        logger.info(f"     Loaded {len(chauffeur_contractors)} chauffeurs, {len(vehicle_contractors)} vehicles")
        
        return {
            'chauffeur_contractors': chauffeur_contractors,
            'vehicle_contractors': vehicle_contractors
        }
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"     Database connection failed for contractor mappings: {e}")
        return {'chauffeur_contractors': {}, 'vehicle_contractors': {}}
    except Exception as e:
        logger.error(f"     Failed to load contractor mappings: {e}", exc_info=True)
        return {'chauffeur_contractors': {}, 'vehicle_contractors': {}}


def validate_model(model, pair_features, chauffeur_data, vehicle_data) -> dict:
    """Validate loaded model is usable"""
    issues = []
    warnings = []
    
    # Check model has required attributes
    required_attrs = ['train_matrix', 'item_similarities', 'user_similarities', 
                     'popularity_scores', 'global_mean']
    for attr in required_attrs:
        if not hasattr(model, attr):
            issues.append(f"Model missing required attribute: {attr}")
    
    # Check training matrix
    if hasattr(model, 'train_matrix'):
        if len(model.train_matrix) == 0:
            issues.append("Training matrix is empty")
        elif len(model.train_matrix) < 10:
            warnings.append(f"Training matrix has only {len(model.train_matrix)} users (very small)")
        
        if len(model.train_matrix.columns) == 0:
            issues.append("Training matrix has no items")
        elif len(model.train_matrix.columns) < 10:
            warnings.append(f"Training matrix has only {len(model.train_matrix.columns)} items (very small)")
    
    # Check pair features
    if pair_features is None or len(pair_features) == 0:
        issues.append("Pair features are empty")
    elif len(pair_features) < 50:
        warnings.append(f"Only {len(pair_features)} pair features (very small)")

    # Check required columns in pair features
    if pair_features is not None and len(pair_features) > 0:
        required_cols = ['completion_rate', 'total_bookings']
        missing_cols = [col for col in required_cols if col not in pair_features.columns]
    if missing_cols:
        warnings.append(f"Pair features missing columns: {missing_cols}")    
    # Check supporting data
    if chauffeur_data is None or len(chauffeur_data) == 0:
        warnings.append("Chauffeur data is empty - filtering will be limited")
    
    if vehicle_data is None or len(vehicle_data) == 0:
        warnings.append("Vehicle data is empty - filtering will be limited")
    
    # Check model freshness (if metadata available) - FIXED ORDER
    model_age_days = None
    if hasattr(model, 'trained_at'):
        model_age = datetime.now() - model.trained_at
        model_age_days = model_age.days
        
        if model_age_days > 180:
            issues.append(f"Model is {model_age_days} days old (STALE - retrain recommended)")
        elif model_age_days > 90:
            warnings.append(f"Model is {model_age_days} days old (consider retraining)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'model_age_days': model_age_days
    }


def test_model_prediction(model, test_user_id, test_item_id) -> bool:
    """Test if model can actually make predictions"""
    try:
        prediction = model.predict_rating(test_user_id, test_item_id)
        
        # Check prediction is valid
        if prediction is None:
            logger.error("Model returned None for test prediction")
            return False
        
        if not (1.0 <= prediction <= 5.0):
            logger.error(f"Model returned invalid prediction: {prediction}")
            return False
        
        logger.info(f"  Model test prediction successful: {prediction:.2f}")
        return True
        
    except (ValueError, KeyError, TypeError) as e:
        logger.error(f"  Model test prediction failed (data error): {e}")
        return False
    except Exception as e:
        logger.error(f"  Model test prediction failed (unexpected): {e}", exc_info=True)
        return False


async def _load_model_into_app_state(config: Config) -> None:
    """
    Load model, feature data, and supporting datasets then write them into
    app_state.  Called at startup and again from POST /admin/reload after
    monthly retraining completes.
    """
    app_state['validation_passed'] = False  # block requests while reloading

    # Load model and data with validation
    logger.info("\n  STEP 1: Loading trained model...")
    ensemble_model = await asyncio.to_thread(load_pickle, config.TRAINED_MODEL_PATH)
    logger.info("     Model file loaded")

    logger.info("\n  STEP 2: Loading feature data...")
    pair_features = await asyncio.to_thread(load_dataframe, config.PAIR_FEATURES_PATH)
    logger.info(f"     Loaded {len(pair_features):,} pair features")

    logger.info("\n  STEP 3: Loading supporting datasets...")
    data_loader = DataLoader(config, strict_mode=True)
    data = await asyncio.to_thread(data_loader.load_all_data)

    chauffeur_data = data['chauffeur']
    vehicle_data = data['vehicle']
    vehicle_type_data = data['vehicle_type']
    chauffeur_status_data = data['chauffeur_status']
    client_data = data['client']

    # VALIDATION STEP
    logger.info("\n STEP 4: Validating loaded data...")
    validation = validate_model(ensemble_model, pair_features, chauffeur_data, vehicle_data)

    if not validation['valid']:
        logger.error("  MODEL VALIDATION FAILED!")
        for issue in validation['issues']:
            logger.error(f"     {issue}")
        raise DataValidationError("Model validation failed. See logs for details.")

    if validation['warnings']:
        logger.warning("   VALIDATION WARNINGS:")
        for warning in validation['warnings']:
            logger.warning(f"      {warning}")

    if validation['model_age_days'] is not None:
        logger.info(f"  Model age: {validation['model_age_days']} days")

    logger.info("     Validation passed")

    # Load port mappings
    logger.info("\n  STEP 5: Loading port mappings...")
    port_mappings = await asyncio.to_thread(load_port_mappings)
    app_state['port_mappings'] = port_mappings

    # Load contractor mappings
    logger.info("\n  STEP 6: Loading contractor mappings...")
    contractor_mappings = await asyncio.to_thread(load_contractor_mappings)
    app_state['contractor_mappings'] = contractor_mappings

    # Initialize filter with port mappings
    logger.info("\n STEP 7: Initializing filter engine...")
    logger.info("      Using STRICT MODE for all filters")
    rec_filter = RecommendationFilter(
        pair_features=pair_features,
        chauffeur_data=chauffeur_data,
        vehicle_data=vehicle_data,
        vehicle_type_data=vehicle_type_data,
        chauffeur_status_data=chauffeur_status_data,
        client_data=client_data,
        strict_mode=True,
        port_mappings=port_mappings,
        contractor_mappings=contractor_mappings
    )
    logger.info("     Filter initialized in STRICT MODE with port and contractor mappings")

    # Initialize service
    logger.info("\n  STEP 8: Initializing recommendation service...")
    service = RecommendationService(
        ensemble_model=ensemble_model,
        recommendation_filter=rec_filter,
        pair_features=pair_features,
        chauffeur_data=chauffeur_data,
        vehicle_data=vehicle_data,
        vehicle_type_data=vehicle_type_data
    )
    logger.info("     Service initialized")

    # Test prediction
    logger.info("\n STEP 9: Testing model prediction...")
    if len(ensemble_model.train_matrix) > 0:
        test_user_id = ensemble_model.train_matrix.index[0]
        test_item_id = ensemble_model.train_matrix.columns[0]

        if not test_model_prediction(ensemble_model, test_user_id, test_item_id):
            raise RuntimeError("Model prediction test failed")

        app_state['test_user_id'] = test_user_id
        app_state['test_item_id'] = test_item_id
    else:
        logger.warning("      Cannot test - no data in training matrix")

    # Store in app state
    app_state['model'] = ensemble_model
    app_state['service'] = service
    app_state['loaded_at'] = datetime.now()
    app_state['validation_passed'] = True
    app_state['model_metadata'] = {
        'n_users': len(ensemble_model.train_matrix),
        'n_items': len(ensemble_model.train_matrix.columns),
        'n_pairs': len(pair_features),
        'model_age_days': validation['model_age_days']
    }

    logger.info("\n" + "="*80)
    logger.info("  SERVICE READY!")
    logger.info("="*80)
    logger.info(f"   Users: {app_state['model_metadata']['n_users']:,}")
    logger.info(f"   Items: {app_state['model_metadata']['n_items']:,}")
    logger.info(f"   Pairs: {app_state['model_metadata']['n_pairs']:,}")
    logger.info(f"   Ports: {len(port_mappings['port_vehicles'])}")
    logger.info("Filter mode: STRICT")
    logger.info("="*80 + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic with validation"""
    # Startup
    logger.info("="*80)
    logger.info("   Starting recommendation service...")
    logger.info("="*80)

    config = Config()
    try:
        config.validate_production_config()
    except EnvironmentError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    try:
        await _load_model_into_app_state(config)
    except DataValidationError as e:
        logger.error("\n  STARTUP FAILED: Data validation error")
        logger.error(f"   {e}")
        logger.error("\n Fix the data issues and restart the service.\n")
        raise
    except Exception as e:
        logger.error(f"\n  STARTUP FAILED: {e}")
        logger.error("   Check that all required files exist and are valid.\n")
        raise

    yield

    # Shutdown
    logger.info("\n Shutting down recommendation service...")
    app_state['validation_passed'] = False


# Create FastAPI app
app = FastAPI(
    title="Chauffeur-Vehicle Recommendation API",
    description="Real-time recommendation service for chauffeur-vehicle pairing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS if Config.ENVIRONMENT != "development" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    
)

from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID to all requests for tracing"""
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Store in request state for access in endpoints
        request.state.request_id = request_id
        logger.info(f"[{request_id}] {request.method} {request.url.path} started")
        
        response = await call_next(request)
        
        response.headers["X-Request-ID"] = request_id
        logger.info(f"[{request_id}] {request.method} {request.url.path} completed - {response.status_code}")
        return response

app.add_middleware(RequestIDMiddleware)


def get_service() -> RecommendationService:
    """Dependency to get recommendation service"""
    if not app_state['validation_passed']:
        raise HTTPException(
            status_code=503, 
            detail="Service failed validation checks. Check server logs."
        )
    if app_state['service'] is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return app_state['service']


@app.get("/", tags=["Status"])
async def root():
    """Root endpoint"""
    return {
        "service": "Chauffeur-Vehicle Recommendation API",
        "version": "1.0.0",
        "status": "running" if app_state['validation_passed'] else "degraded",
        "loaded_at": app_state['loaded_at'].isoformat() if app_state['loaded_at'] else None,
        "filter_mode": "STRICT",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/admin/reload", tags=["Admin"])
async def admin_reload(x_reload_token: str = Header(default="")):
    """
    Hot-reload the collaborative filtering model from disk.

    Called by the monthly retraining orchestrator after a new model has been
    trained.  While the reload is in progress, /health will report 'degraded'
    and the recommendation endpoint will return 503.

    Requires the X-Reload-Token header to match the ADMIN_RELOAD_TOKEN
    environment variable (if that variable is set).
    """
    import os
    expected_token = os.environ.get("ADMIN_RELOAD_TOKEN", "")
    if expected_token and x_reload_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid reload token")

    logger.info("Admin reload requested — reloading CF model from disk…")
    config = Config()
    try:
        await _load_model_into_app_state(config)
    except Exception as exc:
        logger.error("Admin reload failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")

    return {
        "status": "ok",
        "message": "Collaborative filtering model reloaded successfully",
        "reloaded_at": app_state['loaded_at'].isoformat(),
        "model_metadata": app_state['model_metadata'],
    }


async def check_database_connection() -> bool:
    """Check if database is accessible"""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
            Config.DB_CONNECTION_STRING,
            connect_args={"timeout": 5}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return False


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check(service: RecommendationService = Depends(get_service)):
    """Health check that actually tests functionality"""
    model = app_state['model']
    metadata = app_state['model_metadata']
    
    # Determine status
    status = "healthy"
    
    # Check database connection
    db_healthy = await check_database_connection()
    if not db_healthy:
        status = "degraded"
    
    # Check if model is stale
    if metadata and metadata.get('model_age_days'):
        if metadata['model_age_days'] > 180:
            status = "degraded"
        elif metadata['model_age_days'] > 90:
            status = "warning"
    
    # Test prediction if possible
    can_predict = False
    if app_state['test_user_id'] and app_state['test_item_id']:
        try:
            test_pred = model.predict_rating(
                app_state['test_user_id'], 
                app_state['test_item_id']
            )
            can_predict = test_pred is not None and 1.0 <= test_pred <= 5.0
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Health check prediction failed (data error): {e}")
            can_predict = False
            status = "unhealthy"
        except Exception as e:
            logger.error(f"Health check prediction failed (unexpected): {e}", exc_info=True)
            can_predict = False
            status = "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        total_pairs=metadata['n_pairs'] if metadata else 0,
        total_clients=metadata['n_users'] if metadata else 0,
        timestamp=datetime.now(),
        can_predict=can_predict,
        model_age_days=metadata.get('model_age_days') if metadata else None,
        database_connected=db_healthy
    )


@app.post("/api/v1/recommendations", 
          response_model=RecommendationResponse,
          tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    service: RecommendationService = Depends(get_service)
):
    """
    Generate personalized recommendations
    
    This endpoint generates chauffeur-vehicle pair recommendations based on:
    - Client history and preferences (collaborative filtering)
    - Booking context (date, location, distance)
    - Requirements (vehicle type, capacity, features)
    - Filters (availability, ratings, certifications)
    
    Returns ranked recommendations with estimated prices and match scores.
    """
    try:
        logger.info(f"Received recommendation request for client {sanitize_log_input(request.client_id)}")
        logger.info(f"  Requirements: vehicle_type={sanitize_log_input(request.requirements.vehicle_type_id)}, "
                   f"min_level={sanitize_log_input(request.requirements.min_vehicle_level)}, "
                   f"pax={sanitize_log_input(request.booking_context.no_of_pax)}")
        
        response = service.generate_recommendations(request)
        
        # Check if we got zero recommendations
        if len(response.recommendations) == 0:
            # Check if there's an error in metadata
            if 'error' in response.metadata:
                logger.error(f"  No recommendations: {response.metadata['error']}")
                
                # Return 404 with helpful error message
                error_detail = {
                    "error": response.metadata['error'],
                    "details": response.metadata.get('error_details', {}),
                    "filter_breakdown": response.metadata.get('filter_breakdown', {}),
                    "suggestions": []
                }
                
                # Add helpful suggestions
                if request.requirements.vehicle_type_id:
                    error_detail['suggestions'].append(
                        f"Try a different vehicle_type_id (you requested: {request.requirements.vehicle_type_id})"
                    )
                
                if request.requirements.min_vehicle_level:
                    error_detail['suggestions'].append(
                        f"Try lowering min_vehicle_level (you requested: {request.requirements.min_vehicle_level})"
                    )
                
                if request.booking_context.no_of_pax:
                    error_detail['suggestions'].append(
                        f"Try reducing passenger count (you requested: {request.booking_context.no_of_pax})"
                    )
                
                raise HTTPException(
                    status_code=404,
                    detail=error_detail
                )
            else:
                # Generic empty result
                logger.warning("  No recommendations generated (unknown reason)")
        
        if request.save_to_file:
            filepath = save_recommendations_to_json(request.client_id, response)
            logger.info(f" Saved recommendations to {filepath}")
            response.metadata['saved_to_file'] = str(filepath)

        logger.info(f"  Sent {len(response.recommendations)} recommendations")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"  Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def save_recommendations_to_json(client_id: int, response: RecommendationResponse) -> Path:
    """Save recommendations to JSON file (same format as API response)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recommendations_client_{client_id}_{timestamp}.json"
    filepath = RECOMMENDATIONS_DIR / filename
    
    if hasattr(response, 'model_dump'):
        response_dict = response.model_dump()
    else:
        response_dict = response.dict()
        
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(response_dict, f, indent=2, ensure_ascii=False, default=str)
    
    return filepath


@app.post("/api/v1/recommendations/batch", tags=["Recommendations"])
async def get_batch_recommendations(
    requests: list[RecommendationRequest],
    service: RecommendationService = Depends(get_service),
    timeout_per_request: float = Query(default=30.0, ge=1.0, le=120.0)
):
    """
    Generate recommendations for multiple clients with timeout
    
    Accepts a list of recommendation requests and returns recommendations for each.
    Limited to 100 requests per batch.
    Each individual request has a timeout to prevent blocking.
    
    Returns partial success - individual request failures don't fail the entire batch.
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Batch size limited to 100 requests"
        )
    
    logger.info(f" Processing batch of {len(requests)} requests...")
    
    responses = []
    errors = []
    
    for i, req in enumerate(requests):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(service.generate_recommendations, req),
                timeout=timeout_per_request
            )
            
            responses.append({
                "request_index": i,
                "client_id": req.client_id,
                "status": "success",
                "recommendations": response.recommendations,
                "metadata": response.metadata
            })
            
        except asyncio.TimeoutError:
            logger.error(f"  Timeout for client {sanitize_log_input(req.client_id)} (index {i})")
            errors.append({
                "request_index": i,
                "client_id": req.client_id,
                "status": "error",
                "error": f"Request timeout after {timeout_per_request}s",
                "error_type": "TimeoutError"
            })
        
        except Exception as e:
            logger.error(f"  Failed for client {sanitize_log_input(req.client_id)} (index {i}): {e}")
            errors.append({
                "request_index": i,
                "client_id": req.client_id,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
    
    logger.info(f"  Batch complete: {len(responses)} successful, {len(errors)} failed")
    
    return {
        "total_requests": len(requests),
        "successful": len(responses),
        "failed": len(errors),
        "responses": responses,
        "errors": errors if errors else None
    }


@app.get("/api/v1/clients/{client_id}/stats", tags=["Analytics"])
async def get_client_stats(
    client_id: int,
    service: RecommendationService = Depends(get_service)
):
    """Get statistics for a specific client"""
    model = app_state['model']
    
    if client_id not in model.train_matrix.index:
        raise HTTPException(status_code=404, detail="Client not found")
    
    user_ratings = model.train_matrix.loc[client_id]
    rated_items = user_ratings[user_ratings > 0]
    
    return {
        "client_id": client_id,
        "total_interactions": len(rated_items),
        "unique_pairs_used": len(rated_items),
        "avg_rating": float(rated_items.mean()),
        "most_used_pairs": rated_items.nlargest(5).to_dict()
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics_data = []
    
    if app_state['model_metadata']:
        metrics_data.append(f"recommendation_model_users {app_state['model_metadata']['n_users']}")
        metrics_data.append(f"recommendation_model_items {app_state['model_metadata']['n_items']}")
        metrics_data.append(f"recommendation_model_pairs {app_state['model_metadata']['n_pairs']}")
        if app_state['model_metadata'].get('model_age_days'):
            metrics_data.append(f"recommendation_model_age_days {app_state['model_metadata']['model_age_days']}")
    
    metrics_data.append(f"recommendation_service_ready {1 if app_state['validation_passed'] else 0}")
    
    return Response(content="\n".join(metrics_data), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


import warnings
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import uuid

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging
logger = logging.getLogger(__name__)


# ============================================================================
# ENHANCED REQUEST CLASS WITH VALIDATION
# ============================================================================

@dataclass
class EnhancedBookingRequest:
    """Enhanced booking request with comprehensive parameters"""
    
    # REQUIRED
    client_id: Any
    no_of_pax: int
    
    # OPTIONAL CONTEXT
    ref_port: Optional[int] = None 

    ref_vehicle_type: Optional[Any] = None
    appointment_date: Optional[str] = None
    pickup_suburb: Optional[str] = None
    dropoff_suburb: Optional[str] = None
    estimated_distance: Optional[float] = None
    estimated_duration: Optional[int] = None
    # no_of_bags: Optional[int] = None

    # FILTERS
    preferred_chauffeur_ids: Optional[List[Any]] = None
    excluded_vehicle_ids: Optional[List[Any]] = None
    excluded_chauffeur_ids: Optional[List[Any]] = None
    require_baby_seat: bool = False
    require_red_zone_auth: bool = False
    min_vehicle_level: Optional[int] = None
    max_price: Optional[float] = None
    
    # RECOMMENDATION SETTINGS
    top_k_vehicles: int = 20
    top_k_chauffeurs: int = 20
    top_k_pairs: int = 10
    
    # ADVANCED OPTIONS
    include_availability_check: bool = False
    price_weight: float = 0.3
    quality_weight: float = 0.7
    
    def __post_init__(self):
        """Normalize and validate inputs"""
        self.client_id = self._normalize_id(self.client_id)
        if self.ref_port is not None:
            try:
                self.ref_port = int(self.ref_port)
            except (ValueError, TypeError):
                logger.debug(f"  Warning: Invalid ref_port '{self.ref_port}', ignoring")
                self.ref_port = None
       
        if self.ref_vehicle_type is not None:
        # Normalize to float for consistent comparison
            try:
                self.ref_vehicle_type = float(str(self.ref_vehicle_type).strip())
            except (ValueError, TypeError):
                self.ref_vehicle_type = str(self.ref_vehicle_type).strip()
        if self.preferred_chauffeur_ids:
            self.preferred_chauffeur_ids = [self._normalize_id(x) for x in self.preferred_chauffeur_ids]
        if self.excluded_vehicle_ids:
            self.excluded_vehicle_ids = [self._normalize_id(x) for x in self.excluded_vehicle_ids]
        if self.excluded_chauffeur_ids:
            self.excluded_chauffeur_ids = [self._normalize_id(x) for x in self.excluded_chauffeur_ids]
    
    @staticmethod
    def _normalize_id(value):
        """Normalize ID to consistent format"""
        if value is None:
            return None
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return str(value).strip()
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate request parameters"""
        if self.client_id is None:
            return False, "client_id is required"
        
        if self.no_of_pax is None or self.no_of_pax < 1:
            return False, "no_of_pax must be >= 1"
        
        if self.no_of_pax > 100:
            return False, "no_of_pax exceeds maximum (100)"
        
        if self.appointment_date:
            try:
                datetime.fromisoformat(self.appointment_date)
            except ValueError:
                return False, "appointment_date must be in ISO format (YYYY-MM-DD)"
        
        if self.min_vehicle_level and (self.min_vehicle_level < 0 or self.min_vehicle_level > 5):
            return False, "min_vehicle_level must be between 0 and 5"
        
        if self.ref_port is not None:
            if not isinstance(self.ref_port, int) or self.ref_port < 1:
                return False, "ref_port must be a positive integer"

        return True, None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_json(cls, json_str: str):
        return cls(**json.loads(json_str))


# ============================================================================
# ENHANCED RESPONSE CLASS WITH DETAILED INFO
# ============================================================================

@dataclass
class VehicleRecommendation:
    """Detailed vehicle recommendation"""
    vehicle_id: Any
    vehicle_idx: int
    rank: int
    score: float
    
    # Vehicle details
    make: str = "Unknown"
    model: str = "Unknown"
    year: Optional[int] = None
    color: str = "Unknown"
    seat_capacity: Optional[int] = None
    vehicle_level: Optional[int] = None
    
    # Performance metrics
    completion_rate: float = 0.0
    repeat_client_rate: float = 0.0
    total_bookings: int = 0
    avg_price: float = 0.0
    
    # Availability (if checked)
    is_available: bool = True
    availability_note: str = ""


@dataclass
class ChauffeurRecommendation:
    """Detailed chauffeur recommendation"""
    chauffeur_id: Any
    chauffeur_idx: int
    rank: int
    score: float
    
    # Chauffeur details
    code: str = "Unknown"
    experience_score: int = 0
    safety_score: float = 0.0
    
    # Qualifications
    red_zone_authorized: bool = False
    baby_seat_training: bool = False
    driver_training: bool = False
    
    # Performance metrics
    completion_rate: float = 0.0
    repeat_client_rate: float = 0.0
    total_bookings: int = 0
    avg_price: float = 0.0
    
    # Availability (if checked)
    is_available: bool = True
    availability_note: str = ""


@dataclass
class PairRecommendation:
    """Detailed vehicle-chauffeur pair recommendation"""
    rank: int
    combined_score: float
    
    # Component scores
    cv_score: float = 0.0
    cc_score: float = 0.0
    vc_score: float = 0.0
    
    # Vehicle info
    vehicle_id: Any = None
    vehicle_make: str = "Unknown"
    vehicle_model: str = "Unknown"
    vehicle_seat_capacity: Optional[int] = None
    
    # Chauffeur info
    chauffeur_id: Any = None
    chauffeur_experience: int = 0
    chauffeur_safety_score: float = 0.0
    
    # Compatibility
    times_worked_together: int = 0
    compatibility_rate: float = 0.0


@dataclass
class EnhancedRecommendationResponse:
    """Comprehensive recommendation response"""
    
    # Request tracking
    request_id: str
    client_id: Any
    timestamp: str
    
    # Method used
    recommendation_method: str  # 'lightgcn' or 'similarity_based' or 'popularity'
    is_cold_start: bool
    
    # Recommendations
    vehicles: List[Dict]
    chauffeurs: List[Dict]
    pairs: List[Dict]
    
    # Metadata
    total_candidates_vehicles: int
    total_candidates_chauffeurs: int
    filters_applied: List[str]
    warnings: List[str]
    
    # Cold-start specific
    similar_clients_used: int = 0
    similarity_threshold: float = 0.0
    
    # Client context (if available)
    client_info: Dict = field(default_factory=dict)
    
    def to_json(self):
        """
        Convert response to JSON with proper type handling
        
        Ensures all numpy/pandas types are converted to native Python types
        before JSON serialization to avoid TypeError.
        """
        import json
        import numpy as np
        import pandas as pd
        
        def convert_to_native(obj):
            """Convert numpy/pandas types to native Python types"""
            # Handle numpy integer types
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            
            # Handle numpy float types
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            
            # Handle numpy/pandas boolean types
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            
            # Recursively handle dictionaries
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            
            # Recursively handle lists and tuples
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(i) for i in obj]
            
            # Handle numpy arrays
            elif isinstance(obj, np.ndarray):
                return convert_to_native(obj.tolist())
            
            # Handle pandas NaN/None
            elif pd.isna(obj):
                return None
            
            # Handle pandas timestamps
            elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
                return str(obj)
            
            # Handle pandas Series
            elif isinstance(obj, pd.Series):
                return convert_to_native(obj.tolist())
            
            # Return as-is for native Python types
            return obj
        
        # Build response dictionary with explicit type conversions
        response_dict = {
            'request_id': str(self.request_id),
            'client_id': convert_to_native(self.client_id),
            'timestamp': str(self.timestamp),
            'recommendation_method': str(self.recommendation_method),
            'is_cold_start': bool(self.is_cold_start),
            
            # Convert all recommendation lists
            'recommendations': {
                'vehicles': convert_to_native(self.vehicles),
                'chauffeurs': convert_to_native(self.chauffeurs),
                'pairs': convert_to_native(self.pairs)
            },
            
            # Convert metadata fields
            'metadata': {
                'total_candidates_vehicles': int(self.total_candidates_vehicles),
                'total_candidates_chauffeurs': int(self.total_candidates_chauffeurs),
                'filters_applied': [str(f) for f in self.filters_applied],
                'warnings': [str(w) for w in self.warnings],
                'similar_clients_used': int(self.similar_clients_used),
                'similarity_threshold': float(self.similarity_threshold)
            },
            
            # Convert client info dictionary
            'client_info': convert_to_native(self.client_info)
        }
        
        return json.dumps(response_dict, indent=2)


# ============================================================================
# COMPLETE RECOMMENDATION ENGINE
# ============================================================================

class CompleteRecommendationEngine:
    """
    Complete recommendation engine with:
    - LightGCN for in-model clients
    - Similarity-based cold-start
    - Comprehensive filtering
    - Detailed responses
    """
    
    def __init__(self, model, step2_results, step3_results, step4_results, data, adj_sparse):
        logger.debug("\n" + "="*80)
        logger.info("INITIALIZING COMPLETE RECOMMENDATION ENGINE")
        logger.debug("="*80)
        
        self.model = model
        self.model.eval()
        self.data = data
        self.adj_sparse = adj_sparse
        self.mappings = step3_results['mappings']
        
        # Feature tables
        self.vehicle_features = step2_results['vehicle_features']
        self.chauffeur_features = step2_results['chauffeur_features']
        self.client_features = step2_results['client_features']
        self.vc_cooccurrence = step2_results.get('vc_cooccurrence', pd.DataFrame())
        if len(self.vc_cooccurrence) > 0:
            self.vc_cooccurrence['RefVehicle'] = self.vc_cooccurrence['RefVehicle'].astype(float)
            self.vc_cooccurrence['RefChauffeur'] = self.vc_cooccurrence['RefChauffeur'].astype(float)
        ## 
        # In __init__ after loading vehicle_features
        logger.debug("\n Validating vehicle features...")
        logger.debug(f"  Total vehicles: {len(self.vehicle_features)}")
        logger.debug(f"  Unique IDs: {self.vehicle_features['ID'].nunique()}")
        logger.debug(f"  Duplicates: {len(self.vehicle_features) - self.vehicle_features['ID'].nunique()}")

        # Check RefVehicleType coverage
        vtype_null = self.vehicle_features['RefVehicleType'].isna().sum()
        logger.debug(f"  RefVehicleType null: {vtype_null}/{len(self.vehicle_features)}")

        if self.vehicle_features['ID'].duplicated().any():
            logger.debug("   WARNING: Duplicate vehicle IDs found!")
            # Remove duplicates, keeping first occurrence
            self.vehicle_features = self.vehicle_features.drop_duplicates(subset=['ID'], keep='first')

        self.vehicle_features_indexed = self.vehicle_features.set_index('ID')
        # Offsets
        
        self.vehicle_offset = data['n_clients']
        self.chauffeur_offset = data['n_clients'] + data['n_vehicles']
        
        # Precompute embeddings
        logger.debug("\n1. Precomputing embeddings...")
        with torch.no_grad():
            self.all_embeddings = self.model(self.adj_sparse)
        logger.debug(f"    Embeddings shape: {self.all_embeddings.shape}")
        
        # Build lookup indices
        logger.debug("\n2. Building lookup indices...")
        self._build_lookup_indices()
        ################################################### contractor change #####################################
        logger.debug("Loading contractor mappings for recommendation filtering...")
        from data_preparation_and_loading import load_contractor_mappings
        contractor_mappings = load_contractor_mappings()
        self.chauffeur_contractors = contractor_mappings['chauffeur_contractors']
        self.vehicle_contractors = contractor_mappings['vehicle_contractors']
        logger.info(f" Loaded contractor mappings: {len(self.chauffeur_contractors)} chauffeurs, {len(self.vehicle_contractors)} vehicles")


         # Load port mappings BEFORE cold-start prep
        logger.debug("\n3. Loading port mappings...")
        from data_preparation_and_loading import load_port_mappings
        port_mappings = load_port_mappings()
        self.port_vehicles = port_mappings['port_vehicles']
        self.port_chauffeurs = port_mappings['port_chauffeurs']
        self.vehicle_ports = port_mappings['vehicle_ports']
        self.chauffeur_ports = port_mappings['chauffeur_ports']
        logger.info(f"    Loaded {len(self.port_vehicles)} ports with vehicle mappings")
        logger.info(f"    Loaded {len(self.port_chauffeurs)} ports with chauffeur mappings")
        
        # Prepare for cold-start
        logger.debug("\n4. Preparing cold-start features...")
        self._prepare_cold_start_features()
        
        # Precompute popularity rankings (fallback)
        logger.debug("\n5. Computing popularity rankings...")
        self._compute_popularity_rankings()
        
        logger.debug("\n" + "="*80)
        logger.info("ENGINE READY!")
        logger.debug("="*80)
    
    def _build_lookup_indices(self):
        """Build bidirectional ID mappings"""
        # Clients
        self.client_id_to_node = self.mappings['client_to_id']
        self.client_node_to_id = self.mappings['id_to_client']
        
        # Vehicles
        self.vehicle_id_to_idx = {}
        self.vehicle_idx_to_id = {}
        for orig_id, global_id in self.mappings['vehicle_to_id'].items():
            idx = global_id - self.vehicle_offset
            self.vehicle_id_to_idx[orig_id] = idx
            self.vehicle_idx_to_id[idx] = orig_id
        
        # Chauffeurs
        self.chauffeur_id_to_idx = {}
        self.chauffeur_idx_to_id = {}
        for orig_id, global_id in self.mappings['chauffeur_to_id'].items():
            idx = global_id - self.chauffeur_offset
            self.chauffeur_id_to_idx[orig_id] = idx
            self.chauffeur_idx_to_id[idx] = orig_id
        
        # Create indexed feature tables
        #self.vehicle_features_indexed = self.vehicle_features.set_index('ID')
        self.chauffeur_features_indexed = self.chauffeur_features.set_index('ID')
        self.client_features_indexed = self.client_features.set_index('ID')
    
    def _prepare_cold_start_features(self):
        """Prepare features for similarity-based cold-start"""
        
        # Define features for similarity matching
        similarity_features = [
            'RefRegion', 'RefMarketSector', 'RefVehicleType',
            'RefSensitivityLevel', 'tenure_days', 'total_bookings',
            'avg_booking_price', 'lifetime_value', 'activity_score'
        ]
        
        # Filter to available features
        available_features = [f for f in similarity_features 
                             if f in self.client_features.columns]
        
        # Create feature matrix
        self.client_feature_matrix = self.client_features[
            ['ID'] + available_features
        ].copy()
        
        # Handle missing values
        for col in available_features:
            if self.client_feature_matrix[col].dtype == 'object':
                self.client_feature_matrix[col] = self.client_feature_matrix[col].fillna('Unknown')
            else:
                median_val = self.client_feature_matrix[col].median()
                fill_val = median_val if pd.notna(median_val) else 0.0
                self.client_feature_matrix[col] = self.client_feature_matrix[col].fillna(fill_val)
        
        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        self.label_encoders = {}
        categorical_cols = [col for col in available_features 
                           if self.client_feature_matrix[col].dtype == 'object']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.client_feature_matrix[col] = le.fit_transform(
                self.client_feature_matrix[col].astype(str)
            )
            self.label_encoders[col] = le
        
        # Normalize numerical features
        numerical_cols = [col for col in available_features 
                         if col not in categorical_cols]
        
        if len(numerical_cols) > 0:
            self.scaler = StandardScaler()
            self.client_feature_matrix[numerical_cols] = self.scaler.fit_transform(
                self.client_feature_matrix[numerical_cols]
            )
        else:
            self.scaler =None

        self.numerical_cols = numerical_cols    
        self.available_features = available_features
        self.clients_in_model = set(self.client_id_to_node.keys())
        
        logger.debug(f"    Features prepared: {len(available_features)}")
        logger.debug(f"    Clients in model: {len(self.clients_in_model):,}")
    
    def _compute_popularity_rankings(self):
        """Compute global popularity rankings as fallback"""
        
        # Vehicle popularity (by performance metrics)
        # Create a copy with ID as index for proper lookup
        vehicle_df = self.vehicle_features.copy()
        vehicle_df = vehicle_df.set_index('ID')
        
        vehicle_scores = (
            0.4 * vehicle_df['completion_rate'].fillna(0.5) +
            0.3 * vehicle_df['repeat_client_rate'].fillna(0.5) +
            0.3 * (vehicle_df['total_bookings'].fillna(0) / 
                (vehicle_df['total_bookings'].max() + 1))
        )
        
        # Now .index gives us actual vehicle IDs, not row numbers
        self.popular_vehicles = vehicle_scores.nlargest(100).index.tolist()
        
        # Chauffeur popularity
        chauffeur_df = self.chauffeur_features.copy()
        chauffeur_df = chauffeur_df.set_index('ID')
        
        chauffeur_scores = (
            0.4 * chauffeur_df['completion_rate'].fillna(0.5) +
            0.3 * chauffeur_df['repeat_client_rate'].fillna(0.5) +
            0.3 * (chauffeur_df['total_bookings'].fillna(0) / 
                (chauffeur_df['total_bookings'].max() + 1))
        )
        
        self.popular_chauffeurs = chauffeur_scores.nlargest(100).index.tolist()
        
        logger.debug("      Popularity rankings computed")
        logger.debug(f"     Top 3 vehicles: {self.popular_vehicles[:3]}")
        logger.debug(f"     Top 3 chauffeurs: {self.popular_chauffeurs[:3]}")
    
    def _print_summary(self):
        """logger.debug engine summary"""
        logger.debug("\nEngine Statistics:")
        logger.debug(f"  • Clients: {len(self.client_id_to_node):,}")
        logger.debug(f"  • Vehicles: {len(self.vehicle_id_to_idx):,}")
        logger.debug(f"  • Chauffeurs: {len(self.chauffeur_id_to_idx):,}")
        logger.debug(f"  • Features for similarity: {len(self.available_features)}")
    
    # ========================================================================
    # MAIN RECOMMENDATION METHOD
    # ========================================================================
    
    def recommend(self, request: EnhancedBookingRequest) -> EnhancedRecommendationResponse:
        """
        Main recommendation method
        Handles both in-model and cold-start clients
        """
        
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"REQUEST {request_id}")
        logger.debug(f"{'='*80}")
        logger.debug(f"Client: {request.client_id}")
        logger.debug(f"Passengers: {request.no_of_pax}")
        if request.ref_vehicle_type:
            logger.debug(f"Vehicle Type: {request.ref_vehicle_type}")
        
        # Validate request
        is_valid, error_msg = request.validate()
        if not is_valid:
            return self._create_error_response(request_id, request.client_id, 
                                               timestamp, error_msg)
        
        warnings_list = []
        filters_applied = []
        
        # Check if client is in model
        is_cold_start = request.client_id not in self.client_id_to_node
        
        if is_cold_start:
            logger.debug("  COLD START: Client not in model")
            method = 'similarity_based'
        else:
            logger.debug(" Client in model")
            method = 'lightgcn'
        
        # Apply filters
        candidate_vehicles = self._filter_vehicles(request, filters_applied, warnings_list)
        candidate_chauffeurs = self._filter_chauffeurs(request, filters_applied, warnings_list)
        
        logger.debug("\nCandidates after filtering:")
        logger.debug(f"  • Vehicles: {len(candidate_vehicles)}")
        logger.debug(f"  • Chauffeurs: {len(candidate_chauffeurs)}")
        
        if len(candidate_vehicles) == 0:
            warnings_list.append("No vehicles match criteria")
            return self._create_empty_response(
                request_id, request.client_id, timestamp, method,
                is_cold_start, filters_applied, warnings_list
            )
        
        # Generate recommendations based on method
        if is_cold_start:
            # Similarity-based cold-start
            vehicle_recs, chauffeur_recs, pair_recs, similar_clients_count, sim_threshold = \
                self._recommend_cold_start_similarity(
                    request, candidate_vehicles, candidate_chauffeurs
                )
        else:
            # LightGCN recommendations
            client_node_id = self.client_id_to_node[request.client_id]
            vehicle_recs, chauffeur_recs, pair_recs = self._recommend_lightgcn(
                client_node_id, candidate_vehicles, candidate_chauffeurs, request
            )
            similar_clients_count = 0
            sim_threshold = 0.0
        
        # Enrich recommendations with details
        enriched_vehicles = self._enrich_vehicle_recommendations(vehicle_recs)
        enriched_chauffeurs = self._enrich_chauffeur_recommendations(chauffeur_recs)
        enriched_pairs = self._enrich_pair_recommendations(pair_recs)
        
        # Get client info
        client_info = self._get_client_info(request.client_id)
        
        # Create response
        response = EnhancedRecommendationResponse(
            request_id=request_id,
            client_id=request.client_id,
            timestamp=timestamp,
            recommendation_method=method,
            is_cold_start=is_cold_start,
            vehicles=enriched_vehicles[:request.top_k_vehicles],
            chauffeurs=enriched_chauffeurs[:request.top_k_chauffeurs],
            pairs=enriched_pairs[:request.top_k_pairs],
            total_candidates_vehicles=len(candidate_vehicles),
            total_candidates_chauffeurs=len(candidate_chauffeurs),
            filters_applied=filters_applied,
            warnings=warnings_list,
            similar_clients_used=similar_clients_count,
            similarity_threshold=sim_threshold,
            client_info=client_info
        )
        
        logger.info(f"\n Generated {len(enriched_vehicles)} vehicle, "
              f"{len(enriched_chauffeurs)} chauffeur recommendations")
        
        return response
  
 
 
    def _filter_vehicles(self, request, filters_applied, warnings_list):
        """Filter vehicles based on request criteria"""
        candidates = list(range(self.data['n_vehicles']))
        
        logger.debug(f"\n🔍 DEBUG: Initial candidates: {len(candidates)}")
        
        # =====================================================================
        # FILTER BY PORT (DO THIS FIRST - most restrictive)
        # =====================================================================
        if request.ref_port is not None:
            logger.debug(f"🔍 PRIORITY FILTER: Port {request.ref_port}")
            
            # Get allowed vehicles for this port (ORIGINAL DATABASE IDs)
            allowed_vehicle_ids = set(self.port_vehicles.get(request.ref_port, []))
            
            if not allowed_vehicle_ids:
                logger.debug(f"  WARNING: Port {request.ref_port} has no vehicles")
                warnings_list.append(f"Port {request.ref_port} has no vehicles available")
                return []  # Return empty - no vehicles at this port
            
            logger.debug(f"   Port {request.ref_port} has {len(allowed_vehicle_ids)} allowed vehicles")
            logger.debug(f"   First 5 allowed IDs: {list(allowed_vehicle_ids)[:5]}")
            
            # Filter: Keep only candidates whose original ID is in allowed list
            valid = []
            for v_idx in candidates:
                v_id = self.vehicle_idx_to_id[v_idx]  # idx → original database ID
                
                # Type-safe comparison
                if isinstance(v_id, (int, float)):
                    v_id_cmp = float(v_id)
                else:
                    v_id_cmp = v_id
                
                # Check if this vehicle is allowed at this port
                is_allowed = False
                for allowed_id in allowed_vehicle_ids:
                    if isinstance(allowed_id, (int, float)):
                        allowed_id_cmp = float(allowed_id)
                    else:
                        allowed_id_cmp = allowed_id
                    
                    if v_id_cmp == allowed_id_cmp:
                        is_allowed = True
                        break
                
                if is_allowed:
                    valid.append(v_idx)
            
            logger.debug(f"🔍 After port filter: {len(valid)} vehicles remain")
            
            if len(valid) < len(candidates):
                filters_applied.append(f"port={request.ref_port}")
            
            candidates = valid
            
            if len(candidates) == 0:
                warnings_list.append(f"No vehicles available at port {request.ref_port}")
                return []
        
        # =====================================================================
        # FILTER BY PASSENGER CAPACITY
        # =====================================================================
        if request.no_of_pax:
            valid = []
            for v_idx in candidates:
                v_id = self.vehicle_idx_to_id[v_idx]
                if v_id in self.vehicle_features_indexed.index:
                    try:
                        capacity = self.vehicle_features_indexed.loc[v_id, 'SeatCapacity']
                        cap_val = self._safe_extract_value(capacity, None, float)
                        if cap_val is not None and cap_val >= request.no_of_pax:
                            valid.append(v_idx)
                    except (ValueError, TypeError, KeyError):
                        continue
            
            logger.debug(f"🔍 After capacity filter ({request.no_of_pax} pax): {len(valid)}")
            
            if len(valid) < len(candidates):
                filters_applied.append(f"seat_capacity>={request.no_of_pax}")
            
            candidates = valid
        
        # =====================================================================
        # FILTER BY VEHICLE TYPE
        # =====================================================================
        if request.ref_vehicle_type:
            logger.debug(f"🔍 Filtering for vehicle type: {request.ref_vehicle_type}")
            valid = []
            
            # Normalize request type for comparison
            try:
                request_type = float(request.ref_vehicle_type)
            except (ValueError, TypeError, KeyError):
                request_type = str(request.ref_vehicle_type).strip()
            
            for v_idx in candidates:
                v_id = self.vehicle_idx_to_id[v_idx]
                if v_id in self.vehicle_features_indexed.index:
                    try:
                        vtype = self.vehicle_features_indexed.loc[v_id, 'RefVehicleType']
                        vtype_val = self._safe_extract_value(vtype, None, float)
                        
                        # Type-safe comparison
                        if isinstance(vtype_val, (int, float)) and isinstance(request_type, (int, float)):
                            if float(vtype_val) == float(request_type):
                                valid.append(v_idx)
                        elif str(vtype_val) == str(request_type):
                            valid.append(v_idx)
                    except Exception as e:
                        continue
            
            logger.debug(f"After vehicle type filter: {len(valid)}")
            
            if len(valid) < len(candidates):
                filters_applied.append(f"vehicle_type={request.ref_vehicle_type}")
            
            candidates = valid
            
            if len(candidates) == 0:
                warnings_list.append(f"No vehicles match type {request.ref_vehicle_type}")
        # =====================================================================
        # FILTER BY MINIMUM VEHICLE LEVEL
        # =====================================================================
        if request.min_vehicle_level is not None:
            valid = []
            for v_idx in candidates:
                v_id = self.vehicle_idx_to_id[v_idx]
                if v_id in self.vehicle_features_indexed.index:
                    try:
                        level = self.vehicle_features_indexed.loc[v_id, 'VehicleLevel']
                        level_val = self._safe_extract_value(level, None, int)
                        if level_val is not None and level_val >= request.min_vehicle_level:
                            valid.append(v_idx)
                    except (ValueError, TypeError, KeyError):
                        continue
                # If vehicle has no level data, exclude it to be safe

            logger.debug(f"🔍 After vehicle level filter (>={request.min_vehicle_level}): {len(valid)}")

            if len(valid) < len(candidates):
                filters_applied.append(f"vehicle_level>={request.min_vehicle_level}")

            candidates = valid

            if len(candidates) == 0:
                warnings_list.append(f"No vehicles meet minimum level {request.min_vehicle_level}")

        # =====================================================================
        # EXCLUDE SPECIFIC VEHICLES
        # =====================================================================
        if request.excluded_vehicle_ids:
            excluded_indices = [
                self.vehicle_id_to_idx[vid] 
                for vid in request.excluded_vehicle_ids 
                if vid in self.vehicle_id_to_idx
            ]
            candidates = [v for v in candidates if v not in excluded_indices]
            if excluded_indices:
                filters_applied.append(f"excluded {len(excluded_indices)} vehicles")
        
        logger.debug(f"🔍 FINAL vehicle candidates: {len(candidates)}")
        
        return candidates


    def _filter_chauffeurs(self, request, filters_applied,warnings_list):
        """Filter chauffeurs based on request criteria"""
        candidates = list(range(self.data['n_chauffeurs']))
        
        logger.debug(f"\n🔍 DEBUG: Initial chauffeur candidates: {len(candidates)}")
         
        # =====================================================================
        # FILTER BY QUALIFICATIONS
        # =====================================================================
        
        # Red zone authorization
        if request.require_red_zone_auth:
            valid = []
            for c_idx in candidates:
                c_id = self.chauffeur_idx_to_id[c_idx]
                if c_id in self.chauffeur_features_indexed.index:
                    try:
                        rz = self.chauffeur_features_indexed.loc[c_id, 'RedZoneAuthorised']
                        rz_val = self._safe_bool_check(rz, False)
                        if rz_val:
                            valid.append(c_idx)
                    except (ValueError, TypeError):
                        continue
            if len(valid) < len(candidates):
                filters_applied.append("red_zone_auth")
            candidates = valid
        
        # Baby seat training
        if request.require_baby_seat:
            valid = []
            for c_idx in candidates:
                c_id = self.chauffeur_idx_to_id[c_idx]
                if c_id in self.chauffeur_features_indexed.index:
                    try:
                        bs = self.chauffeur_features_indexed.loc[c_id, 'BabySeatTraining']
                        bs_val = self._safe_bool_check(bs, False)
                        if bs_val:
                            valid.append(c_idx)
                    except (ValueError, TypeError):
                        continue
            if len(valid) < len(candidates):
                filters_applied.append("baby_seat_training")
            candidates = valid
        
        # Preferred chauffeurs (ONLY these)
        if request.preferred_chauffeur_ids:
            preferred = [
                self.chauffeur_id_to_idx[cid] 
                for cid in request.preferred_chauffeur_ids 
                if cid in self.chauffeur_id_to_idx
            ]
            if preferred:
                filters_applied.append(f"preferred_chauffeurs ({len(preferred)})")
                candidates = preferred
        
        # Exclude specific chauffeurs
        if request.excluded_chauffeur_ids:
            excluded = [
                self.chauffeur_id_to_idx[cid] 
                for cid in request.excluded_chauffeur_ids 
                if cid in self.chauffeur_id_to_idx
            ]
            candidates = [c for c in candidates if c not in excluded]
            if excluded:
                filters_applied.append(f"excluded {len(excluded)} chauffeurs")
        
        logger.debug(f"🔍 FINAL chauffeur candidates: {len(candidates)}")
        
        return candidates
   
   #########################################contractor change ########################################
    def _filter_by_contractor(self, chauffeur_id, vehicle_recommendations):
            """
            Filter vehicle recommendations to only include same-contractor vehicles
            
            Args:
                chauffeur_id: ID of the chauffeur
                vehicle_recommendations: List of vehicle recommendation dicts
            
            Returns:
                Filtered list of recommendations
            """
            # Get chauffeur's contractor
            chauffeur_contractor = self.chauffeur_contractors.get(int(chauffeur_id))
            
            if chauffeur_contractor is None:
                logger.debug(f"  Warning: No contractor found for chauffeur {chauffeur_id}")
                return vehicle_recommendations  # Return unfiltered as fallback
            
            # Filter to same contractor
            filtered = []
            for rec in vehicle_recommendations:
                vehicle_id = rec.get('vehicle_id')
                vehicle_contractor = self.vehicle_contractors.get(int(vehicle_id))
                
                if vehicle_contractor == chauffeur_contractor:
                    filtered.append(rec)
            
            return filtered
    # ========================================================================
    # LIGHTGCN RECOMMENDATION
    # ========================================================================
    
    def _recommend_lightgcn(self, client_node_id, candidate_vehicles, 
                           candidate_chauffeurs,request):
        """Generate recommendations using LightGCN embeddings"""
        
        client_emb = self.all_embeddings[client_node_id].unsqueeze(0)
        
        # Vehicle recommendations
        vehicle_recs = []
        for v_idx in candidate_vehicles:
            v_node_id = self.vehicle_offset + v_idx
            score = torch.dot(client_emb.squeeze(), 
                            self.all_embeddings[v_node_id]).item()
            vehicle_recs.append({
                'vehicle_idx': v_idx,
                'vehicle_id': self.vehicle_idx_to_id[v_idx],
                'score': float(score)
            })
        vehicle_recs.sort(key=lambda x: x['score'], reverse=True)
        
        # Chauffeur recommendations
        chauffeur_recs = []
        for c_idx in candidate_chauffeurs:
            c_node_id = self.chauffeur_offset + c_idx
            score = torch.dot(client_emb.squeeze(), 
                            self.all_embeddings[c_node_id]).item()
            chauffeur_recs.append({
                'chauffeur_idx': c_idx,
                'chauffeur_id': self.chauffeur_idx_to_id[c_idx],
                'score': float(score)
            })
        chauffeur_recs.sort(key=lambda x: x['score'], reverse=True)
        

        pair_recs = []
        for v_rec in vehicle_recs[:min(len(vehicle_recs), len(vehicle_recs))]:
            v_emb = self.all_embeddings[self.vehicle_offset + v_rec['vehicle_idx']]
            for c_rec in chauffeur_recs[:min(len(chauffeur_recs), len(chauffeur_recs))]:
                c_emb = self.all_embeddings[self.chauffeur_offset + c_rec['chauffeur_idx']]
                
                # NEW: Check contractor match BEFORE computing score
                vehicle_id = int(v_rec['vehicle_id'])
                chauffeur_id = int(c_rec['chauffeur_id'])
                vehicle_contractor = self.vehicle_contractors.get(vehicle_id)
                chauffeur_contractor = self.chauffeur_contractors.get(chauffeur_id)
                
                # Skip if different contractors
                if (vehicle_contractor is None or 
                    chauffeur_contractor is None or
                    vehicle_contractor != chauffeur_contractor):
                    continue
                
                vc_score = torch.dot(v_emb, c_emb).item()
                combined = (0.4 * v_rec['score'] + 
                          0.3 * c_rec['score'] + 
                          0.3 * vc_score)
                pair_recs.append({
                    'vehicle_idx': v_rec['vehicle_idx'],
                    'vehicle_id': v_rec['vehicle_id'],
                    'chauffeur_idx': c_rec['chauffeur_idx'],
                    'chauffeur_id': c_rec['chauffeur_id'],
                    'combined_score': float(combined),
                    'cv_score': float(v_rec['score']),
                    'cc_score': float(c_rec['score']),
                    'vc_score': float(vc_score)
                })
        pair_recs.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Log filtering results
        logger.debug(f"   Generated {len(pair_recs)} valid same-contractor pairs")
        return vehicle_recs, chauffeur_recs, pair_recs
    
    # ========================================================================
    # COLD-START SIMILARITY-BASED RECOMMENDATION
    # ========================================================================
    
    def _recommend_cold_start_similarity(self, request, candidate_vehicles,
                                        candidate_chauffeurs, n_similar=5):
        """
        Generate recommendations for cold-start clients using similar clients
        """
        
        logger.debug("\nFinding similar clients...")
        
        # Find similar clients
        similar_clients = self._find_similar_clients(
            request.client_id, 
            top_k=n_similar
        )
        
        if len(similar_clients) == 0:
            logger.debug("  No similar clients found, using popularity")
            return self._recommend_popularity(
                candidate_vehicles, candidate_chauffeurs
            )
        
        logger.debug(f"  Found {len(similar_clients)} similar clients")
        avg_similarity = np.mean([sim for _, sim in similar_clients])
        logger.debug(f"  Average similarity: {avg_similarity:.3f}")
        
        # Get recommendations from similar clients
        similar_client_node_ids = [
            self.client_id_to_node[client_id] 
            for client_id, _ in similar_clients
        ]
        
        # Aggregate vehicle recommendations
        vehicle_scores = {}
        for sim_client_node_id in similar_client_node_ids:
            client_emb = self.all_embeddings[sim_client_node_id].unsqueeze(0)
            
            for v_idx in candidate_vehicles:
                v_node_id = self.vehicle_offset + v_idx
                score = torch.dot(client_emb.squeeze(), 
                                self.all_embeddings[v_node_id]).item()
                
                # Find similarity weight
                original_client_id = self.client_node_to_id[sim_client_node_id]
                weight = next((w for cid, w in similar_clients 
                             if cid == original_client_id), 0)
                
                if v_idx not in vehicle_scores:
                    vehicle_scores[v_idx] = 0
                vehicle_scores[v_idx] += score * weight
        
        # Create vehicle recommendations
        vehicle_recs = []
        for v_idx, score in vehicle_scores.items():
            vehicle_recs.append({
                'vehicle_idx': v_idx,
                'vehicle_id': self.vehicle_idx_to_id[v_idx],
                'score': float(score)
            })
        vehicle_recs.sort(key=lambda x: x['score'], reverse=True)
        
        # Aggregate chauffeur recommendations
        chauffeur_scores = {}
        for sim_client_node_id in similar_client_node_ids:
            client_emb = self.all_embeddings[sim_client_node_id].unsqueeze(0)
            
            for c_idx in candidate_chauffeurs:
                c_node_id = self.chauffeur_offset + c_idx
                score = torch.dot(client_emb.squeeze(), 
                                self.all_embeddings[c_node_id]).item()
                
                original_client_id = self.client_node_to_id[sim_client_node_id]
                weight = next((w for cid, w in similar_clients 
                             if cid == original_client_id), 0)
                
                if c_idx not in chauffeur_scores:
                    chauffeur_scores[c_idx] = 0
                chauffeur_scores[c_idx] += score * weight
        
        # Create chauffeur recommendations
        chauffeur_recs = []
        for c_idx, score in chauffeur_scores.items():
            chauffeur_recs.append({
                'chauffeur_idx': c_idx,
                'chauffeur_id': self.chauffeur_idx_to_id[c_idx],
                'score': float(score)
            })
        chauffeur_recs.sort(key=lambda x: x['score'], reverse=True)
        
        
        pair_recs = []
   
        for v_rec in vehicle_recs[:min(len(vehicle_recs), len(vehicle_recs))]:
            v_emb = self.all_embeddings[self.vehicle_offset + v_rec['vehicle_idx']]
            for c_rec in chauffeur_recs[:min(len(chauffeur_recs), len(chauffeur_recs))]:
                c_emb = self.all_embeddings[self.chauffeur_offset + c_rec['chauffeur_idx']]
                
                # NEW: Check contractor match BEFORE computing score
                vehicle_id = int(v_rec['vehicle_id'])
                chauffeur_id = int(c_rec['chauffeur_id'])
                vehicle_contractor = self.vehicle_contractors.get(vehicle_id)
                chauffeur_contractor = self.chauffeur_contractors.get(chauffeur_id)
                
                # Skip if different contractors
                if (vehicle_contractor is None or 
                    chauffeur_contractor is None or
                    vehicle_contractor != chauffeur_contractor):
                    continue
                
                vc_score = torch.dot(v_emb, c_emb).item()
                combined = (0.4 * v_rec['score'] + 
                          0.3 * c_rec['score'] + 
                          0.3 * vc_score)
                pair_recs.append({
                    'vehicle_idx': v_rec['vehicle_idx'],
                    'vehicle_id': v_rec['vehicle_id'],
                    'chauffeur_idx': c_rec['chauffeur_idx'],
                    'chauffeur_id': c_rec['chauffeur_id'],
                    'combined_score': float(combined),
                    'cv_score': float(v_rec['score']),
                    'cc_score': float(c_rec['score']),
                    'vc_score': float(vc_score)
                })
        pair_recs.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Log filtering results
        logger.debug(f"   Cold-start: Generated {len(pair_recs)} valid same-contractor pairs")
        return (vehicle_recs, chauffeur_recs, pair_recs, 
                len(similar_clients), avg_similarity)
    
    def _find_similar_clients(self, target_client_id, top_k=5):
        """Find K most similar clients in the model"""
        
        # Get target client features
        target_features = self.client_feature_matrix[
            self.client_feature_matrix['ID'] == target_client_id
        ][self.available_features].values
        
        if len(target_features) == 0:
            return []
        #   NORMALIZE target client using the SAME scaler
        if self.scaler is not None and len(self.numerical_cols) > 0:
            target_features_copy = target_features.copy()
            
            # Get indices of numerical columns in feature array
            num_indices = [self.available_features.index(col) 
                        for col in self.numerical_cols]
            
            # Normalize only numerical columns
            target_features_copy[:, num_indices] = self.scaler.transform(
                target_features[:, num_indices]
            )
            target_features = target_features_copy
        # Get features for clients in model
        model_clients_df = self.client_feature_matrix[
            self.client_feature_matrix['ID'].isin(self.clients_in_model)
        ]
        
        if len(model_clients_df) == 0:
            return []
        
        model_client_ids = model_clients_df['ID'].values
        model_features = model_clients_df[self.available_features].values
        
        # Calculate cosine similarity
        similarities = cosine_similarity(target_features, model_features)[0]
        
        # Get top-K
        top_k_actual = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k_actual:][::-1]
        
        similar_clients = [
            (model_client_ids[idx], similarities[idx]) 
            for idx in top_indices
        ]
        
        return similar_clients
    
    def _recommend_popularity(self, candidate_vehicles, candidate_chauffeurs):
        """Fallback to popularity-based recommendations"""
        
        logger.debug("  Using popularity-based recommendations")
        
        # Filter popular vehicles by candidates
        # popular_vehicles now contains ORIGINAL IDs
        vehicle_recs = []
        for rank, v_id in enumerate(self.popular_vehicles, 1):
            # Check if this vehicle ID exists in our mapping
            if v_id in self.vehicle_id_to_idx:
                v_idx = self.vehicle_id_to_idx[v_id]
                # candidate_vehicles contains 0-based indices
                if v_idx in candidate_vehicles:
                    vehicle_recs.append({
                        'vehicle_idx': v_idx,
                        'vehicle_id': v_id,
                        'score': 1.0 - (rank / 100)  # Rank 1 = 0.99, Rank 100 = 0
                    })
            
            # Stop once we have enough
            if len(vehicle_recs) >= 100:
                break
        
        # Filter popular chauffeurs by candidates
        chauffeur_recs = []
        for rank, c_id in enumerate(self.popular_chauffeurs, 1):
            if c_id in self.chauffeur_id_to_idx:
                c_idx = self.chauffeur_id_to_idx[c_id]
                if c_idx in candidate_chauffeurs:
                    chauffeur_recs.append({
                        'chauffeur_idx': c_idx,
                        'chauffeur_id': c_id,
                        'score': 1.0 - (rank / 100)
                    })
            
            if len(chauffeur_recs) >= 100:
                break
        
        # Warn if we got very few results
        if len(vehicle_recs) < 5:
            logger.debug(f"   WARNING: Only {len(vehicle_recs)} popular vehicles passed filters")
        if len(chauffeur_recs) < 5:
            logger.debug(f"   WARNING: Only {len(chauffeur_recs)} popular chauffeurs passed filters")
        
        # Simple pair combinations
        pair_recs = []
        for v_rec in vehicle_recs[:10]:
            vehicle_id = int(v_rec['vehicle_id'])
            vehicle_contractor = self.vehicle_contractors.get(vehicle_id)
            for c_rec in chauffeur_recs[:10]:
                chauffeur_id = int(c_rec['chauffeur_id'])
                chauffeur_contractor = self.chauffeur_contractors.get(chauffeur_id)
                if (vehicle_contractor is None or
                    chauffeur_contractor is None or
                    vehicle_contractor != chauffeur_contractor):
                    continue
                pair_recs.append({
                    'vehicle_idx': v_rec['vehicle_idx'],
                    'vehicle_id': v_rec['vehicle_id'],
                    'chauffeur_idx': c_rec['chauffeur_idx'],
                    'chauffeur_id': c_rec['chauffeur_id'],
                    'combined_score': (v_rec['score'] + c_rec['score']) / 2,
                    'cv_score': v_rec['score'],
                    'cc_score': c_rec['score'],
                    'vc_score': 0.5
                })
        
        return vehicle_recs, chauffeur_recs, pair_recs, 0, 0.0
    
    # ========================================================================
    # ENRICHMENT METHODS
    # ========================================================================
    
    def _enrich_vehicle_recommendations(self, vehicle_recs):
        """Add detailed vehicle information to recommendations"""
        enriched = []
        
        for rank, rec in enumerate(vehicle_recs, 1):
            v_id = rec['vehicle_id']
            
            details = {
                'vehicle_id': v_id,
                'vehicle_idx': rec['vehicle_idx'],
                'rank': rank,
                'score': rec['score']
            }
            
            if v_id in self.vehicle_features_indexed.index:
                try:
                    f = self.vehicle_features_indexed.loc[v_id]
                    
                    details.update({
                        'make': self._safe_extract_value(f.get('Make'), 'Unknown', str),
                        'model': self._safe_extract_value(f.get('Model'), 'Unknown', str),
                        'year': self._safe_extract_value(f.get('Year'), None, int),
                        'color': self._safe_extract_value(f.get('Colour'), 'Unknown', str),
                        'seat_capacity': self._safe_extract_value(f.get('SeatCapacity'), None, int),
                        'vehicle_level': self._safe_extract_value(f.get('VehicleLevel'), None, int),
                        'completion_rate': self._safe_extract_value(f.get('completion_rate'), 0.0, float),
                        'repeat_client_rate': self._safe_extract_value(f.get('repeat_client_rate'), 0.0, float),
                        'total_bookings': self._safe_extract_value(f.get('total_bookings'), 0, int),
                        'avg_price': self._safe_extract_value(f.get('avg_price'), 0.0, float)
                    })
                except Exception as e:
                    pass
            
            enriched.append(details)
        
        return enriched
    
    def _enrich_chauffeur_recommendations(self, chauffeur_recs):
        """Add detailed chauffeur information to recommendations"""
        enriched = []
        
        for rank, rec in enumerate(chauffeur_recs, 1):
            c_id = rec['chauffeur_id']
            
            details = {
                'chauffeur_id': c_id,
                'chauffeur_idx': rec['chauffeur_idx'],
                'rank': rank,
                'score': rec['score']
            }
            
            if c_id in self.chauffeur_features_indexed.index:
                try:
                    f = self.chauffeur_features_indexed.loc[c_id]
                    
                    details.update({
                        'code': self._safe_extract_value(f.get('Code'), 'Unknown', str),
                        'experience_score': self._safe_extract_value(f.get('experience_score'), 0, int),
                        'safety_score': self._safe_extract_value(f.get('safety_score'), 0.0, float),
                        'red_zone_authorized': self._safe_bool_check(f.get('RedZoneAuthorised'), False),
                        'baby_seat_training': self._safe_bool_check(f.get('BabySeatTraining'), False),
                        'driver_training': self._safe_bool_check(f.get('DriverTraining'), False),
                        'completion_rate': self._safe_extract_value(f.get('completion_rate'), 0.0, float),
                        'repeat_client_rate': self._safe_extract_value(f.get('repeat_client_rate'), 0.0, float),
                        'total_bookings': self._safe_extract_value(f.get('total_bookings'), 0, int),
                        'avg_price': self._safe_extract_value(f.get('avg_price'), 0.0, float)
                    })
                except Exception as e:
                    pass
            
            enriched.append(details)
        
        return enriched
   
            

    
    def _enrich_pair_recommendations(self, pair_recs):
        """
        Add detailed pair information to recommendations
          CORRECTED: Direct idx → original ID mapping
        """
        enriched = []
        pairs_with_history = 0
        pairs_without_history = 0
        lookup_errors = 0
        
        for rank, rec in enumerate(pair_recs, 1):
            # These are 0-based indices used by the model
            vehicle_idx = rec.get('vehicle_idx')
            chauffeur_idx = rec.get('chauffeur_idx')
            
            if vehicle_idx is None or chauffeur_idx is None:
                logger.debug(f"  Rank {rank}: Missing indices in rec")
                lookup_errors += 1
                continue
            
            #   DIRECT MAPPING: idx → original database ID
            v_id = self.vehicle_idx_to_id.get(vehicle_idx)
            c_id = self.chauffeur_idx_to_id.get(chauffeur_idx)
            
            if v_id is None or c_id is None:
                logger.debug(f"  Rank {rank}: Could not map - vehicle_idx={vehicle_idx}, chauffeur_idx={chauffeur_idx}")
                lookup_errors += 1
                continue
            
            #   DEBUG: logger.debug first 3 lookups
            if rank <= 3:
                logger.debug(f"\n🔍 Rank {rank}: vehicle_idx={vehicle_idx} → v_id={v_id}, chauffeur_idx={chauffeur_idx} → c_id={c_id}")
            
            details = {
                'rank': rank,
                'combined_score': rec.get('combined_score', 0.0),
                'cv_score': rec.get('cv_score', 0.0),
                'cc_score': rec.get('cc_score', 0.0),
                'vc_score': rec.get('vc_score', 0.0),
                'vehicle_id': v_id,
                'chauffeur_id': c_id,
                'vehicle_idx': vehicle_idx,
                'chauffeur_idx': chauffeur_idx
            }
            
            # ===== VEHICLE DETAILS =====
            if v_id in self.vehicle_features_indexed.index:
                try:
                    vf = self.vehicle_features_indexed.loc[v_id]
                    details['vehicle_make'] = self._safe_extract_value(vf.get('Make'), 'Unknown', str)
                    details['vehicle_model'] = self._safe_extract_value(vf.get('Model'), 'Unknown', str)
                    details['vehicle_year'] = self._safe_extract_value(vf.get('Year'), None, int)
                    details['vehicle_color'] = self._safe_extract_value(vf.get('Colour'), 'Unknown', str)
                    details['vehicle_seat_capacity'] = self._safe_extract_value(vf.get('SeatCapacity'), None, int)
                    details['vehicle_level'] = self._safe_extract_value(vf.get('VehicleLevel'), None, int)
                    
                    vehicle_type_id = self._safe_extract_value(vf.get('RefVehicleType'), None, float)
                    if vehicle_type_id is not None:
                        details['vehicle_type_id'] = vehicle_type_id
                        details['vehicle_type_description'] = f"Type {int(vehicle_type_id)}"
                    
                    details['vehicle_completion_rate'] = self._safe_extract_value(vf.get('completion_rate'), 0.0, float)
                    details['vehicle_repeat_client_rate'] = self._safe_extract_value(vf.get('repeat_client_rate'), 0.0, float)
                    details['vehicle_total_bookings'] = self._safe_extract_value(vf.get('total_bookings'), 0, int)
                except Exception as e:
                    logger.debug(f"  Error loading vehicle {v_id}: {e}")
            
            # ===== CHAUFFEUR DETAILS =====
            if c_id in self.chauffeur_features_indexed.index:
                try:
                    cf = self.chauffeur_features_indexed.loc[c_id]
                    details['chauffeur_code'] = self._safe_extract_value(cf.get('Code'), 'Unknown', str)
                    details['chauffeur_experience'] = self._safe_extract_value(cf.get('experience_score'), 0, int)
                    details['chauffeur_safety_score'] = self._safe_extract_value(cf.get('safety_score'), 0.0, float)
                    details['chauffeur_red_zone_auth'] = self._safe_bool_check(cf.get('RedZoneAuthorised'), False)
                    details['chauffeur_baby_seat_training'] = self._safe_bool_check(cf.get('BabySeatTraining'), False)
                    details['chauffeur_completion_rate'] = self._safe_extract_value(cf.get('completion_rate'), 0.0, float)
                    details['chauffeur_repeat_client_rate'] = self._safe_extract_value(cf.get('repeat_client_rate'), 0.0, float)
                    details['chauffeur_total_bookings'] = self._safe_extract_value(cf.get('total_bookings'), 0, int)
                except Exception as e:
                    logger.debug(f"  Error loading chauffeur {c_id}: {e}")
            
            # =====   V-C CO-OCCURRENCE HISTORY =====
            try:
                if not hasattr(self, 'vc_cooccurrence') or len(self.vc_cooccurrence) == 0:
                    if rank == 1:  # Only logger.debug once
                        logger.debug("  No V-C co-occurrence data available in engine")
                    details['times_worked_together'] = 0
                    details['successful_trips_together'] = 0
                    details['historical_compatibility_score'] = None
                    details['cooccurrence_quality'] = None
                    details['enhanced_pair_score'] = rec.get('combined_score', 0.0)
                    details['history_weight_used'] = 0.0
                    pairs_without_history += 1
                else:
                    #   DEBUG: Check data types for first 3
                    if rank <= 3:
                        logger.debug(f"   v_id type: {type(v_id)}, c_id type: {type(c_id)}")
                        logger.debug(f"   vc_cooccurrence['RefVehicle'] dtype: {self.vc_cooccurrence['RefVehicle'].dtype}")
                        logger.debug(f"   vc_cooccurrence['RefChauffeur'] dtype: {self.vc_cooccurrence['RefChauffeur'].dtype}")
                    
                    #   ENSURE TYPE MATCHING
                    v_id_lookup = float(v_id) if not isinstance(v_id, str) else v_id
                    c_id_lookup = float(c_id) if not isinstance(c_id, str) else c_id
                    
                    if rank <= 3:
                        logger.debug(f"   Looking up: V={v_id_lookup}, C={c_id_lookup}")
                    
                    # Lookup in vc_cooccurrence
                    vc_pair = self.vc_cooccurrence[
                        (self.vc_cooccurrence['RefVehicle'] == v_id_lookup) & 
                        (self.vc_cooccurrence['RefChauffeur'] == c_id_lookup)
                    ]
                    
                    if rank <= 3:
                        logger.debug(f"   Matches found: {len(vc_pair)}")
                    
                    if len(vc_pair) > 0:
                        pair_data = vc_pair.iloc[0]
                        
                        # Extract historical data
                        times_together = int(self._safe_extract_value(pair_data.get('times_together'), 0, int))
                        details['times_worked_together'] = times_together
                        details['successful_trips_together'] = int(self._safe_extract_value(pair_data.get('successful_together'), 0, int))
                        
                        hist_compat = float(self._safe_extract_value(pair_data.get('compatibility_score'), 0.0, float))
                        details['historical_compatibility_score'] = hist_compat
                        details['cooccurrence_quality'] = float(self._safe_extract_value(pair_data.get('cooccurrence_quality'), 0.0, float))
                        
                        details['avg_price_together'] = float(self._safe_extract_value(pair_data.get('avg_price_together'), 0.0, float))
                        details['total_revenue_together'] = float(self._safe_extract_value(pair_data.get('revenue_together'), 0.0, float))
                        details['days_working_together'] = int(self._safe_extract_value(pair_data.get('days_working_together'), 0, int))
                        
                        #   ENHANCED SCORE
                        if times_together > 0:
                            history_weight = min(0.30, (times_together / 50) * 0.30)
                            
                            details['enhanced_pair_score'] = (
                                (1 - history_weight) * rec.get('combined_score', 0.0) +
                                history_weight * hist_compat
                            )
                            details['history_weight_used'] = history_weight
                            pairs_with_history += 1
                            
                            if rank <= 3:
                                logger.debug(f"     HISTORY FOUND: {times_together} trips, compat={hist_compat:.2%}, weight={history_weight:.3f}")
                        else:
                            details['enhanced_pair_score'] = rec.get('combined_score', 0.0)
                            details['history_weight_used'] = 0.0
                            pairs_without_history += 1
                    else:
                        # No history found
                        details['times_worked_together'] = 0
                        details['successful_trips_together'] = 0
                        details['historical_compatibility_score'] = None
                        details['cooccurrence_quality'] = None
                        details['enhanced_pair_score'] = rec.get('combined_score', 0.0)
                        details['history_weight_used'] = 0.0
                        pairs_without_history += 1
                        
                        if rank <= 3:
                            logger.debug("    NO HISTORY: Pair never worked together")
                        
            except Exception as e:
                logger.debug(f"  ERROR at rank {rank} for V:{v_id}, C:{c_id}: {e}")
                import traceback
                traceback.logger.debug_exc()
                details['times_worked_together'] = 0
                details['enhanced_pair_score'] = rec.get('combined_score', 0.0)
                details['history_weight_used'] = 0.0
                lookup_errors += 1
            
            enriched.append(details)
        
        #   SUMMARY
        logger.debug("\n V-C Lookup Summary:")
        logger.debug(f"   Total pairs: {len(enriched)}")
        logger.debug(f"   WITH history: {pairs_with_history}")
        logger.debug(f"   WITHOUT history: {pairs_without_history}")
        logger.debug(f"   Errors: {lookup_errors}")
        
        if pairs_with_history > 0:
            logger.debug("     Historical data IS WORKING!")
        else:
            logger.debug("     NO pairs with history found - check data types")
        
        #   RE-SORT BY ENHANCED SCORE
        enriched.sort(key=lambda x: x.get('enhanced_pair_score', x.get('combined_score', 0)), reverse=True)
        
        # Update ranks
        for new_rank, item in enumerate(enriched, 1):
            item['rank'] = new_rank
        
        return enriched
    
    def _get_client_info(self, client_id):
        """Get comprehensive client information"""
        if client_id not in self.client_features_indexed.index:
            return {}
        
        try:
            f = self.client_features_indexed.loc[client_id]
            
            client_info = {
                'client_id': client_id,
                
                # ===== BOOKING HISTORY =====
                'total_bookings': self._safe_extract_value(f.get('total_bookings'), 0, int),
                'lifetime_value': self._safe_extract_value(f.get('lifetime_value'), 0.0, float),
                'avg_booking_price': self._safe_extract_value(f.get('avg_booking_price'), 0.0, float),
                
                # ===== CLIENT PROFILE =====
                'client_tier': self._safe_extract_value(f.get('client_tier'), 'Standard', str),
                'activity_score': self._safe_extract_value(f.get('activity_score'), 0, int),
                'tenure_days': self._safe_extract_value(f.get('tenure_days'), 0, int),
                
                # ===== PREFERENCES (if available) =====
                'preferred_vehicle_type': self._safe_extract_value(f.get('RefVehicleType'), None, float),
                'sensitivity_level': self._safe_extract_value(f.get('RefSensitivityLevel'), None, int),
                'market_sector': self._safe_extract_value(f.get('RefMarketSector'), None, str),
                
                # ===== BOOKING PATTERNS =====
                'weekend_preference': self._safe_extract_value(f.get('weekend_preference'), None, float),
                'business_hours_preference': self._safe_extract_value(f.get('business_hours_preference'), None, float),
                'avg_lead_time_days': self._safe_extract_value(f.get('avg_lead_time'), None, float),
                
                # ===== AVERAGES =====
                'avg_passengers': self._safe_extract_value(f.get('avg_pax'), None, float),
                'avg_distance_km': self._safe_extract_value(f.get('avg_distance'), None, float),
                
                # ===== STATUS =====
                'is_active': self._safe_extract_value(f.get('IsActive'), True, bool),
                'days_since_last_booking': self._safe_extract_value(f.get('days_since_last_booking'), None, float)
            }
            
            # Clean up None values
            return {k: v for k, v in client_info.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error getting client info: {e}")
            return {}
        
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    @staticmethod
    def _safe_extract_value(val, default=None, as_type=float):
        """Safely extract value from Series/array/scalar"""
        try:
            if val is None or pd.isna(val):
                return default
            
            if hasattr(val, 'iloc'):
                if len(val) == 0:
                    return default
                extracted = val.iloc[0]
                if pd.isna(extracted):
                    return default
                return as_type(extracted) if default is not None else extracted
            
            elif hasattr(val, '__len__') and not isinstance(val, str):
                if len(val) == 0:
                    return default
                extracted = val[0] if hasattr(val, '__getitem__') else list(val)[0]
                if pd.isna(extracted):
                    return default
                return as_type(extracted) if default is not None else extracted
            
            else:
                if pd.isna(val):
                    return default
                return as_type(val) if default is not None else val
                
        except (ValueError, TypeError, AttributeError):
            return default
    
    @staticmethod
    def _safe_bool_check(val, default=False):
        """Safely check boolean values"""
        extracted = CompleteRecommendationEngine._safe_extract_value(val, default, bool)
        if isinstance(extracted, (int, float)):
            return bool(extracted)
        return extracted
    
    def _create_error_response(self, request_id, client_id, timestamp, error_msg):
        """Create error response"""
        return EnhancedRecommendationResponse(
            request_id=request_id,
            client_id=client_id,
            timestamp=timestamp,
            recommendation_method='error',
            is_cold_start=False,
            vehicles=[],
            chauffeurs=[],
            pairs=[],
            total_candidates_vehicles=0,
            total_candidates_chauffeurs=0,
            filters_applied=[],
            warnings=[error_msg]
        )
    
    def _create_empty_response(self, request_id, client_id, timestamp, method,
                              is_cold_start, filters_applied, warnings_list):
        """Create empty response when no recommendations found"""
        return EnhancedRecommendationResponse(
            request_id=request_id,
            client_id=client_id,
            timestamp=timestamp,
            recommendation_method=method,
            is_cold_start=is_cold_start,
            vehicles=[],
            chauffeurs=[],
            pairs=[],
            total_candidates_vehicles=0,
            total_candidates_chauffeurs=0,
            filters_applied=filters_applied,
            warnings=warnings_list
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_complete_engine(pipeline_results):
    """
    Create complete engine from pipeline results
    
    Args:
        pipeline_results: Output from run_complete_pipeline()
    
    Returns:
        CompleteRecommendationEngine instance
    """
    from model_training import sparse_to_torch, load_graph_data, device
    
    model = pipeline_results['model']
    step2_results = pipeline_results['step2_results']
    step3_results = pipeline_results['step3_results']
    step4_results = pipeline_results['step4_results']
    
    data = load_graph_data(step4_results)
    adj_sparse = sparse_to_torch(data['normalized_adj'], device)
    
    engine = CompleteRecommendationEngine(
        model=model,
        step2_results=step2_results,
        step3_results=step3_results,
        step4_results=step4_results,
        data=data,
        adj_sparse=adj_sparse
    )
    
    return engine


if __name__ == "__main__":
    logger.info("Complete Recommendation API - Ready")
    logger.debug("\nKey Features:")
    logger.debug("   LightGCN for in-model clients")
    logger.debug("   Similarity-based cold-start")
    logger.debug("   Comprehensive filtering")
    logger.debug("   Detailed JSON responses")
    logger.info("   Production-ready")
    # Load port mappings

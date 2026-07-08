"""
models.py - Data models, request classes, and constants for the Recommendation Combiner API

This module contains:
- Feature weight configurations
- CombinerRequest dataclass for unified request handling
- Utility functions for data parsing
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE WEIGHT CONFIGURATIONS
# =============================================================================

class FeatureWeights:
    """
    Centralized configuration for all feature weights used in score computation.
    These weights determine how different features contribute to the final score.
    """
    
    # LightGCN feature weights for computing weighted composite score
    LIGHTGCN: Dict[str, float] = {
        'cv_score': 0.25,
        'cc_score': 0.25,
        'vc_score': 0.25,
        'combined_score': 0.10,
        'vehicle_completion_rate': 0.05,
        'chauffeur_completion_rate': 0.05,
        'vehicle_repeat_client_rate': 0.025,
        'chauffeur_repeat_client_rate': 0.025
    }
    
    # Collaborative Filtering match score weights
    CF_MATCH: Dict[str, float] = {
        'cf_score': 0.4,
        'overall_match': 0.3,
        'quality_score': 0.2,
        'requirement_match': 0.1
    }
    
    # Default fusion weights (used when grid search weights unavailable)
    DEFAULT_FUSION: Dict[str, float] = {
        'alpha1_rrf': 0.3,
        'alpha2_gcn': 0.3,
        'alpha3_item_sim': 0.1,
        'alpha4_user_sim': 0.1,
        'alpha5_confidence': 0.1
    }
    
    @classmethod
    def get_lightgcn_weights(cls) -> Dict[str, float]:
        """Return a copy of LightGCN feature weights"""
        return cls.LIGHTGCN.copy()
    
    @classmethod
    def get_cf_weights(cls) -> Dict[str, float]:
        """Return a copy of CF match weights"""
        return cls.CF_MATCH.copy()
    
    @classmethod
    def get_default_fusion_weights(cls) -> Dict[str, float]:
        """Return a copy of default fusion weights"""
        return cls.DEFAULT_FUSION.copy()


# =============================================================================
# RRF CONFIGURATION
# =============================================================================

class RRFConfig:
    """Configuration for Reciprocal Rank Fusion"""
    K: int = 30  # RRF constant


# =============================================================================
# COMBINER REQUEST MODEL
# =============================================================================

@dataclass
class CombinerRequest:
    """
    Unified request format for the combiner API.
    
    This dataclass encapsulates all possible request parameters and provides
    methods to convert to the specific formats required by each downstream API.
    
    Attributes:
        client_id: Unique identifier for the client
        no_of_pax: Number of passengers
        ref_port: Optional port filtering
        appointment_date: Optional appointment date
        pickup_suburb: Optional pickup location
        dropoff_suburb: Optional dropoff location
        estimated_distance: Optional estimated trip distance
        estimated_duration: Optional estimated trip duration in minutes
        no_of_bags: Optional number of bags
        ref_vehicle_type: Optional vehicle type reference
        min_vehicle_level: Optional minimum vehicle level requirement
        require_red_zone_auth: Whether red zone authorization is required
        require_baby_seat: Whether baby seat is required
        excluded_vehicle_ids: List of vehicle IDs to exclude
        excluded_chauffeur_ids: List of chauffeur IDs to exclude
        preferred_chauffeur_ids: List of preferred chauffeur IDs
        top_k: Number of recommendations to request from each API
    """
    client_id: int
    no_of_pax: int
    ref_port: Optional[int] = None
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
        """
        Convert to LightGCN API format.
        
        Returns:
            Dict formatted for the LightGCN recommendation API
        """
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
        """
        Convert to Collaborative API format.
        
        Returns:
            Dict formatted for the Collaborative Filtering API
        """
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
        """
        Convert to User Content API format.
        
        Returns:
            Dict formatted for the User Content-based recommendation API
        """
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
        """
        Convert to Item Content API format.
        
        Returns:
            Dict formatted for the Item Content-based recommendation API
        """
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_comma_separated_ids(id_string: Optional[str]) -> Optional[List[int]]:
    """
    Parse comma-separated string into list of integers.
    
    Args:
        id_string: Comma-separated string of IDs (e.g., "1,2,3")
        
    Returns:
        List of integers, or None if parsing fails
    """
    if not id_string or pd.isna(id_string):
        return None
    try:
        return [int(x.strip()) for x in str(id_string).split(',') if x.strip()]
    except (ValueError, AttributeError):
        return None


# For backward compatibility with existing code
LIGHTGCN_FEATURE_WEIGHTS = FeatureWeights.LIGHTGCN
CF_MATCH_WEIGHTS = FeatureWeights.CF_MATCH
RRF_K = RRFConfig.K

# Collaborative Filtering API Models - Production Ready

from pydantic import BaseModel, Field
try:
    from pydantic import field_validator  # Pydantic v2
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import validator  # Pydantic v1 fallback
    PYDANTIC_V2 = False

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class PriceSensitivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VehicleLevel(str, Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    LUXURY = "luxury"
    EXECUTIVE = "executive"


class Location(BaseModel):
    suburb_id: Optional[int] = None
    postcode: Optional[str] = Field(default=None, max_length=20)
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class BookingContext(BaseModel):
    appointment_date: datetime
    pickup_location: Optional[Location] = None
    dropoff_location: Optional[Location] = None
    estimated_distance_km: Optional[float] = Field(default=None, ge=0, le=10000)
    estimated_duration_minutes: Optional[int] = Field(default=None, ge=0, le=14400)
    no_of_pax: Optional[int] = Field(default=1, ge=1, le=80)
    ref_port: Optional[int] = Field(default=None, description="Port ID for filtering chauffeurs and vehicles")

    if PYDANTIC_V2:
        @field_validator('estimated_distance_km')
        @classmethod
        def validate_distance(cls, v):
            if v is not None and v < 0:
                raise ValueError("Distance cannot be negative")
            if v is not None and v > 1000:
                raise ValueError("Distance unreasonably large (>1000km)")
            return v
        
        @field_validator('no_of_pax')
        @classmethod
        def validate_passengers(cls, v):
            if v is not None and v < 1:
                raise ValueError("Must have at least 1 passenger")
            if v is not None and v > 20:
                raise ValueError("Passenger count suspiciously high (>20)")
            return v
    else:
        @validator('estimated_distance_km')
        def validate_distance(cls, v):
            if v is not None and v < 0:
                raise ValueError("Distance cannot be negative")
            if v is not None and v > 1000:
                raise ValueError("Distance unreasonably large (>1000km)")
            return v
        
        @validator('no_of_pax')
        def validate_passengers(cls, v):
            if v is not None and v < 1:
                raise ValueError("Must have at least 1 passenger")
            if v is not None and v > 20:
                raise ValueError("Passenger count suspiciously high (>20)")
            return v


class Requirements(BaseModel):
    """
    Hard requirements for vehicle filtering
    
    NOTE: For seat capacity, use no_of_pax in BookingContext instead.
    The system will automatically ensure vehicle has enough seats for passengers.
    """
    vehicle_type_id: Optional[int] = Field(
        default=None, 
        ge=1,
        description="STRICT: Exact vehicle type ID required. Only vehicles with this exact type will be returned."
    )
    min_vehicle_level: Optional[int] = Field(
        default=None, 
        ge=0,
        description="Minimum vehicle level: 0=standard, 1=premium, 3=luxury, 4=executive. Returns this level and higher."
    )
    required_features: Optional[List[str]] = Field(
        default_factory=list,
        description="Required features (baby_seat, red_zone_authorized, etc.)"
    )
    
    if PYDANTIC_V2:
        @field_validator('required_features')
        @classmethod
        def validate_features(cls, v):
            if v is None:
                return v
            valid_features = {
                'baby_seat', 'wheelchair_accessible', 'red_zone_authorized',
                'luxury_amenities', 'wifi', 'privacy_partition'
            }
            invalid = set(v) - valid_features
            if invalid:
                raise ValueError(f"Invalid features: {invalid}. Valid features: {valid_features}")
            return v
    else:
        @validator('required_features')
        def validate_features(cls, v):
            if v is None:
                return v
            valid_features = {
                'baby_seat', 'wheelchair_accessible', 'red_zone_authorized',
                'luxury_amenities', 'wifi', 'privacy_partition'
            }
            invalid = set(v) - valid_features
            if invalid:
                raise ValueError(f"Invalid features: {invalid}. Valid features: {valid_features}")
            return v


class Preferences(BaseModel):
    preferred_chauffeurs: Optional[List[int]] = Field(default_factory=list)
    excluded_chauffeurs: Optional[List[int]] = Field(default_factory=list)
    preferred_vehicles: Optional[List[int]] = Field(default_factory=list)
    excluded_vehicles: Optional[List[int]] = Field(default_factory=list)


class Filters(BaseModel):
    only_available: bool = False
    min_rating: Optional[float] = Field(default=3.0, ge=1.0, le=5.0)
    max_days_since_last_booking: Optional[int] = Field(default=180, ge=0, le=730)
    require_active_certifications: bool = False
    price_sensitivity: PriceSensitivity = PriceSensitivity.MEDIUM
    min_completion_rate: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)


class RecommendationRequest(BaseModel):
    client_id: int = Field(..., ge=1)
    booking_context: BookingContext
    requirements: Optional[Requirements] = Field(default_factory=Requirements)
    preferences: Optional[Preferences] = Field(default_factory=Preferences)
    filters: Optional[Filters] = Field(default_factory=Filters)
    n_recommendations: int = Field(default=10, ge=1, le=100)
    save_to_file: bool = Field(default=False, description="Save recommendations to JSON file")

    if PYDANTIC_V2:
        @field_validator('n_recommendations')
        @classmethod
        def validate_n_recommendations(cls, v):
            if v > 150:
                raise ValueError("Too many recommendations requested (max 100)")
            return v
    else:
        @validator('n_recommendations')
        def validate_n_recommendations(cls, v):
            if v > 150:
                raise ValueError("Too many recommendations requested (max 100)")
            return v
    
    if PYDANTIC_V2:
        model_config = {
            "json_schema_extra": {
                "example": {
                    "client_id": 12345,
                    "booking_context": {
                        "appointment_date": "2025-11-15T14:30:00",
                        "pickup_location": {
                            "suburb_id": 100,
                            "postcode": "3000"
                        },
                        "no_of_pax": 3,
                        "ref_port": 5         
                    },
                    "requirements": {
                        "vehicle_type_id": 10
                    },
                    "n_recommendations": 10
                }
            }
        }
    else:
        class Config:
            schema_extra = {
                "example": {
                    "client_id": 12345,
                    "booking_context": {
                        "appointment_date": "2025-11-15T14:30:00",
                        "pickup_location": {
                            "suburb_id": 100,
                            "postcode": "3000"
                        },
                        "no_of_pax": 3,
                        "ref_port": 5         
                    },
                    "requirements": {
                        "vehicle_type_id": 10
                    },
                    "n_recommendations": 10
                }
            }


class ChauffeurVehicleRecommendation(BaseModel):
    rank: int
    chauffeur_id: int
    vehicle_id: int
    pair_id: str = Field(..., max_length=50)
    predicted_rating: float
    confidence: float
    estimated_price: Optional[float] = None
    
    # Chauffeur details
    chauffeur_name: Optional[str] = Field(default=None, max_length=200)
    chauffeur_rating: Optional[float] = None
    chauffeur_experience_days: Optional[int] = None
    
    # Vehicle details
    vehicle_make: Optional[str] = Field(default=None, max_length=100)
    vehicle_model: Optional[str] = Field(default=None, max_length=100)
    vehicle_year: Optional[int] = None
    vehicle_type: Optional[str] = Field(default=None, max_length=100)
    vehicle_type_id: Optional[int] = None
    seat_capacity: Optional[int] = None
    
    # Match scores
    match_scores: Dict[str, float] = Field(default_factory=dict)
    
    if PYDANTIC_V2:
        model_config = {
            "json_schema_extra": {
                "example": {
                    "rank": 1,
                    "chauffeur_id": 101,
                    "vehicle_id": 301,
                    "pair_id": "101_301",
                    "predicted_rating": 4.85,
                    "confidence": 0.92,
                    "estimated_price": 125.50,
                    "chauffeur_name": "John Smith",
                    "vehicle_make": "Mercedes",
                    "vehicle_model": "E-Class",
                    "vehicle_type": "Premium Sedan",
                    "vehicle_type_id": 10,
                    "seat_capacity": 4,
                    "match_scores": {
                        "cf_score": 4.85,
                        "overall_match": 0.85,
                        "quality_score": 0.90,
                        "requirement_match": 1.0
                    }
                }
            }
        }
    else:
        class Config:
            schema_extra = {
                "example": {
                    "rank": 1,
                    "chauffeur_id": 101,
                    "vehicle_id": 301,
                    "pair_id": "101_301",
                    "predicted_rating": 4.85,
                    "confidence": 0.92,
                    "estimated_price": 125.50,
                    "chauffeur_name": "John Smith",
                    "vehicle_make": "Mercedes",
                    "vehicle_model": "E-Class",
                    "vehicle_type": "Premium Sedan",
                    "vehicle_type_id": 10,
                    "seat_capacity": 4,
                    "match_scores": {
                        "cf_score": 4.85,
                        "overall_match": 0.85,
                        "quality_score": 0.90,
                        "requirement_match": 1.0
                    }
                }
            }


class RecommendationResponse(BaseModel):
    client_id: int
    request_timestamp: datetime
    recommendations: List[ChauffeurVehicleRecommendation]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    if PYDANTIC_V2:
        model_config = {
            "json_schema_extra": {
                "example": {
                    "client_id": 12345,
                    "request_timestamp": "2025-10-31T10:30:00",
                    "recommendations": [],
                    "metadata": {
                        "total_candidates": 50,
                        "filtered_candidates": 35,
                        "processing_time_ms": 245
                    }
                }
            }
        }
    else:
        class Config:
            schema_extra = {
                "example": {
                    "client_id": 12345,
                    "request_timestamp": "2025-10-31T10:30:00",
                    "recommendations": [],
                    "metadata": {
                        "total_candidates": 50,
                        "filtered_candidates": 35,
                        "processing_time_ms": 245
                    }
                }
            }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_pairs: int
    total_clients: int
    timestamp: datetime
    can_predict: Optional[bool] = None
    model_age_days: Optional[int] = None
    database_connected: Optional[bool] = None
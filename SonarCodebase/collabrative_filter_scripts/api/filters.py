# collabrative 

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from datetime import datetime, timedelta
import logging
import re
from config.config import Config

from api.models import RecommendationRequest, PriceSensitivity

logger = logging.getLogger(__name__)


class PairIDValidationError(Exception):
    """Raised when pair ID format is invalid"""
    pass


class RecommendationFilter:
    """Apply filters to candidate recommendations"""
    
    # Valid pair ID format: integer_integer (e.g., "101_302")
    PAIR_ID_PATTERN = re.compile(r'^(\d+)_(\d+)$')
    
    def __init__(self, 
                 pair_features: pd.DataFrame,
                 chauffeur_data: pd.DataFrame,
                 vehicle_data: pd.DataFrame,
                 vehicle_type_data: pd.DataFrame,
                 chauffeur_status_data: pd.DataFrame,
                 client_data: pd.DataFrame,
                 strict_mode: bool = False,
                 port_mappings: dict = None,
                 contractor_mappings: dict = None):
        """
        Args:
            strict_mode: If True, filters are strict (fail on missing data)
                        If False, filters are lenient (assume OK on missing data)
                        NOTE: Vehicle type is ALWAYS strict when requested!
        """
        self.contractor_mappings = contractor_mappings or {
        'chauffeur_contractors': {},
        'vehicle_contractors': {}
    }
        self.port_mappings = port_mappings or {
            'port_vehicles': {},
            'port_chauffeurs': {},
            'vehicle_ports': {},
            'chauffeur_ports': {}
        }
        self.pair_features = pair_features
        self.chauffeur_data = chauffeur_data
        self.vehicle_data = vehicle_data
        self.vehicle_type_data = vehicle_type_data
        self.chauffeur_status_data = chauffeur_status_data
        self.client_data = client_data
        self.strict_mode = strict_mode
        
        # Create lookup indices
        self._create_lookup_indices()
        self.config = Config()

    
    def _parse_pair_id(self, pair_id: str) -> Optional[Tuple[int, int]]:
        """
        Parse and validate pair ID format
        
        Args:
            pair_id: String in format "chauffeur_id_vehicle_id"
            
        Returns:
            Tuple of (chauffeur_id, vehicle_id) or None if invalid
        """
        match = self.PAIR_ID_PATTERN.match(str(pair_id))
        if match:
            try:
                chauffeur_id = int(match.group(1))
                vehicle_id = int(match.group(2))
                return chauffeur_id, vehicle_id
            except ValueError:
                logger.warning(f"Failed to convert pair_id to integers: {pair_id}")
                return None
        else:
            logger.warning(f"Invalid pair_id format: {pair_id} (expected format: '123_456')")
            return None
    
    def _create_lookup_indices(self):
        """Create fast lookup indices with validation"""
        logger.info("   Creating lookup indices...")
        
        # Parse pair IDs with validation
        self.pair_to_chauffeur = {}
        self.pair_to_vehicle = {}
        
        invalid_pairs = []
        for pair_id in self.pair_features.index:
            parsed = self._parse_pair_id(pair_id)
            if parsed:
                chauffeur_id, vehicle_id = parsed
                self.pair_to_chauffeur[pair_id] = chauffeur_id
                self.pair_to_vehicle[pair_id] = vehicle_id
            else:
                invalid_pairs.append(pair_id)
        
        if invalid_pairs:
            logger.warning(f"   Found {len(invalid_pairs)} pairs with invalid ID format")
            if len(invalid_pairs) <= 10:
                logger.warning(f"   Invalid pairs: {invalid_pairs}")
        
        logger.info(f"   Parsed {len(self.pair_to_chauffeur)} valid pair IDs")
        
        # Chauffeur index with safe conversion
        if 'ID' in self.chauffeur_data.columns:
            self.chauffeur_data['ID'] = pd.to_numeric(self.chauffeur_data['ID'], errors='coerce')
            self.chauffeur_data = self.chauffeur_data.dropna(subset=['ID'])
            self.chauffeur_data['ID'] = self.chauffeur_data['ID'].astype(int)
            self.chauffeur_idx = self.chauffeur_data.set_index('ID')
        else:
            self.chauffeur_idx = self.chauffeur_data
        
        # Vehicle index with safe conversion
        if 'ID' in self.vehicle_data.columns:
            self.vehicle_data['ID'] = pd.to_numeric(self.vehicle_data['ID'], errors='coerce')
            self.vehicle_data = self.vehicle_data.dropna(subset=['ID'])
            self.vehicle_data['ID'] = self.vehicle_data['ID'].astype(int)
            self.vehicle_idx = self.vehicle_data.set_index('ID')
        else:
            self.vehicle_idx = self.vehicle_data
        
        # Vehicle type index with safe conversion
        if 'ID' in self.vehicle_type_data.columns:
            self.vehicle_type_data['ID'] = pd.to_numeric(self.vehicle_type_data['ID'], errors='coerce')
            self.vehicle_type_data = self.vehicle_type_data.dropna(subset=['ID'])
            self.vehicle_type_data['ID'] = self.vehicle_type_data['ID'].astype(int)
            self.vehicle_type_idx = self.vehicle_type_data.set_index('ID')
        else:
            self.vehicle_type_idx = self.vehicle_type_data
        
        # DIAGNOSTIC: Log vehicle type index
        logger.info(f"   Indexed {len(self.vehicle_type_idx)} vehicle types")
        logger.info(f"   Vehicle type ID range: {self.vehicle_type_idx.index.min()} - {self.vehicle_type_idx.index.max()}")
        
        # Chauffeur status index
        if 'RefChauffeur' in self.chauffeur_status_data.columns:
            self.status_by_chauffeur = self.chauffeur_status_data.set_index('RefChauffeur')
        else:
            self.status_by_chauffeur = self.chauffeur_status_data
        
        # Client index
        if 'ID' in self.client_data.columns:
            self.client_idx = self.client_data.set_index('ID')
        else:
            self.client_idx = self.client_data
        
        logger.info(f"   Indexed {len(self.chauffeur_idx)} chauffeurs")
        logger.info(f"   Indexed {len(self.vehicle_idx)} vehicles")
    
    def _safe_get_int(self, df: pd.DataFrame, idx, column: str, default: int) -> int:
        """Safe integer extraction with error handling"""
        try:
            value = df.loc[idx, column]
            if pd.isna(value):
                return default
            return int(float(value))  # Convert via float to handle strings like "4.0"
        except (KeyError, ValueError, TypeError):
            return default
    
    def _safe_get_float(self, df: pd.DataFrame, idx, column: str, default: float) -> float:
        """Safe float extraction with error handling"""
        try:
            value = df.loc[idx, column]
            if pd.isna(value):
                return default
            return float(value)
        except (KeyError, ValueError, TypeError):
            return default
    
    def _safe_get_bool(self, df: pd.DataFrame, idx, column: str, default: bool) -> bool:
        """Safe boolean extraction with error handling"""
        try:
            value = df.loc[idx, column]
            if pd.isna(value):
                return default
            return bool(value)
        except (KeyError, ValueError, TypeError):
            return default
    
    def filter_candidates(self, 
                         candidate_pairs: List[str],
                         request: RecommendationRequest) -> Tuple[List[str], Dict[str, int]]:
        """
        Apply all filters to candidate pairs
        
        Returns:
            (filtered_pairs, filter_stats)
        """
        stats = {
            'initial_candidates': len(candidate_pairs),
            'after_contractor_filter': 0,  #    ADD THIS
            'after_port_filter': 0,  # NEW
            'after_hard_requirements': 0,
            'after_preferences': 0,
            'after_rating': 0,
            'final_candidates': 0
        }
        
        logger.info(f"🔍 Filtering {len(candidate_pairs)} candidates (strict_mode={self.strict_mode})...")
        
        # Start with all candidates
        candidates = candidate_pairs.copy()
          #    ADD THIS - FILTER 0: CONTRACTOR FILTER (MOST CRITICAL)
        candidates = self._apply_contractor_filter(candidates)
        logger.info(f"   After contractor filter: {len(candidates)}")
        stats['after_contractor_filter'] = len(candidates)
        
        if len(candidates) == 0:
            logger.error("     No pairs with matching contractors")
            return [], stats
        
        # 0. PORT FILTER - FIRST AND STRICTEST
        if request.booking_context.ref_port:
            candidates = self._apply_port_filter(candidates, request.booking_context.ref_port)
            logger.info(f"   After port filter: {len(candidates)}")
            stats['after_port_filter'] = len(candidates)
            
            if len(candidates) == 0:
                logger.error(f"     No pairs available for port {request.booking_context.ref_port}")
                return [], stats
        
        # 1. Hard requirements (OPTIONAL - only if specified)
        if self._has_hard_requirements(request):
            candidates = self._apply_hard_requirements(candidates, request)
            logger.info(f"   After hard requirements: {len(candidates)}")
        else:
            logger.info("   Skipping hard requirements (none specified)")
        stats['after_hard_requirements'] = len(candidates)
        
        # 2. Preferences (preferred/excluded) - ALWAYS APPLY
        candidates = self._apply_preferences(candidates, request)
        stats['after_preferences'] = len(candidates)
        logger.info(f"   After preferences: {len(candidates)}")
        
        # If we have no candidates left, return ALL original candidates
        if len(candidates) == 0:
            logger.warning("     All candidates filtered out by requirements")
            if request.requirements and request.requirements.vehicle_type_id:
                logger.error(f"   No vehicles found with vehicle_type_id={request.requirements.vehicle_type_id}")
            if request.booking_context.no_of_pax:
                logger.error(f"   Required capacity: {request.booking_context.no_of_pax} passengers")
            if request.requirements and request.requirements.min_vehicle_level:
                logger.error(f"   Required minimum level: {request.requirements.min_vehicle_level}")
            
            stats['final_candidates'] = 0
            # Return empty - API will handle this

            if self._has_hard_requirements(request):
                logger.info("   Hard requirements specified - NOT falling back to originals")
                return [], stats
            else:
                logger.warning("   No hard requirements - this shouldn't happen")
                return [], stats  
                  
        # 3. Rating filter (OPTIONAL)
        if request.filters.min_rating and request.filters.min_rating > 1.0:
            before = len(candidates)
            candidates = self._filter_by_rating(candidates, request.filters.min_rating)
            logger.info(f"   After rating: {len(candidates)} (filtered {before - len(candidates)})")
            stats['after_rating'] = len(candidates)
        
        # Final fallback
        if len(candidates) == 0:
            if not self._has_hard_requirements(request):
                logger.warning("   All candidates filtered out by soft filters - returning top 10 originals")
                candidates = candidate_pairs[:10]
            else:
                logger.error("   All candidates filtered out by hard requirements - returning empty")
        
        stats['final_candidates'] = len(candidates)
        
        return candidates, stats
    
    def _apply_contractor_filter(self, pairs: List[str]) -> List[str]:
        """
        Filter pairs to only those where chauffeur and vehicle belong to same contractor
        CRITICAL: This is the FIRST filter - ensures valid business logic
        """
        if not self.contractor_mappings or not self.contractor_mappings['chauffeur_contractors']:
            logger.warning("     No contractor mappings available - skipping contractor filter")
            return pairs
        
        chauffeur_contractors = self.contractor_mappings['chauffeur_contractors']
        vehicle_contractors = self.contractor_mappings['vehicle_contractors']
        
        filtered = []
        rejected_no_chauffeur = 0
        rejected_no_vehicle = 0
        rejected_mismatch = 0
        
        for pair_id in pairs:
            chauffeur_id = self.pair_to_chauffeur.get(pair_id)
            vehicle_id = self.pair_to_vehicle.get(pair_id)
            
            if not chauffeur_id or not vehicle_id:
                continue
            
            # Get contractors
            chauffeur_contractor = chauffeur_contractors.get(chauffeur_id)
            vehicle_contractor = vehicle_contractors.get(vehicle_id)
            
            # Reject if either is missing
            if not chauffeur_contractor:
                rejected_no_chauffeur += 1
                continue
            
            if not vehicle_contractor:
                rejected_no_vehicle += 1
                continue
            
            # STRICT: Must match
            if chauffeur_contractor != vehicle_contractor:
                rejected_mismatch += 1
                continue
            
            # Passed - same contractor
            filtered.append(pair_id)
        
        logger.info(f"   Rejected: {rejected_no_chauffeur} (no chauffeur contractor), "
                    f"{rejected_no_vehicle} (no vehicle contractor), "
                    f"{rejected_mismatch} (contractor mismatch)")
        
        return filtered




    def _apply_port_filter(self, pairs: List[str], ref_port: int) -> List[str]:
        """
        Filter pairs to only those belonging to the specified port
        CRITICAL: Both chauffeur AND vehicle must belong to the port
        
        Logic:
        - Vehicle: vehicle_id → check if in port_vehicles[ref_port]
        - Chauffeur: chauffeur_id → contractor_id (via chauffeur_contractors) → port_id (via contractor_ports) → compare with ref_port
        """
        if not ref_port or not self.port_mappings:
            return pairs
        
        # Get valid vehicles for this port
        valid_vehicles = set(self.port_mappings['port_vehicles'].get(ref_port, []))
        
        # Get contractor_ports mapping (Contractor → Port)
        contractor_ports = self.port_mappings.get('contractor_ports', {})
        
        # Get chauffeur → contractor mapping
        chauffeur_contractors = self.contractor_mappings.get('chauffeur_contractors', {})
        
        logger.info(f"   Port {ref_port}: {len(valid_vehicles)} vehicles, {len(contractor_ports)} contractors mapped")
        
        if not valid_vehicles:
            logger.warning(f"     Port {ref_port} has no vehicles")
            return []
        
        filtered = []
        rejected_chauffeur_no_contractor = 0
        rejected_chauffeur_wrong_port = 0
        rejected_vehicle = 0
        
        for pair_id in pairs:
            chauffeur_id = self.pair_to_chauffeur.get(pair_id)
            vehicle_id = self.pair_to_vehicle.get(pair_id)
            
            if not chauffeur_id or not vehicle_id:
                continue
            
            # Check vehicle belongs to port
            if vehicle_id not in valid_vehicles:
                rejected_vehicle += 1
                continue
            
            # Check chauffeur belongs to port via: Chauffeur → Contractor → Port
            chauffeur_contractor = chauffeur_contractors.get(chauffeur_id)
            if not chauffeur_contractor:
                # No contractor info for chauffeur - reject (strict mode)
                rejected_chauffeur_no_contractor += 1
                continue
            
            chauffeur_port = contractor_ports.get(chauffeur_contractor)
            if chauffeur_port != ref_port:
                rejected_chauffeur_wrong_port += 1
                continue
            
            # Both chauffeur and vehicle belong to the requested port
            filtered.append(pair_id)
        
        logger.info(f"   Rejected: {rejected_chauffeur_no_contractor} (chauffeur no contractor), "
                    f"{rejected_chauffeur_wrong_port} (chauffeur wrong port), "
                    f"{rejected_vehicle} (vehicle not in port)")
        return filtered
    def _has_hard_requirements(self, request: RecommendationRequest) -> bool:
        """Check if request has any hard requirements"""
        reqs = request.requirements
        return (
            reqs.vehicle_type_id is not None or
            reqs.min_vehicle_level is not None or
            (reqs.required_features and len(reqs.required_features) > 0)
        )
    
    def _apply_hard_requirements(self, pairs: List[str], request: RecommendationRequest) -> List[str]:
        """
        FIXED: Filter by hard requirements with STRICT vehicle type enforcement
        
        Vehicle type is ALWAYS strictly enforced when requested, regardless of strict_mode.
        This prevents returning wrong vehicle types.
        """
        filtered = []
        
        # CRITICAL: Vehicle type is ALWAYS strict when requested
        vehicle_type_is_strict = request.requirements.vehicle_type_id is not None
        
        rejected_stats = {
            'unknown_vehicle': 0,
            'vehicle_not_indexed': 0,
            'vehicle_type_missing': 0,
            'vehicle_type_not_indexed': 0,
            'wrong_vehicle_type': 0,
            'insufficient_capacity': 0,
            'insufficient_level': 0,
            'missing_features': 0
        }
        
        for pair_id in pairs:
            vehicle_id = self.pair_to_vehicle.get(pair_id)
            if not vehicle_id:
                rejected_stats['unknown_vehicle'] += 1
                if self.strict_mode or vehicle_type_is_strict:
                    continue  # Skip unknown pairs
                else:
                    filtered.append(pair_id)  # Keep in lenient mode
                    continue
            
            if vehicle_id not in self.vehicle_idx.index:
                rejected_stats['vehicle_not_indexed'] += 1
                if self.strict_mode or vehicle_type_is_strict:
                    continue
                else:
                    filtered.append(pair_id)
                    continue
            
           
            
            # Get vehicle type with safe conversion
            vehicle_type_id = self._safe_get_int(self.vehicle_idx, vehicle_id, 'RefVehicleType', -1)
            
            # CRITICAL: If vehicle type is requested, missing vehicle type = REJECT
            if vehicle_type_id == -1:
                rejected_stats['vehicle_type_missing'] += 1
                if self.strict_mode or vehicle_type_is_strict:
                    logger.debug(f"Rejecting {pair_id}: vehicle {vehicle_id} has no vehicle type")
                    continue  # Cannot verify - reject
                else:
                    filtered.append(pair_id)
                    continue
            
            # CRITICAL: If vehicle type is requested, missing from index = REJECT  
            if vehicle_type_id not in self.vehicle_type_idx.index:
                rejected_stats['vehicle_type_not_indexed'] += 1
                if self.strict_mode or vehicle_type_is_strict:
                    logger.debug(f"Rejecting {pair_id}: vehicle type {vehicle_type_id} not in index")
                    continue  # Cannot verify - reject
                else:
                    filtered.append(pair_id)
                    continue
            
            
            
            # Check vehicle type requirement (ALWAYS STRICT - this is a hard requirement)
            if request.requirements.vehicle_type_id:
                if vehicle_type_id != request.requirements.vehicle_type_id:
                    rejected_stats['wrong_vehicle_type'] += 1
                    logger.debug(f"Rejecting {pair_id}: vehicle type {vehicle_type_id} != requested {request.requirements.vehicle_type_id}")
                    continue  # Wrong vehicle type - reject
            
            # Check seat capacity with safe conversion
            if request.booking_context.no_of_pax:
                seat_capacity = self._safe_get_int(
                    self.vehicle_type_idx, 
                    vehicle_type_id, 
                    'SeatCapacity', 
                    0
                )
                # Vehicle must have at least as many seats as passengers
                if seat_capacity < request.booking_context.no_of_pax:
                    rejected_stats['insufficient_capacity'] += 1
                    continue
            
            # Check vehicle level
            if request.requirements.min_vehicle_level is not None:
                vehicle_level = self._safe_get_int(
                    self.vehicle_type_idx, 
                    vehicle_type_id, 
                    'VehicleLevel', 
                    -1
                )
                # Filter out if vehicle level is LESS than requested minimum
                if vehicle_level < request.requirements.min_vehicle_level:
                    rejected_stats['insufficient_level'] += 1
                    continue
            
            # Check required features
            if request.requirements.required_features:
                chauffeur_id = self.pair_to_chauffeur.get(pair_id)
                if chauffeur_id and not self._has_required_features(chauffeur_id, vehicle_id, request.requirements.required_features):
                    rejected_stats['missing_features'] += 1
                    continue
            
            # Passed all checks
            filtered.append(pair_id)
        
        # Log rejection stats
        if sum(rejected_stats.values()) > 0:
            logger.info("   Rejection breakdown:")
            for reason, count in rejected_stats.items():
                if count > 0:
                    logger.info(f"     - {reason}: {count}")
        
        return filtered
    
    def _has_required_features(self, chauffeur_id: int, vehicle_id: int, features: List[str]) -> bool:
        """Check if chauffeur-vehicle has required features"""
        if not chauffeur_id or chauffeur_id not in self.chauffeur_idx.index:
            return not self.strict_mode  # Lenient: assume OK if data missing
        
        chauffeur = self.chauffeur_idx.loc[chauffeur_id]
        
        for feature in features:
            if feature == 'baby_seat':
                if not self._safe_get_bool(self.chauffeur_idx, chauffeur_id, 'BabySeatTraining', False):
                    return False
            elif feature == 'red_zone_authorized':
                if not self._safe_get_bool(self.chauffeur_idx, chauffeur_id, 'RedZoneAuthorised', False):
                    return False
        
        return True
    
    def _apply_preferences(self, pairs: List[str], request: RecommendationRequest) -> List[str]:
        """Apply user preferences (boost preferred, exclude blacklisted)"""
        if not request.preferences:
            return pairs
        
        filtered = []
        
        excluded_chauffeurs = set(request.preferences.excluded_chauffeurs or [])
        excluded_vehicles = set(request.preferences.excluded_vehicles or [])
        
        for pair_id in pairs:
            chauffeur_id = self.pair_to_chauffeur.get(pair_id)
            vehicle_id = self.pair_to_vehicle.get(pair_id)
            
            # Exclude blacklisted
            if chauffeur_id in excluded_chauffeurs or vehicle_id in excluded_vehicles:
                continue
            
            filtered.append(pair_id)
        
        return filtered
    
    def _filter_by_rating(self, pairs: List[str], min_rating: float) -> List[str]:
        """Filter by minimum average rating"""
        filtered = []
        
        for pair_id in pairs:
            if pair_id not in self.pair_features.index:
                if not self.strict_mode:
                    filtered.append(pair_id)
                continue
            
            try:
                completion_rate = self._safe_get_float(self.pair_features, pair_id, 'completion_rate', 0.8)
                
                # Use completion rate as proxy for quality
                estimated_rating = 1 + (completion_rate * 4)  # Scale 0-1 to 1-5
                
                if estimated_rating >= min_rating:
                    filtered.append(pair_id)
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Error checking rating for {pair_id}: {e}")
                if not self.strict_mode:
                    filtered.append(pair_id)
        
        return filtered
    
    def calculate_match_scores(self, 
                              pair_id: str, 
                              request: RecommendationRequest) -> Dict[str, float]:
        """Calculate detailed match scores with data quality"""
        scores = {}
        
        chauffeur_id = self.pair_to_chauffeur.get(pair_id)
        vehicle_id = self.pair_to_vehicle.get(pair_id)
        
        # Preference match
        scores['preference_match'] = self._calculate_preference_score(chauffeur_id, vehicle_id, request)
        
        # Requirement match
        scores['requirement_match'] = self._calculate_requirement_score(vehicle_id, request)
        
        # Quality score with data quality consideration
        scores['quality_score'] = self._calculate_quality_score(pair_id)
        
        # Data quality score
        scores['data_quality'] = self._calculate_data_quality(pair_id)
        
        # Overall match score (weighted by data quality)
        base_match = np.mean(list(scores.values()))
        scores['overall_match'] = base_match * scores['data_quality']
        
        return scores
    
    def _calculate_data_quality(self, pair_id: str) -> float:
        """Calculate data quality/freshness score"""
        if pair_id not in self.pair_features.index:
            return self.config.MIN_DATA_QUALITY_SCORE  # Low quality if no data
        
        try:
            pair = self.pair_features.loc[pair_id]
            
            # Number of bookings (more = better)
            total_bookings = self._safe_get_int(self.pair_features, pair_id, 'total_bookings', 0)
            booking_score = min(1.0, total_bookings / self.config.BOOKING_QUALITY_THRESHOLD)
            
            # Recency (recent = better)
            days_since_last = self._safe_get_int(self.pair_features, pair_id, 'days_since_last_booking', 365)
            recency_score = max(0, 1 - (days_since_last / self.config.RECENCY_DECAY_DAYS))
            
            # Combine (70% volume, 30% recency)
            quality = (booking_score * 0.7) + (recency_score * 0.3)
            
            return max(self.config.MIN_DATA_QUALITY_SCORE, quality)
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error calculating data quality for {pair_id}: {e}")
            return 0.5
    
    def _calculate_preference_score(self, chauffeur_id: int, vehicle_id: int, request: RecommendationRequest) -> float:
        """Calculate preference match score"""
        score = 0.5  # Neutral
        
        if request.preferences:
            if chauffeur_id in request.preferences.preferred_chauffeurs:
                score += 0.3
            if vehicle_id in request.preferences.preferred_vehicles:
                score += 0.2
        
        return min(1.0, score)
    
    def _calculate_requirement_score(self, vehicle_id: int, request: RecommendationRequest) -> float:
        """Calculate how well vehicle matches requirements"""
        if not vehicle_id or vehicle_id not in self.vehicle_idx.index:
            return 0.5
        
        try:
            vehicle_type_id = self._safe_get_int(self.vehicle_idx, vehicle_id, 'RefVehicleType', -1)
            
            if vehicle_type_id == -1 or vehicle_type_id not in self.vehicle_type_idx.index:
                return 0.5
            
            score = 0.0
            checks = 0
            
            # Vehicle type exact match
            if request.requirements.vehicle_type_id:
                checks += 1
                if vehicle_type_id == request.requirements.vehicle_type_id:
                    score += 1.0
            
            # Seat capacity (bonus for excess capacity)
            if request.booking_context.no_of_pax:
                checks += 1
                capacity = self._safe_get_int(self.vehicle_type_idx, vehicle_type_id, 'SeatCapacity', 4)
                if capacity >= request.booking_context.no_of_pax:
                    excess = capacity - request.booking_context.no_of_pax
                    score += min(1.0, 0.8 + (excess * 0.05))
            
            return score / checks if checks > 0 else 0.5
        except (KeyError, ValueError, TypeError)  as e:
            logger.debug(f"Error calculating requirement score for vehicle {vehicle_id}: {e}")
            return 0.5
    
    def _calculate_quality_score(self, pair_id: str) -> float:
        """Calculate quality score from historical data"""
        if pair_id not in self.pair_features.index:
            return 0.5
        
        try:
            # Completion rate
            completion = self._safe_get_float(self.pair_features, pair_id, 'completion_rate', 0.5)
            
            # Price consistency
            consistency = self._safe_get_float(self.pair_features, pair_id, 'price_consistency', 0.5)
            
            # Booking volume (normalized)
            total_bookings = self._safe_get_int(self.pair_features, pair_id, 'total_bookings', 0)
            volume_score = min(1.0, total_bookings / 100)
            
            # Weighted quality score
            quality = (completion * 0.5) + (consistency * 0.3) + (volume_score * 0.2)
            
            return quality
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error calculating quality score for {pair_id}: {e}")
            return 0.5
    
    def estimate_price(self, pair_id: str, request: RecommendationRequest) -> float:
        """Estimate price for this booking"""
        try:
            if pair_id in self.pair_features.index:
                base_price = self._safe_get_float(self.pair_features, pair_id, 'avg_price', 100.0)
            else:
                base_price = 100.0
            
            # Adjust for distance
            if request.booking_context.estimated_distance_km:
                distance = request.booking_context.estimated_distance_km
                distance_price = distance * 2.0
                base_price = max(base_price, distance_price)
            
            return round(base_price, 2)
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"Error estimating price for {pair_id}: {e}")
            return 100.0
# collabrative 


import time
from typing import List, Dict
from datetime import datetime
import logging
import pandas as pd

from api.models import (
    RecommendationRequest, 
    RecommendationResponse,
    ChauffeurVehicleRecommendation
)
from api.filters import RecommendationFilter
from models.ensemble_model import EnsembleRecommender
from config.config import Config

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating real-time recommendations"""
    
    def __init__(self,
                 ensemble_model: EnsembleRecommender,
                 recommendation_filter: RecommendationFilter,
                 pair_features,
                 chauffeur_data,
                 vehicle_data,
                 vehicle_type_data):
        self.model = ensemble_model
        self.filter = recommendation_filter
        self.pair_features = pair_features
        self.chauffeur_data = chauffeur_data
        self.vehicle_data = vehicle_data
        self.vehicle_type_data = vehicle_type_data
        self.config = Config()

        # Create lookup indices
        if 'ID' in chauffeur_data.columns:
            self.chauffeur_idx = chauffeur_data.set_index('ID')
        else:
            self.chauffeur_idx = chauffeur_data
            
        if 'ID' in vehicle_data.columns:
            self.vehicle_idx = vehicle_data.set_index('ID')
        else:
            self.vehicle_idx = vehicle_data
            
        if 'ID' in vehicle_type_data.columns:
            self.vehicle_type_idx = vehicle_type_data.set_index('ID')
        else:
            self.vehicle_type_idx = vehicle_type_data
        
        # Diagnostic info
        logger.info(f"   Service initialized with {len(self.vehicle_type_idx)} vehicle types indexed")
    
    def generate_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Generate recommendations for a single request"""
        start_time = time.time()
        
        # Check if cold start (new client)
        is_cold_start = request.client_id not in self.model.train_matrix.index
        
        if is_cold_start:
            logger.info(f" Cold start client {request.client_id} - using context-aware popularity")
            recommendations, metadata = self._generate_cold_start_recommendations(request)
        else:
            logger.info(f"   Existing client {request.client_id} - using CF model")
            recommendations, metadata = self._generate_cf_recommendations(request)
        
        processing_time = (time.time() - start_time) * 1000
        metadata['processing_time_ms'] = round(processing_time, 2)
        metadata['cold_start'] = is_cold_start
        metadata['model_type'] = 'context_aware_popularity' if is_cold_start else 'ensemble_cf'
        
        response = RecommendationResponse(
            client_id=request.client_id,
            request_timestamp=datetime.now(),
            recommendations=recommendations,
            metadata=metadata
        )
        
        logger.info(f"      Generated {len(recommendations)} recommendations in {processing_time:.0f}ms")
        
        return response
    
    def _generate_cold_start_recommendations(self, request: RecommendationRequest) -> tuple:
        """Context-aware recommendations for new clients - FILTER FIRST"""
        
        # STEP 1: Get all possible pairs
        all_candidate_pairs = list(self.model.train_matrix.columns)
        logger.info(f"   Total available pairs: {len(all_candidate_pairs)}")
        
        # STEP 2: Filter FIRST
        filtered_pairs, filter_stats = self.filter.filter_candidates(
            all_candidate_pairs,
            request
        )
        
        logger.info(f"   After filtering: {len(filtered_pairs)} candidates")
        
        if len(filtered_pairs) == 0:
            logger.error("     No vehicles match the specified requirements")
            
            metadata = {
                'strategy': 'cold_start_filter_first',
                'total_candidates': len(all_candidate_pairs),
                'filtered_candidates': 0,
                'filter_breakdown': filter_stats,
                'error': 'No vehicles match the requirements',
                'error_details': {}
            }
            
            # Add helpful error details
            if request.requirements:
                if request.requirements.vehicle_type_id:
                    metadata['error_details']['vehicle_type_id'] = request.requirements.vehicle_type_id
                    metadata['error_details']['message'] = f"No chauffeur-vehicle pairs found with vehicle_type_id={request.requirements.vehicle_type_id}"
                
                if request.requirements.min_vehicle_level:
                    metadata['error_details']['min_vehicle_level'] = request.requirements.min_vehicle_level
                
                if request.booking_context.no_of_pax:
                    metadata['error_details']['required_capacity'] = request.booking_context.no_of_pax
            
            return [], metadata
        # STEP 3: Score by popularity + context
        scored_pairs = []
        popularity_dict = dict(self.model.top_items)
        
        for pair_id in filtered_pairs:
            pop_score = popularity_dict.get(pair_id, 3.0)
            match_scores = self.filter.calculate_match_scores(pair_id, request)
            
            # Weighted scoring: 40% popularity + 60% context match
            final_score = (
                pop_score * self.config.COLD_START_POPULARITY_WEIGHT + 
                match_scores['overall_match'] * self.config.CONTEXT_SCORE_MULTIPLIER * self.config.COLD_START_CONTEXT_WEIGHT
            )            
            # THIS FORMAT: (pair_id, score, confidence, match_scores)
            scored_pairs.append((pair_id, final_score, 0.5, match_scores))
        
        # STEP 4: Sort and take top N
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:request.n_recommendations]
        
        # STEP 5: Build recommendations
        recommendations = self._build_detailed_recommendations(top_pairs, request)
        
        metadata = {
            'strategy': 'cold_start_filter_first',
            'total_candidates': len(all_candidate_pairs),
            'filtered_candidates': len(filtered_pairs),
            'filter_breakdown': filter_stats
        }
        
        return recommendations, metadata
    
    def _generate_cf_recommendations(self, request: RecommendationRequest) -> tuple:
        """CF-based recommendations for existing clients"""
        
        # STEP 1: Get ALL possible pairs from the model
        all_candidate_pairs = list(self.model.train_matrix.columns)
        logger.info(f"   Total available pairs: {len(all_candidate_pairs)}")
        
        # STEP 2: Apply ALL filters FIRST
        filtered_pairs, filter_stats = self.filter.filter_candidates(
            all_candidate_pairs, 
            request
        )
        
        logger.info(f"   After filtering: {len(filtered_pairs)} candidates remain")
        
        # STEP 3: If no candidates passed filters, return error response
        if len(filtered_pairs) == 0:
            logger.error("     No vehicles match the specified requirements")
            
            # Build error metadata
            metadata = {
                'total_candidates': len(all_candidate_pairs),
                'filtered_candidates': 0,
                'filter_breakdown': filter_stats,
                'error': 'No vehicles match the requirements',
                'error_details': {}
            }
            
            # Add helpful error details
            if request.requirements:
                if request.requirements.vehicle_type_id:
                    metadata['error_details']['vehicle_type_id'] = request.requirements.vehicle_type_id
                    metadata['error_details']['message'] = f"No chauffeur-vehicle pairs found with vehicle_type_id={request.requirements.vehicle_type_id}"
                
                if request.requirements.min_vehicle_level:
                    metadata['error_details']['min_vehicle_level'] = request.requirements.min_vehicle_level
                
                if request.booking_context.no_of_pax:
                    metadata['error_details']['required_capacity'] = request.booking_context.no_of_pax
            
            # Return empty recommendations with error metadata
            return [], metadata
    
        # STEP 4: Score ONLY the filtered pairs
        logger.info(f"   Scoring {len(filtered_pairs)} filtered candidates...")
        scored_pairs = self.model.score_items(
            user_id=request.client_id,
            item_list=filtered_pairs,
            return_scores=True
        )
        
        logger.info(f"   Scored {len(scored_pairs)} pairs")
        
        # STEP 5: Add match_scores to each prediction
        scored_pairs_with_match = []
        for pair_id, score, confidence in scored_pairs:
            match_scores = self.filter.calculate_match_scores(pair_id, request)
            scored_pairs_with_match.append((pair_id, score, confidence, match_scores))
        
        # STEP 6: Take top N
        top_pairs = scored_pairs_with_match[:request.n_recommendations]
        
        # STEP 7: Build detailed recommendations
        recommendations = self._build_detailed_recommendations(top_pairs, request)
        
        metadata = {
            'total_candidates': len(all_candidate_pairs),
            'filtered_candidates': len(filtered_pairs),
            'filter_breakdown': filter_stats,
            'scored_candidates': len(scored_pairs)
        }
        
        return recommendations, metadata

    def _smart_fallback_recommendations(self, candidate_pairs: List[str], request: RecommendationRequest) -> List[str]:
        """
        Smart fallback that RESPECTS hard requirements
        """
        logger.info("   Applying smart fallback with hard requirement validation...")
        
        # FIXED: Never relax hard requirements!
        hard_requirements = request.requirements
        
        valid_pairs = []
        
        for pair_id in candidate_pairs:
            # CRITICAL: Validate against hard requirements
            if not self._meets_hard_requirements(pair_id, hard_requirements, request.booking_context):
                continue
            
            # Passed hard requirements
            valid_pairs.append(pair_id)
            
            # Stop once we have enough
            if len(valid_pairs) >= request.n_recommendations * 2:
                break
        
        if len(valid_pairs) == 0:
            logger.error("     NO PAIRS meet hard requirements!")
            logger.error(f"   Requirements: {hard_requirements}")
            
            # Last resort: return POPULAR pairs that at least match vehicle type
            if hard_requirements.vehicle_type_id:
                # Try to find pairs with correct vehicle type
                for pair_id in self.model.top_item_ids[:100]:
                    vehicle_id = self.filter.pair_to_vehicle.get(pair_id)
                    if vehicle_id and vehicle_id in self.filter.vehicle_idx.index:
                        vehicle = self.filter.vehicle_idx.loc[vehicle_id]
                        vtype_id = vehicle.get('RefVehicleType')
                        if vtype_id == hard_requirements.vehicle_type_id:
                            valid_pairs.append(pair_id)
                            if len(valid_pairs) >= 5:
                                break
            
            # If still nothing, we must return something
            if len(valid_pairs) == 0:
                logger.error("     EMERGENCY: Returning top popular without validation")
                valid_pairs = self.model.top_item_ids[:10]
        
        logger.info(f"   Smart fallback found {len(valid_pairs)} valid candidates")
        return valid_pairs
    
    def _meets_hard_requirements(self, pair_id: str, requirements, booking_context) -> bool:
        """Check if pair meets hard requirements"""
        if not requirements:
            return True
        
        vehicle_id = self.filter.pair_to_vehicle.get(pair_id)
        chauffeur_id = self.filter.pair_to_chauffeur.get(pair_id)
        
        if not vehicle_id or vehicle_id not in self.filter.vehicle_idx.index:
            return False
        
        vehicle = self.filter.vehicle_idx.loc[vehicle_id]
        vehicle_type_id = vehicle.get('RefVehicleType')
        
        if pd.isna(vehicle_type_id):
            return False
        
        try:
            vehicle_type_id = int(vehicle_type_id)
        except (ValueError, TypeError):
            return False
        
        if vehicle_type_id not in self.filter.vehicle_type_idx.index:
            return False
        
        vehicle_type = self.filter.vehicle_type_idx.loc[vehicle_type_id]
        
        # Check vehicle type ID (STRICT)
        if requirements.vehicle_type_id:
            if vehicle_type_id != requirements.vehicle_type_id:
                return False
        
        # Check seat capacity using no_of_pax from booking_context (STRICT)
        if booking_context.no_of_pax:
            seat_capacity = vehicle_type.get('SeatCapacity')
            if pd.notna(seat_capacity):
                try:
                    if int(seat_capacity) < booking_context.no_of_pax:
                        return False
                except (ValueError, TypeError):
                    return False
        
        # Check MINIMUM vehicle level (STRICT)
        if requirements.min_vehicle_level is not None:
            vehicle_level = vehicle_type.get('VehicleLevel')
            if pd.notna(vehicle_level):
                try:
                    if int(vehicle_level) < requirements.min_vehicle_level:
                        return False
                except (ValueError, TypeError):
                    return False
        
        # Check required features (STRICT)
        if requirements.required_features and chauffeur_id and not self._has_required_features(
                chauffeur_id, vehicle_id, requirements.required_features
            ):
                return False
        
        return True
    
    def _has_required_features(self, chauffeur_id: int, vehicle_id: int, features: List[str]) -> bool:
        """Check if chauffeur-vehicle has required features"""
        if not chauffeur_id or chauffeur_id not in self.filter.chauffeur_idx.index:
            return False
        
        chauffeur = self.filter.chauffeur_idx.loc[chauffeur_id]
        
        for feature in features:
            if feature == 'baby_seat':
                if not chauffeur.get('BabySeatTraining', False):
                    return False
            elif feature == 'red_zone_authorized':
                if not chauffeur.get('RedZoneAuthorised', False):
                    return False
        
        return True
    
    def _build_detailed_recommendations(self, 
                                       ranked_pairs: List[tuple],
                                       request: RecommendationRequest) -> List[ChauffeurVehicleRecommendation]:
        """Build detailed recommendation objects with improved vehicle type handling"""
        recommendations = []
        
        for rank, (pair_id, final_score, confidence, match_scores) in enumerate(ranked_pairs, 1):
            chauffeur_id = self.filter.pair_to_chauffeur.get(pair_id)
            vehicle_id = self.filter.pair_to_vehicle.get(pair_id)
            
            if not chauffeur_id or not vehicle_id:
                continue
            
            # Get chauffeur details
            chauffeur_info = self._get_chauffeur_info(chauffeur_id)
            
            # Get vehicle details (IMPROVED with vehicle_type_id)
            vehicle_info = self._get_vehicle_info(vehicle_id)
            
            # Estimate price
            estimated_price = self.filter.estimate_price(pair_id, request)
            
            # Create recommendation
            rec = ChauffeurVehicleRecommendation(
                rank=rank,
                chauffeur_id=chauffeur_id,
                vehicle_id=vehicle_id,
                pair_id=pair_id,
                predicted_rating=round(final_score, 2),
                confidence=round(confidence, 3),
                estimated_price=estimated_price,
                **chauffeur_info,
                **vehicle_info,
                match_scores={
                    'cf_score': round(final_score, 2),
                    'overall_match': round(match_scores.get('overall_match', 0.5), 2),
                    'quality_score': round(match_scores.get('quality_score', 0.5), 2),
                    'requirement_match': round(match_scores.get('requirement_match', 0.5), 2)
                }
            )
            
            recommendations.append(rec)
        
        return recommendations
    
    def _get_chauffeur_info(self, chauffeur_id: int) -> Dict:
        """Get chauffeur details"""
        info = {}
        
        if chauffeur_id in self.chauffeur_idx.index:
            chauffeur = self.chauffeur_idx.loc[chauffeur_id]
            info['chauffeur_name'] = str(chauffeur.get('Code', f"Chauffeur {chauffeur_id}"))
            info['chauffeur_experience_days'] = int(chauffeur.get('employee_duration_days', 0))
            
            # Calculate rating from pair features if available
            pair_matches = [p for p in self.pair_features.index if str(p).startswith(f"{chauffeur_id}_")]
            if pair_matches:
                completion_rates = []
                for p in pair_matches:
                    cr = self.pair_features.loc[p].get('completion_rate', 0.5)
                    completion_rates.append(cr)
                avg_completion = sum(completion_rates) / len(completion_rates)
                info['chauffeur_rating'] = round(1 + (avg_completion * 4), 2)
        
        return info
    
    def _get_vehicle_info(self, vehicle_id: int) -> Dict:
        """
        IMPROVED: Get vehicle details with better vehicle type handling
        """
        info = {}
        
        if vehicle_id not in self.vehicle_idx.index:
            logger.warning(f"Vehicle {vehicle_id} not in index")
            return info
        
        vehicle = self.vehicle_idx.loc[vehicle_id]
        info['vehicle_make'] = str(vehicle.get('Make', 'Unknown'))
        info['vehicle_model'] = str(vehicle.get('Model', 'Unknown'))
        
        year = vehicle.get('Year')
        if year and not pd.isna(year):
            try:
                info['vehicle_year'] = int(year)
            except (ValueError, TypeError):
                info['vehicle_year'] = None
        
        # Get vehicle type info with improved logging
        vehicle_type_id = vehicle.get('RefVehicleType')
        
        if pd.isna(vehicle_type_id):
            logger.warning(f"Vehicle {vehicle_id} has no RefVehicleType")
            info['vehicle_type_id'] = None
            return info
        
        try:
            vehicle_type_id = int(vehicle_type_id)
            info['vehicle_type_id'] = vehicle_type_id  # NEW: Always include raw ID
            
            if vehicle_type_id not in self.vehicle_type_idx.index:
                logger.warning(f"Vehicle type {vehicle_type_id} not found in vehicle_type index!")
                logger.warning(f"   Available types: {list(self.vehicle_type_idx.index[:10])}...")
                # Return with vehicle_type_id but NULL other fields
                return info
            
            vtype = self.vehicle_type_idx.loc[vehicle_type_id]
            info['vehicle_type'] = str(vtype.get('DisplayName', 'Standard'))
            
            capacity = vtype.get('SeatCapacity')
            if capacity and not pd.isna(capacity):
                try:
                    info['seat_capacity'] = int(capacity)
                except Exception as e:
                    logger.warning(f"Failed to parse SeatCapacity for type {vehicle_type_id}: {e}")
                    info['seat_capacity'] = 4
        except Exception as e:
            logger.error(f"Error processing vehicle type for vehicle {vehicle_id}: {e}")
        
        return info
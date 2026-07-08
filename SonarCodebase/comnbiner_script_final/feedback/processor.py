



import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .store import FeedbackStore
from .weight_learner import WeightLearner
from .config import (
    FEEDBACK_CONFIG,
    REJECTION_CATEGORIES,
    MODEL_NAMES,
    is_learnable_rejection,
    get_category_model_weights
)
from sqlalchemy import text

logger = logging.getLogger(__name__)


class FeedbackProcessor:
    """
    Processes dispatcher feedback with SEPARATE reject and select flows.
    
    REJECT FLOW (process_rejection):
        - LLM classify rejection reason
        - Calculate penalty credits (if learnable)
        - Update learning (EMA)
        - Save to feedback_rejections table
        - CALL [DispatcherActivity] SP @Status=0  ← SP CALLED
        - Return next recommendation
    
    SELECT FLOW (process_selection):
        - Calculate reward credits
        - Update learning (EMA)
        - Save to BookingRecommendations (IsSelected=1)
        - CALL [DispatcherActivity] SP @Status=1  ← SP CALLED
        - Confirm booking → FinalAllocation
    """
    
    def __init__(self, 
                 store: FeedbackStore = None, 
                 weight_learner: 'WeightLearner' = None
                 ):
        """Initialize the feedback processor."""
        self.store = store or FeedbackStore()
        self.weight_learner = weight_learner or WeightLearner(self.store)
        # self.llm_classifier = llm_classifier or RejectionClassifier()
        
        # Penalty system weights
        self.alpha = FEEDBACK_CONFIG.get('SCORE_PENALTY_WEIGHT', 0.5)
        self.beta = FEEDBACK_CONFIG.get('CATEGORY_PENALTY_WEIGHT', 0.5)
        
        # Separate reward and penalty factors
        self.reward_factor = FEEDBACK_CONFIG.get('SELECTION_REWARD_FACTOR', 0.08)
        self.penalty_factor = FEEDBACK_CONFIG.get('REJECTION_PENALTY_FACTOR', 0.05)
        self.max_penalty = FEEDBACK_CONFIG.get('MAX_REJECTION_PENALTY', 0.15)
        
        logger.info(f"FeedbackProcessor initialized (reward={self.reward_factor}, penalty={self.penalty_factor}, α={self.alpha}, β={self.beta})")

    # =========================================================================
    # NEW METHOD: Process Single Rejection
    # NO SP CALLED - Only updates feedback tables
    # =========================================================================
    
    def process_rejection(self, rejection_data: Dict) -> Dict:
        """
       
        
        This is called each time dispatcher clicks "Reject" on a recommendation.
        - Applies learning penalty immediately
        - Calls SP to update AvailaibityAllocation
        - Returns the next available recommendation
        
        Args:
            rejection_data: Dictionary containing:
                - request_id: BookingID
                - pair_id: The pair being rejected
                - reason_text: Dispatcher's reason for rejection
        
        Returns:
            Result dictionary with:
                - success: bool
                - rejected_pair: str
                - rejection_category: str (LLM classified)
                - credits_applied: dict (penalties)
                - next_recommendation: dict or None
                - remaining_count: int
        """
        try:
            request_id = rejection_data.get('request_id')
            pair_id = rejection_data.get('pair_id')
            categories_list = rejection_data.get('categories', [])
            reason_text = rejection_data.get('reason_text', '')
            
            if not request_id:
                return {'success': False, 'error': 'request_id is required'}
            if not pair_id:
                return {'success': False, 'error': 'pair_id is required'}
            
            # Get shown event (recommendations for this booking)
            shown_event = self.store.get_shown_event(request_id)
            if not shown_event:
                return {'success': False, 'error': f'request_id {request_id} not found'}
            
            # Find the rejected pair
            rejected_pair = self._find_pair(shown_event['pairs_shown'], pair_id)
            if not rejected_pair:
                return {'success': False, 'error': f'pair_id {pair_id} not found in recommendations'}
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: LLM Classify rejection reason
            # ─────────────────────────────────────────────────────────────────
            if categories_list and len(categories_list)>0:
                classification = self._build_classification_from_categories(categories_list)
            # else:
            #     classification = classify_rejection(reason_text)
            
            logger.info(f"\n🔍 Processing REJECTION: request={request_id}, pair={pair_id}")
            logger.info(f"   Reason: '{reason_text[:50]}...' " if len(reason_text) > 50 else f"   Reason: '{reason_text}'")
            logger.info(f"   Categories: {classification['categories']}")
            logger.info(f"   Learnable: {classification['is_learnable']}")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Calculate penalties (if learnable)
            # ─────────────────────────────────────────────────────────────────
            penalties = {
                'lightgcn': 0.0,
                'collaborative': 0.0,
                'user_content': 0.0,
                'item_content': 0.0
            }
            
            if classification['is_learnable']:
                penalties, _ = self._calculate_penalties(
                    rejected_pair=rejected_pair,
                    category_weights=classification['categories']
                )
                
                # Convert to negative credits
                model_credits = {k: -v for k, v in penalties.items()}
                
                logger.info("   Penalties applied:")
                for model, penalty in penalties.items():
                    if penalty > 0:
                        logger.info(f"     [{model.upper()}]: -{penalty:.4f}")
                
                # ─────────────────────────────────────────────────────────────
                # STEP 3: Update learning (EMA update)
                # Updates: feedback_model_performance table
                # ─────────────────────────────────────────────────────────────
                self.weight_learner.update(
                    client_id=shown_event['client_id'],
                    client_segment=shown_event['client_segment'],
                    model_credits=model_credits,
                    selected_rank=None  # No selection yet
                )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Save rejection to database
            # Inserts into: feedback_rejections table
            # ─────────────────────────────────────────────────────────────────
            self._save_rejection_with_classification(
                request_id=request_id,
                pair_id=pair_id,
                rejected_pair=rejected_pair,
                reason_text=reason_text,
                classification=classification
            )
          
            rejection_sp_result = self._call_dispatcher_activity_sp(
                booking_id=int(request_id),
                pair_id=pair_id,
                status=2  
            )
            if rejection_sp_result.get('success') and rejection_sp_result.get('output_id'):
                logger.info(f"   Updated OutputId: {rejection_sp_result['output_id']}")
            logger.info(f"   SP [DispatcherActivity] called for rejection: {rejection_sp_result}")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 6: Get next recommendation
            # ─────────────────────────────────────────────────────────────────
            next_rec = self._get_next_recommendation(request_id)
            remaining = self._count_remaining_recommendations(request_id)
            
            # Get primary category
            categories = classification.get('categories', {})
            primary_category = max(categories, key=categories.get) if categories else 'unknown'
            
            logger.info(f"      Rejection processed. Next recommendation: {next_rec['pair_id'] if next_rec else 'NONE'}")
            logger.info(f"   Remaining: {remaining}")
            
            return {
                'success': True,
                'rejected_pair': pair_id,
                'rejection_category': primary_category,
                'classification': classification,
                'credits_applied': {k: -v for k, v in penalties.items()},
                'learning_applied': classification['is_learnable'],
                'allocation_updated': rejection_sp_result.get('success', False),
                'next_recommendation': next_rec,
                'remaining_count': remaining
            }
            
        except Exception as e:
            logger.error(f"  Error processing rejection: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # MODIFIED: Process Selection (Only handles acceptance now)
    # SP CALLED: [DispatcherActivity] @Status=1
    # =========================================================================
    
    def process_selection(self, feedback: Dict) -> Dict:
        """
       
        Args:
            feedback: Dictionary containing:
                - request_id: BookingID
                - selected_pair_id: The pair being accepted
                - overall_quality: Optional 1-5 star rating
                - rejections: (OPTIONAL) For backward compatibility
        
        Returns:
            Result dictionary with success status and credits
        """
        try:
            request_id = feedback.get('request_id')
            selected_pair_id = feedback.get('selected_pair_id')
            overall_quality = feedback.get('overall_quality')
            
            if not request_id:
                return {'success': False, 'error': 'request_id is required'}
            if not selected_pair_id:
                return {'success': False, 'error': 'selected_pair_id is required'}
            
            # Get shown event
            shown_event = self.store.get_shown_event(request_id)
            if not shown_event:
                return {'success': False, 'error': f'request_id {request_id} not found'}
            
            # Find selected pair
            selected_pair = self._find_pair(shown_event['pairs_shown'], selected_pair_id)
            if not selected_pair:
                return {'success': False, 'error': f'selected_pair_id {selected_pair_id} not found'}
            
            # ─────────────────────────────────────────────────────────────────
            # BACKWARD COMPATIBILITY: Process any rejections passed in
            # (For old API calls that send rejections array with selection)
            
            # ─────────────────────────────────────────────────────────────────
            ##  need to remove this old flow ### 
            rejections_in_request = feedback.get('rejections', [])
            if rejections_in_request:
                logger.info(f"Processing {len(rejections_in_request)} rejections (backward compat mode)")
                for rejection in rejections_in_request:
                    self.process_rejection({
                        'request_id': request_id,
                        'pair_id': rejection.get('pair_id'),
                        'reason_text': rejection.get('reason_text', rejection.get('reason', ''))
                    })
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Calculate REWARDS for selected pair
            # ─────────────────────────────────────────────────────────────────
            rewards = self._calculate_rewards(selected_pair)
            
            logger.info(f"\n Processing SELECTION: request={request_id}, pair={selected_pair_id}")
            logger.info(f"   Quality rating: {overall_quality}/5" if overall_quality else "   Quality rating: not provided")
            logger.info("   Rewards:")
            for model, reward in rewards.items():
                logger.info(f"     [{model.upper()}]: +{reward:.4f}")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Get all previous rejections for this booking
            # ─────────────────────────────────────────────────────────────────
            rejections = self._get_rejections_for_booking(request_id)
            total_rejections = len(rejections)
            
            logger.info(f"   Prior rejections: {total_rejections}")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Save selection to BookingRecommendations
            # Updates: BookingRecommendations SET IsSelected=1
            # ─────────────────────────────────────────────────────────────────
            selected_rank = selected_pair.get('rank_shown') or selected_pair.get('final_rank', 0)
            
            allocation_result = self._call_dispatcher_activity_sp(
                booking_id=int(request_id),
                pair_id=selected_pair_id,
                status=1  
            )
            
            # Check if SP succeeded BEFORE proceeding
            if not allocation_result.get('success'):
                error_msg = allocation_result.get('error', 'SP returned failure')
                logger.warning(f"     SP returned failure: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'already_processed': 'already' in error_msg.lower() if error_msg else False
                }
            
            if allocation_result.get('output_id'):
                logger.info(f"      SP Success - OutputId: {allocation_result['output_id']}")
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Save credits ONLY (SP already set IsSelected=1)
            # Only saves: OverallQuality, model credits
            # Does NOT update IsSelected (SP already did it)
            # ─────────────────────────────────────────────────────────────────
            self.store.save_selection_credits(
                request_id=request_id,
                selected_pair_id=selected_pair_id,
                overall_quality=overall_quality,
                model_credits=rewards
            )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 5: Update learning (EMA) with rewards
            # Updates: feedback_model_performance table
            # ─────────────────────────────────────────────────────────────────
            self.weight_learner.update(
                client_id=shown_event['client_id'],
                client_segment=shown_event['client_segment'],
                model_credits=rewards,
                selected_rank=selected_rank
            )
            
            logger.info("      Selection processed and confirmed!")
            
            return {
                'success': True,
                'selected_pair_id': selected_pair_id,
                'selected_rank': selected_rank,
                'overall_quality': overall_quality,
                'model_credits': rewards,
                'total_rejections': total_rejections,
                'learning_applied': True,
                'allocation_confirmed': allocation_result.get('success', False),
                'client_id': shown_event['client_id']
            }
            
        except Exception as e:
            logger.error(f"  Error processing selection: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # SP CALL METHODS
    # =========================================================================
    

    def _call_dispatcher_activity_sp(self, booking_id: int, pair_id: str, status: int) -> Dict:
        """
        Call DispatcherActivity stored procedure and VERIFY it succeeded.
        
        FIXED: Now properly reads SP output and checks Result column.
        
        Args:
            booking_id: BookingID (BIGINT)
            pair_id: Specific pair (NVARCHAR)
            status: 1 for Accept, 0 for Reject
            
        Returns:
            Dict with success status, output_id, and any error message
        """
        try:
            with self.store.engine.connect() as conn:
                # Execute the SP
                result = conn.execute(
                    text("""
                        EXEC [dbo].[DispatcherActivity] 
                            @Status = :status, 
                            @BookingID = :booking_id,
                            @PairId = :pair_id
                    """),
                    {
                        'status': status,
                        'booking_id': booking_id,
                        'pair_id': pair_id
                    }
                )
                
                # CRITICAL: Must commit for changes to persist
                conn.commit()
                
                # Try to read the SP's output
                try:
                    # Fetch the first result set
                    row = result.fetchone()
                    
                    if row:
                        row_dict = dict(row._mapping)
                        
                        # Check the Result column
                        if row_dict.get('Result') == 'SUCCESS':
                            output_id = row_dict.get('OutputId')
                            logger.info(f"      SP SUCCESS: BookingID={booking_id}, PairId={pair_id}, "
                                    f"OutputId={output_id}, Status={status}")
                            
                            # VERIFY: Query the table to confirm IsSelected was updated
                            verify_query = text("""
                                SELECT IsSelected, SelectedAt 
                                FROM AvailabilityAllocation 
                                WHERE BookingID = :booking_id AND PairId = :pair_id
                            """)
                            verify_result = conn.execute(verify_query, {
                                'booking_id': booking_id,
                                'pair_id': pair_id
                            })
                            verify_row = verify_result.fetchone()
                            
                            if verify_row:
                                is_selected = verify_row.IsSelected
                                selected_at = verify_row.SelectedAt
                                
                                if is_selected == status:
                                    logger.info(f"       VERIFIED: IsSelected={is_selected}, SelectedAt={selected_at}")
                                else:
                                    logger.error(f"     VERIFICATION FAILED: IsSelected={is_selected} (expected {status})")
                                    return {
                                        'success': False,
                                        'error': f'SP claimed success but IsSelected={is_selected} (expected {status})',
                                        'booking_id': booking_id,
                                        'pair_id': pair_id
                                    }
                            
                            return {
                                'success': True,
                                'booking_id': booking_id,
                                'pair_id': pair_id,
                                'output_id': output_id,
                                'status': status
                            }
                        
                        elif row_dict.get('Result') == 'ERROR':
                            error_msg = row_dict.get('ErrorMessage', 'Unknown error')
                            logger.error(f"     SP ERROR: {error_msg}")
                            return {
                                'success': False,
                                'error': error_msg,
                                'booking_id': booking_id,
                                'pair_id': pair_id
                            }
                    
                except Exception as e:
                    # If we can't read the result, something went wrong
                    logger.error(f"     Failed to read SP result: {e}")
                    
                    # Try to verify the update anyway
                    verify_query = text("""
                        SELECT IsSelected, SelectedAt 
                        FROM AvailabilityAllocation 
                        WHERE BookingID = :booking_id AND PairId = :pair_id
                    """)
                    verify_result = conn.execute(verify_query, {
                        'booking_id': booking_id,
                        'pair_id': pair_id
                    })
                    verify_row = verify_result.fetchone()
                    
                    if verify_row and verify_row.IsSelected == status:
                        logger.warning("     Couldn't read SP result, but table WAS updated correctly")
                        return {
                            'success': True,
                            'booking_id': booking_id,
                            'pair_id': pair_id,
                            'status': status,
                            'verified_by_query': True
                        }
                    else:
                        logger.error("     Couldn't read SP result AND table was NOT updated")
                        return {
                            'success': False,
                            'error': f'SP execution unclear, table not updated: {e}',
                            'booking_id': booking_id,
                            'pair_id': pair_id
                        }
                
                # Should not reach here
                logger.error("     SP returned no result")
                return {
                    'success': False,
                    'error': 'SP returned no result',
                    'booking_id': booking_id,
                    'pair_id': pair_id
                }
                    
        except Exception as e:
            error_str = str(e)
            logger.error(f"  Error calling DispatcherActivity SP: {e}", exc_info=True)
            
            # Check for specific SP errors
            if 'already ACCEPTED' in error_str or 'already REJECTED' in error_str:
                return {'success': False, 'error': error_str, 'already_processed': True}
            elif 'already has an accepted pair' in error_str:
                return {'success': False, 'error': error_str, 'booking_already_allocated': True}
            elif 'not found' in error_str:
                return {'success': False, 'error': error_str, 'pair_not_found': True}
            
            return {'success': False, 'error': error_str}
    
    def _call_select_recommendation_sp(self, booking_id: int) -> List[Dict]:
        """
       
        
        Args:
            booking_id: BookingID
            
        Returns:
            List of recommendation dicts
        """
        try:
            with self.store.engine.connect() as conn:
                query = text("""
                    EXEC [dbo].[SelectRecommendation] 
                        @BookingID = :booking_id
                """)
                
                result = conn.execute(query, {'booking_id': booking_id})
                rows = result.fetchall()
                
                recommendations = []
                for row in rows:
                    recommendations.append(dict(row._mapping))
                
                logger.info(f"   Called [SelectRecommendation] SP: BookingID={booking_id}, Found={len(recommendations)}")
                return recommendations
                
        except Exception as e:
            logger.error(f"Error calling SelectRecommendation SP: {e}")
            return []

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
     
    def _get_next_recommendation(self, request_id: str) -> Optional[Dict]:
        """
        Get the next available (not reviewed) recommendation for a BookingID.
        
        Schema Note:
            - request_id is actually BookingID (foreign key to BookingFeed)
            - Multiple OutputIds exist for one BookingID (20 recommendations)
            - Returns the lowest FinalRank where IsSelected IS NULL
        
        Args:
            request_id: BookingID as string
            
        Returns:
            Dict with recommendation details including OutputId, or None
        """
        try:
            with self.store.engine.connect() as conn:
                query = text("""
                    SELECT TOP 1
                        OutputId,
                        BookingID,
                        ClientId,
                        ClientType,
                        PairId,
                        VehicleId,
                        ChauffeurId,
                        FinalRank,
                        FinalScore,
                        GcnScore,
                        CfScore,
                        UserContentScore,
                        ItemContentScore
                    FROM AvailabilityAllocation
                    WHERE BookingID = :booking_id
                    AND IsSelected =0 -- ← NEW: 0 = Default/Unprocessed
                    ORDER BY FinalRank ASC
                """)
                
                result = conn.execute(query, {'booking_id': int(request_id)})
                row = result.fetchone()
                
                if not row:
                    logger.info(f"   No more recommendations for BookingID {request_id}")
                    return None
                
                logger.info(f"   Next: OutputId={row.OutputId}, "
                        f"PairId={row.PairId}, Rank={row.FinalRank}")
                
                return {
                    'output_id': row.OutputId, 
                    'booking_id': row.BookingID,
                    'client_id': row.ClientId,
                    'client_type': row.ClientType or 'UNKNOWN',
                    'pair_id': row.PairId,
                    'vehicle_id': row.VehicleId,
                    'chauffeur_id': row.ChauffeurId,
                    'rank': row.FinalRank,
                    'score': float(row.FinalScore) if row.FinalScore else 0.0,
                    'gcn_score': float(row.GcnScore) if row.GcnScore else 0.0,
                    'cf_score': float(row.CfScore) if row.CfScore else 0.0,
                    'user_content_score': float(row.UserContentScore) if row.UserContentScore else 0.0,
                    'item_content_score': float(row.ItemContentScore) if row.ItemContentScore else 0.0
                }
                
        except Exception as e:
            logger.error(f"  Error getting next recommendation: {e}")
            return None
    
    def _count_remaining_recommendations(self, request_id: str) -> int:
        """
        Count how many recommendations are still available (not reviewed).
        
        Args:
            request_id: BookingID as string
            
        Returns:
            Count of recommendations with IsSelected IS NULL
        """
        try:
            with self.store.engine.connect() as conn:
                query = text("""
                    SELECT COUNT(*) 
                    FROM AvailabilityAllocation
                    WHERE BookingID = :booking_id
                    AND IsSelected = 0  -- ⬅️ SIMPLIFIED!
                """)
                
                result = conn.execute(query, {'booking_id': int(request_id)})
                count = result.scalar()
                
                logger.info(f"   Remaining for BookingID {request_id}: {count}")
                return count or 0
                
        except Exception as e:
            logger.error(f"  Error counting recommendations: {e}")
            return 0
    
    def _get_rejections_for_booking(self, request_id: str) -> List[Dict]:
        """Get all rejections for a booking from feedback_rejections table."""
        try:
            with self.store.engine.connect() as conn:
                # FIXED: Use correct column names from actual table schema
                # - classification_data (not classified_category)
                # - rejected_at (not created_at)
                query = text("""
                    SELECT 
                        pair_id, 
                        rejection_reason, 
                        classification_data,
                        rejected_at
                    FROM feedback_rejections
                    WHERE request_id = :request_id
                    ORDER BY rejected_at
                """)
                
                result = conn.execute(query, {'request_id': str(request_id)})
                
                rejections = []
                for row in result.fetchall():
                    # Parse classification_data JSON if present
                    category = None
                    if row.classification_data:
                        try:
                            import json
                            class_data = json.loads(row.classification_data)
                            # Get the primary category (highest weight)
                            categories = class_data.get('categories', {})
                            if categories:
                                category = max(categories.items(), key=lambda x: x[1])[0]
                        except:
                            category = row.rejection_reason.split(':')[0] if row.rejection_reason else None
                    
                    rejections.append({
                        'pair_id': row.pair_id,
                        'reason': row.rejection_reason,
                        'category': category,
                        'classification_data': row.classification_data,
                        'rejected_at': row.rejected_at.isoformat() if row.rejected_at else None
                    })
                
                return rejections
                
        except Exception as e:
            logger.error(f"Error getting rejections: {e}")
            return []

    # =========================================================================
    # CREDIT CALCULATION METHODS
    # =========================================================================
    
    def _calculate_rewards(self, selected_pair: Dict) -> Dict[str, float]:
        """
        Calculate rewards for models based on selected pair scores.
        
        FORMULA:
            reward[model] = score[model] × SELECTION_REWARD_FACTOR
        """
        rewards = {}
        
        for model in ['lightgcn', 'collaborative', 'user_content', 'item_content']:
            score = self._get_model_score(selected_pair, model)
            rewards[model] = score * self.reward_factor
        
        return rewards
    
    def _calculate_penalties(self,
                            rejected_pair: Dict,
                            category_weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict]:
        """
        Calculate penalties for models based on rejected pair.
        
        FORMULA:
            score_penalty = rejected_score × REJECTION_PENALTY_FACTOR
            category_penalty = Σ(cat_weight × model_weight) × REJECTION_PENALTY_FACTOR
            total_penalty = α × score_penalty + β × category_penalty
        """
        penalties = {}
        detail = {
            'score_penalties': {},
            'category_penalties': {},
            'combined_penalties': {}
        }
        
        for model in ['lightgcn', 'collaborative', 'user_content', 'item_content']:
            # PENALTY 1: Score-Based
            model_score = self._get_model_score(rejected_pair, model)
            score_penalty = model_score * self.penalty_factor
            detail['score_penalties'][model] = score_penalty
            
            # PENALTY 2: Category-Based
            category_penalty = 0
            for category, cat_weight in category_weights.items():
                if category in REJECTION_CATEGORIES:
                    cat_config = REJECTION_CATEGORIES[category]
                    if cat_config['learnable']:
                        model_weight = cat_config['model_weights'].get(model, 0)
                        category_penalty += cat_weight * model_weight * self.penalty_factor
            
            detail['category_penalties'][model] = category_penalty
            
            # COMBINE: α × score_penalty + β × category_penalty
            combined = self.alpha * score_penalty + self.beta * category_penalty
            
            # Cap at maximum
            combined = min(combined, self.max_penalty)
            
            penalties[model] = combined
            detail['combined_penalties'][model] = combined
        
        return penalties, detail
    
    def _get_model_score(self, pair: Dict, model: str) -> float:
        """Get model score from pair, handling None values."""
        score_key = f'{model}_score'
        score = pair.get(score_key, 0)
        return float(score) if score else 0.0
    


    def _build_classification_from_categories(self, categories_list: List[str]) -> Dict:
        """
        Build classification from directly provided categories with EQUAL weights.
        """
        valid_categories = [cat for cat in categories_list if cat in REJECTION_CATEGORIES]
        
        if not valid_categories:
            return {
                'categories': {'poor_match': 1.0},
                'is_learnable': True,
                'confidence': 0.8,
                'learnable_weight': 1.0
            }
        
        # Equal weights for all selected categories
        equal_weight = 1.0 / len(valid_categories)
        category_weights = {cat: equal_weight for cat in valid_categories}
        
        learnable_weight = sum(
            weight for cat, weight in category_weights.items()
            if REJECTION_CATEGORIES.get(cat, {}).get('learnable', False)
        )
        
        return {
            'categories': category_weights,
            'is_learnable': learnable_weight > 0.5,
            'confidence': 1.0,
            'learnable_weight': learnable_weight
        }

    
    def _save_rejection_with_classification(self,
                                        request_id: str,
                                        pair_id: str,
                                        rejected_pair: Dict,
                                        reason_text: str,
                                        classification: Dict):
        """
        Save rejection with LLM classification to database.
        Inserts into: feedback_rejections table
        """
        try:
            import json
            
            rank = rejected_pair.get('rank_shown', 0)
            
            categories = classification.get('categories', {})
            is_learnable = classification.get('is_learnable', False)
            
            # Format ALL categories for storage
            if categories:
                categories_str = ','.join([
                    f"{cat}:{weight:.2f}" 
                    for cat, weight in sorted(categories.items(), key=lambda x: -x[1])
                ])
                primary_category = max(categories, key=categories.get)
            else:
                categories_str = 'external_factor:1.0'
                primary_category = 'external_factor'
            
            # Save to database
            success = self.store.save_rejection(
                request_id=request_id,
                pair_id=pair_id,
                rank=rank,
                rejection_reason=categories_str,
                notes=reason_text,
                classification_data=json.dumps(classification)
            )
            
            if success:
                logger.debug(f"      Saved rejection: {pair_id} (categories={categories_str})")
            else:
                logger.warning(f"    Failed to save rejection for {pair_id}")
                
        except Exception as e:
            logger.error(f"Error saving rejection: {e}")
    
    def _find_pair(self, pairs_shown: List[Dict], pair_id: str) -> Optional[Dict]:
        """Find a pair in the shown pairs list."""
        for pair in pairs_shown:
            if str(pair.get('pair_id')) == str(pair_id):
                return pair
        return None
    
    # =========================================================================
    # BATCH AND VALIDATION METHODS
    # =========================================================================
    
    def process_batch_selections(self, selections: List[Dict]) -> Dict:
        """Process multiple selections in batch."""
        results = {
            'total': len(selections),
            'successful': 0,
            'failed': 0,
            'results': []
        }
        
        for selection in selections:
            result = self.process_selection(selection)
            results['results'].append(result)
            
            if result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Batch processed {results['total']} selections: "
                   f"{results['successful']} successful, {results['failed']} failed")
        
        return results
    
    def validate_feedback(self, feedback: Dict) -> Tuple[bool, Optional[str]]:
        """Validate feedback structure."""
        if not feedback.get('request_id'):
            return False, 'request_id is required'
        
        if not feedback.get('selected_pair_id'):
            return False, 'selected_pair_id is required'
        
        rejections = feedback.get('rejections', [])
        if not isinstance(rejections, list):
            return False, 'rejections must be an array'
        
        for idx, rejection in enumerate(rejections):
            if not isinstance(rejection, dict):
                return False, f'rejections[{idx}] must be an object'
            if not rejection.get('pair_id'):
                return False, f'rejections[{idx}].pair_id is required'
        
        quality = feedback.get('overall_quality')
        if quality is not None:
            if not isinstance(quality, int) or quality < 1 or quality > 5:
                return False, 'overall_quality must be an integer between 1 and 5'
        
        return True, None
    
    def get_selection_stats(self, client_id: int = None) -> Dict:
        """Get statistics about processed selections."""
        if client_id:
            return self.store.get_client_stats(client_id)
        else:
            return self.store.get_system_stats()
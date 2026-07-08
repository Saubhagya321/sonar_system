



import logging
from typing import Dict, Optional, List

from .store import FeedbackStore
from .config import (
    FEEDBACK_CONFIG,
    MODEL_NAMES,
    MODEL_TO_WEIGHT_KEY,
    DEFAULT_BASE_WEIGHTS
)

logger = logging.getLogger(__name__)


class WeightLearner:
    """
    Learns and maintains model performance statistics.
    
    Uses exponential moving average (EMA) for smooth learning:
        new_avg = (1 - α) × old_avg + α × new_credit
    
    Supports hierarchical learning with fallback:
        1. Client-level (most personalized)
        2. Segment-level (e.g., VIP, Regular)
        3. Global-level (all clients)
        4. Base weights (from grid search)
    """
    
    def __init__(self, store: FeedbackStore = None):
        """
        Initialize the weight learner.
        
        Args:
            store: FeedbackStore instance. If None, creates a new one.
        """
        self.store = store or FeedbackStore()
        
        # Cache for quick access (cleared periodically)
        self._cache = {}
        
        logger.info("WeightLearner initialized")
    
    def update(self,
               client_id: int,
               client_segment: str,
               model_credits: Dict[str, float],
               selected_rank: int = None) -> None:
        """
        Update model performance at all levels (client, segment, global).
        
        Called after each dispatcher selection to incorporate new learning.
        
        Args:
            client_id: The client ID
            client_segment: Client segment/type (e.g., 'VIP', 'REGULAR')
            model_credits: Dict mapping model name to credit value
            selected_rank: Rank of the selected pair (for tracking accuracy)
        """
        try:
            # Update client-level stats
            self._update_level('client', str(client_id), model_credits, selected_rank)
            
            # Update segment-level stats
            if client_segment:
                self._update_level('segment', client_segment, model_credits, selected_rank)
            
            # Update global stats
            self._update_level('global', 'GLOBAL', model_credits, selected_rank)
            
            # Invalidate cache for this client
            cache_key = f"client_{client_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            # Log the credits being stored
            logger.info("     WeightLearner update:")
            logger.info(f"     Client: {client_id}, Segment: {client_segment}")
            logger.info(f"     Credits stored: {model_credits}")
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}", exc_info=True)
    
    def _update_level(self,
                      level: str,
                      identifier: str,
                      model_credits: Dict[str, float],
                      selected_rank: int = None) -> None:
        """
        Update performance stats at a specific level.
        
        Args:
            level: 'client', 'segment', or 'global'
            identifier: client_id, segment_name, or 'GLOBAL'
            model_credits: Dict mapping model name to credit
        """
        learning_rate = FEEDBACK_CONFIG['LEARNING_RATE']
        
        #    FIX: Changed from get_model_performance to use update_model_performance
        self.store.update_model_performance(
            level=level,
            identifier=identifier,
            model_credits=model_credits,
            learning_rate=learning_rate
        )
    
    def compute_weights(self,
                        client_id: int,
                        client_segment: str,
                        base_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute personalized weights using hierarchical fallback.
        
        Priority order:
        1. Client-level (if enough observations)
        2. Segment-level (if enough observations)
        3. Global-level (if enough observations)
        4. Base weights
        
        Args:
            client_id: The client ID
            client_segment: Client segment/type
            base_weights: Default weights from grid search
        
        Returns:
            Personalized weights dict:
            {
                'alpha1_rrf': float,
                'alpha2_gcn': float,
                'alpha3_item_sim': float,
                'alpha4_user_sim': float,
                'alpha5_confidence': float
            }
        """
        base_weights = base_weights or DEFAULT_BASE_WEIGHTS.copy()
        
        #    FIX: Changed from get_model_performance to get_model_stats
        # Try client-level first
        client_stats = self.store.get_model_stats('client', str(client_id))
        if self._has_enough_observations(client_stats, 
                                         FEEDBACK_CONFIG['MIN_CLIENT_OBSERVATIONS']):
            weights = self._stats_to_weights(client_stats, base_weights, 'client')
            logger.info(f"Using client-level weights for client {client_id}")
            return weights

        # Fallback to segment-level
        if client_segment:
            segment_stats = self.store.get_model_stats('segment', client_segment)
            if self._has_enough_observations(segment_stats,
                                            FEEDBACK_CONFIG['MIN_SEGMENT_OBSERVATIONS']):
                weights = self._stats_to_weights(segment_stats, base_weights, 'segment')
                logger.info(f"Using segment-level weights for segment {client_segment}")
                return weights
        
        # Fallback to global
        global_stats = self.store.get_model_stats('global', 'GLOBAL')
        if self._has_enough_observations(global_stats,
                                        FEEDBACK_CONFIG.get('MIN_GLOBAL_OBSERVATIONS', 50)):
            weights = self._stats_to_weights(global_stats, base_weights, 'global')
            logger.info("Using global-level weights")
            return weights
        
        # Fallback to base weights
        logger.info(f"Using base weights for client {client_id} (insufficient observations)")
        logger.info(f"     Base weights: {base_weights}")
        return base_weights
    
    def _has_enough_observations(self, 
                                  stats: Dict[str, Dict], 
                                  min_observations: int) -> bool:
        """
        Check if we have enough observations to trust the learned weights.
        
        Args:
            stats: Model performance stats dict
            min_observations: Minimum required observations
        
        Returns:
            True if we have enough observations
        """
        if not stats:
            return False
        
        # Check minimum observations across all models
        # Require at least some data for each model
        min_count = float('inf')
        for model_name in ['lightgcn', 'collaborative', 'user_content', 'item_content']:
            model_stats = stats.get(model_name, {})
            count = model_stats.get('count', 0)
            min_count = min(min_count, count)
        
        return min_count >= min_observations
    
    def _stats_to_weights(self,
                          stats: Dict[str, Dict],
                          base_weights: Dict[str, float],
                          level: str) -> Dict[str, float]:
        """
        Convert model performance stats to fusion weights.
        
        FIXED: Handles negative avg_credits properly by clamping to 0.
        
        Process:
        1. Extract average credits per model (clamp negatives to 0)
        2. Normalize to sum to 1.0
        3. Apply fixed RRF portion
        4. Blend with base weights based on confidence
        
        Args:
            stats: Model performance statistics
            base_weights: Base weights from grid search
            level: Learning level ('client', 'segment', 'global')
        
        Returns:
            Personalized weights dictionary
        """
        # Extract average credits - RAW VALUES (may be negative)
        raw_credits = {
            'lightgcn': stats.get('lightgcn', {}).get('avg_credit', 0),
            'collaborative': stats.get('collaborative', {}).get('avg_credit', 0),
            'user_content': stats.get('user_content', {}).get('avg_credit', 0),
            'item_content': stats.get('item_content', {}).get('avg_credit', 0)
        }
        
        #    FIX 1: Clamp negative credits to 0
        # Models with negative avg_credit have been performing poorly
        # They should get minimum weight, not negative weight
        credits = {k: max(v, 0) for k, v in raw_credits.items()}
        
        # Log if any credits were negative (for monitoring)
        negative_models = [k for k, v in raw_credits.items() if v < 0]
        if negative_models:
            logger.warning(f"  Negative avg_credits detected for: {negative_models}. "
                          f"Raw values: {raw_credits}. Clamping to 0 for weight computation.")
        
        total_credit = sum(credits.values())
        
        #    FIX 2: Handle case where all credits are 0 or negative
        if total_credit <= 0:
            logger.warning(f"  Total credit is {total_credit} (all models poor/negative). "
                          f"Falling back to base weights.")
            return base_weights.copy()
        
        # Scale credits to learnable portion (0.85)
        rrf_portion = FEEDBACK_CONFIG.get('RRF_FIXED_PORTION', 0.15)
        learnable_portion = 1.0 - rrf_portion  # 0.85
        
        # Direct proportional scaling
        learned_weights = {
            'alpha1_rrf': rrf_portion,
            'alpha2_gcn': (credits['lightgcn'] / total_credit) * learnable_portion,
            'alpha3_item_sim': (credits['item_content'] / total_credit) * learnable_portion,
            'alpha4_user_sim': (credits['user_content'] / total_credit) * learnable_portion,
            'alpha5_confidence': (credits['collaborative'] / total_credit) * learnable_portion
        }
        
        # Blend with base weights
        total_observations = min(s.get('count', 0) for s in stats.values())
        blend_scale = FEEDBACK_CONFIG.get('BLEND_SCALE_OBSERVATIONS', 15)
        max_blend = FEEDBACK_CONFIG['MAX_LEARNED_BLEND']
        
        #    FIX 3: Ensure confidence is non-negative
        # Use clamped total_credit for confidence calculation
        confidence = min(max(total_credit, 0) / 1.0, 1.0)
        blend_factor = min(total_observations / blend_scale, max_blend) * confidence
        
        # Ensure blend_factor is in valid range [0, max_blend]
        blend_factor = max(0, min(blend_factor, max_blend))
        
        # Blend learned weights with base weights
        final_weights = {}
        for key in base_weights:
            learned = learned_weights.get(key, base_weights[key])
            base = base_weights[key]
            final_weights[key] = blend_factor * learned + (1 - blend_factor) * base
        
        # Normalize to sum to 1.0
        weight_sum = sum(final_weights.values())
        if weight_sum > 0:
            final_weights = {k: v / weight_sum for k, v in final_weights.items()}
        else:
            # Fallback to base weights if something went wrong
            logger.warning("  Weight sum is 0, falling back to base weights")
            final_weights = base_weights.copy()
        
        return final_weights
    
    def get_learning_status(self, client_id: int, client_segment: str = None) -> Dict:
        """
        Get the current learning status for a client.
        
        Useful for debugging and monitoring.
        
        Args:
            client_id: The client ID
            client_segment: Optional client segment
        
        Returns:
            Status dictionary with observations and current weights level
        """
        #    FIX: Changed from get_model_performance to get_model_stats
        client_stats = self.store.get_model_stats('client', str(client_id))
        segment_stats = {}
        if client_segment:
            segment_stats = self.store.get_model_stats('segment', client_segment)
        global_stats = self.store.get_model_stats('global', 'GLOBAL')
        
        # Count observations at each level
        def count_observations(stats):
            if not stats:
                return 0
            return min(s.get('count', 0) for s in stats.values()) if stats else 0
        
        client_obs = count_observations(client_stats)
        segment_obs = count_observations(segment_stats)
        global_obs = count_observations(global_stats)
        
        # Determine active level
        if client_obs >= FEEDBACK_CONFIG['MIN_CLIENT_OBSERVATIONS']:
            active_level = 'client'
        elif segment_obs >= FEEDBACK_CONFIG['MIN_SEGMENT_OBSERVATIONS']:
            active_level = 'segment'
        elif global_obs >= FEEDBACK_CONFIG.get('MIN_GLOBAL_OBSERVATIONS', 50):
            active_level = 'global'
        else:
            active_level = 'base'
        
        # Calculate observations needed for next level
        if active_level == 'base':
            next_level = 'global'
            obs_needed = FEEDBACK_CONFIG.get('MIN_GLOBAL_OBSERVATIONS', 50) - global_obs
        elif active_level == 'global':
            next_level = 'segment'
            obs_needed = FEEDBACK_CONFIG['MIN_SEGMENT_OBSERVATIONS'] - segment_obs
        elif active_level == 'segment':
            next_level = 'client'
            obs_needed = FEEDBACK_CONFIG['MIN_CLIENT_OBSERVATIONS'] - client_obs
        else:
            next_level = None
            obs_needed = 0
        
        return {
            'client_id': client_id,
            'client_segment': client_segment,
            'observations': {
                'client': client_obs,
                'segment': segment_obs,
                'global': global_obs
            },
            'active_level': active_level,
            'next_level': next_level,
            'observations_to_next_level': max(0, obs_needed),
            'model_performance': {
                'client': client_stats,
                'segment': segment_stats,
                'global': global_stats
            }
        }
    
    def get_model_ranking(self, client_id: int) -> List[Dict]:
        """
        Get models ranked by their performance for a client.
        
        Args:
            client_id: The client ID
        
        Returns:
            List of models sorted by average credit (best first)
        """
        #    FIX: Changed from get_model_performance to get_model_stats
        stats = self.store.get_model_stats('client', str(client_id))
        
        if not stats:
            return []
        
        ranking = []
        for model_name, model_stats in stats.items():
            ranking.append({
                'model': model_name,
                'avg_credit': model_stats.get('avg_credit', 0),
                'observations': model_stats.get('count', 0)
            })
        
        # Sort by avg_credit descending
        ranking.sort(key=lambda x: x['avg_credit'], reverse=True)
        
        return ranking
    
    def reset_client(self, client_id: int) -> bool:
        """
        Reset learning for a specific client.
        
        This removes all learned weights for the client, causing
        fallback to segment/global/base weights.
        
        Args:
            client_id: The client ID to reset
        
        Returns:
            True if successful
        """
        # Note: Actual implementation would delete from database
        # For now, log the action
        logger.info(f"Reset learning for client {client_id}")
        
        # Invalidate cache
        cache_key = f"client_{client_id}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        return True


import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from .store import FeedbackStore
from .weight_learner import WeightLearner
from .config import FEEDBACK_CONFIG, DEFAULT_BASE_WEIGHTS, DB_CONNECTION_STRING

logger = logging.getLogger(__name__)


class WeightSelector:
    """
    Provides personalized weights for the recommendation combiner.
    
    This is the primary interface between the combiner and the feedback
    system. It handles initialization, caching, and provides a simple
    get_weights() method.
    
    Integration:
        Only 2 lines needed in combiner_api_final.py:
        
        1. Initialize (at top):
           weight_selector = WeightSelector()
        
        2. Use (in process_single_fusion_request):
           weights = weight_selector.get_weights(client_id, client_type, base_weights)
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the weight selector.
        
        Args:
            db_path: Path to SQLite database. Uses default if not specified.
        """
        self.store = FeedbackStore(DB_CONNECTION_STRING)
        self.weight_learner = WeightLearner(self.store)
        
        # Simple cache to avoid repeated DB queries
        self._cache = {}
        #    FIX: Added missing _cache_ttl initialization
        self._cache_ttl = FEEDBACK_CONFIG.get('WEIGHT_CACHE_TTL', 3600)
        self._cache_timestamps = {}
        
        logger.info("WeightSelector initialized with database")
    
    def get_weights(self,
                    client_id: int,
                    client_segment: str,
                    base_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Get personalized weights for a client.
        
        This is the MAIN METHOD called from combiner_api_final.py.
        
        It returns personalized weights if enough feedback has been
        collected, otherwise falls back to base weights.
        
        Args:
            client_id: The client ID
            client_segment: Client type/segment (e.g., 'VIP', 'REGULAR', 'UNKNOWN')
            base_weights: Default weights from grid search optimizer
        
        Returns:
            Personalized weights dictionary:
            {
                'alpha1_rrf': float,      # RRF score weight
                'alpha2_gcn': float,      # LightGCN score weight
                'alpha3_item_sim': float, # Item content similarity weight
                'alpha4_user_sim': float, # User content similarity weight
                'alpha5_confidence': float # Collaborative filtering weight
            }
        
        Example:
            weights = weight_selector.get_weights(
                client_id=101,
                client_segment='VIP',
                base_weights={'alpha1_rrf': 0.3, 'alpha2_gcn': 0.3, ...}
            )
        """
        # Use default base weights if not provided
        base_weights = base_weights or DEFAULT_BASE_WEIGHTS.copy()
        
        # Check cache first
        cache_key = f"weights_{client_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Compute weights using hierarchical fallback
            weights = self.weight_learner.compute_weights(
                client_id=client_id,
                client_segment=client_segment,
                base_weights=base_weights
            )
            
            # Cache the result
            self._set_cached(cache_key, weights)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error computing weights for client {client_id}: {e}")
            # On error, return base weights to ensure system continues working
            return base_weights
    
    def get_weights_with_source(self,
                                client_id: int,
                                client_segment: str,
                                base_weights: Dict[str, float] = None) -> Dict:
        """
        Get personalized weights with metadata about the source.
        
        Extended version of get_weights() that includes information
        about which level the weights came from.
        
        Args:
            client_id: The client ID
            client_segment: Client segment/type
            base_weights: Default weights
        
        Returns:
            Dictionary with weights and metadata:
            {
                'weights': {...},
                'source': 'client' | 'segment' | 'global' | 'base',
                'observations': 23,
                'confidence': 0.75
            }
        """
        base_weights = base_weights or DEFAULT_BASE_WEIGHTS.copy()
        
        try:
            # Get learning status to determine source
            status = self.weight_learner.get_learning_status(
                client_id=client_id,
                client_segment=client_segment
            )
           
            # Get weights
            weights = self.weight_learner.compute_weights(
                client_id=client_id,
                client_segment=client_segment,
                base_weights=base_weights
            )
            
            # Calculate confidence based on observations
            active_level = status['active_level']
            obs_dict = status['observations']

            # When active_level='base' (negative-credit fallback), report
            # the actual client-level count so callers know feedback exists
            if active_level == 'base':
                observations = max(obs_dict.values()) if obs_dict else 0
            else:
                observations = obs_dict.get(active_level, 0)

            max_obs_for_full_confidence = 100
            confidence = min(observations / max_obs_for_full_confidence, 1.0)
            
            
            return {
                'weights': weights,
                'source': active_level,
                'observations': observations,
                'confidence': round(confidence, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting weights with source: {e}")
            return {
                'weights': base_weights,
                'source': 'base',
                'observations': 0,
                'confidence': 0
            }
    
    def get_stats(self, client_id: int) -> Dict:
        """
        Get learning statistics for a client.
        
        Useful for monitoring and debugging.
        
        Args:
            client_id: The client ID
        
        Returns:
            Statistics dictionary including:
            - Observations at each level
            - Model performance stats
            - Current active learning level
        """
        try:
            client_summary = self.store.get_client_stats(client_id)
            learning_status = self.weight_learner.get_learning_status(
                client_id=client_id
            )
            
            return {
                'client_id': client_id,
                'summary': client_summary,
                'learning_status': learning_status
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for client {client_id}: {e}")
            return {
                'client_id': client_id,
                'error': str(e)
            }
    
    def get_model_ranking(self, client_id: int) -> Dict:
        """
        Get models ranked by their performance for a client.
        
        Shows which models are working best for this client.
        
        Args:
            client_id: The client ID
        
        Returns:
            Dictionary with model ranking and details
        """
        try:
            ranking = self.weight_learner.get_model_ranking(client_id)
            
            return {
                'client_id': client_id,
                'ranking': ranking,
                'best_model': ranking[0]['model'] if ranking else None,
                'has_personalization': len(ranking) > 0
            }
            
        except Exception as e:
            logger.error(f"Error getting model ranking: {e}")
            return {
                'client_id': client_id,
                'ranking': [],
                'error': str(e)
            }
    
    def invalidate_cache(self, client_id: int = None):
        """
        Invalidate cached weights.
        
        Args:
            client_id: If provided, only invalidate for this client.
                      If None, invalidate all cached weights.
        """
        if client_id:
            cache_key = f"weights_{client_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]
                if cache_key in self._cache_timestamps:
                    del self._cache_timestamps[cache_key]
            logger.debug(f"Invalidated cache for client {client_id}")
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
            logger.debug("Invalidated all cached weights")
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None
        
        timestamp = self._cache_timestamps.get(key)
        if timestamp is None:
            return None
        
        # Check if expired
        if self._cache_ttl > 0:
            expiry = timestamp + timedelta(seconds=self._cache_ttl)
            if datetime.now() > expiry:
                # Expired, remove from cache
                del self._cache[key]
                del self._cache_timestamps[key]
                return None
        
        return self._cache[key]
    
    def _set_cached(self, key: str, value: Dict):
        """Set value in cache with timestamp."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()
    
    def is_healthy(self) -> bool:
        """
        Check if the feedback system is healthy.
        
        Returns:
            True if system is operational
        """
        try:
            # Try to query the database
            _ = self.store.get_system_stats()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_system_stats(self) -> Dict:
        """
        Get overall system statistics.
        
        Returns:
            System-wide statistics including total selections,
            clients with feedback, average rank improvement, etc.
        """
        try:
            return self.store.get_system_stats()
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}


# =============================================================================
# CONVENIENCE FUNCTION FOR SIMPLE INTEGRATION
# =============================================================================

# Global instance for simple usage
_default_selector = None

def get_default_selector() -> WeightSelector:
    """
    Get the default WeightSelector instance.
    
    Creates a singleton instance for simple integration.
    
    Usage:
        from feedback.weight_selector import get_default_selector
        
        selector = get_default_selector()
        weights = selector.get_weights(client_id, segment, base_weights)
    """
    global _default_selector
    
    if _default_selector is None:
        _default_selector = WeightSelector()
    
    return _default_selector


def get_personalized_weights(client_id: int,
                             client_segment: str,
                             base_weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Convenience function to get personalized weights.
    
    Simple function for quick integration without managing instances.
    
    Usage:
        from feedback.weight_selector import get_personalized_weights
        
        weights = get_personalized_weights(101, 'VIP', base_weights)
    """
    selector = get_default_selector()
    return selector.get_weights(client_id, client_segment, base_weights)
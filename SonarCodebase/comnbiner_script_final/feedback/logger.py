

import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
import pandas as pd

from .store import FeedbackStore
from .config import MODEL_NAMES

logger = logging.getLogger(__name__)


class FeedbackLogger:
    """
    Logs recommendation events for later feedback processing.
    
    This class captures what was shown to the dispatcher, including
    how each model ranked each pair, enabling attribution when
    feedback is received.
    """
    
    def __init__(self, store: FeedbackStore = None):
        """
        Initialize the feedback logger.
        
        Args:
            store: FeedbackStore instance. If None, creates a new one.
        """
        self.store = store or FeedbackStore()
        logger.info("FeedbackLogger initialized")
    
    def log_shown_event(self,
                        request_id: str,
                        client_id: int,
                        client_segment: str,
                        fused_df: pd.DataFrame,
                        weights_used: Dict[str, float]) -> str:
        """
        Log what was shown to dispatcher with per-model rankings.
        
        This method extracts ranking and score information from each model's
        output and stores it for later feedback attribution.
        
        Args:
            request_id: Unique identifier for the request
            client_id: Client ID
            client_segment: Client segment/type (e.g., 'VIP', 'REGULAR')
            fused_df: DataFrame with fused recommendations (final output)
            lightgcn_df: DataFrame from LightGCN model
            collaborative_df: DataFrame from Collaborative Filtering model
            user_content_df: DataFrame from User Content model
            item_content_df: DataFrame from Item Content model
            weights_used: Weights used for fusion
        
        Returns:
            request_id for later matching with feedback
        """
        try:
            
            
            # Build pairs_shown list with all metadata
            pairs_shown = []
            
            for _, row in fused_df.iterrows():
                pair_id = row['pair_id']
                
                pair_info = {
                    # Basic info
                    'pair_id': pair_id,
                    'final_rank': int(row.get('final_rank', 0)),
                    'final_score': float(row.get('final_score', 0)),
              
                    'lightgcn_score': float(row.get('gcn_score', 0)),
                    'collaborative_score': float(row.get('cf_score', 0)),
                    'user_content_score': float(row.get('user_content_score', 0)),
                    'item_content_score': float(row.get('item_content_score', 0)),

                    
                    # Component scores from fusion
                    'rrf_score': float(row.get('rrf_score', 0)),
                    
                    # Metadata (if available)
                    'vehicle_id': row.get('vehicle_id'),
                    'chauffeur_id': row.get('chauffeur_id')
                }
                
                pairs_shown.append(pair_info)
            
            #    FIX: Build model_rankings dict for store.save_shown_event()
          
            #    FIX: Call store.save_shown_event with individual parameters
            success = self.store.save_shown_event(
                request_id=request_id,
                client_id=client_id,
                client_segment=client_segment,
                pairs_shown=pairs_shown,
                weights_used=weights_used,
                batch_id=None  # Optional, can be added later
            )
            
            if success:
                logger.info(f"Logged shown_event: {request_id} with {len(pairs_shown)} pairs "
                           f"for client {client_id} ({client_segment})")
            else:
                logger.error(f"Failed to log shown_event: {request_id}")
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error logging shown event: {e}", exc_info=True)
            # Don't fail the recommendation - just log the error
            return request_id
    
    
    def _build_score_lookup(self, 
                            df: pd.DataFrame, 
                            primary_col: str,
                            fallback_cols: List[str]) -> Dict[str, float]:
        """
        Build a lookup dictionary mapping pair_id to score.
        
        Args:
            df: Model output DataFrame
            primary_col: Primary score column to look for
            fallback_cols: Fallback columns if primary not found
        
        Returns:
            Dict mapping pair_id to score
        """
        if df is None or df.empty:
            return {}
        
        lookup = {}
        
        # Find score column
        score_col = None
        if primary_col in df.columns:
            score_col = primary_col
        else:
            for col in fallback_cols:
                if col in df.columns:
                    score_col = col
                    break
        
        if score_col is None:
            return {}
        
        for _, row in df.iterrows():
            pair_id = str(row.get('pair_id', ''))
            if pair_id:
                score = row.get(score_col, 0)
                lookup[pair_id] = float(score) if pd.notna(score) else 0.0
        
        return lookup
    
   

    
    def log_shown_event_simple(self,
                               request_id: str,
                               client_id: int,
                               client_segment: str,
                               recommendations: List[Dict],
                               weights_used: Dict[str, float]) -> str:
        """
        Simplified logging when model DataFrames are not available.
        
        Use this when you only have the final recommendations list
        without access to individual model outputs.
        
        Args:
            request_id: Unique identifier for the request
            client_id: Client ID
            client_segment: Client segment/type
            recommendations: List of recommendation dicts from combiner
            weights_used: Weights used for fusion
        
        Returns:
            request_id for later matching
        """
        try:
            pairs_shown = []
            
            for idx, rec in enumerate(recommendations):
                pair_info = {
                    'pair_id': rec.get('pair_id'),
                    'final_rank': rec.get('final_rank', idx + 1),
                    'final_score': rec.get('final_score', 0),
                    
                    # Extract model ranks if available in recommendation
                    'lightgcn_rank': rec.get('lightgcn_rank'),
                    'collaborative_rank': rec.get('collaborative_rank'),
                    'user_content_rank': rec.get('user_content_rank'),
                    'item_content_rank': rec.get('item_content_rank'),
                    
                    # Scores from fusion result
                    'gcn_score': rec.get('gcn_score', 0),
                    'cf_score': rec.get('cf_score', 0),
                    'user_content_score': rec.get('user_content_score', 0),
                    'item_content_score': rec.get('item_content_score', 0),
                    'rrf_score': rec.get('rrf_score', 0),
                    
                    # Metadata
                    'vehicle_id': rec.get('vehicle_id'),
                    'chauffeur_id': rec.get('chauffeur_id')
                }
                pairs_shown.append(pair_info)
            
            #    FIX: Call with correct parameters
            success = self.store.save_shown_event(
                request_id=request_id,
                client_id=client_id,
                client_segment=client_segment,
                pairs_shown=pairs_shown,
                weights_used=weights_used,
                batch_id=None
            )
            
            if success:
                logger.info(f"Logged shown_event (simple): {request_id} with {len(pairs_shown)} pairs")
            
            return request_id
            
        except Exception as e:
            logger.error(f"Error logging shown event (simple): {e}", exc_info=True)
            return request_id
    
    def get_shown_event(self, request_id: str) -> Optional[Dict]:
        """
        Retrieve a previously logged shown event.
        
        Args:
            request_id: The request identifier
        
        Returns:
            Event dictionary or None if not found
        """
        return self.store.get_shown_event(request_id)
    
    def has_shown_event(self, request_id: str) -> bool:
        """
        Check if a shown event exists for a request.
        
        Args:
            request_id: The request identifier
        
        Returns:
            True if event exists
        """
        return self.store.get_shown_event(request_id) is not None
#feedback file

from feedback.config import DB_CONFIG

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
from sqlalchemy import create_engine, text
import pandas as pd
from .config import DB_CONNECTION_STRING
logger = logging.getLogger(__name__)


class FeedbackStore:
    """
    Feedback storage using SQL Server instead of SQLite
    """
    
    def __init__(self, connection_string: str = None):
        """
        Initialize SQL Server connection
        
        Args:
            connection_string: SQL Server connection string
                              Default: localhost/TransElite.Q3
        """
        if connection_string is None:
            connection_string = DB_CONNECTION_STRING
    
     
        
        self.connection_string = connection_string
        self.engine = create_engine(connection_string)
        logger.info("   SQL Server Feedback Store initialized")
    
    # =========================================================================
    # SHOWN EVENTS - Log what was shown to dispatcher
    # =========================================================================
    
    def save_shown_event(self, request_id: str, client_id: int, 
                    client_segment: str, pairs_shown: List[Dict],
                    weights_used: Dict,
                    batch_id: str = None) -> bool:
        """
        DEPRECATED: Data is now stored directly in BookingRecommendations.
        This method is kept for backward compatibility but does nothing.
        """
        logger.info("save_shown_event() is deprecated - data stored in BookingRecommendations")
        return True  # No-op, data already in BookingRecommendations
    
    def get_shown_event(self, request_id: str) -> Optional[Dict]:
        """
        Get shown event from AvailabilityAllocation table.
        
        UPDATED: Now reads directly from AvailabilityAllocation instead of
        feedback_shown_events/feedback_shown_pairs tables.
        
        Args:
            request_id: BookingID as string
            
        Returns:
            Dict with event details and pairs_shown list
        """
        try:
            with self.engine.connect() as conn:
                # Get recommendations from AvailabilityAllocation
                query = text("""
                    SELECT 
                        BookingID,
                        ClientId,
                        ClientType,
                        PairId,
                        VehicleId,
                        ChauffeurId,
                        FinalRank,
                        FinalScore,
                        RrfScore,
                        GcnScore,
                        CfScore,
                        UserContentScore,
                        ItemContentScore,
                        WeightsUsed,
                        BatchId,
                        CreatedDate
                    FROM AvailabilityAllocation
                    WHERE BookingID = :booking_id
                    ORDER BY FinalRank
                """)
                
                result = conn.execute(query, {'booking_id': int(request_id)})
                rows = result.fetchall()
                
                if not rows:
                    return None
                
                # Build event dict from first row
                first_row = rows[0]
                
                # Parse weights
                weights_used = {}
                if first_row.WeightsUsed:
                    try:
                        weights_used = json.loads(first_row.WeightsUsed)
                    except Exception as e:
                        weights_used = {}
                
                event = {
                    'request_id': request_id,
                    'batch_id': first_row.BatchId,
                    'client_id': first_row.ClientId,
                    'client_segment': first_row.ClientType,
                    'total_shown': len(rows),
                    'weights_used': weights_used,
                    'shown_at': first_row.CreatedDate
                }
                
                # Build pairs_shown list
                pairs_shown = []
                for row in rows:
                    pairs_shown.append({
                        'pair_id': row.PairId,
                        'vehicle_id': row.VehicleId,
                        'chauffeur_id': row.ChauffeurId,
                        'rank_shown': row.FinalRank,
                        'final_score': float(row.FinalScore) if row.FinalScore else 0.0,
                        'rrf_score': float(row.RrfScore) if row.RrfScore else 0.0,
                        'lightgcn_score': float(row.GcnScore) if row.GcnScore else 0.0,
                        'collaborative_score': float(row.CfScore) if row.CfScore else 0.0,
                        'user_content_score': float(row.UserContentScore) if row.UserContentScore else 0.0,
                        'item_content_score': float(row.ItemContentScore) if row.ItemContentScore else 0.0
                    })
                
                event['pairs_shown'] = pairs_shown
                
                return event
                
        except Exception as e:
            logger.error(f"  Error getting shown event: {e}", exc_info=True)
            return None
        
    # =========================================================================
    # SELECTIONS - Save dispatcher selections
    # =========================================================================
    
    def save_selection(self, request_id: str, selected_pair_id: str,
                  selected_rank: int, overall_quality: int,
                  client_id: int, client_segment: str = None,
                  model_credits: Dict = None) -> bool:
        """
        Save dispatcher selection by updating AvailabilityAllocation table.
        
        UPDATED: Now updates AvailabilityAllocation (sets IsSelected=1 on the
        selected pair) instead of inserting into feedback_selected_pairs.
        
        Args:
            request_id: BookingID as string
            selected_pair_id: The selected pair ID
            selected_rank: Rank of selected pair
            overall_quality: Quality rating 1-5
            client_id: Client ID (not used - already in AvailabilityAllocation)
            client_segment: Client segment (not used - already in AvailabilityAllocation)
            model_credits: Dict of model credits
            
        Returns:
            True if successful
        """
        try:
            with self.engine.connect() as conn:
                credits = model_credits or {}
                
                # Update the selected pair in AvailabilityAllocation
                query = text("""
                    UPDATE AvailabilityAllocation
                    SET 
                        IsSelected = 1,
                        OverallQuality = :overall_quality,
                        LightGcnCredit = :lightgcn_credit,
                        CollaborativeCredit = :collaborative_credit,
                        UserContentCredit = :user_content_credit,
                        ItemContentCredit = :item_content_credit,
                        SelectedAt = GETDATE()
                    WHERE BookingID = :booking_id
                    AND PairId = :pair_id
                """)
                
                result = conn.execute(query, {
                    'booking_id': int(request_id),
                    'pair_id': selected_pair_id,
                    'overall_quality': overall_quality,
                    'lightgcn_credit': credits.get('lightgcn', 0.0),
                    'collaborative_credit': credits.get('collaborative', 0.0),
                    'user_content_credit': credits.get('user_content', 0.0),
                    'item_content_credit': credits.get('item_content', 0.0)
                })
                
                conn.commit()
                
                if result.rowcount > 0:
                    logger.info(f"   Saved selection: {selected_pair_id} at rank {selected_rank} for BookingID {request_id}")
                    return True
                else:
                    logger.warning(f"  No row updated for BookingID {request_id}, PairId {selected_pair_id}")
                    return False
                
        except Exception as e:
            logger.error(f"  Error saving selection: {e}", exc_info=True)
            return False

    def save_selection_credits(self, request_id: str, selected_pair_id: str,
                              overall_quality: int, model_credits: Dict = None) -> bool:
        """
        Save ONLY the credits/quality for a selection.
        
        IMPORTANT: This does NOT set IsSelected=1 - the SP does that!
        This method only updates:
            - OverallQuality
            - LightGcnCredit, CollaborativeCredit, UserContentCredit, ItemContentCredit
        
        Args:
            request_id: BookingID as string
            selected_pair_id: The selected pair ID
            overall_quality: Quality rating 1-5
            model_credits: Dict of model credits
            
        Returns:
            True if successful
        """
        try:
            with self.engine.connect() as conn:
                credits = model_credits or {}
                
                # Only update credits - NOT IsSelected (SP handles that)
                query = text("""
                    UPDATE AvailabilityAllocation
                    SET 
                        OverallQuality = :overall_quality,
                        LightGcnCredit = :lightgcn_credit,
                        CollaborativeCredit = :collaborative_credit,
                        UserContentCredit = :user_content_credit,
                        ItemContentCredit = :item_content_credit
                    WHERE BookingID = :booking_id
                    AND PairId = :pair_id
                """)
                
                result = conn.execute(query, {
                    'booking_id': int(request_id),
                    'pair_id': selected_pair_id,
                    'overall_quality': overall_quality,
                    'lightgcn_credit': credits.get('lightgcn', 0.0),
                    'collaborative_credit': credits.get('collaborative', 0.0),
                    'user_content_credit': credits.get('user_content', 0.0),
                    'item_content_credit': credits.get('item_content', 0.0)
                })
                
                conn.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Saved credits for: {selected_pair_id} (BookingID {request_id})")
                    return True
                else:
                    logger.warning(f"No row updated for BookingID {request_id}, PairId {selected_pair_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Error saving selection credits: {e}", exc_info=True)
            return False 
    # =========================================================================
    # REJECTIONS - Save rejected recommendations
    # =========================================================================
    
    def save_rejection(self, request_id: str, pair_id: str, rank: int,
                  rejection_reason: str, notes: str = None,
                  classification_data: str = None) -> bool:
        """
        Save rejection with reason and full classification data.
        
        UPDATED: Now accepts classification_data parameter to store
        the complete LLM classification including all categories.
        
        Args:
            request_id: The request ID
            pair_id: The rejected pair ID
            rank: Rank of rejected pair
            rejection_reason: All categories with weights (comma-separated)
                            e.g., "chauffeur_quality:0.60,past_negative_exp:0.40"
            notes: Original dispatcher text
            classification_data: Full JSON of LLM classification (optional)
        """
        try:
            # Check if ANY learnable category is present
            is_learnable = self._is_learnable_reason(rejection_reason)
            
            with self.engine.connect() as conn:
                query = text("""
                    INSERT INTO dbo.feedback_rejections
                    (request_id, pair_id, rank_rejected, rejection_reason, 
                    is_learnable, dispatcher_notes, classification_data)
                    VALUES
                    (:request_id, :pair_id, :rank_rejected, :rejection_reason,
                    :is_learnable, :dispatcher_notes, :classification_data)
                """)
                
                conn.execute(query, {
                    'request_id': request_id,
                    'pair_id': pair_id,
                    'rank_rejected': rank,
                    'rejection_reason': rejection_reason,
                    'is_learnable': is_learnable,
                    'dispatcher_notes': notes,
                    'classification_data': classification_data  # NEW!
                })
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"  Error saving rejection: {e}", exc_info=True)
            return False
        
    def _is_learnable_reason(self, rejection_reason: str) -> bool:
        """
        Check if rejection contains any learnable category.
        
        UPDATED: Now handles comma-separated format with multiple categories.
        
        Args:
            rejection_reason: Can be:
                - Single category: "chauffeur_quality"
                - Multiple categories: "chauffeur_quality:0.60,past_negative_exp:0.40"
        
        Returns:
            True if ANY category is learnable
        """
        from .config import REJECTION_CATEGORIES
        
        # Handle comma-separated format: "cat1:0.6,cat2:0.4"
        if ':' in rejection_reason:
            # Extract just the category names
            categories = [part.split(':')[0].strip() for part in rejection_reason.split(',')]
        else:
            # Single category (backward compatibility)
            categories = [rejection_reason]
        
        # Check if ANY category is learnable
        for category in categories:
            if category in REJECTION_CATEGORIES:
                if REJECTION_CATEGORIES[category].get('learnable', False):
                    return True
        
        return False 
    # =========================================================================
    # WEIGHTS - Client personalized weights
    # =========================================================================
    

    def save_client_weights(self, client_id: int, weights: Dict,
                           observations: int, client_segment: str = None) -> bool:
        """
        Save or update client weights
        """
        try:
            with self.engine.connect() as conn:
                # Check if exists
                check_query = text("""
                    SELECT weight_id FROM dbo.feedback_client_weights
                    WHERE client_id = :client_id
                """)
                
                result = conn.execute(check_query, {'client_id': client_id})
                exists = result.fetchone() is not None
                
                if exists:
                    # Update
                    query = text("""
                        UPDATE dbo.feedback_client_weights
                        SET alpha1_rrf = :alpha1_rrf,
                            alpha2_gcn = :alpha2_gcn,
                            alpha3_item_sim = :alpha3_item_sim,
                            alpha4_user_sim = :alpha4_user_sim,
                            alpha5_confidence = :alpha5_confidence,
                            observations = :observations,
                            updated_at = GETDATE()
                        WHERE client_id = :client_id
                    """)
                else:
                    # Insert
                    query = text("""
                        INSERT INTO dbo.feedback_client_weights
                        (client_id, client_segment, alpha1_rrf, alpha2_gcn, 
                         alpha3_item_sim, alpha4_user_sim, alpha5_confidence, observations)
                        VALUES
                        (:client_id, :client_segment, :alpha1_rrf, :alpha2_gcn,
                         :alpha3_item_sim, :alpha4_user_sim, :alpha5_confidence, :observations)
                    """)
                
                conn.execute(query, {
                    'client_id': client_id,
                    'client_segment': client_segment,
                    'alpha1_rrf': weights.get('alpha1_rrf', 0.2),
                    'alpha2_gcn': weights.get('alpha2_gcn', 0.35),
                    'alpha3_item_sim': weights.get('alpha3_item_sim', 0.225),
                    'alpha4_user_sim': weights.get('alpha4_user_sim', 0.125),
                    'alpha5_confidence': weights.get('alpha5_confidence', 0.1),
                    'observations': observations
                })
                
                conn.commit()
                logger.info(f"   Saved weights for client {client_id} ({observations} observations)")
                return True
                
        except Exception as e:
            logger.error(f"  Error saving weights: {e}", exc_info=True)
            return False
    
    def get_client_weights(self, client_id: int) -> Optional[Dict]:
        """
        Get client weights from SQL Server
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT * FROM dbo.feedback_client_weights
                    WHERE client_id = :client_id
                """)
                
                result = conn.execute(query, {'client_id': client_id})
                row = result.fetchone()
                
                if row:
                    return dict(row._mapping)
                return None
                
        except Exception as e:
            logger.error(f"  Error getting client weights: {e}")
            return None
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
   # In store.py - replace get_system_stats()
    def get_system_stats(self) -> Dict:
        """Get system-wide statistics from BookingRecommendations"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT 
                        COUNT(DISTINCT BookingID) as total_shown_events,
                        COUNT(CASE WHEN IsSelected = 1 THEN 1 END) as total_selections,
                        COUNT(DISTINCT CASE WHEN IsSelected = 1 THEN ClientId END) as unique_clients_with_feedback,
                        AVG(CASE WHEN IsSelected = 1 THEN CAST(FinalRank as FLOAT) END) as average_selected_rank
                    FROM AvailabilityAllocation
                """)
                result = conn.execute(query)
                row = result.fetchone()
                
                if row:
                    return dict(row._mapping)
                return {}
                
        except Exception as e:
            logger.error(f"  Error getting system stats: {e}")
            return {}

    # In store.py - replace get_client_stats()
    def get_client_stats(self, client_id: int) -> Dict:
        """Get client-specific statistics from BookingRecommendations"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT 
                        ClientId as client_id,
                        COUNT(DISTINCT BookingID) as total_shown,
                        COUNT(CASE WHEN IsSelected = 1 THEN 1 END) as total_selections,
                        AVG(CASE WHEN IsSelected = 1 THEN CAST(FinalRank as FLOAT) END) as avg_selected_rank
                    FROM AvailabilityAllocation
                    WHERE ClientId = :client_id
                    GROUP BY ClientId
                """)
                
                result = conn.execute(query, {'client_id': client_id})
                row = result.fetchone()
                
                if row:
                    return dict(row._mapping)
                return {}
                
        except Exception as e:
            logger.error(f"  Error getting client stats: {e}")
            return {}
        


    # =========================================================================
    # MODEL PERFORMANCE - Track per-model statistics for weight learning
    # =========================================================================
    
    def get_model_stats(self, level: str, identifier: str) -> Dict[str, Dict]:
        """
        Get model performance statistics for a given level and identifier.
        
        This is the CORE method that weight_learner.py uses to retrieve
        per-model performance data.
        
        Args:
            level: 'client', 'segment', or 'global'
            identifier: client_id (as string), segment name, or 'GLOBAL'
        
        Returns:
            Dict mapping model name to stats:
            {
                'lightgcn': {
                    'avg_credit': 0.45,
                    'total_credit': 11.25,
                    'count': 25,
                    'updated_at': '2025-01-15T10:30:00'
                },
                'collaborative': {...},
                'user_content': {...},
                'item_content': {...}
            }
            
            Returns empty dict {} if no stats found.
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT 
                        model_name,
                        avg_credit,
                        total_credit,
                        count,
                        updated_at
                    FROM dbo.feedback_model_performance
                    WHERE level = :level
                      AND identifier = :identifier
                """)
                
                result = conn.execute(query, {
                    'level': level,
                    'identifier': str(identifier)
                })
                
                stats = {}
                for row in result.fetchall():
                    stats[row.model_name] = {
                        'avg_credit': float(row.avg_credit),
                        'total_credit': float(row.total_credit),
                        'count': int(row.count),
                        'updated_at': row.updated_at.isoformat() if row.updated_at else None
                    }
                
                return stats
                
        except Exception as e:
            logger.error(f"  Error getting model stats for {level}/{identifier}: {e}", exc_info=True)
            return {}
    
    def save_model_stats(self, level: str, identifier: str, 
                        model_name: str, avg_credit: float, 
                        total_credit: float, count: int) -> bool:
        """
        Save or update model performance statistics.
        
        Uses UPSERT pattern (update if exists, insert if not).
        
        Args:
            level: 'client', 'segment', or 'global'
            identifier: client_id (as string), segment name, or 'GLOBAL'
            model_name: 'lightgcn', 'collaborative', 'user_content', 'item_content'
            avg_credit: Current average credit (EMA)
            total_credit: Total accumulated credit
            count: Number of observations
        """
        try:
            with self.engine.connect() as conn:
                # Check if exists
                check_query = text("""
                    SELECT performance_id 
                    FROM dbo.feedback_model_performance
                    WHERE level = :level
                      AND identifier = :identifier
                      AND model_name = :model_name
                """)
                
                result = conn.execute(check_query, {
                    'level': level,
                    'identifier': str(identifier),
                    'model_name': model_name
                })
                
                exists = result.fetchone() is not None
                
                if exists:
                    # Update existing
                    query = text("""
                        UPDATE dbo.feedback_model_performance
                        SET avg_credit = :avg_credit,
                            total_credit = :total_credit,
                            count = :count,
                            updated_at = GETDATE()
                        WHERE level = :level
                          AND identifier = :identifier
                          AND model_name = :model_name
                    """)
                else:
                    # Insert new
                    query = text("""
                        INSERT INTO dbo.feedback_model_performance
                        (level, identifier, model_name, avg_credit, total_credit, count)
                        VALUES
                        (:level, :identifier, :model_name, :avg_credit, :total_credit, :count)
                    """)
                
                conn.execute(query, {
                    'level': level,
                    'identifier': str(identifier),
                    'model_name': model_name,
                    'avg_credit': avg_credit,
                    'total_credit': total_credit,
                    'count': count
                })
                
                conn.commit()
                
                action = "Updated" if exists else "Inserted"
                logger.debug(f"   {action} model stats: {level}/{identifier}/{model_name} "
                           f"(avg={avg_credit:.4f}, count={count})")
                return True
                
        except Exception as e:
            logger.error(f"  Error saving model stats: {e}", exc_info=True)
            return False
    
    def update_model_performance(self, level: str, identifier: str,
                                model_credits: Dict[str, float],
                                learning_rate: float = 0.1) -> bool:
        """
        Update model performance using Exponential Moving Average (EMA).
        
        This is the main method called after each dispatcher selection.
        
        Formula: new_avg = (1 - α) × old_avg + α × new_credit
        
        Args:
            level: 'client', 'segment', or 'global'
            identifier: client_id (as string), segment name, or 'GLOBAL'
            model_credits: Dict mapping model name to credit value
                          e.g., {'lightgcn': 0.333, 'collaborative': 0.125, ...}
            learning_rate: α parameter for EMA (default 0.1)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current stats
            current_stats = self.get_model_stats(level, identifier)
            
            # Update each model
            for model_name, new_credit in model_credits.items():
                if model_name not in current_stats:
                    # First observation for this model
                    avg_credit = new_credit
                    total_credit = new_credit
                    count = 1
                else:
                    # EMA update
                    old_avg = current_stats[model_name]['avg_credit']
                    old_total = current_stats[model_name]['total_credit']
                    old_count = current_stats[model_name]['count']
                    
                    # Exponential Moving Average
                    avg_credit = (1 - learning_rate) * old_avg + learning_rate * new_credit
                    total_credit = old_total + new_credit
                    count = old_count + 1
                
                # Save updated stats
                success = self.save_model_stats(
                    level=level,
                    identifier=identifier,
                    model_name=model_name,
                    avg_credit=avg_credit,
                    total_credit=total_credit,
                    count=count
                )
                
                if not success:
                    logger.warning(f"Failed to update {model_name} for {level}/{identifier}")
            
            logger.info(f"   Updated model performance for {level}/{identifier}")
            return True
            
        except Exception as e:
            logger.error(f"  Error updating model performance: {e}", exc_info=True)
            return False
    
    def get_model_performance_summary(self, min_observations: int = 1) -> List[Dict]:
        """
        Get summary of all model performance data.
        
        Useful for debugging and monitoring.
        
        Args:
            min_observations: Only include models with at least this many observations
        
        Returns:
            List of dicts with performance data
        """
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT 
                        level,
                        identifier,
                        model_name,
                        avg_credit,
                        total_credit,
                        count,
                        updated_at
                    FROM dbo.feedback_model_performance
                    WHERE count >= :min_observations
                    ORDER BY level, identifier, avg_credit DESC
                """)
                
                result = conn.execute(query, {'min_observations': min_observations})
                
                summary = []
                for row in result.fetchall():
                    summary.append({
                        'level': row.level,
                        'identifier': row.identifier,
                        'model_name': row.model_name,
                        'avg_credit': float(row.avg_credit),
                        'total_credit': float(row.total_credit),
                        'count': int(row.count),
                        'updated_at': row.updated_at.isoformat() if row.updated_at else None
                    })
                
                return summary
                
        except Exception as e:
            logger.error(f"  Error getting performance summary: {e}")
            return []

    # ==============================================================================
    # USAGE EXAMPLE
    # ==============================================================================
    """
    # After a dispatcher selection, the weight_learner calls:

    learner = WeightLearner(store)

    # 1. Update model performance at all levels
    learner.update(
        client_id=101,
        client_segment='VIP',
        model_credits={
            'lightgcn': 0.333,      # Selected pair was rank 3 in LightGCN
            'collaborative': 0.125,  # Rank 8 in Collaborative
            'user_content': 0.500,   # Rank 2 in User Content
            'item_content': 0.100    # Rank 10 in Item Content
        }
    )

    # This calls store.update_model_performance() which:
    # 1. Gets current stats from feedback_model_performance
    # 2. Updates each model using EMA
    # 3. Saves back to feedback_model_performance

    # 2. Later, when getting weights for next request:
    stats = store.get_model_stats('client', '101')
    # Returns:
    # {
    #     'lightgcn': {'avg_credit': 0.45, 'count': 25},
    #     'collaborative': {'avg_credit': 0.15, 'count': 25},
    #     'user_content': {'avg_credit': 0.30, 'count': 25},
    #     'item_content': {'avg_credit': 0.10, 'count': 25}
    # }

    # 3. Weight learner converts these to fusion weights:
    # Higher avg_credit → Higher weight in fusion
    """









    # =========================================================================
    # HELPERS
    # =========================================================================
    
    


    def _is_learnable_reason(self, reason: str) -> bool:
        """Check if rejection reason is learnable"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT is_learnable FROM dbo.feedback_rejection_reasons
                    WHERE reason_code = :reason
                """)
                
                result = conn.execute(query, {'reason': reason})
                row = result.fetchone()
                
                return row[0] if row else False
                
        except Exception as e :
            return False
    
    def is_healthy(self) -> bool:
        """Check if SQL Server connection is healthy"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Error logging shown event (simple): {e}")
            return False
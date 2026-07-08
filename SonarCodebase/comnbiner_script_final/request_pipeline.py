# combiner
"""
Real-time Request Pipeline - UPDATED
Extracts all match_scores from Collaborative
Applies weighted feature scoring from LightGCN
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from config import BUSINESS_RULES
from data_loaders import DataLoader
from fusion_models import RecommendationFusion
from grid_search import GridSearchOptimizer

from config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """
    Real-time recommendation pipeline with comprehensive feature scoring
    """
    
    def __init__(self):
        """Initialize pipeline"""
        self.fusion = RecommendationFusion()
        self.optimizer = GridSearchOptimizer()
        self.loader = DataLoader()
        
        logger.info("Loading client types from client_final.csv...")
        self.client_types = self.loader.load_client_types()
        logger.info(f"Loaded {len(self.client_types)} client types")
        
        # Weights for LightGCN feature scoring
        self.lightgcn_feature_weights = {
            'cv_score': 0.25,
            'cc_score': 0.25,
            'vc_score': 0.25,
            'combined_score': 0.10,
            'vehicle_completion_rate': 0.05,
            'chauffeur_completion_rate': 0.05,
            'vehicle_repeat_client_rate': 0.025,
            'chauffeur_repeat_client_rate': 0.025
        }
        
        # Weights for Collaborative match_scores
        self.cf_match_weights = {
            'cf_score': 0.4,  # predicted_rating
            'overall_match': 0.3,
            'quality_score': 0.2,
            'requirement_match': 0.1
        }
    
    def extract_collaborative_scores(self, cf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all match_scores from Collaborative API response
        Creates a combined CF score from: cf_score, overall_match, quality_score, requirement_match
        """
        if cf_df.empty:
            return cf_df
        
        logger.info("Extracting Collaborative match_scores...")
        
        # Check if match_scores column exists (nested dict)
        if 'match_scores' in cf_df.columns:
            # Extract match_scores sub-fields
            match_scores_df = pd.json_normalize(cf_df['match_scores'])
            
            # Add to main dataframe
            for col in ['cf_score', 'overall_match', 'quality_score', 'requirement_match']:
                if col in match_scores_df.columns:
                    cf_df[col] = match_scores_df[col]
        
        # Compute weighted CF combined score from all match_scores
        cf_df['cf_combined_score'] = 0
        
        for score_col, weight in self.cf_match_weights.items():
            if score_col in cf_df.columns:
                cf_df['cf_combined_score'] += weight * cf_df[score_col].fillna(0)
        
        # Also keep original predicted_rating * confidence
        if 'predicted_rating' in cf_df.columns and 'confidence' in cf_df.columns:
            cf_df['cf_rating_confidence'] = cf_df['predicted_rating'] * cf_df['confidence']
        
        # Normalize combined score
        min_val = cf_df['cf_combined_score'].min()
        max_val = cf_df['cf_combined_score'].max()
        if max_val > min_val:
            cf_df['cf_combined_score_normalized'] = (
                (cf_df['cf_combined_score'] - min_val) / (max_val - min_val)
            )
        else:
            cf_df['cf_combined_score_normalized'] = 0.5
        
        logger.info(f"  CF combined score range: [{cf_df['cf_combined_score'].min():.3f}, {cf_df['cf_combined_score'].max():.3f}]")
        
        return cf_df
    
    def extract_lightgcn_scores(self, gcn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and weight all features from LightGCN response
        Creates weighted composite score from multiple features
        """
        if gcn_df.empty:
            return gcn_df
        
        logger.info("Extracting LightGCN features...")
        
        # Compute weighted feature score
        gcn_df['lightgcn_weighted_score'] = 0
        
        for feature, weight in self.lightgcn_feature_weights.items():
            if feature in gcn_df.columns:
                # Normalize feature to 0-1
                col_min = gcn_df[feature].min()
                col_max = gcn_df[feature].max()
                
                if col_max > col_min:
                    normalized = (gcn_df[feature] - col_min) / (col_max - col_min)
                else:
                    normalized = 0.5
                
                gcn_df['lightgcn_weighted_score'] += weight * normalized
        
        # Also keep the simple GCN average
        if all(col in gcn_df.columns for col in ['cv_score', 'cc_score', 'vc_score']):
            gcn_df['gcn_avg_score'] = (
                gcn_df['cv_score'] + gcn_df['cc_score'] + gcn_df['vc_score']
            ) / 3
            
            # Normalize GCN average
            min_val = gcn_df['gcn_avg_score'].min()
            max_val = gcn_df['gcn_avg_score'].max()
            if max_val > min_val:
                gcn_df['gcn_avg_score'] = (gcn_df['gcn_avg_score'] - min_val) / (max_val - min_val)
        
        logger.info(f"  LightGCN weighted score range: [{gcn_df['lightgcn_weighted_score'].min():.3f}, {gcn_df['lightgcn_weighted_score'].max():.3f}]")
        
        return gcn_df
    
    def normalize_model_scores(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Normalize all model scores to 0-1 range for fair comparison
        """
        logger.info("Normalizing model scores...")
        
        normalized_data = {}
        
        # 1. LightGCN - extract weighted scores
        if not all_data['lightgcn'].empty:
            gcn_df = self.extract_lightgcn_scores(all_data['lightgcn'].copy())
            normalized_data['lightgcn'] = gcn_df
        else:
            normalized_data['lightgcn'] = pd.DataFrame()
        
        # 2. Collaborative - extract match_scores
        if not all_data['collaborative'].empty:
            cf_df = self.extract_collaborative_scores(all_data['collaborative'].copy())
            normalized_data['collaborative'] = cf_df
        else:
            normalized_data['collaborative'] = pd.DataFrame()
        
        # 3. Item-Item Similarity (already 0-1)
        if not all_data['item_similarity'].empty:
            item_df = all_data['item_similarity'].copy()
            if 'similarity' in item_df.columns:
                item_df['similarity'] = item_df['similarity'].clip(0, 1)
            normalized_data['item_similarity'] = item_df
        else:
            normalized_data['item_similarity'] = pd.DataFrame()
        
        # 4. User-Item Similarity (already 0-1)
        if not all_data['user_similarity'].empty:
            user_df = all_data['user_similarity'].copy()
            if 'similarity' in user_df.columns:
                user_df['similarity'] = user_df['similarity'].clip(0, 1)
            normalized_data['user_similarity'] = user_df
        else:
            normalized_data['user_similarity'] = pd.DataFrame()
        
        logger.info("    Score normalization complete")
        
        return normalized_data
    
    def adjust_weights_for_cold_start(self, weights: Dict[str, float], is_cold_start: bool) -> Dict[str, float]:
        """
        Adjust weights for cold-start scenario
        """
        if not is_cold_start:
            return weights
        
        logger.info("   Cold-start detected - boosting similarity model weights (×4)")
        
        adjusted = weights.copy()
        
        adjusted['alpha3_item_sim'] = weights['alpha3_item_sim'] * 4
        adjusted['alpha4_user_sim'] = weights['alpha4_user_sim'] * 4
        
        adjusted['alpha1_rrf'] = weights['alpha1_rrf'] * 0.1
        adjusted['alpha2_gcn'] = weights['alpha2_gcn'] * 0.1
        adjusted['alpha5_confidence'] = weights['alpha5_confidence'] * 0.1
        
        weight_sum = sum(adjusted.values())
        if weight_sum > 0:
            for key in adjusted:
                adjusted[key] = adjusted[key] / weight_sum
        
        logger.info(f"  Adjusted weights: {adjusted}")
        
        return adjusted
    
    def get_client_type(self, client_id: int) -> str:
        """
        Get client type from loaded client_types dictionary
        """
        client_type = self.client_types.get(client_id, 'UNKNOWN')
        
        if client_type == 'UNKNOWN':
            logger.debug(f"Client {client_id} not found in client types, using UNKNOWN")
        else:
            logger.debug(f"Client {client_id} type: {client_type}")
        
        return client_type
    
    def apply_business_filters(self, 
                              recommendations_df: pd.DataFrame,
                              request_context: Optional[Dict] = None) -> pd.DataFrame:
        """
        Apply business rules and filters
        """
        logger.info("Applying business filters...")
        
        filtered = recommendations_df.copy()
        
        # Filter by confidence threshold
        if 'cf_confidence' in filtered.columns:
            min_conf = BUSINESS_RULES.get('min_confidence_threshold', 0.0)
            if min_conf > 0:
                filtered = filtered[
                    (filtered['cf_confidence'] >= min_conf) | 
                    (filtered['cf_confidence'] == 0)
                ]
                logger.info(f"  After confidence filter: {len(filtered)} pairs")
        
        # Filter by minimum score
        if 'final_score' in filtered.columns:
            min_score = BUSINESS_RULES.get('min_combined_score', 0.0)
            if min_score > 0:
                filtered = filtered[filtered['final_score'] >= min_score]
                logger.info(f"  After score filter: {len(filtered)} pairs")
        
        # Apply request context filters
        if request_context:
            if 'required_vehicle_type' in request_context and 'vehicle_type_id' in filtered.columns:
                req_type = request_context['required_vehicle_type']
                filtered = filtered[filtered['vehicle_type_id'] == req_type]
                logger.info(f"  After vehicle type filter: {len(filtered)} pairs")
            
            if request_context.get('baby_seat_required') and 'chauffeur_baby_seat_training' in filtered.columns:
                filtered = filtered[filtered['chauffeur_baby_seat_training'] == True]
                logger.info(f"  After baby seat filter: {len(filtered)} pairs")
            
            if request_context.get('red_zone_trip') and 'chauffeur_red_zone_auth' in filtered.columns:
                filtered = filtered[filtered['chauffeur_red_zone_auth'] == True]
                logger.info(f"  After red zone filter: {len(filtered)} pairs")
        
        # Limit to max recommendations
        max_recs = BUSINESS_RULES.get('max_recommendations', 100)
        filtered = filtered.head(max_recs)
        
        logger.info(f"    Final recommendations after filters: {len(filtered)}")
        
        return filtered
    
    def boost_scores_by_context(self,
                               recommendations_df: pd.DataFrame,
                               request_context: Optional[Dict] = None) -> pd.DataFrame:
        """
        Dynamically boost scores based on request context
        """
        if not request_context:
            return recommendations_df
        
        logger.info("Applying context-based score boosts...")
        
        df = recommendations_df.copy()
        
        if request_context.get('important_trip') and 'vehicle_completion_rate' in df.columns:
            df['final_score'] += 0.1 * df['vehicle_completion_rate']
            logger.info("  Boosted scores for high vehicle completion rate")
        
        if request_context.get('vip_client') and 'chauffeur_experience' in df.columns:
            df['final_score'] += 0.05 * (df['chauffeur_experience'] / df['chauffeur_experience'].max())
            logger.info("  Boosted scores for experienced chauffeurs")
        
        if 'times_worked_together' in df.columns:
            df['final_score'] += 0.02 * df['times_worked_together']
            logger.info("  Boosted scores for pairs with history")
        
        df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
        df['final_rank'] = range(1, len(df) + 1)
        
        return df
    
    def handle_cold_start(self, 
                         client_id: int,
                         request_context: Optional[Dict] = None) -> pd.DataFrame:
        """
        Handle cold-start scenario (new client or no history)
        """
        logger.info(f"   Cold-start detected for client {client_id}")
        logger.info("Using similarity-based recommendations...")
        
        item_sim = self.loader.load_item_similarity(client_id)
        user_sim = self.loader.load_user_item_similarity(client_id)
        
        if not item_sim.empty or not user_sim.empty:
            all_pairs = set()
            
            if not item_sim.empty:
                all_pairs.update(item_sim['pair_id'].unique())
            if not user_sim.empty:
                all_pairs.update(user_sim['pair_id'].unique())
            
            cold_start_recs = pd.DataFrame({'pair_id': list(all_pairs)})
            
            if not item_sim.empty:
                item_scores = dict(zip(item_sim['pair_id'], item_sim['similarity']))
                cold_start_recs['item_similarity'] = cold_start_recs['pair_id'].map(item_scores).fillna(0)
            else:
                cold_start_recs['item_similarity'] = 0
            
            if not user_sim.empty:
                user_scores = dict(zip(user_sim['pair_id'], user_sim['similarity']))
                cold_start_recs['user_similarity'] = cold_start_recs['pair_id'].map(user_scores).fillna(0)
            else:
                cold_start_recs['user_similarity'] = 0
            
            cold_start_recs['final_score'] = (
                0.5 * cold_start_recs['item_similarity'] +
                0.5 * cold_start_recs['user_similarity']
            )
            
            cold_start_recs = cold_start_recs.sort_values('final_score', ascending=False)
            cold_start_recs['final_rank'] = range(1, len(cold_start_recs) + 1)
            
            logger.info(f"    Generated {len(cold_start_recs)} cold-start recommendations")
            
            return cold_start_recs.head(BUSINESS_RULES['max_recommendations'])
        
        logger.warning("No similarity data available for cold-start")
        return pd.DataFrame()
    
    def generate_recommendations(self,
                                client_id: int,
                                request_context: Optional[Dict] = None) -> Dict:
        """
        Main pipeline: Generate recommendations for a client request
        """
        start_time = datetime.now()
        
        logger.info("\n" + "="*60)
        logger.info(f"GENERATING RECOMMENDATIONS FOR CLIENT {client_id}")
        logger.info("="*60)
        
        # Step 1: Load client type
        client_type = self.get_client_type(client_id)
        logger.info(f"Client type: {client_type}")
        
        # Step 2: Load weights
        weights = self.optimizer.load_weights(client_type)
        logger.info(f"Base weights: {weights}")
        
        # Step 3: Load all recommendation data
        all_data = self.loader.load_all_request_data(client_id)
        
        # Check for cold-start
        is_cold_start = (
            all_data['lightgcn'].empty and 
            all_data['collaborative'].empty
        )
        
        if is_cold_start:
            logger.info("   Cold-start scenario detected")
            
            adjusted_weights = self.adjust_weights_for_cold_start(weights, is_cold_start=True)
            
            recommendations = self.handle_cold_start(client_id, request_context)
            
            if recommendations.empty:
                logger.error("Failed to generate cold-start recommendations")
                return {
                    'client_id': client_id,
                    'error': 'No recommendations available',
                    'is_cold_start': True
                }
        else:
            # Normalize scores and extract features
            logger.info("\n🔄 Normalizing scores and extracting features...")
            normalized_data = self.normalize_model_scores(all_data)
            
            adjusted_weights = self.adjust_weights_for_cold_start(weights, is_cold_start=False)
            
            # Fuse recommendations
            recommendations_dict = {
                'lightgcn': normalized_data['lightgcn'],
                'collaborative': normalized_data['collaborative'],
                'item_similarity': normalized_data['item_similarity'],
                'user_similarity': normalized_data['user_similarity']
            }
            
            recommendations = self.fusion.fuse_recommendations(recommendations_dict, adjusted_weights)
            
            if recommendations.empty:
                logger.error("Fusion produced no recommendations")
                return {
                    'client_id': client_id,
                    'error': 'Fusion failed',
                    'is_cold_start': False
                }
        
        # Apply context boosting
        recommendations = self.boost_scores_by_context(recommendations, request_context)
        
        # Apply business filters
        recommendations = self.apply_business_filters(recommendations, request_context)
        
        # Format output
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'client_id': client_id,
            'client_type': client_type,
            'is_cold_start': is_cold_start,
            'request_context': request_context or {},
            'total_recommendations': len(recommendations),
            'recommendations': recommendations.to_dict('records'),
            'weights_used': adjusted_weights if is_cold_start else weights,
            'processing_time_seconds': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'normalization_applied': True,
            'feature_extraction_applied': True
        }
        
        logger.info(f"\n    Generated {len(recommendations)} recommendations")
        logger.info(f"    Processing time: {elapsed_time:.3f} seconds")
        logger.info("="*60 + "\n")
        
        return result
    
    def get_top_k(self, result: Dict, k: int = 10) -> List[Dict]:
        """
        Get top-k recommendations from result
        """
        if 'recommendations' not in result:
            return []
        
        return result['recommendations'][:k]
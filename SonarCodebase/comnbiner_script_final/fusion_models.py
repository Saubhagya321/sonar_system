# combiner

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from config import RRF_K

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationFusion:
    """
    Handles fusion of multiple recommendation models using:
    1. Reciprocal Rank Fusion (RRF) 
    2. Weighted Score Fusion
    
    UPDATED: Uses predicted_rating * confidence for CF score
    """
    
    def __init__(self, rrf_k: int = RRF_K, quiet: bool = False):
        """
        Initialize fusion model
        """
        self.rrf_k = rrf_k
        self.quiet = quiet
    
    def compute_rrf(self, ranks: List[int]) -> float:
        """
        Compute Reciprocal Rank Fusion score
        """
        if not ranks:
            return 0.0
        
        rrf_score = sum(1.0 / (self.rrf_k + rank) for rank in ranks if rank > 0)
        return rrf_score
    
    def compute_rrf_for_dataframe(self, df: pd.DataFrame, 
                                  rank_columns: List[str]) -> pd.Series:
        """
        Compute RRF for all rows in a DataFrame
        """
        def row_rrf(row):
            ranks = [row[col] for col in rank_columns if col in df.columns and pd.notna(row[col])]
            return self.compute_rrf(ranks)
        
        return df.apply(row_rrf, axis=1)
    
    def fuse_recommendations(self, 
                            recommendations: Dict[str, pd.DataFrame],
                            weights: Dict[str, float]) -> pd.DataFrame:
        """
        Fuse recommendations from all 4 models
        
        UPDATED: Uses cf_combined_score (predicted_rating * confidence) for CF
        
        Final Score = α1 * RRF 
                    + α2 * gcn_avg_score (NORMALIZED)
                    + α3 * item_item_similarity
                    + α4 * user_item_similarity
                    + α5 * cf_combined_score (rating * confidence, normalized)
        """
        if not self.quiet:
            logger.info("Starting recommendation fusion...")
        
        # Get all unique pair_ids
        all_pairs = set()
        for model_name, df in recommendations.items():
            if not df.empty and 'pair_id' in df.columns:
                all_pairs.update(df['pair_id'].unique())
        
        if not all_pairs:
            logger.warning("No pairs found across all models!")
            return pd.DataFrame()
        
        if not self.quiet:
            logger.info(f"Total unique pairs across all models: {len(all_pairs)}")
        
        # Initialize result DataFrame
        result = pd.DataFrame({'pair_id': list(all_pairs)})
        
        # Step 1: Compute RRF scores
        if not self.quiet:
            logger.info("Step 1: Computing RRF scores...")
        
        rrf_scores = {}
        
        for pair_id in all_pairs:
            ranks = []
            
            # Get rank from LightGCN
            if not recommendations['lightgcn'].empty:
                gcn_row = recommendations['lightgcn'][
                    recommendations['lightgcn']['pair_id'] == pair_id
                ]
                if not gcn_row.empty and 'rank' in gcn_row.columns:
                    ranks.append(gcn_row.iloc[0]['rank'])
            
            # Get rank from Collaborative
            if not recommendations['collaborative'].empty:
                cf_row = recommendations['collaborative'][
                    recommendations['collaborative']['pair_id'] == pair_id
                ]
                if not cf_row.empty and 'rank' in cf_row.columns:
                    ranks.append(cf_row.iloc[0]['rank'])
            
            rrf_scores[pair_id] = self.compute_rrf(ranks)
        
        result['rrf_score'] = result['pair_id'].map(rrf_scores)
        
        # Step 2: Get GCN average score
        if not self.quiet:
            logger.info("Step 2: Extracting GCN scores...")
        gcn_scores = {}
        
        if not recommendations['lightgcn'].empty:
            gcn_df = recommendations['lightgcn']
            
            if 'gcn_avg_score' in gcn_df.columns:
                for _, row in gcn_df.iterrows():
                    pair_id = row['pair_id']
                    gcn_scores[pair_id] = row['gcn_avg_score']
            
            elif all(f'{col}_normalized' in gcn_df.columns for col in ['cc_score', 'cv_score', 'vc_score']):
                for _, row in gcn_df.iterrows():
                    pair_id = row['pair_id']
                    scores = []
                    
                    if pd.notna(row.get('cc_score_normalized')):
                        scores.append(row['cc_score_normalized'])
                    if pd.notna(row.get('cv_score_normalized')):
                        scores.append(row['cv_score_normalized'])
                    if pd.notna(row.get('vc_score_normalized')):
                        scores.append(row['vc_score_normalized'])
                    
                    if scores:
                        gcn_scores[pair_id] = np.mean(scores)
        
        result['gcn_avg_score'] = result['pair_id'].map(gcn_scores).fillna(0)
        
        # Step 3: Get item-item similarity
        if not self.quiet:
            logger.info("Step 3: Extracting item-item similarity...")
        item_sim = {}
        
        if not recommendations['item_similarity'].empty:
            for _, row in recommendations['item_similarity'].iterrows():
                if 'pair_id' in row and 'similarity' in row:
                    item_sim[row['pair_id']] = row['similarity']
        
        result['item_similarity'] = result['pair_id'].map(item_sim).fillna(0)
        
        # Step 4: Get user-item similarity
        if not self.quiet:
            logger.info("Step 4: Extracting user-item similarity...")
        user_sim = {}
        
        if not recommendations['user_similarity'].empty:
            for _, row in recommendations['user_similarity'].iterrows():
                if 'pair_id' in row and 'similarity' in row:
                    user_sim[row['pair_id']] = row['similarity']
        
        result['user_similarity'] = result['pair_id'].map(user_sim).fillna(0)
        
        # Step 5: Get CF combined score (rating * confidence, normalized)
        if not self.quiet:
            logger.info("Step 5: Extracting CF combined score (rating * confidence)...")
        cf_combined = {}
        
        if not recommendations['collaborative'].empty:
            cf_df = recommendations['collaborative']
            
            # Check if normalized combined score exists
            if 'cf_combined_score_normalized' in cf_df.columns:
                for _, row in cf_df.iterrows():
                    if 'pair_id' in row:
                        cf_combined[row['pair_id']] = row['cf_combined_score_normalized']
            # Fallback: use confidence only if combined score not available
            elif 'confidence' in cf_df.columns:
                for _, row in cf_df.iterrows():
                    if 'pair_id' in row:
                        cf_combined[row['pair_id']] = row['confidence']
        
        result['cf_confidence'] = result['pair_id'].map(cf_combined).fillna(0)
        
        # Step 6: Compute final weighted score
        if not self.quiet:
            logger.info("Step 6: Computing final weighted scores...")
        
        result['final_score'] = (
            weights.get('alpha1_rrf', 0.3) * result['rrf_score'] +
            weights.get('alpha2_gcn', 0.3) * result['gcn_avg_score'] +
            weights.get('alpha3_item_sim', 0.15) * result['item_similarity'] +
            weights.get('alpha4_user_sim', 0.15) * result['user_similarity'] +
            weights.get('alpha5_confidence', 0.1) * result['cf_confidence']
        )
        
        # Sort by final score
        result = result.sort_values('final_score', ascending=False).reset_index(drop=True)
        result['final_rank'] = range(1, len(result) + 1)
        
        # Add metadata from models
        result = self._enrich_with_metadata(result, recommendations)
        
        if not self.quiet:
            logger.info(f"    Fusion complete: {len(result)} pairs ranked")
            
            non_zero_counts = {
                'rrf_score': (result['rrf_score'] > 0).sum(),
                'gcn_avg_score': (result['gcn_avg_score'] > 0).sum(),
                'item_similarity': (result['item_similarity'] > 0).sum(),
                'user_similarity': (result['user_similarity'] > 0).sum(),
                'cf_confidence': (result['cf_confidence'] > 0).sum()
            }
            logger.info(f"  Non-zero score counts: {non_zero_counts}")
        
        return result
    
    def _enrich_with_metadata(self, 
                             fused_df: pd.DataFrame,
                             recommendations: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Enrich fused results with metadata from original models
        """
        # Try to add metadata from LightGCN
        if not recommendations['lightgcn'].empty:
            gcn_df = recommendations['lightgcn']
            
            meta_cols = ['pair_id', 'vehicle_id', 'chauffeur_id',
                        'vehicle_make', 'vehicle_model', 'vehicle_year',
                        'chauffeur_code', 'chauffeur_experience']
            
            available_cols = [col for col in meta_cols if col in gcn_df.columns]
            
            if available_cols:
                gcn_meta = gcn_df[available_cols].drop_duplicates(subset=['pair_id'])
                fused_df = fused_df.merge(gcn_meta, on='pair_id', how='left')
        
        # If no metadata from GCN, try collaborative
        if 'vehicle_id' not in fused_df.columns and not recommendations['collaborative'].empty:
            cf_df = recommendations['collaborative']
            
            meta_cols = ['pair_id', 'vehicle_id', 'chauffeur_id',
                        'vehicle_make', 'vehicle_model', 'chauffeur_name']
            
            available_cols = [col for col in meta_cols if col in cf_df.columns]
            
            if available_cols:
                cf_meta = cf_df[available_cols].drop_duplicates(subset=['pair_id'])
                fused_df = fused_df.merge(cf_meta, on='pair_id', how='left')
        
        return fused_df
    
    def get_top_k(self, fused_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
        """
        Get top-k recommendations
        """
        return fused_df.head(k).copy()


class NDCGCalculator:
    """
    Calculate NDCG for validation during grid search
    """
    
    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """
        Compute Discounted Cumulative Gain at k
        """
        relevances = relevances[:k]
        if not relevances:
            return 0.0
        
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))
    
    @staticmethod
    def ndcg_at_k(predicted_relevances: List[float],
                  ideal_relevances: List[float],
                  k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k
        """
        dcg = NDCGCalculator.dcg_at_k(predicted_relevances, k)
        idcg = NDCGCalculator.dcg_at_k(sorted(ideal_relevances, reverse=True), k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def compute_ndcg_for_client(client_predictions: pd.DataFrame,
                                ground_truth: pd.DataFrame,
                                k: int = 10) -> float:
        """
        Compute NDCG@k for a single client's predictions
        """
        merged = client_predictions.merge(
            ground_truth,
            on='pair_id',
            how='left'
        )
        
        if 'rating' in merged.columns:
            relevances = merged['rating'].fillna(0).tolist()
        elif 'interaction' in merged.columns:
            relevances = merged['interaction'].fillna(0).tolist()
        else:
            relevances = [1 if pd.notna(r) else 0 for r in merged.get('rating', merged.get('interaction', [0]*len(merged)))]
        
        ideal_relevances = relevances.copy()
        ndcg = NDCGCalculator.ndcg_at_k(relevances, ideal_relevances, k)
        
        return ndcg
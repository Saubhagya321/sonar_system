"""
services.py - Business logic services for the Recommendation Combiner API

This module contains:
- ScoreExtractor: Extract and normalize scores from API responses
- ResponseProcessor: Process raw API responses into DataFrames
- RecommendationFuser: Fuse recommendations from multiple systems
- SensitivityFilter: Filter recommendations based on sensitivity levels
- WeightManager: Handle weight redistribution and cold start adjustments
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json

from models import FeatureWeights, RRFConfig

logger = logging.getLogger(__name__)


# =============================================================================
# SCORE EXTRACTION SERVICE
# =============================================================================

class ScoreExtractor:
    """
    Service for extracting and normalizing scores from API responses.
    
    This class handles the conversion of raw API response data into
    normalized scores suitable for fusion.
    """
    
    def __init__(self):
        self.lightgcn_weights = FeatureWeights.get_lightgcn_weights()
        self.cf_weights = FeatureWeights.get_cf_weights()
    
    @staticmethod
    def normalize_scores(values: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range using min-max normalization.
        
        Args:
            values: List of raw scores
            
        Returns:
            List of normalized scores in [0, 1] range
        """
        if not values or len(values) == 0:
            return values
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return [0.5] * len(values)
        
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    def extract_collaborative_scores(self, cf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all match_scores from Collaborative API response.
        Creates weighted CF combined score from 4 match_scores.
        
        Args:
            cf_df: DataFrame from collaborative API response
            
        Returns:
            DataFrame with extracted and weighted scores
        """
        if cf_df.empty:
            return cf_df
        
        logger.info("    Extracting Collaborative match_scores...")
        
        # Extract match_scores if nested
        if 'match_scores' in cf_df.columns:
            match_scores_df = pd.json_normalize(cf_df['match_scores'])
            
            for col in ['cf_score', 'overall_match', 'quality_score', 'requirement_match']:
                if col in match_scores_df.columns:
                    cf_df[col] = match_scores_df[col]
                    logger.info(f"      Extracted {col}")
        
        # Compute weighted CF combined score
        cf_df['cf_combined_score'] = 0
        
        for score_col, weight in self.cf_weights.items():
            if score_col in cf_df.columns:
                cf_df['cf_combined_score'] += weight * cf_df[score_col].fillna(0)
                logger.info(f"      Applied weight {weight:.2f} to {score_col}")
        
        # Also keep original predicted_rating * confidence
        if 'predicted_rating' in cf_df.columns and 'confidence' in cf_df.columns:
            cf_df['cf_rating_confidence'] = cf_df['predicted_rating'] * cf_df['confidence']
            logger.info("      Computed cf_rating_confidence = rating × confidence")
        
        # Normalize combined score
        min_val = cf_df['cf_combined_score'].min()
        max_val = cf_df['cf_combined_score'].max()
        if max_val > min_val:
            cf_df['cf_combined_score_normalized'] = (
                (cf_df['cf_combined_score'] - min_val) / (max_val - min_val)
            )
            logger.info(f"      Normalized CF combined score: [{min_val:.3f}, {max_val:.3f}] → [0.0, 1.0]")
        else:
            cf_df['cf_combined_score_normalized'] = 0.5
            logger.info("  ⚠ All CF scores equal, using 0.5")
        
        return cf_df
    
    def extract_lightgcn_scores(self, gcn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and weight all features from LightGCN response.
        Creates weighted composite score from 8+ features.
        
        Args:
            gcn_df: DataFrame from LightGCN API response
            
        Returns:
            DataFrame with extracted and weighted scores
        """
        if gcn_df.empty:
            return gcn_df
        
        logger.info("    Extracting LightGCN features...")
        
        # Compute weighted feature score
        gcn_df['lightgcn_weighted_score'] = 0
        features_found = []
        
        for feature, weight in self.lightgcn_weights.items():
            if feature in gcn_df.columns:
                col_min = gcn_df[feature].min()
                col_max = gcn_df[feature].max()
                
                if col_max > col_min:
                    normalized = (gcn_df[feature] - col_min) / (col_max - col_min)
                else:
                    normalized = 0.5
                
                gcn_df['lightgcn_weighted_score'] += weight * normalized
                features_found.append(feature)
                logger.info(f"      Applied weight {weight:.3f} to {feature}")
        
        logger.info(f"      Total features used: {len(features_found)}/{len(self.lightgcn_weights)}")
        
        # Also keep simple GCN average (3 core scores)
        if all(col in gcn_df.columns for col in ['cv_score', 'cc_score', 'vc_score']):
            # Normalize each score
            for col in ['cv_score', 'cc_score', 'vc_score']:
                values = gcn_df[col].fillna(0).tolist()
                gcn_df[f'{col}_normalized'] = self.normalize_scores(values)
            
            # Calculate average normalized score
            gcn_df['gcn_avg_score'] = (
                gcn_df['cv_score_normalized'] +
                gcn_df['cc_score_normalized'] +
                gcn_df['vc_score_normalized']
            ) / 3
            logger.info("      Computed gcn_avg_score (normalized average of 3 core scores)")
        else:
            gcn_df['gcn_avg_score'] = 0
            logger.info("  ⚠ Core scores missing, gcn_avg_score = 0")
        
        weighted_min = gcn_df['lightgcn_weighted_score'].min()
        weighted_max = gcn_df['lightgcn_weighted_score'].max()
        logger.info(f"      LightGCN weighted score range: [{weighted_min:.3f}, {weighted_max:.3f}]")
        
        return gcn_df


# =============================================================================
# RESPONSE PROCESSOR SERVICE
# =============================================================================

class ResponseProcessor:
    """
    Service for processing raw API responses into standardized DataFrames.
    
    Each API returns data in a different format. This class normalizes
    all responses into a consistent DataFrame structure.
    """
    
    def __init__(self):
        self.score_extractor = ScoreExtractor()
    
    def process_lightgcn_response(self, lightgcn_data: Dict) -> pd.DataFrame:
        """
        Process LightGCN API response with feature extraction.
        
        Args:
            lightgcn_data: Raw response from LightGCN API
            
        Returns:
            Processed DataFrame with standardized columns
        """
        logger.info("Processing LightGCN response...")
        
        if not lightgcn_data:
            logger.warning("⚠ No LightGCN data received")
            return pd.DataFrame()
        ## {---}
        logger.info(f"Response keys: {list(lightgcn_data.keys())}")
        
        # Handle nested structure
        if 'recommendations' in lightgcn_data and 'pairs' in lightgcn_data['recommendations']:
            pairs = lightgcn_data['recommendations']['pairs']
        elif 'pairs' in lightgcn_data:
            pairs = lightgcn_data['pairs']
        else:
            logger.warning("⚠ No pairs found in LightGCN response")
            return pd.DataFrame()
        ### { ?}
        if not pairs:
            logger.warning("⚠ LightGCN pairs list is empty")
            return pd.DataFrame()
        
        logger.info(f"Found {len(pairs)} pairs in response")
        
        df = pd.DataFrame(pairs)
        
        if 'chauffeur_id' not in df.columns or 'vehicle_id' not in df.columns:
            logger.error("  Missing chauffeur_id or vehicle_id in LightGCN response")
            return pd.DataFrame()
        
        # Convert IDs to strings
        df['chauffeur_id'] = df['chauffeur_id'].astype(float).astype(int).astype(str)
        df['vehicle_id'] = df['vehicle_id'].astype(float).astype(int).astype(str)
        df['pair_id'] = df['chauffeur_id'] + '_' + df['vehicle_id']
        
        logger.info(f"Sample pair_ids: {df['pair_id'].head(3).tolist()}")
        
        # Extract weighted features
        df = self.score_extractor.extract_lightgcn_scores(df)
        
        if 'rank' not in df.columns:
            df['rank'] = range(1, len(df) + 1)
        
        logger.info(f"  Processed {len(df)} LightGCN pairs with enriched features")
        
        return df
    
    def process_collaborative_response(self, collab_data: Dict) -> pd.DataFrame:
        """
        Process Collaborative API response with match_scores extraction.
        
        Args:
            collab_data: Raw response from Collaborative API
            
        Returns:
            Processed DataFrame with standardized columns
        """
        logger.info("Processing Collaborative response...")
        
        if not collab_data or 'recommendations' not in collab_data:
            logger.warning("⚠ No recommendations found in Collaborative response")
            return pd.DataFrame()
        
        logger.info(f"Response keys: {list(collab_data.keys())}")
        
        recommendations = collab_data['recommendations']
        
        if not recommendations:
            logger.warning("⚠ Collaborative recommendations list is empty")
            return pd.DataFrame()
        
        logger.info(f"Found {len(recommendations)} recommendations in response")
        
        df = pd.DataFrame(recommendations)
        
        # Handle pair_id
        if 'pair_id' in df.columns:
            df['pair_id'] = df['pair_id'].astype(str)
        elif 'vehicle_id' in df.columns and 'chauffeur_id' in df.columns:
            df['vehicle_id'] = df['vehicle_id'].astype(str)
            df['chauffeur_id'] = df['chauffeur_id'].astype(str)
            df['pair_id'] = df['chauffeur_id'] + '_' + df['vehicle_id']
        else:
            logger.error("  Cannot determine pair_id format in Collaborative response")
            return pd.DataFrame()
        
        logger.info(f"Sample pair_ids: {df['pair_id'].head(3).tolist()}")
        
        # Extract match_scores
        df = self.score_extractor.extract_collaborative_scores(df)
        
        if 'rank' not in df.columns:
            df['rank'] = range(1, len(df) + 1)
        
        logger.info(f"  Processed {len(df)} Collaborative pairs with match_scores")
        
        return df
    
    def process_user_content_response(self, user_content_data: Dict) -> pd.DataFrame:
        """
        Process User Content API response.
        
        Args:
            user_content_data: Raw response from User Content API
            
        Returns:
            Processed DataFrame with standardized columns
        """
        logger.info("Processing User Content response...")
        
        if not user_content_data or not user_content_data.get('success'):
            logger.warning("⚠ No User Content data received")
            return pd.DataFrame()
        
        data = user_content_data.get('data', [])
        
        if not data:
            logger.warning("⚠ User Content data list is empty")
            return pd.DataFrame()
        
        logger.info(f"Found {len(data)} recommendations in response")
        
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        if 'RefChauffeur' not in df.columns or 'RefVehicle' not in df.columns:
            logger.error("  Missing RefChauffeur or RefVehicle in User Content response")
            return pd.DataFrame()
        
        # Convert IDs to strings
        df['RefChauffeur'] = df['RefChauffeur'].astype(int).astype(str)
        df['RefVehicle'] = df['RefVehicle'].astype(int).astype(str)
        df['pair_id'] = df['RefChauffeur'] + '_' + df['RefVehicle']
        
        # Use similarity score
        if 'similarity' not in df.columns:
            logger.error("  Missing similarity in User Content response")
            return pd.DataFrame()
        
        df['user_content_score'] = df['similarity']
        
        # Add rank if not present
        if 'rank' not in df.columns:
            df['rank'] = range(1, len(df) + 1)
        
        logger.info(f"Sample pair_ids: {df['pair_id'].head(3).tolist()}")
        logger.info(f"  Processed {len(df)} User Content pairs")
        
        return df
    
    def process_item_content_response(self, item_content_data: Dict) -> pd.DataFrame:
        """
        Process Item Content API response.
        
        Args:
            item_content_data: Raw response from Item Content API
            
        Returns:
            Processed DataFrame with standardized columns
        """
        logger.info("Processing Item Content response...")
        
        if not item_content_data or not item_content_data.get('success'):
            logger.warning("⚠ No Item Content data received")
            return pd.DataFrame()
        
        data = item_content_data.get('data', [])
        
        if not data:
            logger.warning("⚠ Item Content data list is empty")
            return pd.DataFrame()
        
        logger.info(f"Found {len(data)} recommendations in response")
        
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        if 'RefChauffeur' not in df.columns or 'RefVehicle' not in df.columns:
            logger.error("  Missing RefChauffeur or RefVehicle in Item Content response")
            return pd.DataFrame()
        
        # Convert IDs to strings
        df['RefChauffeur'] = df['RefChauffeur'].astype(int).astype(str)
        df['RefVehicle'] = df['RefVehicle'].astype(int).astype(str)
        df['pair_id'] = df['RefChauffeur'] + '_' + df['RefVehicle']
        
        # Use similarity score
        if 'similarity' not in df.columns:
            logger.error("  Missing similarity in Item Content response")
            return pd.DataFrame()
        
        df['item_content_score'] = df['similarity']
        
        # Add rank if not present
        if 'rank' not in df.columns:
            df['rank'] = range(1, len(df) + 1)
        
        logger.info(f"Sample pair_ids: {df['pair_id'].head(3).tolist()}")
        logger.info(f"  Processed {len(df)} Item Content pairs")
        
        return df


# =============================================================================
# RECOMMENDATION FUSER SERVICE
# =============================================================================

class RecommendationFuser:
    """
    Service for fusing recommendations from multiple systems.
    
    This class implements the core fusion logic using:
    - Reciprocal Rank Fusion (RRF)
    - Weighted score combination
    - Metadata enrichment
    """
    
    def __init__(self):
        self.rrf_k = RRFConfig.K
    
    def compute_rrf(self, ranks: List[int]) -> float:
        """
        Compute Reciprocal Rank Fusion score.
        
        Args:
            ranks: List of ranks from different systems
            
        Returns:
            RRF score
        """
        if not ranks:
            return 0.0
        return sum(1.0 / (self.rrf_k + rank) for rank in ranks if rank > 0)
    
    def analyze_overlap(self, lightgcn_df: pd.DataFrame,
                        collaborative_df: pd.DataFrame,
                        user_content_df: pd.DataFrame,
                        item_content_df: pd.DataFrame) -> Dict:
        """
        Analyze overlap between all 4 recommendation systems.
        
        Args:
            lightgcn_df: DataFrame from LightGCN
            collaborative_df: DataFrame from Collaborative Filtering
            user_content_df: DataFrame from User Content
            item_content_df: DataFrame from Item Content
            
        Returns:
            Dict with overlap statistics
        """
        logger.info("\n" + "="*70)
        logger.info("    4-SYSTEM RECOMMENDATION OVERLAP ANALYSIS")
        logger.info("="*70)
        
        # Get pair sets
        gcn_pairs = set(lightgcn_df['pair_id'].unique()) if not lightgcn_df.empty else set()
        cf_pairs = set(collaborative_df['pair_id'].unique()) if not collaborative_df.empty else set()
        uc_pairs = set(user_content_df['pair_id'].unique()) if not user_content_df.empty else set()
        ic_pairs = set(item_content_df['pair_id'].unique()) if not item_content_df.empty else set()
        
        total_unique = gcn_pairs | cf_pairs | uc_pairs | ic_pairs
        
        # All 4 systems
        all_four = gcn_pairs & cf_pairs & uc_pairs & ic_pairs
        
        # Any 3 systems
        gcn_cf_uc = gcn_pairs & cf_pairs & uc_pairs
        gcn_cf_ic = gcn_pairs & cf_pairs & ic_pairs
        gcn_uc_ic = gcn_pairs & uc_pairs & ic_pairs
        cf_uc_ic = cf_pairs & uc_pairs & ic_pairs
        any_three = gcn_cf_uc | gcn_cf_ic | gcn_uc_ic | cf_uc_ic
        
        logger.info("\n System Counts:")
        logger.info(f"  • LightGCN:       {len(gcn_pairs):5d}")
        logger.info(f"  • Collaborative:  {len(cf_pairs):5d}")
        logger.info(f"  • User Content:   {len(uc_pairs):5d}")
        logger.info(f"  • Item Content:   {len(ic_pairs):5d}")
        logger.info(f"  • Total Unique:   {len(total_unique):5d}")
        
        overlap_pct = len(all_four)/len(total_unique)*100 if total_unique else 0
        logger.info("\n  Overlap Statistics:")
        logger.info(f"  • All 4 systems:  {len(all_four):5d}  ({overlap_pct:.1f}%)")
        logger.info(f"  • Any 3 systems:  {len(any_three):5d}  ({len(any_three)/len(total_unique)*100 if total_unique else 0:.1f}%)")
        
        logger.info("="*70 + "\n")
        
        return {
            "total_unique_pairs": len(total_unique),
            "lightgcn_pairs": len(gcn_pairs),
            "collaborative_pairs": len(cf_pairs),
            "user_content_pairs": len(uc_pairs),
            "item_content_pairs": len(ic_pairs),
            "all_four_systems": len(all_four),
            "any_three_systems": len(any_three),
            "overlap_percentage": round(overlap_pct, 2)
        }
    
    def enrich_with_metadata(self, fused_df: pd.DataFrame,
                             lightgcn_df: pd.DataFrame,
                             collaborative_df: pd.DataFrame,
                             user_content_df: pd.DataFrame,
                             item_content_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata from all 4 systems.
        Priority: LightGCN > User Content > Item Content > Collaborative
        
        Args:
            fused_df: Fused recommendations DataFrame
            lightgcn_df: LightGCN source DataFrame
            collaborative_df: Collaborative source DataFrame
            user_content_df: User Content source DataFrame
            item_content_df: Item Content source DataFrame
            
        Returns:
            Enriched DataFrame with metadata from source systems
        """
        # Try LightGCN first
        if not lightgcn_df.empty:
            meta_cols = ['pair_id', 'vehicle_id', 'chauffeur_id']
            for col in lightgcn_df.columns:
                if col not in meta_cols and col not in fused_df.columns:
                    meta_cols.append(col)
            
            available_cols = [col for col in meta_cols if col in lightgcn_df.columns]
            if available_cols:
                gcn_meta = lightgcn_df[available_cols].drop_duplicates(subset=['pair_id'])
                fused_df = fused_df.merge(gcn_meta, on='pair_id', how='left')
                logger.info(f"      Enriched with {len(available_cols)} columns from LightGCN")
        
        # Fill missing with User Content (use RefVehicle/RefChauffeur)
        if not user_content_df.empty:
            meta_cols = ['pair_id', 'RefVehicle', 'RefChauffeur', 'similarity']
            available_cols = [col for col in meta_cols if col in user_content_df.columns]
            
            if available_cols:
                uc_meta = user_content_df[available_cols].drop_duplicates(subset=['pair_id'])
                fused_df = fused_df.merge(uc_meta, on='pair_id', how='left', suffixes=('', '_uc'))
                
                # Copy RefVehicle/RefChauffeur if main IDs are missing
                if 'RefVehicle' in fused_df.columns and 'vehicle_id' not in fused_df.columns:
                    fused_df['vehicle_id'] = fused_df['RefVehicle']
                elif 'RefVehicle' in fused_df.columns and 'vehicle_id' in fused_df.columns:
                    fused_df['vehicle_id'] = fused_df['vehicle_id'].fillna(fused_df['RefVehicle'])
                
                if 'RefChauffeur' in fused_df.columns and 'chauffeur_id' not in fused_df.columns:
                    fused_df['chauffeur_id'] = fused_df['RefChauffeur']
                elif 'RefChauffeur' in fused_df.columns and 'chauffeur_id' in fused_df.columns:
                    fused_df['chauffeur_id'] = fused_df['chauffeur_id'].fillna(fused_df['RefChauffeur'])
                
                logger.info(f"      Enriched with {len(available_cols)} columns from User Content")
        
        # Fill missing with Item Content
        if not item_content_df.empty:
            meta_cols = ['pair_id', 'RefVehicle', 'RefChauffeur', 'similarity']
            available_cols = [col for col in meta_cols if col in item_content_df.columns]
            
            if available_cols:
                ic_meta = item_content_df[available_cols].drop_duplicates(subset=['pair_id'])
                fused_df = fused_df.merge(ic_meta, on='pair_id', how='left', suffixes=('', '_ic'))
                
                # Copy RefVehicle/RefChauffeur if still missing
                if 'RefVehicle_ic' in fused_df.columns:
                    if 'vehicle_id' not in fused_df.columns:
                        fused_df['vehicle_id'] = fused_df['RefVehicle_ic']
                    else:
                        fused_df['vehicle_id'] = fused_df['vehicle_id'].fillna(fused_df['RefVehicle_ic'])
                
                if 'RefChauffeur_ic' in fused_df.columns:
                    if 'chauffeur_id' not in fused_df.columns:
                        fused_df['chauffeur_id'] = fused_df['RefChauffeur_ic']
                    else:
                        fused_df['chauffeur_id'] = fused_df['chauffeur_id'].fillna(fused_df['RefChauffeur_ic'])
                
                logger.info(f"      Enriched with {len(available_cols)} columns from Item Content")
        
        # Finally Collaborative
        if not collaborative_df.empty:
            needs_ids = 'vehicle_id' not in fused_df.columns or 'chauffeur_id' not in fused_df.columns
            
            if needs_ids:
                meta_cols = ['pair_id', 'vehicle_id', 'chauffeur_id']
                available_cols = [col for col in meta_cols if col in collaborative_df.columns]
                if available_cols:
                    cf_meta = collaborative_df[available_cols].drop_duplicates(subset=['pair_id'])
                    fused_df = fused_df.merge(cf_meta, on='pair_id', how='left', suffixes=('', '_cf'))
                    logger.info(f"      Enriched with {len(available_cols)} columns from Collaborative")
        
        return fused_df
    
    def _ensure_ids_exist(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure vehicle_id and chauffeur_id columns exist in the result.
        Falls back to extracting from pair_id if necessary.
        """
        logger.info("\n POST-ENRICHMENT ID VALIDATION")
        
        has_vehicle_id = 'vehicle_id' in result.columns
        has_chauffeur_id = 'chauffeur_id' in result.columns
        has_refvehicle = 'RefVehicle' in result.columns
        has_refchauffeur = 'RefChauffeur' in result.columns
        
        logger.info(f"  • vehicle_id present: {has_vehicle_id}")
        logger.info(f"  • chauffeur_id present: {has_chauffeur_id}")
        logger.info(f"  • RefVehicle present: {has_refvehicle}")
        logger.info(f"  • RefChauffeur present: {has_refchauffeur}")
        
        # Strategy 1: Copy from Ref* columns if they exist
        if not has_vehicle_id and has_refvehicle:
            result['vehicle_id'] = result['RefVehicle']
            logger.info("      Copied vehicle_id from RefVehicle")
        
        if not has_chauffeur_id and has_refchauffeur:
            result['chauffeur_id'] = result['RefChauffeur']
            logger.info("      Copied chauffeur_id from RefChauffeur")
        
        # Strategy 2: Extract from pair_id if still missing
        if 'vehicle_id' not in result.columns or result['vehicle_id'].isna().any():
            logger.info("    Extracting IDs from pair_id as fallback")
            
            # Split pair_id (format: "chauffeur_vehicle")
            split_ids = result['pair_id'].str.split('_', n=1, expand=True)
            
            if 'chauffeur_id' not in result.columns or result['chauffeur_id'].isna().any():
                result['chauffeur_id'] = result['chauffeur_id'].fillna(split_ids[0]) if 'chauffeur_id' in result.columns else split_ids[0]
            
            if 'vehicle_id' not in result.columns or result['vehicle_id'].isna().any():
                result['vehicle_id'] = result['vehicle_id'].fillna(split_ids[1]) if 'vehicle_id' in result.columns else split_ids[1]
        
        return result
    
    def fuse(self, lightgcn_df: pd.DataFrame,
             collaborative_df: pd.DataFrame,
             user_content_df: pd.DataFrame,
             item_content_df: pd.DataFrame,
             weights: Dict[str, float]) -> pd.DataFrame:
        """
        Fuse recommendations using enriched scores from all 4 systems.
        Uses grid search optimized weights.
        
        Args:
            lightgcn_df: LightGCN recommendations
            collaborative_df: Collaborative recommendations
            user_content_df: User Content recommendations
            item_content_df: Item Content recommendations
            weights: Fusion weights dictionary
            
        Returns:
            Fused DataFrame with final scores and rankings
        """
        logger.info(" Fusing recommendations from 4 systems...")
        
        # Get all unique pair_ids
        all_pairs = set()
        if not lightgcn_df.empty:
            all_pairs.update(lightgcn_df['pair_id'].unique())
        if not collaborative_df.empty:
            all_pairs.update(collaborative_df['pair_id'].unique())
        if not user_content_df.empty:
            all_pairs.update(user_content_df['pair_id'].unique())
        if not item_content_df.empty:
            all_pairs.update(item_content_df['pair_id'].unique())
        
        if not all_pairs:
            logger.warning(" No pairs found to fuse")
            return pd.DataFrame()
        
        # Analyze overlap
        self.analyze_overlap(lightgcn_df, collaborative_df, user_content_df, item_content_df)
        
        logger.info(f"Total unique pairs: {len(all_pairs)}")
        logger.info(f"  • From LightGCN: {len(lightgcn_df['pair_id'].unique()) if not lightgcn_df.empty else 0}")
        logger.info(f"  • From Collaborative: {len(collaborative_df['pair_id'].unique()) if not collaborative_df.empty else 0}")
        logger.info(f"  • From User Content: {len(user_content_df['pair_id'].unique()) if not user_content_df.empty else 0}")
        logger.info(f"  • From Item Content: {len(item_content_df['pair_id'].unique()) if not item_content_df.empty else 0}")
        
        # Initialize result DataFrame
        result = pd.DataFrame({'pair_id': list(all_pairs)})
        
        # 1. Compute RRF scores (from rankings)
        logger.info("\n Computing RRF scores...")
        rrf_scores = {}
        
        for pair_id in all_pairs:
            ranks = []
            
            if not lightgcn_df.empty:
                gcn_row = lightgcn_df[lightgcn_df['pair_id'] == pair_id]
                if not gcn_row.empty and 'rank' in gcn_row.columns:
                    ranks.append(gcn_row.iloc[0]['rank'])
            
            if not collaborative_df.empty:
                cf_row = collaborative_df[collaborative_df['pair_id'] == pair_id]
                if not cf_row.empty and 'rank' in cf_row.columns:
                    ranks.append(cf_row.iloc[0]['rank'])
            
            if not user_content_df.empty:
                uc_row = user_content_df[user_content_df['pair_id'] == pair_id]
                if not uc_row.empty and 'rank' in uc_row.columns:
                    ranks.append(uc_row.iloc[0]['rank'])
            
            if not item_content_df.empty:
                ic_row = item_content_df[item_content_df['pair_id'] == pair_id]
                if not ic_row.empty and 'rank' in ic_row.columns:
                    ranks.append(ic_row.iloc[0]['rank'])
            
            rrf_scores[pair_id] = self.compute_rrf(ranks)
        
        result['rrf_score'] = result['pair_id'].map(rrf_scores)
        logger.info(f"      RRF scores computed for {len(all_pairs)} pairs")
        
        # 2. Get GCN weighted score (normalized)
        logger.info("\n Extracting GCN scores...")
        gcn_scores = {}
        
        if not lightgcn_df.empty:
            for _, row in lightgcn_df.iterrows():
                if 'lightgcn_weighted_score' in row and pd.notna(row['lightgcn_weighted_score']):
                    gcn_scores[row['pair_id']] = row['lightgcn_weighted_score']
                elif 'gcn_avg_score' in row:
                    gcn_scores[row['pair_id']] = row.get('gcn_avg_score', 0)
        
        result['gcn_score'] = result['pair_id'].map(gcn_scores).fillna(0)
        
        if result['gcn_score'].max() > 0:
            result['gcn_score'] = (result['gcn_score'] - result['gcn_score'].min()) / \
                                  (result['gcn_score'].max() - result['gcn_score'].min())
        
        logger.info(f"      GCN scores extracted and normalized: {(result['gcn_score'] > 0).sum()} non-zero")
        
        # 3. Get CF combined score (normalized)
        logger.info("\n Extracting CF scores...")
        cf_scores = {}
        
        if not collaborative_df.empty:
            for _, row in collaborative_df.iterrows():
                if 'cf_combined_score_normalized' in row and pd.notna(row['cf_combined_score_normalized']):
                    cf_scores[row['pair_id']] = row['cf_combined_score_normalized']
                elif 'confidence' in row:
                    cf_scores[row['pair_id']] = row.get('confidence', 0)
        
        result['cf_score'] = result['pair_id'].map(cf_scores).fillna(0)
        
        if result['cf_score'].max() > 0:
            result['cf_score'] = (result['cf_score'] - result['cf_score'].min()) / \
                                 (result['cf_score'].max() - result['cf_score'].min())
        
        logger.info(f"      CF scores extracted and normalized: {(result['cf_score'] > 0).sum()} non-zero")
        
        # 4. Get User Content similarity (already 0-1)
        logger.info("\n Extracting User Content scores...")
        user_content_scores = {}
        
        if not user_content_df.empty:
            for _, row in user_content_df.iterrows():
                if 'user_content_score' in row:
                    user_content_scores[row['pair_id']] = row['user_content_score']
        
        result['user_content_score'] = result['pair_id'].map(user_content_scores).fillna(0)
        logger.info(f"      User Content scores extracted: {(result['user_content_score'] > 0).sum()} non-zero")
        
        # 5. Get Item Content similarity (already 0-1)
        logger.info("\n Extracting Item Content scores...")
        item_content_scores = {}
        
        if not item_content_df.empty:
            for _, row in item_content_df.iterrows():
                if 'item_content_score' in row:
                    item_content_scores[row['pair_id']] = row['item_content_score']
        
        result['item_content_score'] = result['pair_id'].map(item_content_scores).fillna(0)
        logger.info(f"      Item Content scores extracted: {(result['item_content_score'] > 0).sum()} non-zero")
        
        # 6. Compute final weighted score
        logger.info("\n Computing final weighted scores...")
        logger.info("Weights applied:")
        logger.info(f"  • alpha1_rrf: {weights.get('alpha1_rrf', 0.3):.3f}")
        logger.info(f"  • alpha2_gcn: {weights.get('alpha2_gcn', 0.3):.3f}")
        logger.info(f"  • alpha3_item_sim: {weights.get('alpha3_item_sim', 0.1):.3f}")
        logger.info(f"  • alpha4_user_sim: {weights.get('alpha4_user_sim', 0.1):.3f}")
        logger.info(f"  • alpha5_confidence: {weights.get('alpha5_confidence', 0.1):.3f}")
        
        result['final_score'] = (
            weights.get('alpha1_rrf', 0.3) * result['rrf_score'] +
            weights.get('alpha2_gcn', 0.3) * result['gcn_score'] +
            weights.get('alpha3_item_sim', 0.1) * result['item_content_score'] +
            weights.get('alpha4_user_sim', 0.1) * result['user_content_score'] +
            weights.get('alpha5_confidence', 0.1) * result['cf_score']
        )
        
        # Sort by final score
        result = result.sort_values('final_score', ascending=False).reset_index(drop=True)
        result['final_rank'] = range(1, len(result) + 1)
        
        # Enrich with metadata
        result = self.enrich_with_metadata(result, lightgcn_df, collaborative_df,
                                           user_content_df, item_content_df)
        
        # Ensure IDs exist
        result = self._ensure_ids_exist(result)
        
        return result


# =============================================================================
# SENSITIVITY FILTER SERVICE
# =============================================================================

class SensitivityFilter:
    """
    Service for filtering recommendations based on sensitivity levels.
    
    Ensures that:
    client_sensitivity <= vehicle_sensitivity AND client_sensitivity <= chauffeur_sensitivity
    """
    
    def apply(self, fused_df: pd.DataFrame,
              client_id: int,
              sensitivity_levels: Dict) -> pd.DataFrame:
        """
        Filter recommendations based on sensitivity levels.
        
        Args:
            fused_df: DataFrame with fused recommendations
            client_id: Client ID to get sensitivity level
            sensitivity_levels: Dict with client/chauffeur/vehicle sensitivity mappings
            
        Returns:
            Filtered DataFrame with re-ranked results
        """
        if fused_df.empty:
            return fused_df
        
        logger.info("\n APPLYING SENSITIVITY FILTER")
        logger.info(f"{'='*60}")
        
        initial_count = len(fused_df)
        
        # Get client sensitivity level (default to 0 if not found)
        client_sensitivity = sensitivity_levels['clients'].get(client_id, 0)
        logger.info(f"Client {client_id} sensitivity level: {client_sensitivity}")
        
        # Extract vehicle_id and chauffeur_id (handle both naming conventions)
        if 'vehicle_id' in fused_df.columns:
            fused_df['vehicle_id_int'] = pd.to_numeric(fused_df['vehicle_id'], errors='coerce').fillna(0).astype(int)
        elif 'RefVehicle' in fused_df.columns:
            fused_df['vehicle_id_int'] = pd.to_numeric(fused_df['RefVehicle'], errors='coerce').fillna(0).astype(int)
        else:
            logger.error("  No vehicle_id column found")
            return fused_df
        
        if 'chauffeur_id' in fused_df.columns:
            fused_df['chauffeur_id_int'] = pd.to_numeric(fused_df['chauffeur_id'], errors='coerce').fillna(0).astype(int)
        elif 'RefChauffeur' in fused_df.columns:
            fused_df['chauffeur_id_int'] = pd.to_numeric(fused_df['RefChauffeur'], errors='coerce').fillna(0).astype(int)
        else:
            logger.error("  No chauffeur_id column found")
            return fused_df
        
        # Map sensitivity levels (default to 0 if not found)
        fused_df['vehicle_sensitivity'] = fused_df['vehicle_id_int'].map(
            sensitivity_levels['vehicles']
        ).fillna(0).astype(int)
        
        fused_df['chauffeur_sensitivity'] = fused_df['chauffeur_id_int'].map(
            sensitivity_levels['chauffeurs']
        ).fillna(0).astype(int)
        
        # Apply filter
        filtered_df = fused_df[
            (fused_df['vehicle_sensitivity'] >= client_sensitivity) &
            (fused_df['chauffeur_sensitivity'] >= client_sensitivity)
        ].copy()
        
        filtered_count = len(filtered_df)
        removed_count = initial_count - filtered_count
        
        logger.info("\nFilter Results:")
        logger.info(f"  • Initial recommendations: {initial_count}")
        logger.info(f"  • After sensitivity filter: {filtered_count}")
        logger.info(f"  • Removed: {removed_count} ({removed_count/initial_count*100 if initial_count > 0 else 0:.1f}%)")
        
        if removed_count > 0 and removed_count <= 5:
            logger.info("\nRemoved pairs (sensitivity mismatch):")
            removed_df = fused_df[~fused_df.index.isin(filtered_df.index)]
            for idx, row in removed_df.iterrows():
                logger.info(f"  • Pair {row['pair_id']}: "
                           f"Vehicle sensitivity={row['vehicle_sensitivity']}, "
                           f"Chauffeur sensitivity={row['chauffeur_sensitivity']}, "
                           f"Client requires >={client_sensitivity}")
        
        logger.info(f"{'='*60}\n")
        
        # Drop temporary columns
        filtered_df = filtered_df.drop(columns=['vehicle_id_int', 'chauffeur_id_int',
                                                'vehicle_sensitivity', 'chauffeur_sensitivity'])
        
        # Re-rank after filtering
        if not filtered_df.empty:
            filtered_df = filtered_df.sort_values('final_score', ascending=False).reset_index(drop=True)
            filtered_df['final_rank'] = range(1, len(filtered_df) + 1)
        
        return filtered_df


# =============================================================================
# WEIGHT MANAGER SERVICE
# =============================================================================

class WeightManager:
    """
    Service for managing fusion weights.
    
    Handles:
    - Weight redistribution when APIs are unavailable
    - Cold start weight adjustments
    """
    
    @staticmethod
    def redistribute_for_available_apis(weights: Dict[str, float],
                                        lightgcn_available: bool,
                                        collaborative_available: bool,
                                        user_content_available: bool,
                                        item_content_available: bool) -> Dict[str, float]:
        """
        Redistribute weights when some APIs are unavailable.
        Maintains relative proportions among available sources.
        
        Args:
            weights: Original fusion weights
            lightgcn_available: Whether LightGCN returned data
            collaborative_available: Whether CF returned data
            user_content_available: Whether User Content returned data
            item_content_available: Whether Item Content returned data
            
        Returns:
            Redistributed weights dictionary
        """
        availability = {
            'alpha1_rrf': lightgcn_available or collaborative_available or user_content_available or item_content_available,
            'alpha2_gcn': lightgcn_available,
            'alpha3_item_sim': item_content_available,
            'alpha4_user_sim': user_content_available,
            'alpha5_confidence': collaborative_available
        }
        
        available_count = sum(availability.values())
        
        if available_count == 0:
            logger.warning("⚠️ No APIs available - returning original weights")
            return weights
        
        if available_count == 5:  # All available (including RRF which needs at least one)
            return weights
        
        # Calculate sum of available weights
        available_weight_sum = sum(
            weights.get(key, 0) for key, available in availability.items() if available
        )
        
        if available_weight_sum == 0:
            logger.warning("⚠️ Available weight sum is 0 - using original weights")
            return weights
        
        # Redistribute: scale up available weights proportionally
        new_weights = {}
        for key, available in availability.items():
            if available:
                new_weights[key] = weights.get(key, 0) / available_weight_sum
            else:
                new_weights[key] = 0.0
        
        logger.info(" WEIGHT REDISTRIBUTION (some APIs unavailable):")
        logger.info(f"   APIs Available: LightGCN={lightgcn_available}, CF={collaborative_available}, "
                    f"User={user_content_available}, Item={item_content_available}")
        logger.info(f"   Original weights: {weights}")
        logger.info(f"   Redistributed:    {new_weights}")
        logger.info(f"   New sum: {sum(new_weights.values()):.4f}")
        
        return new_weights
    
    @staticmethod
    def adjust_for_cold_start(lightgcn_response: Dict, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust weights for cold start scenarios.
        
        When LightGCN reports cold start with popular items,
        boost content-based similarity weights.
        
        Args:
            lightgcn_response: Response from LightGCN API
            weights: Current fusion weights
            
        Returns:
            Adjusted weights dictionary
        """
        if not lightgcn_response:
            return weights
        
        is_cold_start = lightgcn_response.get('is_cold_start', False)
        recommendation_method = lightgcn_response.get('recommendation_method', '')
        
        if is_cold_start and recommendation_method == 'similarity_based':
            logger.info("\n❄️ COLD START DETECTED (similarity_based) - Boosting content similarity weights by 5x")
            adjusted_weights = weights.copy()
            
            # Multiply BOTH similarity weights by 5
            original_item_sim = adjusted_weights.get('alpha3_item_sim', 0.1)
            original_user_sim = adjusted_weights.get('alpha4_user_sim', 0.1)
            
            adjusted_weights['alpha3_item_sim'] = original_item_sim * 5
            adjusted_weights['alpha4_user_sim'] = original_user_sim * 5
            
            # Normalize to sum to 1.0
            total_weight = sum(adjusted_weights.values())
            for key in adjusted_weights:
                adjusted_weights[key] = adjusted_weights[key] / total_weight
            
            logger.info(f"   Original item similarity weight: {original_item_sim:.3f} → {adjusted_weights['alpha3_item_sim']:.3f}")
            logger.info(f"   Original user similarity weight: {original_user_sim:.3f} → {adjusted_weights['alpha4_user_sim']:.3f}")
            logger.info(f"   All adjusted weights (normalized): {json.dumps(adjusted_weights, indent=2)}")
            
            return adjusted_weights
        
        elif is_cold_start:
            logger.info(f"\n❄️ COLD START DETECTED but method is '{recommendation_method}', keeping original weights")
        
        return weights


# =============================================================================
# FACADE CLASS - UNIFIED ACCESS TO ALL SERVICES
# =============================================================================

class RecommendationService:
    """
    Facade class providing unified access to all recommendation services.
    
    This class coordinates the various services and provides a simpler
    interface for the API routes.
    """
    
    def __init__(self):
        self.response_processor = ResponseProcessor()
        self.fuser = RecommendationFuser()
        self.sensitivity_filter = SensitivityFilter()
        self.weight_manager = WeightManager()
    
    def process_lightgcn(self, response: Dict) -> pd.DataFrame:
        """Process LightGCN API response"""
        return self.response_processor.process_lightgcn_response(response)
    
    def process_collaborative(self, response: Dict) -> pd.DataFrame:
        """Process Collaborative API response"""
        return self.response_processor.process_collaborative_response(response)
    
    def process_user_content(self, response: Dict) -> pd.DataFrame:
        """Process User Content API response"""
        return self.response_processor.process_user_content_response(response)
    
    def process_item_content(self, response: Dict) -> pd.DataFrame:
        """Process Item Content API response"""
        return self.response_processor.process_item_content_response(response)
    
    def fuse_recommendations(self, lightgcn_df: pd.DataFrame,
                             collaborative_df: pd.DataFrame,
                             user_content_df: pd.DataFrame,
                             item_content_df: pd.DataFrame,
                             weights: Dict[str, float]) -> pd.DataFrame:
        """Fuse recommendations from all systems"""
        return self.fuser.fuse(lightgcn_df, collaborative_df, 
                               user_content_df, item_content_df, weights)
    
    def apply_sensitivity_filter(self, fused_df: pd.DataFrame,
                                 client_id: int,
                                 sensitivity_levels: Dict) -> pd.DataFrame:
        """Apply sensitivity filtering"""
        return self.sensitivity_filter.apply(fused_df, client_id, sensitivity_levels)
    
    def redistribute_weights(self, weights: Dict[str, float],
                             lightgcn_available: bool,
                             collaborative_available: bool,
                             user_content_available: bool,
                             item_content_available: bool) -> Dict[str, float]:
        """Redistribute weights for available APIs"""
        return self.weight_manager.redistribute_for_available_apis(
            weights, lightgcn_available, collaborative_available,
            user_content_available, item_content_available
        )
    
    def adjust_weights_for_cold_start(self, lightgcn_response: Dict,
                                      weights: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights for cold start scenarios"""
        return self.weight_manager.adjust_for_cold_start(lightgcn_response, weights)


# Create singleton instance for easy import
recommendation_service = RecommendationService()

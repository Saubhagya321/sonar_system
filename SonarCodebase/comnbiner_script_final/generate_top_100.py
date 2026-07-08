
from request_pipeline import RecommendationPipeline
import pandas as pd
import json
from datetime import datetime
import sys
import numpy as np
import logging 

# ADD THIS
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_score_contribution(recommendations_df: pd.DataFrame, weights: dict):
    """
    Analyze how much each model contributes to final scores
    
    Args:
        recommendations_df: DataFrame with all scores
        weights: Weight dictionary used
    """
    logger.info("=" * 70)
    logger.info("  SCORE CONTRIBUTION ANALYSIS")
   
    
    score_cols = {
        'rrf_score': weights.get('alpha1_rrf', 0),
        'gcn_avg_score': weights.get('alpha2_gcn', 0),
        'item_similarity': weights.get('alpha3_item_sim', 0),
        'user_similarity': weights.get('alpha4_user_sim', 0),
        'cf_confidence': weights.get('alpha5_confidence', 0)
    }
    
    logger.info("Per-Model Statistics:")
   
    
    total_contribution = {}
    
    for col, weight in score_cols.items():
        if col not in recommendations_df.columns:
            continue
        
        non_zero = (recommendations_df[col] > 0).sum()
        mean_score = recommendations_df[col].mean()
        max_score = recommendations_df[col].max()
        min_score = recommendations_df[col].min()
        
        # Calculate weighted contribution
        weighted_mean = mean_score * weight
        total_contribution[col] = weighted_mean
        
        logger.info(f"{col}:")
        logger.info(f"  Weight: {weight:.3f}")
        logger.info(f"  Non-zero pairs: {non_zero}/{len(recommendations_df)} ({non_zero/len(recommendations_df)*100:.1f}%)")
        logger.info(f"  Score range: [{min_score:.4f}, {max_score:.4f}]")
        logger.info(f"  Mean score: {mean_score:.4f}")
        logger.info(f"  Weighted contribution: {weighted_mean:.4f}")
        
        if non_zero == 0:
            logger.warning(f"Model '{col}' is not contributing any scores")
    
    # Show relative contributions
    
    logger.info(" Relative Contribution to Final Score:")
    
    total = sum(total_contribution.values())
    if total > 0:
        for col, contrib in sorted(total_contribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (contrib / total) * 100
            bar = "" * int(percentage / 2)
            logger.info(f"{col:20s}: {bar} {percentage:.1f}%")
    
    logger.info("="*70 + "\n")


def compare_before_after_normalization(all_data: dict):
    """
    Compare score distributions before and after normalization
    
    Args:
        all_data: Dictionary with all model data (unnormalized)
    """
    logger.info("\n" + "="*70)
    logger.info("   NORMALIZATION IMPACT ANALYSIS")
    logger.info("="*70)
    
    logger.info("\n BEFORE Normalization:")
    logger.info("-" * 70)
    
    # LightGCN
    if not all_data['lightgcn'].empty:
        gcn_scores = []
        for col in ['cv_score', 'cc_score', 'vc_score']:
            if col in all_data['lightgcn'].columns:
                gcn_scores.extend(all_data['lightgcn'][col].dropna().tolist())
        
        if gcn_scores:
            logger.info(f"LightGCN scores:")
            logger.info(f"  Range: [{min(gcn_scores):.2f}, {max(gcn_scores):.2f}]")
            logger.info(f"  Mean:  {np.mean(gcn_scores):.2f}")
            logger.info(f"  Std:   {np.std(gcn_scores):.2f}")
    
    # Collaborative
    if not all_data['collaborative'].empty and 'predicted_rating' in all_data['collaborative'].columns:
        cf_scores = all_data['collaborative']['predicted_rating'].dropna()
        logger.info(f"\nCollaborative predicted_rating:")
        logger.info(f"  Range: [{cf_scores.min():.2f}, {cf_scores.max():.2f}]")
        logger.info(f"  Mean:  {cf_scores.mean():.2f}")
        logger.info(f"  Std:   {cf_scores.std():.2f}")
    
    # Similarities
    if not all_data['item_similarity'].empty and 'similarity' in all_data['item_similarity'].columns:
        item_sim = all_data['item_similarity']['similarity'].dropna()
        logger.info(f"\nItem-Item similarity:")
        logger.info(f"  Range: [{item_sim.min():.2f}, {item_sim.max():.2f}]")
        logger.info(f"  Mean:  {item_sim.mean():.2f}")
    
    if not all_data['user_similarity'].empty and 'similarity' in all_data['user_similarity'].columns:
        user_sim = all_data['user_similarity']['similarity'].dropna()
        logger.info(f"\nUser-Item similarity:")
        logger.info(f"  Range: [{user_sim.min():.2f}, {user_sim.max():.2f}]")
        logger.info(f"  Mean:  {user_sim.mean():.2f}")
    
    logger.info("\n" + "-" * 70)
    logger.info(" KEY INSIGHT:")
    logger.info("  Without normalization, GCN scores (~4-5) dominate over similarities (~0.7)")
    logger.info("  Even with weights, 4.5 × 0.3 = 1.35 >> 0.7 × 0.1 = 0.07")
    logger.info("  This makes other models almost irrelevant!")
    logger.info("\n  After normalization, all scores are 0-1, so weights work properly.")
    logger.info("="*70 + "\n")


def main():
    """Generate and save top 100 recommendations with normalization"""
    
    logger.info("\n" + "="*70)
    logger.info("  GENERATING TOP 100 RECOMMENDATIONS - WITH NORMALIZATION")
    logger.info("="*70 + "\n")
    
    # Initialize pipeline
    try:
        pipeline = RecommendationPipeline()
    except Exception as e:
        logger.error(f"  Error initializing pipeline: {str(e)}")
        sys.exit(1)
    
    # Load data for diagnostics
    logger.info(" Loading recommendation data for diagnostics...")
    try:
        from data_loaders import DataLoader
        import numpy as np
        loader = DataLoader()
        all_data = loader.load_all_request_data(client_id=811)
    except Exception as e:
        logger.error(f"  Error loading data: {str(e)}")
        sys.exit(1)
    
    # Show before/after normalization comparison
    compare_before_after_normalization(all_data)
    
    # Generate recommendations
    logger.info("\n Generating recommendations with normalization...")
    try:
        result = pipeline.generate_recommendations(client_id=811)
    except Exception as e:
        logger.error(f"  Error generating recommendations: {str(e)}")
        import traceback
        traceback.logger.info_exc()
        sys.exit(1)
    
    # Check if successful
    if 'error' in result:
        logger.error(f"  Error: {result['error']}")
        sys.exit(1)
    
    # Get top 100
    logger.info("   Recommendations generated successfully!")
    logger.info(f"   Total recommendations: {result['total_recommendations']}")
    logger.info(f"   Processing time: {result['processing_time_seconds']:.3f}s")
    logger.info(f"   Normalization applied: {result.get('normalization_applied', False)}")
    
    top_100 = pipeline.get_top_k(result, k=100)
    
    # Convert to DataFrame
    recommendations_df = pd.DataFrame(top_100)
    
    # Analyze score contributions
    analyze_score_contribution(recommendations_df, result['weights_used'])
    
    # Generate filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'top_100_normalized_client_811_{timestamp}.csv'
    json_filename = f'full_result_normalized_client_811_{timestamp}.json'
    
    # Save to CSV
    logger.info(f"\n Saving results...")
    try:
        recommendations_df.to_csv(csv_filename, index=False)
        logger.info(f"      CSV saved: {csv_filename}")
    except Exception as e:
        logger.info(f"     Error saving CSV: {str(e)}")
    
    # Save full result to JSON
    try:
        result_copy = result.copy()
        result_copy['recommendations'] = top_100
        
        with open(json_filename, 'w') as f:
            json.dump(result_copy, f, indent=2)
        logger.info(f"      JSON saved: {json_filename}")
    except Exception as e:
        logger.error(f"     Error saving JSON: {str(e)}")
    
    # Display summary
    logger.info("\n" + "="*70)
    logger.info("  SUMMARY")
    logger.info("="*70)
    logger.info(f"Client ID:              {result['client_id']}")
    logger.info(f"Client Type:            {result['client_type']}")
    logger.info(f"Cold Start:             {'Yes' if result['is_cold_start'] else 'No'}")
    logger.info(f"Total Recommendations:  {result['total_recommendations']}")
    logger.info(f"Processing Time:        {result['processing_time_seconds']:.3f}s")
    logger.info(f"Normalization Applied:  {result.get('normalization_applied', False)}")
    logger.info(f"Timestamp:              {result['timestamp']}")
    
    logger.info(f"\n{'='*70}")
    logger.info("  WEIGHTS USED")
    logger.info("="*70)
    for key, value in result['weights_used'].items():
        logger.info(f"{key:20s}: {value:.4f}")
    
    # Show top 10 with ALL score details
    logger.info(f"\n{'='*70}")
    logger.info("  TOP 10 PAIRS (WITH NORMALIZED SCORES)")
    logger.info("="*70)
    
    display_cols = ['pair_id', 'final_rank', 'final_score', 'rrf_score', 
                    'gcn_avg_score', 'item_similarity', 'user_similarity', 'cf_confidence']
    
    available_cols = [col for col in display_cols if col in recommendations_df.columns]
    
    if available_cols:
        top10 = recommendations_df[available_cols].head(10)
        logger.info(top10.to_string(index=False))
    
    # Score distribution summary
    if 'final_score' in recommendations_df.columns:
        logger.info(f"\n{'='*70}")
        logger.info("  FINAL SCORE DISTRIBUTION")
        logger.info("="*70)
        logger.info(f"Highest Score:   {recommendations_df['final_score'].max():.4f}")
        logger.info(f"Lowest Score:    {recommendations_df['final_score'].min():.4f}")
        logger.info(f"Average Score:   {recommendations_df['final_score'].mean():.4f}")
        logger.info(f"Median Score:    {recommendations_df['final_score'].median():.4f}")
        logger.info(f"Std Deviation:   {recommendations_df['final_score'].std():.4f}")
    
    # Final recommendations
    logger.info("\n" + "="*70)
    logger.info("     COMPLETE")
    logger.info("="*70)
    
    logger.info("\n Output Files:")
    logger.info(f"   • {csv_filename}")
    logger.info(f"   • {json_filename}")
    
    logger.info("\n Key Improvements:")
    logger.info("      All scores normalized to 0-1 range before fusion")
    logger.info("      All models now contribute fairly based on weights")
    logger.info("      Cold-start handling with 4× boosted similarity weights")
    logger.info("      Score contribution analysis shows model effectiveness")
    
    logger.info("\n Next Steps:")
    logger.info("   1. Compare this CSV with your previous unnormalized output")
    logger.info("   2. Check that all score columns have values 0-1")
    logger.info("   3. Verify that item/user similarity now contribute to rankings")
    logger.info("   4. Review score contribution analysis above")
    logger.info("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n  Unexpected error: {str(e)}")
        import traceback
        traceback.logger.info_exc()
        sys.exit(1)
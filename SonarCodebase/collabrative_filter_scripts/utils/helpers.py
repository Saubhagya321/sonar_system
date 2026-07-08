import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def save_pickle(obj: Any, filepath: Path):
    """Save object to pickle file"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f" Saved to {filepath}")
    except (IOError, OSError, pickle.PicklingError) as e:
        logger.error(f"  Error saving to {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"  Unexpected error saving to {filepath}: {e}", exc_info=True)
        raise


def load_pickle(filepath: Path) -> Any:
    """Load object from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"  Loaded from {filepath}")
        return obj
    except (IOError, OSError, pickle.UnpicklingError) as e:
        logger.error(f"  Error loading from {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"  Unexpected error loading from {filepath}: {e}", exc_info=True)
        raise

def save_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """Save DataFrame to pickle"""
    save_pickle(df, filepath)


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """Load DataFrame from pickle"""
    return load_pickle(filepath)


def setup_logging(log_level: str = "INFO", log_file: Path = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def print_summary_statistics(recommendations: Dict, ratings: pd.DataFrame) -> None:
    """
    Print summary statistics for recommendations
    FROM: Document 4 - summary printing sections
    
    Args:
        recommendations: Dictionary of recommendations
        ratings: Ratings dataframe
    """
    logger.info("\n" + "="*80)
    logger.info(" RECOMMENDATION SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Calculate statistics
    all_scores = []
    all_confidences = []
    
    for client_recs in recommendations.values():
        for rec in client_recs:
            all_scores.append(rec['predicted_rating'])
            all_confidences.append(rec['confidence'])
    
    logger.info("\n RECOMMENDATION STATISTICS:")
    logger.info(f"   • Clients processed: {len(recommendations):,}")
    logger.info(f"   • Total recommendations: {len(all_scores):,}")
    logger.info(f"   • Avg predicted rating: {np.mean(all_scores):.2f}")
    logger.info(f"   • Rating std: {np.std(all_scores):.2f}")
    logger.info(f"   • Avg confidence: {np.mean(all_confidences):.3f}")
    logger.info(f"   • High confidence recs (>0.7): {sum(1 for c in all_confidences if c > 0.7):,} "
               f"({sum(1 for c in all_confidences if c > 0.7)/len(all_confidences)*100:.1f}%)")
    
    # Rating distribution
    logger.info("\n PREDICTED RATING DISTRIBUTION:")
    for i in range(1, 6):
        count = sum(1 for s in all_scores if i <= s < i+1)
        pct = count / len(all_scores) * 100 if all_scores else 0
        logger.info(f"   • {i}.0-{i}.9: {count:,} ({pct:.1f}%)")


def export_recommendations_csv(recommendations: Dict, output_path: Path) -> None:
    """
    Export recommendations to CSV file
    
    Args:
        recommendations: Dictionary of recommendations
        output_path: Output CSV file path
    """
    logger.info(" Exporting recommendations to CSV...")
    
    rows = []
    for client_id, recs in recommendations.items():
        for rec in recs:
            rows.append({
                'client_id': client_id,
                'chauffeur_vehicle_pair': rec['chauffeur_vehicle_pair'],
                'predicted_rating': rec['predicted_rating'],
                'confidence': rec['confidence'],
                'rank': rec['recommendation_rank']
            })
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"   Exported {len(rows):,} recommendations to {output_path}")


__all__ = [
    'save_pickle',
    'load_pickle', 
    'save_dataframe',
    'load_dataframe',
    'setup_logging',
    'print_summary_statistics',
    'export_recommendations_csv'
]
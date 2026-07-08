# combiner
"""
Utility Functions for Recommendation Combiner
Helper functions used across the system
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from .config import setup_logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
setup_logging()
logger = logging.getLogger(__name__)


def normalize_scores(df: pd.DataFrame, 
                     columns: List[str],
                     method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize score columns to 0-1 range
    
    Args:
        df: DataFrame with scores
        columns: List of column names to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        DataFrame with normalized columns
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            result[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val + 1e-10)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            result[f'{col}_normalized'] = (df[col] - mean_val) / (std_val + 1e-10)
    
    return result


def create_pair_id(chauffeur_id: Any, vehicle_id: Any) -> str:
    """
    Create standardized pair ID from chauffeur and vehicle IDs
    
    Args:
        chauffeur_id: Chauffeur ID
        vehicle_id: Vehicle ID
        
    Returns:
        Pair ID string
    """
    return f"{chauffeur_id}_{vehicle_id}"


def parse_pair_id(pair_id: str) -> tuple:
    """
    Parse pair ID into chauffeur and vehicle IDs
    
    Args:
        pair_id: Pair ID string (format: chauffeur_vehicle)
        
    Returns:
        (chauffeur_id, vehicle_id) tuple
    """
    parts = pair_id.split('_')
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def load_json(file_path: Path) -> Dict:
    """
    Load JSON file safely
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON or empty dict if fails
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return {}


def save_json(data: Dict, file_path: Path, indent: int = 2):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        file_path: Path to save to
        indent: JSON indentation
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f"    Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"    Failed to save JSON: {str(e)}")


def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str],
                       name: str = "DataFrame") -> bool:
    """
    Validate that DataFrame has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for logging
        
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        logger.warning(f"{name} is empty")
        return False
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        logger.error(f"{name} missing columns: {missing_cols}")
        return False
    
    logger.info(f"    {name} validated: {len(df)} rows, all required columns present")
    return True


def merge_recommendations(dfs: List[pd.DataFrame],
                         on: str = 'pair_id',
                         how: str = 'outer') -> pd.DataFrame:
    """
    Merge multiple recommendation DataFrames
    
    Args:
        dfs: List of DataFrames to merge
        on: Column to merge on
        how: Merge method ('outer', 'inner', 'left', 'right')
        
    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    result = dfs[0].copy()
    
    for df in dfs[1:]:
        result = result.merge(df, on=on, how=how, suffixes=('', '_dup'))
    
    # Remove duplicate columns
    result = result.loc[:, ~result.columns.str.endswith('_dup')]
    
    return result


def compute_rank_correlation(ranks1: List[int], ranks2: List[int]) -> float:
    """
    Compute Spearman rank correlation between two rankings
    
    Args:
        ranks1: First ranking
        ranks2: Second ranking
        
    Returns:
        Correlation coefficient (-1 to 1)
    """
    from scipy.stats import spearmanr
    
    if len(ranks1) != len(ranks2):
        logger.warning("Rank lists have different lengths")
        return 0.0
    
    if len(ranks1) < 2:
        return 0.0
    
    try:
        correlation, _ = spearmanr(ranks1, ranks2)
        return correlation
    except Exception as e:
        logger.warning(f"Spearman correlation calculation failed: {e}")       
        return 0.0


def get_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> Dict:
    """
    Get summary statistics for a DataFrame
    
    Args:
        df: DataFrame to summarize
        name: Name for the summary
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'name': name,
        'rows': len(df),
        'columns': list(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'dtypes': df.dtypes.to_dict()
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = {
            col: {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing': df[col].isna().sum()
            }
            for col in numeric_cols
        }
    
    return summary


def print_data_summary(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Print formatted data summary
    
    Args:
        df: DataFrame to summarize
        name: Name for the summary
    """
    summary = get_data_summary(df, name)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Data Summary: {summary['name']}")
    logger.info(f"{'='*60}")
    logger.info(f"Rows: {summary['rows']}")
    logger.info(f"Columns: {len(summary['columns'])}")
    logger.info(f"Memory: {summary['memory_mb']:.2f} MB")
    
    if 'numeric_stats' in summary:
        logger.info(f"\nNumeric Columns:")
        for col, stats in summary['numeric_stats'].items():
            logger.info(f"  {col}:")
            logger.info(f"    Mean: {stats['mean']:.3f}")
            logger.info(f"    Std:  {stats['std']:.3f}")
            logger.info(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            if stats['missing'] > 0:
                logger.info(f"    Missing: {stats['missing']}")
    
    logger.info(f"{'='*60}\n")


def export_to_csv(df: pd.DataFrame, 
                 file_path: Path,
                 include_index: bool = False):
    """
    Export DataFrame to CSV with error handling
    
    Args:
        df: DataFrame to export
        file_path: Path to save to
        include_index: Whether to include index
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=include_index)
        logger.info(f"    Exported to CSV: {file_path}")
    except Exception as e:
        logger.error(f"    Failed to export CSV: {str(e)}")


def export_to_json(df: pd.DataFrame,
                  file_path: Path,
                  orient: str = 'records',
                  indent: int = 2):
    """
    Export DataFrame to JSON with error handling
    
    Args:
        df: DataFrame to export
        file_path: Path to save to
        orient: JSON orientation
        indent: Indentation level
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(file_path, orient=orient, indent=indent)
        logger.info(f"    Exported to JSON: {file_path}")
    except Exception as e:
        logger.error(f"    Failed to export JSON: {str(e)}")


def timing_decorator(func):
    """
    Decorator to measure function execution time
    
    Usage:
        @timing_decorator
        def my_function():
            ...
    """
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"⏱ {func.__name__} took {elapsed_time:.3f} seconds")
        return result
    
    return wrapper


# Example usage
if __name__ == "__main__":
    # Test utilities
    logger.info("Testing utilities...")
    
    # Test pair ID functions
    pair_id = create_pair_id(100, 200)
    logger.info(f"Pair ID: {pair_id}")
    
    chauffeur, vehicle = parse_pair_id(pair_id)
    logger.info(f"Parsed: chauffeur={chauffeur}, vehicle={vehicle}")
    
    # Test DataFrame operations
    sample_df = pd.DataFrame({
        'score1': [1, 2, 3, 4, 5],
        'score2': [10, 20, 30, 40, 50],
        'rank': [1, 2, 3, 4, 5]
    })
    
    print_data_summary(sample_df, "Sample Data")
    
    # Test normalization
    normalized = normalize_scores(sample_df, ['score1', 'score2'])
    logger.info("\nNormalized scores:")
    logger.info(normalized[['score1', 'score1_normalized', 'score2', 'score2_normalized']])

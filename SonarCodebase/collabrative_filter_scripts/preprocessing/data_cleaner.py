import pandas as pd
import numpy as np
import logging
from typing import Tuple

from config.config import Config

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and validate booking data"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def clean_booking_data(self, booking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean booking data - fix dates and handle missing values
        FROM: Document 2 - clean_booking_data function
        
        Args:
            booking_df: Raw booking dataframe
            
        Returns:
            Cleaned booking dataframe
        """
        logger.info(" Cleaning booking data...")
        df = booking_df.copy()
        original_rows = len(df)
        
        # 1. Fix date issues
        df = self._fix_date_columns(df)
        
        # 2. Handle missing critical values
        df = self._handle_missing_values(df)
        
        # 3. Fix price anomalies
        df = self._fix_price_anomalies(df)
        
        rows_removed = original_rows - len(df)
        removal_pct = (rows_removed / original_rows) * 100
        
        logger.info("   Data cleaning complete:")
        logger.info(f"   Original rows: {original_rows:,}")
        logger.info(f"   Cleaned rows: {len(df):,}")
        logger.info(f"   Rows removed: {rows_removed:,} ({removal_pct:.1f}%)")
        
        return df
    
    def _fix_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix date parsing issues
        FROM: Document 2 - clean_booking_data function (date fixing section)
        """
        logger.info("  Fixing date columns...")
        
        min_valid = pd.Timestamp(self.config.MIN_VALID_DATE)
        max_valid = pd.Timestamp(self.config.MAX_VALID_DATE)
        
        date_columns = [
            'BookingDate', 'DateCompleted', 'AppointmentDate',
            'BookingStatusDate', 'CreatedOn'
        ]
        
        for col in date_columns:
            if col not in df.columns:
                continue
                
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Filter invalid dates
            date_mask = (df[col] >= min_valid) & (df[col] <= max_valid)
            invalid_count = (~date_mask & df[col].notna()).sum()
            
            if invalid_count > 0:
                logger.info(f"   {col}: {invalid_count:,} invalid dates found")
                df.loc[~date_mask, col] = pd.NaT
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle critical missing values
        FROM: Document 2 - clean_booking_data function (missing values section)
        """
        logger.info(" Handling missing values...")
        
        # Remove bookings without chauffeur AND vehicle
        before_filter = len(df)
        df = df.dropna(subset=['RefChauffeur', 'RefVehicle'])
        removed = before_filter - len(df)
        logger.info(f"   Removed {removed:,} bookings missing chauffeur or vehicle")
        
        # Remove bookings without valid dates
        before_filter = len(df)
        df = df.dropna(subset=['BookingDate'])
        removed = before_filter - len(df)
        logger.info(f"   Removed {removed:,} bookings without valid BookingDate")
        
        return df
    
    def _fix_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix price data anomalies
        FROM: Document 2 - clean_booking_data function (price fixing section)
        """
        logger.info(" Cleaning price data...")
        
        # Remove negative prices
        if 'PriceComputed' in df.columns:
            negative_count = (df['PriceComputed'] < 0).sum()
            if negative_count > 0:
                logger.info(f"   Fixing {negative_count:,} negative prices")
                df.loc[df['PriceComputed'] < 0, 'PriceComputed'] = np.nan
        
        # Cap extreme prices
        price_cols = ['PriceComputed', 'QuoteToClient']
        for col in price_cols:
            if col not in df.columns:
                continue
                
            q99 = df[col].quantile(0.99)
            extreme_threshold = q99 * 5
            extreme_count = (df[col] > extreme_threshold).sum()
            
            if extreme_count > 0:
                logger.info(f"   Capping {extreme_count:,} extreme {col} values above ${extreme_threshold:.2f}")
                df.loc[df[col] > extreme_threshold, col] = q99
        
        return df
# collabrative 
"""
Feature engineering for collaborative filtering - FIXED VERSION
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict

from config.config import Config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for collaborative filtering"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def create_pair_features(self, booking_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create chauffeur-vehicle pair features with validation
        
        Args:
            booking_df: Cleaned booking dataframe
            
        Returns:
            Tuple of (booking_with_pairs, pair_features)
        """
        logger.info("   Creating chauffeur-vehicle pair features...")
        df = booking_df.copy()
        
        original_rows = len(df)
        
        # CRITICAL FIX: Validate and clean IDs before creating pairs
        logger.info("   Validating chauffeur and vehicle IDs...")
        
        # Convert to numeric, coercing errors to NaN
        df['RefChauffeur'] = pd.to_numeric(df['RefChauffeur'], errors='coerce')
        df['RefVehicle'] = pd.to_numeric(df['RefVehicle'], errors='coerce')
        
        # Check for invalid IDs
        invalid_chauffeur = df['RefChauffeur'].isna().sum()
        invalid_vehicle = df['RefVehicle'].isna().sum()
        
        if invalid_chauffeur > 0:
            logger.warning(f"     Found {invalid_chauffeur} bookings with invalid chauffeur IDs")
        if invalid_vehicle > 0:
            logger.warning(f"     Found {invalid_vehicle} bookings with invalid vehicle IDs")
        
        # Drop rows with invalid IDs
        df = df.dropna(subset=['RefChauffeur', 'RefVehicle'])
        
        removed_rows = original_rows - len(df)
        if removed_rows > 0:
            logger.warning(f"   Removed {removed_rows} bookings with invalid IDs ({removed_rows/original_rows*100:.1f}%)")
        
        # Ensure IDs are integers
        df['RefChauffeur'] = df['RefChauffeur'].astype(int)
        df['RefVehicle'] = df['RefVehicle'].astype(int)
        
        # Create pair identifier with validation
        logger.info("   Creating pair identifiers...")
        try:
            df['ChauffeurVehiclePair'] = (
                df['RefChauffeur'].astype(str) + '_' + 
                df['RefVehicle'].astype(str)
            )
            logger.info(f"      Created {df['ChauffeurVehiclePair'].nunique():,} unique pairs")
        except Exception as e:
            logger.error(f"     Failed to create pair IDs: {e}")
            raise ValueError(f"Failed to create pair identifiers: {e}")
        
        # Validate pair format
        sample_pairs = df['ChauffeurVehiclePair'].head(5).tolist()
        logger.info(f"   Sample pair IDs: {sample_pairs}")
        
       
         #Check for invalid pairs but DON'T crash - just warn and remove them
        invalid_pairs = df[~df['ChauffeurVehiclePair'].str.match(r'^\d+_\d+$')]
        if len(invalid_pairs) > 0:
            logger.warning(f"     Found {len(invalid_pairs)} pairs with invalid format - removing them")
            logger.warning(f"   Examples: {invalid_pairs['ChauffeurVehiclePair'].head(3).tolist()}")
            
            # Remove invalid pairs instead of crashing
            df = df[df['ChauffeurVehiclePair'].str.match(r'^\d+_\d+$')]
            logger.info(f"   Continuing with {len(df):,} valid bookings")
        # Calculate pair statistics
        logger.info("   Calculating pair statistics...")
        pair_stats = df.groupby('ChauffeurVehiclePair').agg({
            'ID': 'count',
            'DateCompleted': 'count',
            'PriceComputed': ['mean', 'std'],
            'BookingDate': ['min', 'max'],
            'RefClient': 'nunique'
        }).round(2)
        
        # Flatten column names
        pair_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                             for col in pair_stats.columns]
        pair_stats = pair_stats.rename(columns={
            'ID_count': 'total_bookings',
            'DateCompleted_count': 'completed_bookings',
            'PriceComputed_mean': 'avg_price',
            'PriceComputed_std': 'price_std',
            'BookingDate_min': 'first_booking',
            'BookingDate_max': 'last_booking',
            'RefClient_nunique': 'unique_clients'
        })
        
        # Calculate quality metrics
        pair_stats['completion_rate'] = (
            pair_stats['completed_bookings'] / pair_stats['total_bookings']
        )
        pair_stats['price_consistency'] = (
            1 - (pair_stats['price_std'] / pair_stats['avg_price'])
        ).fillna(0).clip(0, 1)
        
        # Calculate recency
        current_date = df['BookingDate'].max()
        pair_stats['days_since_last_booking'] = (
            current_date - pair_stats['last_booking']
        ).dt.days
        pair_stats['pair_age_days'] = (
            pair_stats['last_booking'] - pair_stats['first_booking']
        ).dt.days
        
        logger.info(f"      Created features for {len(pair_stats):,} chauffeur-vehicle pairs")
        logger.info(f"   Average bookings per pair: {pair_stats['total_bookings'].mean():.1f}")
        logger.info(f"   Average completion rate: {pair_stats['completion_rate'].mean():.2%}")
        
        return df, pair_stats
    
    def build_implicit_ratings(self, 
                              booking_df: pd.DataFrame, 
                              pair_features: pd.DataFrame) -> pd.DataFrame:
        """
        Build implicit rating signals from booking behavior
        
        Args:
            booking_df: Booking dataframe with pairs
            pair_features: Pair feature dataframe
            
        Returns:
            Implicit ratings dataframe
        """
        logger.info(" Building implicit rating signals...")
        
        # 1. Frequency-based rating
        frequency_ratings = self._calculate_frequency_ratings(booking_df)
        
        # 2. Recency-based rating
        # recency_ratings = self._calculate_recency_ratings(booking_df)
        
        # 3. Completion-based rating
        completion_ratings = self._calculate_completion_ratings(booking_df)
        
        # 4. Price acceptance rating
        price_ratings = self._calculate_price_ratings(booking_df, pair_features)
        
        # 5. Combine signals
        implicit_ratings = self._combine_implicit_signals(
            frequency_ratings, completion_ratings, price_ratings
        )
        
        logger.info(f"   Created implicit ratings for {len(implicit_ratings):,} client-pair combinations")
        logger.info(f"   Average implicit rating: {implicit_ratings['implicit_rating'].mean():.2f}")
        
        return implicit_ratings
    
    def _calculate_frequency_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate frequency-based preferences"""
        logger.info("    Calculating frequency-based preferences...")
        
        client_pair_counts = df.groupby(['RefClient', 'ChauffeurVehiclePair']).size().reset_index(name='pair_bookings')
        client_total_counts = df.groupby('RefClient').size().reset_index(name='total_bookings')
        
        frequency_ratings = client_pair_counts.merge(client_total_counts, on='RefClient')
        frequency_ratings['frequency_score'] = (
            frequency_ratings['pair_bookings'] / frequency_ratings['total_bookings']
        )
        frequency_ratings['frequency_rating'] = (
            1 + (frequency_ratings['frequency_score'] * 4)
        ).clip(1, 5)
        
        return frequency_ratings[['RefClient', 'ChauffeurVehiclePair', 'frequency_rating']]
    
  
    def _calculate_completion_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate completion-based preferences"""
        logger.info("      Calculating completion-based preferences...")
        
        completion_data = df.groupby(['RefClient', 'ChauffeurVehiclePair']).agg({
            'ID': 'count',
            'DateCompleted': 'count'
        }).reset_index()
        completion_data.columns = ['RefClient', 'ChauffeurVehiclePair', 'total_bookings', 'completed_bookings']
        
        completion_data['completion_rate'] = (
            completion_data['completed_bookings'] / completion_data['total_bookings']
        )
        completion_data['completion_rating'] = (
            1 + (completion_data['completion_rate'] * 4)
        ).clip(1, 5)
        
        return completion_data[['RefClient', 'ChauffeurVehiclePair', 'completion_rating']]
    
    def _calculate_price_ratings(self, df: pd.DataFrame, pair_features: pd.DataFrame) -> pd.DataFrame:
        """Calculate price acceptance preferences"""
        logger.info("    Calculating price acceptance preferences...")
        
        price_data = df[df['PriceComputed'].notna()].groupby(['RefClient', 'ChauffeurVehiclePair']).agg({
            'PriceComputed': ['mean', 'count']
        }).reset_index()
        price_data.columns = ['RefClient', 'ChauffeurVehiclePair', 'avg_price_paid', 'price_bookings']
        
        # Merge with pair averages
        pair_avg = pair_features[['avg_price']].reset_index()
        price_data = price_data.merge(pair_avg, on='ChauffeurVehiclePair', how='left')
        
        # Calculate acceptance ratio
        price_data['price_ratio'] = (price_data['avg_price_paid'] / price_data['avg_price']).fillna(1.0).clip(0.5, 2.0)
        price_data['price_rating'] = (1 + ((price_data['price_ratio'] - 0.5) / 1.5) * 4).clip(1, 5)
        
        return price_data[['RefClient', 'ChauffeurVehiclePair', 'price_rating']]
    
    def _combine_implicit_signals(self, 
                                  frequency: pd.DataFrame,
                                  completion: pd.DataFrame,
                                  price: pd.DataFrame) -> pd.DataFrame:
        """Combine all implicit signals"""
        logger.info("    Combining implicit rating signals...")
        
        # Start with frequency
        implicit = frequency.copy()
        
        # Merge other signals
        # implicit = implicit.merge(recency, on=['RefClient', 'ChauffeurVehiclePair'], how='left')
        implicit = implicit.merge(completion, on=['RefClient', 'ChauffeurVehiclePair'], how='left')
        implicit = implicit.merge(price, on=['RefClient', 'ChauffeurVehiclePair'], how='left')
        
        # Fill missing with neutral
        implicit = implicit.fillna(3.0)
        
        # Weighted combination
        implicit['implicit_rating'] = (
            implicit['frequency_rating'] * self.config.FREQUENCY_WEIGHT +
            # implicit['recency_rating'] * self.config.RECENCY_WEIGHT +
            implicit['completion_rating'] * self.config.COMPLETION_WEIGHT +
            implicit['price_rating'] * self.config.PRICE_WEIGHT
        )
        
        return implicit
    
    def integrate_explicit_ratings(self,
                                   implicit_ratings: pd.DataFrame,
                                   rating_df: pd.DataFrame,
                                   booking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate explicit ratings with implicit ratings
        
        Args:
            implicit_ratings: Implicit ratings dataframe
            rating_df: Explicit ratings dataframe
            booking_df: Booking dataframe
            
        Returns:
            Final ratings dataframe
        """
        logger.info(" Integrating explicit & implicit ratings...")
        
        if len(rating_df) == 0:
            logger.warning("   No explicit ratings available - using implicit only")
            final_ratings = implicit_ratings.copy()
            final_ratings['final_rating'] = final_ratings['implicit_rating']
            final_ratings['has_explicit'] = False
            return final_ratings
        
        # Map ratings to bookings
        booking_map = booking_df[['ID', 'RefClient', 'RefChauffeur', 'RefVehicle']].copy()
        booking_map['ChauffeurVehiclePair'] = (
            booking_map['RefChauffeur'].astype(int).astype(str) + '_' +
            booking_map['RefVehicle'].astype(int).astype(str)
        )
        
        # Merge explicit ratings
        explicit_with_client = rating_df.merge(
            booking_map[['ID', 'RefClient', 'ChauffeurVehiclePair']],
            left_on='RefBooking', right_on='ID', how='inner'
        )
        
        # Aggregate by client-pair
        explicit_agg = explicit_with_client.groupby(['RefClient', 'ChauffeurVehiclePair']).agg({
            'RatingStars': 'mean'
        }).reset_index()
        explicit_agg.columns = ['RefClient', 'ChauffeurVehiclePair', 'explicit_rating']
        
        logger.info(f"   Processed {len(explicit_agg):,} explicit client-pair ratings")
        
        # Combine with implicit
        final_ratings = implicit_ratings.merge(explicit_agg, on=['RefClient', 'ChauffeurVehiclePair'], how='left')
        final_ratings['has_explicit'] = final_ratings['explicit_rating'].notna()
        
        # Weighted combination
        final_ratings['final_rating'] = np.where(
            final_ratings['has_explicit'],
            (final_ratings['explicit_rating'] * self.config.EXPLICIT_RATING_WEIGHT +
             final_ratings['implicit_rating'] * self.config.IMPLICIT_RATING_WEIGHT),
            final_ratings['implicit_rating']
        ).clip(1, 5)
        
        logger.info(f"   Final ratings: {len(final_ratings):,} (with explicit: {final_ratings['has_explicit'].sum():,})")
        
        return final_ratings
    
    def create_interaction_matrix(self, 
                                  ratings_df: pd.DataFrame,
                                  min_interactions: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create user-item interaction matrix
        
        Args:
            ratings_df: Final ratings dataframe
            min_interactions: Minimum interactions to keep user/item
            
        Returns:
            Tuple of (interaction_matrix, filtered_ratings)
        """
        logger.info(" Creating user-item interaction matrix...")
        
        min_interactions = min_interactions or self.config.MIN_INTERACTIONS
        
        # Filter by minimum interactions
        user_counts = ratings_df['RefClient'].value_counts()
        item_counts = ratings_df['ChauffeurVehiclePair'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        filtered_ratings = ratings_df[
            (ratings_df['RefClient'].isin(valid_users)) &
            (ratings_df['ChauffeurVehiclePair'].isin(valid_items))
        ].copy()
        
        logger.info(f"   Original: {len(ratings_df):,} ratings, "
                   f"{ratings_df['RefClient'].nunique():,} users, "
                   f"{ratings_df['ChauffeurVehiclePair'].nunique():,} items")
        logger.info(f"   Filtered: {len(filtered_ratings):,} ratings, "
                   f"{filtered_ratings['RefClient'].nunique():,} users, "
                   f"{filtered_ratings['ChauffeurVehiclePair'].nunique():,} items")
        
        # Create pivot table
        interaction_matrix = filtered_ratings.pivot_table(
            index='RefClient',
            columns='ChauffeurVehiclePair',
            values='final_rating',
            fill_value=0
        )
        
        # Calculate sparsity
        total_cells = interaction_matrix.shape[0] * interaction_matrix.shape[1]
        non_zero = (interaction_matrix > 0).sum().sum()
        sparsity = (1 - non_zero / total_cells) * 100
        
        logger.info(f"   Matrix shape: {interaction_matrix.shape[0]:,} × {interaction_matrix.shape[1]:,}")
        logger.info(f"   Sparsity: {sparsity:.2f}%")
        
        return interaction_matrix, filtered_ratings
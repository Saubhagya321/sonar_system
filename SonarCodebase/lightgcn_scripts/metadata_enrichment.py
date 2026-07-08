
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import dask.dataframe as dd

from base_classes import BaseDataLoader, BaseProcessor, BasePipeline
from config import METADATA_FILES, MIN_VALID_DATE, MAX_VALID_DATE




class BookingsComputer:
    """
    Safely computes dask dataframes with date validation.
    
    REPLACES: safe_compute_bookings() [lines 77-113]
    """
    
    DATE_COLUMNS = ['BookingDate', 'AppointmentDate', 'DateCompleted']
    MIN_DATE = pd.Timestamp(MIN_VALID_DATE)
    MAX_DATE = pd.Timestamp(MAX_VALID_DATE)
    
    def __init__(self):
        """Initialize bookings computer."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def safe_compute(self, bookings_enhanced: dd.DataFrame) -> pd.DataFrame:
        """
        Safely compute dask dataframe with date validation.
        
        REPLACES: safe_compute_bookings() [lines 77-113]
        
        Args:
            bookings_enhanced: Dask DataFrame with booking data
            
        Returns:
            Pandas DataFrame with validated dates
        """
        self.logger.debug("Validating dates before computing...")
 
        if hasattr(bookings_enhanced, 'compute'):
            # Filter to valid date range before computing
            for col in self.DATE_COLUMNS:
                if col in bookings_enhanced.columns:
                    bookings_enhanced = bookings_enhanced[
                        (bookings_enhanced[col].isna()) |
                        ((bookings_enhanced[col] >= self.MIN_DATE) &
                         (bookings_enhanced[col] <= self.MAX_DATE))
                    ]
            
            bookings_pd = bookings_enhanced.compute()
        else:
            bookings_pd = bookings_enhanced
        
        # Final safety check
        for col in self.DATE_COLUMNS:
            if col in bookings_pd.columns:
                mask = bookings_pd[col].notna()
                if mask.any():
                    bookings_pd.loc[mask, col] = bookings_pd.loc[mask, col].clip(
                        lower=self.MIN_DATE,
                        upper=self.MAX_DATE
                    )
        
        self.logger.debug(f"Safely computed {len(bookings_pd)} records")
        return bookings_pd


class MetadataLoader(BaseDataLoader):
    """
    Loads metadata from CSV files.
    
    REPLACES: load_metadata_tables() [lines 23-68]
    """
    
    VEHICLE_COLUMNS = [
        'ID', 'RefVehicleType', 'Year', 'Code', 'Colour', 'Make', 'Model',
        'RefChauffeurCurrent', 'RefChauffeurDefault', 'RefPreviousSuburb',
        'LeaseCost', 'LeaseDuration_AvgImputed', 'LeaseDuration_TillToday',
        'LeaseIsActive', 'InsuranceDaysTillExpiry', 'InsuranceIsActive',
        'RWCDaysTillExpiry', 'RWCIsActive', 'PublicLiabilityDaysTillExpiry',
        'PublicLiabilityIsActive'
    ]
    
    VEHICLE_TYPE_COLUMNS = [
        'ID', 'SeatCapacity', 'VehicleLevel', 'Code', 'Description',
        'DisplayName', 'TollVehicleType', 'IsLevyApplied'
    ]
    
    PERSON_CHARGE_RATE_COLUMNS = [
        'RefVehicleType', 'PersonRate1', 'PersonRate2', 'PersonRate3', 'PersonRate4'
    ]
    
    CHAUFFEUR_COLUMNS = [
        'ID', 'employee_duration_days', 'Code', 'RefSuburb', 'RefChauffeurStatus',
        'CommissionRate', 'RedZoneAuthorised', 'DriverTraining', 'BabySeatTraining',
        'days_to_license_expiry', 'license_expired', 'age', 'chauffeur_status_duration',
        'days_since_police_check', 'days_since_wwc', 'days_since_medical_cert'
    ]
    
    CLIENT_COLUMNS = [
        'ID', 'Code', 'Description', 'StartDate', 'LastBookingDate', 'IsActive',
        'RefPort', 'RefSuburb', 'PostCode', 'RefMarketSector', 'RefVehicleType',
        'VehicleTypeIds', 'RefSensitivityLevel', 'ChauffeurInstructions',
        'ClientSpecialInstruction', 'IsLevyApplied', 'RefPaymentType',
        'CommissionReserve', 'PayVariance', 'RefDiscount', 'IsControlledAccount'
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metadata loader.
        
        Args:
            config: Configuration with metadata file paths
        """
        super().__init__(config or {'metadata_files': METADATA_FILES})
        self.metadata_files = self.config.get('metadata_files', METADATA_FILES)
    
    def load(self) -> Dict[str, pd.DataFrame]:
        """
        Load all metadata tables.
        
        REPLACES: load_metadata_tables() [lines 23-68]
        
        Returns:
            Dictionary of metadata DataFrames
        """
        return self.load_all()
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all metadata tables.
        
        REPLACES: load_metadata_tables() [lines 23-68]
        
        Returns:
            Dictionary of metadata DataFrames
        """
        
        result = {
            'vehicle_df': self._load_vehicle_data(),
            'vehicle_type_df': self._load_vehicle_type_data(),
            'person_charge_rate_df': self._load_person_charge_rate_data(),
            'chauffeur_df': self._load_chauffeur_data(),
            'client_df': self._load_client_data()
        }
        
        self.logger.info("Metadata tables loaded successfully!")
        return result
    
    def validate(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate loaded metadata."""
        required_keys = ['vehicle_df', 'vehicle_type_df', 'person_charge_rate_df',
                         'chauffeur_df', 'client_df']
        return all(key in data for key in required_keys)
    
    def _load_vehicle_data(self) -> pd.DataFrame:
        """Load vehicle metadata."""
        return pd.read_csv(self.metadata_files['vehicle'])[self.VEHICLE_COLUMNS]
    
    def _load_vehicle_type_data(self) -> pd.DataFrame:
        """Load vehicle type metadata."""
        return pd.read_csv(self.metadata_files['vehicle_type'])[self.VEHICLE_TYPE_COLUMNS]
    
    def _load_person_charge_rate_data(self) -> pd.DataFrame:
        """Load person charge rate metadata."""
        return pd.read_csv(self.metadata_files['person_charge_rate'])[self.PERSON_CHARGE_RATE_COLUMNS]
    
    def _load_chauffeur_data(self) -> pd.DataFrame:
        """Load chauffeur metadata."""
        return pd.read_csv(self.metadata_files['chauffeur'])[self.CHAUFFEUR_COLUMNS]
    
    def _load_client_data(self) -> pd.DataFrame:
        """Load client metadata."""
        return pd.read_csv(self.metadata_files['client'])[self.CLIENT_COLUMNS]


class VehicleFeatureCreator(BaseProcessor):
    """
    Creates comprehensive vehicle features with performance metrics.
    
    REPLACES: create_vehicle_features() [lines 115-301]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize vehicle feature creator."""
        super().__init__(config)
        self.bookings_computer = BookingsComputer()
        self.current_year = datetime.now().year
    
    def process(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process data to create vehicle features.
        
        Args:
            data: Dictionary with vehicle_df, vehicle_type_df, 
                  person_charge_rate_df, bookings_enhanced
                  
        Returns:
            DataFrame with vehicle features
        """
        return self.create_features(
            data['vehicle_df'],
            data['vehicle_type_df'],
            data['person_charge_rate_df'],
            data['bookings_enhanced']
        )
    
    def create_features(
        self,
        vehicle_df: pd.DataFrame,
        vehicle_type_df: pd.DataFrame,
        person_charge_rate_df: pd.DataFrame,
        bookings_enhanced: dd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive vehicle features with performance metrics.
        
        REPLACES: create_vehicle_features() [lines 115-301]
        
        Args:
            vehicle_df: Vehicle metadata DataFrame
            vehicle_type_df: Vehicle type metadata DataFrame
            person_charge_rate_df: Person charge rate DataFrame
            bookings_enhanced: Enhanced bookings DataFrame
            
        Returns:
            DataFrame with vehicle features
        """
        self.logger.info("\n=== Step 2.1: Vehicle Features ===")
        
        # Clean vehicle data
        vehicle_df_clean = self._clean_vehicle_data(vehicle_df)
        
        # Merge with metadata
        vehicle_meta = self._merge_metadata(
            vehicle_df_clean, vehicle_type_df, person_charge_rate_df
        )
        
        # Create base features
        vehicle_features = self._create_base_features(vehicle_meta)
        
        # Add performance metrics
        bookings_pd = self.bookings_computer.safe_compute(bookings_enhanced)
        vehicle_features = self._add_performance_metrics(vehicle_features, bookings_pd)
        
        self._log_summary(vehicle_features)
        
        return vehicle_features
    
    def _clean_vehicle_data(self, vehicle_df: pd.DataFrame) -> pd.DataFrame:
        """Clean vehicle year data."""
        vehicle_df_clean = vehicle_df.copy()
        
        def clean_year(year):
            if pd.isna(year) or year <= 0:
                return np.nan
            elif year < 1950 or year > self.current_year + 1:
                return np.nan
            return year
        
        vehicle_df_clean['Year'] = vehicle_df_clean['Year'].apply(clean_year)
        
        # Impute missing years
        if vehicle_df_clean['Year'].isna().any():
            median_year = vehicle_df_clean['Year'].median()
            if pd.isna(median_year):
                median_year = self.current_year - 5
            
            make_model_median = vehicle_df_clean.groupby(['Make', 'Model'])['Year'].transform('median')
            vehicle_df_clean['Year'] = vehicle_df_clean['Year'].fillna(make_model_median)
            vehicle_df_clean['Year'] = vehicle_df_clean['Year'].fillna(median_year)
        
        return vehicle_df_clean
    
    def _merge_metadata(
        self,
        vehicle_df: pd.DataFrame,
        vehicle_type_df: pd.DataFrame,
        person_charge_rate_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge vehicle with type and rate metadata."""
        # Merge with vehicle type
        vehicle_type_for_merge = vehicle_type_df.rename(columns={'ID': 'VehicleTypeID'})
        
        vehicle_meta = vehicle_df.merge(
            vehicle_type_for_merge[[
                'VehicleTypeID', 'SeatCapacity', 'VehicleLevel',
                'TollVehicleType', 'IsLevyApplied', 'Description', 'DisplayName'
            ]],
            left_on='RefVehicleType',
            right_on='VehicleTypeID',
            how='left'
        )
        
        vehicle_meta = vehicle_meta.drop(columns=['VehicleTypeID'], errors='ignore')
        
        # Merge with person charge rates
        vehicle_meta = vehicle_meta.merge(
            person_charge_rate_df,
            on='RefVehicleType',
            how='left'
        )
        
        self.logger.debug(f"  Before deduplication: {len(vehicle_meta)}")
        vehicle_meta = vehicle_meta.drop_duplicates(subset=['ID'], keep='first')
        self.logger.debug(f"  After deduplication: {len(vehicle_meta)}")
        
        return vehicle_meta
    
    def _create_base_features(self, vehicle_meta: pd.DataFrame) -> pd.DataFrame:
        """Create base vehicle features."""
        vehicle_features = vehicle_meta[[
            'ID', 'Code', 'RefVehicleType', 'Colour', 'Make', 'Model', 'Year',
            'RefChauffeurCurrent', 'RefChauffeurDefault', 'LeaseCost',
            'SeatCapacity', 'VehicleLevel', 'TollVehicleType', 'IsLevyApplied',
            'PersonRate1', 'PersonRate2', 'PersonRate3', 'PersonRate4'
        ]].copy()
        
        # Vehicle age
        vehicle_features['vehicle_age'] = self.current_year - vehicle_features['Year']
        vehicle_features['vehicle_age'] = vehicle_features['vehicle_age'].clip(0, 50)
        
        # Luxury score
        vehicle_features['luxury_score'] = vehicle_features['VehicleLevel'].fillna(1)
        
        # Average person rate
        rate_cols = ['PersonRate1', 'PersonRate2', 'PersonRate3', 'PersonRate4']
        vehicle_features['avg_person_rate'] = vehicle_features[rate_cols].mean(axis=1, skipna=True)
        
        # Capacity category
        def categorize_capacity(capacity):
            if pd.isna(capacity):
                return 'Unknown'
            elif capacity <= 4:
                return 'Small'
            elif capacity <= 7:
                return 'Medium'
            return 'Large'
        
        vehicle_features['capacity_category'] = vehicle_features['SeatCapacity'].apply(categorize_capacity)
        vehicle_features['is_premium'] = (vehicle_features['VehicleLevel'] >= 3).astype(int)
        
        return vehicle_features
    
    def _add_performance_metrics(
        self,
        vehicle_features: pd.DataFrame,
        bookings_pd: pd.DataFrame
    ) -> pd.DataFrame:
        """Add performance metrics to vehicle features."""
        self.logger.debug("Calculating vehicle performance metrics...")
        
        # Calculate vehicle statistics
        vehicle_stats = bookings_pd.groupby('RefVehicle').agg({
            'ID': 'count',
            'DateCompleted': lambda x: x.notna().sum(),
            'PriceComputed': ['mean', 'sum', 'std'],
            'NoOfPax': 'mean',
            'TotalDistance': ['sum', 'mean'],
            'BookingDate': ['min', 'max']
        }).reset_index()
        
        vehicle_stats.columns = [
            'RefVehicle', 'total_bookings', 'completed_bookings',
            'avg_price', 'total_revenue', 'price_std',
            'avg_pax', 'total_distance', 'avg_distance',
            'first_booking', 'last_booking'
        ]
        
        # Completion rate
        vehicle_stats['completion_rate'] = (
            vehicle_stats['completed_bookings'] / vehicle_stats['total_bookings']
        ).fillna(0)
        
        # Days active
        valid_dates = vehicle_stats['first_booking'].notna() & vehicle_stats['last_booking'].notna()
        vehicle_stats['days_active'] = 1
        
        if valid_dates.sum() > 0:
            vehicle_stats.loc[valid_dates, 'days_active'] = (
                vehicle_stats.loc[valid_dates, 'last_booking'] -
                vehicle_stats.loc[valid_dates, 'first_booking']
            ).dt.days.clip(1, 36500)
        
        vehicle_stats['bookings_per_month'] = (
            vehicle_stats['total_bookings'] / (vehicle_stats['days_active'] / 30)
        ).clip(upper=2000)
        
        # Revenue per booking
        vehicle_stats['revenue_per_booking'] = (
            vehicle_stats['total_revenue'] / vehicle_stats['total_bookings']
        ).fillna(0)
        
        # Repeat client rate
        repeat_clients = bookings_pd.groupby('RefVehicle')['RefClient'].apply(
            lambda x: (x.value_counts() > 1).sum() / x.nunique() if x.nunique() > 0 else 0
        ).reset_index()
        repeat_clients.columns = ['RefVehicle', 'repeat_client_rate']
        
        # Days since last booking
        valid_last = vehicle_stats['last_booking'].notna()
        vehicle_stats['days_since_last_booking'] = 36500
        
        if valid_last.sum() > 0:
            max_date = vehicle_stats.loc[valid_last, 'last_booking'].max()
            vehicle_stats.loc[valid_last, 'days_since_last_booking'] = (
                max_date - vehicle_stats.loc[valid_last, 'last_booking']
            ).dt.days.clip(0, 36500)
        
        # Merge performance metrics
        vehicle_features = vehicle_features.merge(
            vehicle_stats,
            left_on='ID',
            right_on='RefVehicle',
            how='left'
        )
        
        vehicle_features = vehicle_features.merge(
            repeat_clients,
            left_on='ID',
            right_on='RefVehicle',
            how='left'
        )
        
        # Fill missing values
        perf_cols = ['total_bookings', 'completion_rate', 'repeat_client_rate',
                     'bookings_per_month', 'avg_price', 'total_revenue']
        for col in perf_cols:
            if col in vehicle_features.columns:
                vehicle_features[col] = vehicle_features[col].fillna(0)
        
        # Quality score
        vehicle_features['vehicle_quality_score'] = (
            0.3 * (vehicle_features['luxury_score'] / 5.0) +
            0.3 * vehicle_features['completion_rate'] +
            0.4 * vehicle_features['repeat_client_rate']
        ).fillna(0.5)
        
        return vehicle_features
    
    def _log_summary(self, vehicle_features: pd.DataFrame) -> None:
        """Log vehicle feature summary."""
        self.logger.debug(f"Vehicle features created for {len(vehicle_features)} vehicles")
        self.logger.debug(f"Average completion rate: {vehicle_features['completion_rate'].mean():.2%}")
        self.logger.debug(f"Average repeat client rate: {vehicle_features['repeat_client_rate'].mean():.2%}")
        self.logger.debug(f"Vehicles with >5 bookings: {(vehicle_features['total_bookings'] > 5).sum()}")

class ChauffeurFeatureCreator(BaseProcessor):
    """
    Creates comprehensive chauffeur features with performance metrics.
    
    REPLACES: create_chauffeur_features() [lines 307-478]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize chauffeur feature creator."""
        super().__init__(config)
        self.bookings_computer = BookingsComputer()
    
    def process(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Process data to create chauffeur features."""
        return self.create_features(
            data['chauffeur_df'],
            data['bookings_enhanced']
        )
    
    def create_features(
        self,
        chauffeur_df: pd.DataFrame,
        bookings_enhanced: dd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive chauffeur features with performance metrics.
        
        REPLACES: create_chauffeur_features() [lines 307-478]
        
        Args:
            chauffeur_df: Chauffeur metadata DataFrame
            bookings_enhanced: Enhanced bookings DataFrame
            
        Returns:
            DataFrame with chauffeur features
        """
        self.logger.info("\n=== Step 2.2: Chauffeur Features ===")
        
        chauffeur_features = chauffeur_df.copy()
        
        # Add scores
        chauffeur_features['experience_score'] = chauffeur_features['employee_duration_days'].apply(
            self._calculate_experience_score
        )
        chauffeur_features['safety_score'] = chauffeur_features.apply(
            self._calculate_safety_score, axis=1
        )
        chauffeur_features['training_score'] = chauffeur_features.apply(
            self._calculate_training_score, axis=1
        )
        
        # Add performance metrics
        bookings_pd = self.bookings_computer.safe_compute(bookings_enhanced)
        chauffeur_features = self._add_performance_metrics(chauffeur_features, bookings_pd)
        
        # Add age category
        chauffeur_features['age_category'] = chauffeur_features['age'].apply(
            self._categorize_age
        )
        
        self._log_summary(chauffeur_features)
        
        return chauffeur_features
    
    @staticmethod
    def _calculate_experience_score(duration_days: float) -> int:
        """Calculate experience score from duration days."""
        if pd.isna(duration_days):
            return 0
        elif duration_days < 90:
            return 1  # New
        elif duration_days < 365:
            return 2  # Junior
        elif duration_days < 1825:
            return 3  # Experienced
        return 4  # Senior
    
    @staticmethod
    def _calculate_safety_score(row: pd.Series) -> float:
        """Calculate safety score from compliance metrics."""
        score = 0
        
        if not pd.isna(row.get('license_expired')) and not row['license_expired']:
            score += 1
        elif pd.isna(row.get('license_expired')):
            score += 0.5
        
        if not pd.isna(row.get('days_since_police_check')) and row['days_since_police_check'] < 730:
            score += 1
        elif pd.isna(row.get('days_since_police_check')):
            score += 0.5
        
        if not pd.isna(row.get('days_since_wwc')) and row['days_since_wwc'] < 1825:
            score += 1
        elif pd.isna(row.get('days_since_wwc')):
            score += 0.5
        
        if not pd.isna(row.get('days_since_medical_cert')) and row['days_since_medical_cert'] < 365:
            score += 1
        elif pd.isna(row.get('days_since_medical_cert')):
            score += 0.5
        
        return score / 4
    
    @staticmethod
    def _calculate_training_score(row: pd.Series) -> float:
        """Calculate training score."""
        score = 0
        if not pd.isna(row.get('DriverTraining')) and row['DriverTraining']:
            score += 1
        if not pd.isna(row.get('BabySeatTraining')) and row['BabySeatTraining']:
            score += 1
        return score / 2
    
    @staticmethod
    def _categorize_age(age: float) -> str:
        """Categorize age."""
        if pd.isna(age):
            return 'Unknown'
        elif age < 30:
            return 'Young'
        elif age < 50:
            return 'Middle'
        return 'Experienced'
    
    def _add_performance_metrics(
        self,
        chauffeur_features: pd.DataFrame,
        bookings_pd: pd.DataFrame
    ) -> pd.DataFrame:
        """Add performance metrics to chauffeur features."""
        self.logger.debug("Calculating chauffeur performance metrics...")
        
        # Calculate chauffeur statistics
        chauffeur_stats = bookings_pd.groupby('RefChauffeur').agg({
            'ID': 'count',
            'DateCompleted': lambda x: x.notna().sum(),
            'PriceComputed': ['mean', 'sum'],
            'lead_time_days': 'mean',
            'NoOfPax': 'mean',
            'TotalDistance': 'sum',
            'BookingDate': ['min', 'max']
        }).reset_index()
        
        chauffeur_stats.columns = [
            'RefChauffeur', 'total_bookings', 'completed_bookings',
            'avg_price', 'total_revenue', 'avg_lead_time',
            'avg_pax', 'total_distance', 'first_booking', 'last_booking'
        ]
        
        # Completion rate
        chauffeur_stats['completion_rate'] = (
            chauffeur_stats['completed_bookings'] / chauffeur_stats['total_bookings']
        ).fillna(0)
        
        # Days active
        valid_dates = chauffeur_stats['first_booking'].notna() & chauffeur_stats['last_booking'].notna()
        chauffeur_stats['days_active'] = 1
        
        if valid_dates.sum() > 0:
            chauffeur_stats.loc[valid_dates, 'days_active'] = (
                chauffeur_stats.loc[valid_dates, 'last_booking'] -
                chauffeur_stats.loc[valid_dates, 'first_booking']
            ).dt.days.clip(1, 36500)
        
        chauffeur_stats['bookings_per_month'] = (
            chauffeur_stats['total_bookings'] / (chauffeur_stats['days_active'] / 30)
        ).clip(upper=1000)
        
        # Repeat client rate
        repeat_clients = bookings_pd.groupby('RefChauffeur')['RefClient'].apply(
            lambda x: (x.value_counts() > 1).sum() / x.nunique() if x.nunique() > 0 else 0
        ).reset_index()
        repeat_clients.columns = ['RefChauffeur', 'repeat_client_rate']
        
        # Days since last booking
        valid_last = chauffeur_stats['last_booking'].notna()
        chauffeur_stats['days_since_last_booking'] = 36500
        
        if valid_last.sum() > 0:
            max_date = chauffeur_stats.loc[valid_last, 'last_booking'].max()
            chauffeur_stats.loc[valid_last, 'days_since_last_booking'] = (
                max_date - chauffeur_stats.loc[valid_last, 'last_booking']
            ).dt.days.clip(0, 36500)
        
        chauffeur_stats['avg_advance_booking_days'] = chauffeur_stats['avg_lead_time']
        
        # Merge performance metrics
        chauffeur_features = chauffeur_features.merge(
            chauffeur_stats,
            left_on='ID',
            right_on='RefChauffeur',
            how='left'
        )
        
        chauffeur_features = chauffeur_features.merge(
            repeat_clients,
            left_on='ID',
            right_on='RefChauffeur',
            how='left'
        )
        
        # Fill missing values
        perf_cols = ['total_bookings', 'completion_rate', 'repeat_client_rate',
                     'bookings_per_month', 'avg_price', 'total_revenue']
        for col in perf_cols:
            if col in chauffeur_features.columns:
                chauffeur_features[col] = chauffeur_features[col].fillna(0)
        
        # Quality score
        chauffeur_features['chauffeur_quality_score'] = (
            0.2 * (chauffeur_features['experience_score'] / 4.0) +
            0.2 * chauffeur_features['safety_score'] +
            0.2 * chauffeur_features['training_score'] +
            0.2 * chauffeur_features['completion_rate'] +
            0.2 * chauffeur_features['repeat_client_rate']
        ).fillna(0.5)
        
        return chauffeur_features
    
    def _log_summary(self, chauffeur_features: pd.DataFrame) -> None:
        """Log chauffeur feature summary."""
        self.logger.debug(f"Chauffeur features created for {len(chauffeur_features)} chauffeurs")
        self.logger.debug(f"Average completion rate: {chauffeur_features['completion_rate'].mean():.2%}")
        self.logger.debug(f"Average repeat client rate: {chauffeur_features['repeat_client_rate'].mean():.2%}")
        self.logger.debug(f"Chauffeurs with >5 bookings: {(chauffeur_features['total_bookings'] > 5).sum()}")

class ClientFeatureCreator(BaseProcessor):
    """
    Creates comprehensive client features with booking patterns.
    
    REPLACES: create_client_features() [lines 484-637]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize client feature creator."""
        super().__init__(config)
        self.bookings_computer = BookingsComputer()
    
    def process(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Process data to create client features."""
        return self.create_features(
            data['client_df'],
            data['bookings_enhanced']
        )
    
    def create_features(
        self,
        client_df: pd.DataFrame,
        bookings_enhanced: dd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive client features with booking patterns.
        
        REPLACES: create_client_features() [lines 484-637]
        
        Args:
            client_df: Client metadata DataFrame
            bookings_enhanced: Enhanced bookings DataFrame
            
        Returns:
            DataFrame with client features
        """
        self.logger.info("\n=== Step 2.3: Client Features ===")
        
        client_features = client_df.copy()
        
        # Convert and validate dates
        client_features = self._process_dates(client_features)
        
        # Add activity scores and tiers
        client_features = self._add_activity_features(client_features)
        
        # Add booking pattern features
        bookings_pd = self.bookings_computer.safe_compute(bookings_enhanced)
        client_features = self._add_booking_patterns(client_features, bookings_pd)
        
        self._log_summary(client_features)
        
        return client_features
    
    def _process_dates(self, client_features: pd.DataFrame) -> pd.DataFrame:
        """Process and validate date columns."""
        date_cols = ['StartDate', 'LastBookingDate']
        current_date = datetime.now()
        
        for col in date_cols:
            if col in client_features.columns:
                client_features[col] = pd.to_datetime(client_features[col], errors='coerce')
                client_features[col] = client_features[col].clip(
                    lower=pd.Timestamp(MIN_VALID_DATE),
                    upper=pd.Timestamp(MAX_VALID_DATE)
                )
        
        # Calculate tenure
        if 'StartDate' in client_features.columns:
            valid_start = client_features['StartDate'].notna()
            client_features['tenure_days'] = 0.0
            
            if valid_start.sum() > 0:
                client_features.loc[valid_start, 'tenure_days'] = (
                    current_date - client_features.loc[valid_start, 'StartDate']
                ).dt.days.clip(0, 36500)
            
            self.logger.debug(f"  Tenure calculated for {valid_start.sum()}/{len(client_features)} clients")
        
        # Days since last booking
        if 'LastBookingDate' in client_features.columns:
            valid_last = client_features['LastBookingDate'].notna()
            client_features['days_since_last_booking'] = 999.0
            
            if valid_last.sum() > 0:
                client_features.loc[valid_last, 'days_since_last_booking'] = (
                    current_date - client_features.loc[valid_last, 'LastBookingDate']
                ).dt.days.clip(0, 36500)
            
            self.logger.debug(f"  Last booking date for {valid_last.sum()}/{len(client_features)} clients")
        
        return client_features
    
    def _add_activity_features(self, client_features: pd.DataFrame) -> pd.DataFrame:
        """Add activity scores and tier features."""
        # Activity score
        if 'days_since_last_booking' in client_features.columns:
            def calculate_activity_score(days):
                if pd.isna(days) or days > 365:
                    return 0
                elif days <= 30:
                    return 3
                elif days <= 90:
                    return 2
                return 1
            
            client_features['activity_score'] = client_features['days_since_last_booking'].apply(
                calculate_activity_score
            )
        
        # Client tier
        if 'RefSensitivityLevel' in client_features.columns:
            def categorize_sensitivity(level):
                if pd.isna(level):
                    return 'Standard'
                elif level >= 4:
                    return 'VIP'
                elif level >= 3:
                    return 'Premium'
                elif level >= 2:
                    return 'Priority'
                return 'Standard'
            
            client_features['client_tier'] = client_features['RefSensitivityLevel'].apply(
                categorize_sensitivity
            )
        
        # Special indicators
        if 'IsControlledAccount' in client_features.columns:
            client_features['is_controlled'] = client_features['IsControlledAccount'].fillna(False).astype(int)
        
        if 'ChauffeurInstructions' in client_features.columns:
            client_features['has_chauffeur_instructions'] = (
                ~client_features['ChauffeurInstructions'].isna()
            ).astype(int)
        
        if 'ClientSpecialInstruction' in client_features.columns:
            client_features['has_special_instructions'] = (
                ~client_features['ClientSpecialInstruction'].isna()
            ).astype(int)
        
        return client_features
    
    def _add_booking_patterns(
        self,
        client_features: pd.DataFrame,
        bookings_pd: pd.DataFrame
    ) -> pd.DataFrame:
        """Add booking pattern features."""
        self.logger.debug("Calculating client booking patterns...")
        
        agg_dict = {
            'ID': 'count',
            'PriceComputed': ['mean', 'sum', 'std'],
            'NoOfPax': 'mean',
            'TotalDistance': 'mean'
        }
        
        if 'is_weekend' in bookings_pd.columns:
            agg_dict['is_weekend'] = 'mean'
        if 'is_business_hours' in bookings_pd.columns:
            agg_dict['is_business_hours'] = 'mean'
        if 'lead_time_days' in bookings_pd.columns:
            agg_dict['lead_time_days'] = 'mean'
        
        client_stats = bookings_pd.groupby('RefClient').agg(agg_dict).reset_index()
        
        # Flatten columns
        new_cols = ['RefClient', 'total_bookings', 'avg_booking_price', 'lifetime_value',
                    'price_std', 'avg_pax', 'avg_distance']
        
        if 'is_weekend' in bookings_pd.columns:
            new_cols.append('weekend_preference')
        if 'is_business_hours' in bookings_pd.columns:
            new_cols.append('business_hours_preference')
        if 'lead_time_days' in bookings_pd.columns:
            new_cols.append('avg_lead_time')
        
        client_stats.columns = new_cols
        
        # Booking frequency
        client_stats['bookings_per_month'] = client_stats['total_bookings'] / 12
        
        # Client value score
        client_stats['client_value_score'] = (
            0.5 * np.log1p(client_stats['lifetime_value']) / np.log1p(client_stats['lifetime_value'].max()) +
            0.5 * np.log1p(client_stats['total_bookings']) / np.log1p(client_stats['total_bookings'].max())
        ).fillna(0)
        
        # Merge
        client_features = client_features.merge(
            client_stats,
            left_on='ID',
            right_on='RefClient',
            how='left'
        )
        
        # Fill missing values
        stat_cols = ['total_bookings', 'avg_booking_price', 'lifetime_value',
                     'weekend_preference', 'business_hours_preference']
        for col in stat_cols:
            if col in client_features.columns:
                client_features[col] = client_features[col].fillna(0)
        
        return client_features
    
    def _log_summary(self, client_features: pd.DataFrame) -> None:
        """Log client feature summary."""
        self.logger.debug(f"Client features created for {len(client_features)} clients")
        if 'client_tier' in client_features.columns:
            self.logger.debug(f"Client tier distribution: {client_features['client_tier'].value_counts().to_dict()}")
        if 'total_bookings' in client_features.columns:
            self.logger.debug(f"Average bookings per client: {client_features['total_bookings'].mean():.1f}")


class CooccurrenceFeatureCreator(BaseProcessor):
    """
    Creates vehicle-chauffeur co-occurrence features.
    
    REPLACES: create_cooccurrence_features() [lines 643-725]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize co-occurrence feature creator."""
        super().__init__(config)
        self.bookings_computer = BookingsComputer()
    
    def process(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Process data to create co-occurrence features."""
        return self.create_features(data['bookings_enhanced'])
    
    def create_features(self, bookings_enhanced: dd.DataFrame) -> pd.DataFrame:
        """
        Create features for vehicle-chauffeur co-occurrence edges.
        
        REPLACES: create_cooccurrence_features() [lines 643-725]
        
        Args:
            bookings_enhanced: Enhanced bookings DataFrame
            
        Returns:
            DataFrame with co-occurrence features
        """
        self.logger.debug("\n=== Step 2.4: Vehicle-Chauffeur Co-occurrence Features ===")
        
        bookings_pd = self.bookings_computer.safe_compute(bookings_enhanced)
        
        # Group by vehicle-chauffeur pairs
        vc_pairs = bookings_pd.groupby(['RefVehicle', 'RefChauffeur']).agg({
            'ID': 'count',
            'DateCompleted': lambda x: x.notna().sum(),
            'PriceComputed': ['mean', 'sum'],
            'TotalDistance': 'sum',
            'NoOfPax': 'mean',
            'BookingDate': ['min', 'max']
        }).reset_index()
        
        vc_pairs.columns = [
            'RefVehicle', 'RefChauffeur',
            'times_together', 'successful_together',
            'avg_price_together', 'revenue_together',
            'distance_together', 'avg_pax_together',
            'first_trip', 'last_trip'
        ]
        
        # Compatibility score
        vc_pairs['compatibility_score'] = (
            vc_pairs['successful_together'] / vc_pairs['times_together']
        ).fillna(0)
        
        # Days working together
        valid_trips = vc_pairs['first_trip'].notna() & vc_pairs['last_trip'].notna()
        vc_pairs['days_working_together'] = 0
        
        if valid_trips.sum() > 0:
            vc_pairs.loc[valid_trips, 'days_working_together'] = (
                vc_pairs.loc[valid_trips, 'last_trip'] -
                vc_pairs.loc[valid_trips, 'first_trip']
            ).dt.days.clip(0, 36500)
            
            self.logger.debug(f"  Valid trip dates: {valid_trips.sum()}/{len(vc_pairs)}")
        
        # Trips per month together
        vc_pairs['trips_per_month_together'] = (
            vc_pairs['times_together'] / ((vc_pairs['days_working_together'] + 1) / 30)
        ).clip(upper=100)
        
        # Association strength
        vehicle_totals = bookings_pd.groupby('RefVehicle')['ID'].count()
        chauffeur_totals = bookings_pd.groupby('RefChauffeur')['ID'].count()
        
        vc_pairs = vc_pairs.merge(
            vehicle_totals.reset_index().rename(columns={'ID': 'vehicle_total'}),
            on='RefVehicle', how='left'
        )
        vc_pairs = vc_pairs.merge(
            chauffeur_totals.reset_index().rename(columns={'ID': 'chauffeur_total'}),
            on='RefChauffeur', how='left'
        )
        
        vc_pairs['association_strength'] = (
            vc_pairs['times_together'] /
            (vc_pairs['vehicle_total'] + vc_pairs['chauffeur_total'] - vc_pairs['times_together'])
        ).fillna(0)
        
        # Co-occurrence quality
        vc_pairs['cooccurrence_quality'] = (
            0.4 * vc_pairs['compatibility_score'] +
            0.3 * (np.log1p(vc_pairs['times_together']) / np.log1p(vc_pairs['times_together'].max())) +
            0.3 * vc_pairs['association_strength']
        ).fillna(0)
        
        self._log_summary(vc_pairs)
        
        return vc_pairs
    
    def _log_summary(self, vc_pairs: pd.DataFrame) -> None:
        """Log co-occurrence feature summary."""
        self.logger.debug(f"Co-occurrence features created for {len(vc_pairs)} vehicle-chauffeur pairs")
        self.logger.debug(f"Average compatibility score: {vc_pairs['compatibility_score'].mean():.2%}")
        self.logger.debug(f"Average times worked together: {vc_pairs['times_together'].mean():.1f}")
        self.logger.debug(f"Pairs with >5 trips together: {(vc_pairs['times_together'] > 5).sum()}")


class MetadataEnrichmentPipeline(BasePipeline):
    """
    Orchestrates the metadata enrichment process.
    
    REPLACES: execute_step2_revised() [lines 731-786]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metadata enrichment pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config or {})
        
        self.metadata_loader = MetadataLoader(config)
        self.vehicle_creator = VehicleFeatureCreator(config)
        self.chauffeur_creator = ChauffeurFeatureCreator(config)
        self.client_creator = ClientFeatureCreator(config)
        self.cooccurrence_creator = CooccurrenceFeatureCreator(config)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute metadata enrichment pipeline.
        
        REPLACES: execute_step2_revised() [lines 731-786]
        
        Args:
            input_data: Step 1 results containing bookings_enhanced
            
        Returns:
            Dictionary containing step 2 outputs
        """
        self._log_start("Step 2: Revised Metadata Enrichment (3-Way Graph)")
        
        try:
            # Load metadata
            metadata = self.metadata_loader.load_all()
            
            bookings_enhanced = input_data['bookings_enhanced']
            
            # Create features
            vehicle_features = self.vehicle_creator.create_features(
                metadata['vehicle_df'],
                metadata['vehicle_type_df'],
                metadata['person_charge_rate_df'],
                bookings_enhanced
            )
            
            chauffeur_features = self.chauffeur_creator.create_features(
                metadata['chauffeur_df'],
                bookings_enhanced
            )
            
            client_features = self.client_creator.create_features(
                metadata['client_df'],
                bookings_enhanced
            )
            
            vc_cooccurrence = self.cooccurrence_creator.create_features(bookings_enhanced)
            
            # Combine outputs
            step2_outputs = {
                **input_data,
                'vehicle_features': vehicle_features,
                'chauffeur_features': chauffeur_features,
                'client_features': client_features,
                'vc_cooccurrence': vc_cooccurrence
            }
            
            self._log_summary(step2_outputs)
            self._log_complete("Step 2")
            
            return step2_outputs
            
        except Exception as e:
            self.logger.error(f"Error in Step 2: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_summary(self, outputs: Dict[str, Any]) -> None:
        """Log step 2 summary."""
        self.logger.debug("\n" + "=" * 60)
        self.logger.info("Step 2 completed successfully!")
        self.logger.debug("\nFeature Summary:")
        self.logger.debug(f"- Vehicles: {len(outputs['vehicle_features'])} with performance metrics")
        self.logger.debug(f"- Chauffeurs: {len(outputs['chauffeur_features'])} with performance metrics")
        self.logger.debug(f"- Clients: {len(outputs['client_features'])} with booking patterns")
        self.logger.debug(f"- V-C Co-occurrences: {len(outputs['vc_cooccurrence'])} pairs with compatibility scores")
        self.logger.debug("\nKey Metrics (Better than ratings!):")
        self.logger.debug("  [OK] Completion rates (reliability)")
        self.logger.debug("  [OK] Repeat client rates (satisfaction proxy)")
        self.logger.debug("  [OK] Revenue and utilization (business value)")
        self.logger.debug("  [OK] Co-occurrence compatibility (teamwork quality)")
        self.logger.info("\nReady for Step 3: Graph Construction")


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def load_metadata_tables() -> Dict[str, pd.DataFrame]:
    """
    Backward compatible wrapper for MetadataLoader.load_all().
    
    WRAPS: MetadataLoader.load_all()
    """
    loader = MetadataLoader()
    return loader.load_all()


def safe_compute_bookings(bookings_enhanced) -> pd.DataFrame:
    """
    Backward compatible wrapper for BookingsComputer.safe_compute().
    
    WRAPS: BookingsComputer.safe_compute()
    """
    computer = BookingsComputer()
    return computer.safe_compute(bookings_enhanced)


def create_vehicle_features(vehicle_df, vehicle_type_df, person_charge_rate_df, 
                            bookings_enhanced) -> pd.DataFrame:
    """
    Backward compatible wrapper for VehicleFeatureCreator.create_features().
    
    WRAPS: VehicleFeatureCreator.create_features()
    """
    creator = VehicleFeatureCreator()
    return creator.create_features(
        vehicle_df, vehicle_type_df, person_charge_rate_df, bookings_enhanced
    )


def create_chauffeur_features(chauffeur_df, bookings_enhanced) -> pd.DataFrame:
    """
    Backward compatible wrapper for ChauffeurFeatureCreator.create_features().
    
    WRAPS: ChauffeurFeatureCreator.create_features()
    """
    creator = ChauffeurFeatureCreator()
    return creator.create_features(chauffeur_df, bookings_enhanced)


def create_client_features(client_df, bookings_enhanced) -> pd.DataFrame:
    """
    Backward compatible wrapper for ClientFeatureCreator.create_features().
    
    WRAPS: ClientFeatureCreator.create_features()
    """
    creator = ClientFeatureCreator()
    return creator.create_features(client_df, bookings_enhanced)


def create_cooccurrence_features(bookings_enhanced) -> pd.DataFrame:
    """
    Backward compatible wrapper for CooccurrenceFeatureCreator.create_features().
    
    WRAPS: CooccurrenceFeatureCreator.create_features()
    """
    creator = CooccurrenceFeatureCreator()
    return creator.create_features(bookings_enhanced)


def execute_step2_revised(step1_results, vehicle_df, vehicle_type_df, person_charge_rate_df,
                          chauffeur_df, client_df, bookings_enhanced) -> Dict[str, Any]:
    """
    Backward compatible wrapper for MetadataEnrichmentPipeline.execute().
    
    WRAPS: MetadataEnrichmentPipeline.execute()
    
    Note: This wrapper handles the old function signature where metadata was 
    passed as separate arguments. The new pipeline loads metadata internally.
    """
    # Create step1 results with bookings_enhanced
    input_data = {**step1_results, 'bookings_enhanced': bookings_enhanced}
    
    pipeline = MetadataEnrichmentPipeline()
    return pipeline.execute(input_data)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("Metadata Enrichment Module (Refactored) - Ready")
    logger.debug("\nClasses available:")
    logger.debug("  - MetadataLoader")
    logger.debug("  - VehicleFeatureCreator")
    logger.debug("  - ChauffeurFeatureCreator")
    logger.debug("  - ClientFeatureCreator")
    logger.debug("  - CooccurrenceFeatureCreator")
    logger.info("  - MetadataEnrichmentPipeline")
    logger.debug("\nBackward compatible functions:")
    logger.debug("  - load_metadata_tables()")
    logger.debug("  - execute_step2_revised()")
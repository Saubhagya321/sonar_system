

import gc
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import dask.dataframe as dd

from base_classes import BaseDataLoader, BaseProcessor, BasePipeline, DatabaseConnector
from config import DB_CONNECTION_STRING, DB_NAME, BOOKING_DATA_PATH, MIN_VALID_DATE, MAX_VALID_DATE



class DateCalculator:
    """
    Static utility class for date calculations.
    
    REPLACES:
    - safe_date_diff_days() [lines 19-45]
    - safe_timedelta_calculation() [lines 154-180]
    """
    
    @staticmethod
    def safe_date_diff_days(
        date1: pd.Series, 
        date2: pd.Series, 
        default: float = np.nan, 
        clip_range: Tuple[int, int] = (0, 36500)
    ) -> pd.Series:
        """
        Safely calculate difference between dates in days.
        Handles NaT values properly.
        
        REPLACES: safe_date_diff_days() [lines 19-45]
        
        Args:
            date1: First date series
            date2: Second date series
            default: Default value for invalid dates
            clip_range: Tuple of (min, max) days to clip result
            
        Returns:
            Series of day differences
        """
        result = pd.Series(
            default, 
            index=date2.index if hasattr(date2, 'index') else range(len(date2))
        )
        
        if isinstance(date1, pd.Series):
            valid_mask = date1.notna() & date2.notna()
        else:
            valid_mask = date2.notna()
        
        if valid_mask.sum() > 0:
            if isinstance(date1, pd.Series):
                diff_days = (date1 - date2[valid_mask]).dt.days
            else:
                diff_days = (date1 - date2[valid_mask]).dt.days
            
            if clip_range:
                diff_days = diff_days.clip(clip_range[0], clip_range[1])
            
            result[valid_mask] = diff_days
        
        return result
    
    @staticmethod
    def safe_timedelta_calculation(
        date1: pd.Series, 
        date2: pd.Series, 
        unit: str = 'days', 
        max_value: int = 36500
    ) -> Optional[pd.Series]:
        """
        Safely calculate time difference with overflow protection.
        
        REPLACES: safe_timedelta_calculation() [lines 154-180]
        
        Args:
            date1: First date series
            date2: Second date series
            unit: Unit of result ('days', 'hours', 'seconds')
            max_value: Maximum allowed value
            
        Returns:
            Series of time differences or None on error
        """
        try:
            delta = date1 - date2
            
            if unit == 'days':
                result = delta.dt.days
            elif unit == 'hours':
                result = delta.dt.total_seconds() / 3600
            elif unit == 'seconds':
                result = delta.dt.total_seconds()
            else:
                result = delta.dt.total_seconds() / (24 * 3600)
            
            result = result.clip(lower=-max_value, upper=max_value)
            return result
            
        except Exception as e:
            logging.warning(f"Warning in timedelta calculation: {e}")
            return None


class BookingDataLoader(BaseDataLoader):
    """
    Loads booking data from parquet files.
    
    REPLACES: load_booking_data() [lines 53-66]
    """
    
    REQUIRED_COLUMNS = [
        'ID', 'RefClient', 'RefChauffeur', 'RefVehicle',
        'BookingDate', 'AppointmentDate', 'PriceComputed', 'QuoteToClient',
        'NoOfPax', 'NoOfBags', 'TotalDistance', 'DurationToStartPoint',
        'DateCompleted', 'BookingStatusDate', 'CreatedOn',
        'RefVehicleTypeRequest', 'LevyAmount', 'AirportPickupFee',
        'BaseContractorCost', 'ContractorPayAmount', 'RefBookingSource'
    ]
    
    def __init__(self, config: Dict[str, Any], filepath: Optional[str] = None):
        """
        Initialize booking data loader.
        
        Args:
            config: Configuration dictionary
            filepath: Optional path to parquet file (overrides config)
        """
        super().__init__(config)
        self.filepath = filepath or str(config.get('booking_data_path', BOOKING_DATA_PATH))
    
    def load(self) -> dd.DataFrame:
        """
        Load booking data from parquet file.
        
        REPLACES: load_booking_data() [lines 53-66]
        
        Returns:
            Dask DataFrame with booking data
        """
        self.logger.info(f"Loading booking data from {self.filepath}")
        
        booking_df = dd.read_parquet(self.filepath, engine="pyarrow")
        booking_df = booking_df[self.REQUIRED_COLUMNS]
        
        self.logger.info(f"Loaded booking data with {len(self.REQUIRED_COLUMNS)} columns")
        return booking_df
    
    def validate(self, data: dd.DataFrame) -> bool:
        """
        Validate loaded booking data.
        
        Args:
            data: Booking DataFrame to validate
            
        Returns:
            True if valid
        """
        missing_cols = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing_cols:
            self.logger.error(f"Missing columns: {missing_cols}")
            return False
        return True


class DateFeatureEngineer(BaseProcessor):
    """
    Engineers date-based features from booking data.
    
    REPLACES: 
    - safe_date_conversion() [lines 104-150]
    - engineer_date_features() [lines 183-361]
    """
    
    MIN_REASONABLE_DATE = MIN_VALID_DATE
    MAX_REASONABLE_DATE = MAX_VALID_DATE
    
    DATE_COLUMNS = [
        'BookingDate', 'AppointmentDate', 'DateCompleted',
        'BookingStatusDate', 'CreatedOn'
    ]
    
    SEASON_MAPPING = {
        12: 'Summer', 1: 'Summer', 2: 'Summer',
        3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
        6: 'Winter', 7: 'Winter', 8: 'Winter',
        9: 'Spring', 10: 'Spring', 11: 'Spring'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize date feature engineer."""
        super().__init__(config)
    
    def _safe_date_conversion(
        self, 
        series: pd.Series, 
        column_name: str
    ) -> pd.Series:
        """
        Safely convert a series to datetime with bounds checking.
        
        REPLACES: safe_date_conversion() [lines 104-150]
        
        Args:
            series: Series to convert
            column_name: Name of column (for logging)
            
        Returns:
            Converted datetime series
        """
        self.logger.info(f"Processing {column_name}...")
        
        try:
            series_pd = series.compute() if hasattr(series, 'compute') else series
            converted = pd.to_datetime(series_pd, errors='coerce')
            
            initial_nat_count = converted.isna().sum()
            
            min_date = pd.Timestamp(self.MIN_REASONABLE_DATE)
            max_date = pd.Timestamp(self.MAX_REASONABLE_DATE)
            
            valid_mask = (
                converted.notna() &
                (converted >= min_date) &
                (converted <= max_date)
            )
            
            converted = converted.where(valid_mask)
            
            out_of_range_count = (~valid_mask & converted.notna()).sum()
            final_nat_count = converted.isna().sum()
            
            self.logger.info(
                f"  {column_name}: {initial_nat_count} initially invalid, "
                f"{out_of_range_count} out-of-range, {final_nat_count} total NaT"
            )
            
            return converted
            
        except Exception as e:
            self.logger.error(f"Error processing {column_name}: {e}")
            return series
    
    def process(self, data: dd.DataFrame) -> dd.DataFrame:
        """
        Process booking data to engineer date features.
        
        REPLACES: engineer_date_features() [lines 183-361]
        
        Args:
            data: Booking DataFrame
            
        Returns:
            DataFrame with engineered date features
        """
        return self.engineer_features(data)
    
    def engineer_features(self, bookings_df: dd.DataFrame) -> pd.DataFrame:
        """
        Engineer temporal features from booking dates.
        
        REPLACES: engineer_date_features() [lines 183-361]
        
        Args:
            bookings_df: Booking DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("=== Date Feature Engineering (Robust Version) ===")
        self.logger.debug("Converting date columns safely...")
        
        bookings_pd = bookings_df.compute() if hasattr(bookings_df, 'compute') else bookings_df.copy()
        
        # Convert date columns
        for col in self.DATE_COLUMNS:
            if col in bookings_pd.columns:
                bookings_pd[col] = self._safe_date_conversion(bookings_pd[col], col)
        
        # Validate critical date column
        self._validate_booking_dates(bookings_pd)
        
        # Filter records with invalid BookingDate
        bookings_pd = self._filter_invalid_dates(bookings_pd)
        
        # Extract booking date features
        bookings_pd = self._extract_booking_features(bookings_pd)
        
        # Extract appointment features
        bookings_pd = self._extract_appointment_features(bookings_pd)
        
        # Calculate completion delay
        bookings_pd = self._calculate_completion_delay(bookings_pd)
        
        # Calculate recency features
        bookings_pd = self._calculate_recency_features(bookings_pd)
        
        # Add season
        bookings_pd['season'] = bookings_pd['booking_month'].map(self.SEASON_MAPPING)
        
        # Log statistics
        self._log_statistics(bookings_pd)
        
        self.logger.info(f"\nDate features engineered successfully!")
        self.logger.debug(f"Final record count: {len(bookings_pd)}")

        # Return plain pandas instead of re-wrapping into a fake single-partition
        # dask frame: every downstream consumer (metadata_enrichment.safe_compute,
        # data_filtering.create_all) already handles plain pandas input via
        # hasattr(x, 'compute') checks, so the round-trip bought nothing but an
        # extra full-frame copy (and was a repeated source of MemoryErrors).
        gc.collect()
        return bookings_pd
    
    def _validate_booking_dates(self, df: pd.DataFrame) -> None:
        """Validate booking dates."""
        self.logger.debug("\nValidating critical date column...")
        valid_count = df['BookingDate'].notna().sum()
        total = len(df)
        self.logger.debug(f"Valid BookingDate records: {valid_count}/{total} ({valid_count/total*100:.2f}%)")
        
        if valid_count == 0:
            raise ValueError("No valid booking dates found!")
    
    def _filter_invalid_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter records with invalid BookingDate."""
        self.logger.debug("\nFiltering records with invalid BookingDate...")
        initial_count = len(df)
        df = df[df['BookingDate'].notna()]
        filtered_count = len(df)
        self.logger.debug(f"Kept {filtered_count}/{initial_count} records ({filtered_count/initial_count*100:.2f}%)")
        return df
    
    def _extract_booking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from BookingDate."""
        self.logger.debug("\nExtracting BookingDate features...")
        df['booking_hour'] = df['BookingDate'].dt.hour
        df['booking_dayofweek'] = df['BookingDate'].dt.dayofweek
        df['booking_month'] = df['BookingDate'].dt.month
        df['booking_year'] = df['BookingDate'].dt.year
        df['is_weekend'] = df['booking_dayofweek'].isin([5, 6]).astype(int)
        return df
    
    def _extract_appointment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from AppointmentDate."""
        if 'AppointmentDate' not in df.columns:
            return df
        
        valid_count = df['AppointmentDate'].notna().sum()
        self.logger.debug(f"\nValid AppointmentDate records: {valid_count}/{len(df)}")
        
        if valid_count == 0:
            return df
        
        df['appointment_hour'] = df['AppointmentDate'].dt.hour
        df['appointment_dayofweek'] = df['AppointmentDate'].dt.dayofweek
        df['appointment_month'] = df['AppointmentDate'].dt.month
        df['appointment_is_weekend'] = df['appointment_dayofweek'].isin([5, 6]).astype(int)
        
        # Calculate lead time
        self.logger.debug("Calculating lead time with overflow protection...")
        valid_pair_mask = df['AppointmentDate'].notna() & df['BookingDate'].notna()
        
        df['lead_time_days'] = np.nan
        if valid_pair_mask.sum() > 0:
            lead_time_seconds = (
                df.loc[valid_pair_mask, 'AppointmentDate'] -
                df.loc[valid_pair_mask, 'BookingDate']
            ).dt.total_seconds()
            
            lead_time_days = (lead_time_seconds / (24 * 3600)).clip(-365, 365)
            df.loc[valid_pair_mask, 'lead_time_days'] = lead_time_days
            self.logger.debug(f"  Valid lead time calculations: {valid_pair_mask.sum()}/{len(df)}")
        
        # Business hours flag
        df['is_business_hours'] = (
            (df['appointment_hour'] >= 7) &
            (df['appointment_hour'] <= 18) &
            (df['appointment_dayofweek'] < 5)
        ).astype(int)
        
        return df
    
    def _calculate_completion_delay(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate completion delay."""
        if 'DateCompleted' not in df.columns or 'AppointmentDate' not in df.columns:
            return df
        
        valid_count = df['DateCompleted'].notna().sum()
        self.logger.info(f"\nValid DateCompleted records: {valid_count}/{len(df)}")
        
        if valid_count == 0:
            return df
        
        self.logger.debug("Calculating completion delay with overflow protection...")
        valid_mask = df['DateCompleted'].notna() & df['AppointmentDate'].notna()
        
        df['completion_delay_days'] = np.nan
        if valid_mask.sum() > 0:
            delay_seconds = (
                df.loc[valid_mask, 'DateCompleted'] -
                df.loc[valid_mask, 'AppointmentDate']
            ).dt.total_seconds()
            
            delay_days = (delay_seconds / (24 * 3600)).clip(-30, 30)
            df.loc[valid_mask, 'completion_delay_days'] = delay_days
            self.logger.debug(f"  Valid completion delay calculations: {valid_mask.sum()}/{len(df)}")
        
        return df
    
    def _calculate_recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate recency features."""
        try:
            self.logger.debug("\nCalculating recency features (SAFE VERSION)...")
            valid_dates = df['BookingDate'].notna()
            
            if valid_dates.sum() > 0:
                max_date = df.loc[valid_dates, 'BookingDate'].max()
                self.logger.debug(f"Reference max date for recency: {max_date}")
                
                df['days_since_booking'] = np.nan
                df.loc[valid_dates, 'days_since_booking'] = (
                    max_date - df.loc[valid_dates, 'BookingDate']
                ).dt.days.clip(0, 36500)
                
                self.logger.debug(f"  Valid recency values: {valid_dates.sum()}/{len(df)}")
            else:
                self.logger.warning("Warning: No valid dates for recency calculation")
                df['days_since_booking'] = np.nan
                
        except Exception as e:
            self.logger.error(f"Warning: Could not calculate recency: {e}")
            df['days_since_booking'] = np.nan
        
        return df
    
    def _log_statistics(self, df: pd.DataFrame) -> None:
        """Log date statistics."""
        try:
            if 'lead_time_days' in df.columns:
                valid_lead = df['lead_time_days'].dropna()
                if len(valid_lead) > 0:
                    self.logger.warning(f"\nLead time stats: Mean={valid_lead.mean():.2f} days, "
                          f"Median={valid_lead.median():.2f} days")
            
            if 'is_weekend' in df.columns:
                weekend_sum = df['is_weekend'].sum()
                total_valid = df['is_weekend'].count()
                if total_valid > 0:
                    logger.debug(f"Weekend bookings: {weekend_sum}/{total_valid}")
        except Exception as e:
            self.logger.warning(f"Warning: Could not compute date statistics: {e}")


class PortMappingLoader:
    """
    Loads port mappings from database.
    
    REPLACES: load_port_mappings() [lines 550-651]
    """
    
    def __init__(self, db_connector: DatabaseConnector):
        """
        Initialize port mapping loader.
        
        Args:
            db_connector: Database connector instance
        """
        self.db_connector = db_connector
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self) -> Dict[str, Dict]:
        """
        Load port mappings for vehicles and chauffeurs from database.
        
        REPLACES: load_port_mappings() [lines 550-651]
        
        Returns:
            Dictionary containing:
            - port_vehicles: {port_id: [vehicle_ids]}
            - port_chauffeurs: {port_id: [chauffeur_ids]}
            - vehicle_ports: {vehicle_id: port_id}
            - chauffeur_ports: {chauffeur_id: port_id}
        """
        port_vehicles = {}
        port_chauffeurs = {}
        vehicle_ports = {}
        chauffeur_ports = {}
        
        try:
            # Load port-vehicle mappings
            port_vehicles, vehicle_ports = self._load_vehicle_mappings()
            
            # Load port-chauffeur mappings
            port_chauffeurs, chauffeur_ports = self._load_chauffeur_mappings()
            
            
            logger.debug(f"Loaded {len(port_chauffeurs)} ports with chauffeur mappings")
            logger.debug(f"Total chauffeurs mapped: {len(chauffeur_ports)}")
            
        except Exception as e:
            logger.error(f"Error loading port mappings: {str(e)}")
        
        return {
            'port_vehicles': port_vehicles,
            'port_chauffeurs': port_chauffeurs,
            'vehicle_ports': vehicle_ports,
            'chauffeur_ports': chauffeur_ports
        }
    
    def _load_vehicle_mappings(self) -> Tuple[Dict, Dict]:
        """Load vehicle-port mappings."""
        port_vehicles = {}
        vehicle_ports = {}
        
        query = f"""
            SELECT
                P.ID AS PortID,
                COUNT(V.ID) AS VehicleCount,
                STRING_AGG(CAST(V.ID AS VARCHAR(20)), ',') AS VehicleIDs
            FROM [{DB_NAME}].[dbo].[Port] P
            LEFT JOIN [{DB_NAME}].[dbo].[Contractor] C
                ON C.RefPort = P.ID
            LEFT JOIN [{DB_NAME}].[dbo].[Vehicle] V
                ON V.RefContractor = C.ID
            GROUP BY P.ID
        """
        
        rows = self.db_connector.execute_query(query)
        
        for _, row in rows.iterrows():
            port_id = int(row['PortID'])
            vehicle_ids_str = row['VehicleIDs']

            # STRING_AGG returns NULL for ports with no vehicles; pandas
            # reads that as float NaN, which is truthy — guard on str.
            if isinstance(vehicle_ids_str, str) and vehicle_ids_str.strip():
                vehicle_ids = [int(vid.strip()) for vid in vehicle_ids_str.split(',') if vid.strip()]
                port_vehicles[port_id] = vehicle_ids
                
                for vid in vehicle_ids:
                    vehicle_ports[vid] = port_id
            else:
                port_vehicles[port_id] = []
        
        return port_vehicles, vehicle_ports
    
    def _load_chauffeur_mappings(self) -> Tuple[Dict, Dict]:
        """Load chauffeur-port mappings."""
        port_chauffeurs = {}
        chauffeur_ports = {}
        
        query = f"""
            SELECT
                RefPort,
                count([RefChauffeur]) as countofchauffeurs,
                STRING_AGG(CAST(RefChauffeur AS VARCHAR(20)), ',') AS ChauffeurList
            FROM [{DB_NAME}].[dbo].[PortPreferredChauffeur]
            GROUP BY RefPort
        """
        
        rows = self.db_connector.execute_query(query)
        
        for _, row in rows.iterrows():
            port_id = int(row['RefPort'])
            chauffeur_ids_str = row['ChauffeurList']

            # Same NaN guard as vehicle mappings above.
            if isinstance(chauffeur_ids_str, str) and chauffeur_ids_str.strip():
                chauffeur_ids = [int(cid.strip()) for cid in chauffeur_ids_str.split(',') if cid.strip()]
                port_chauffeurs[port_id] = chauffeur_ids
                
                for cid in chauffeur_ids:
                    chauffeur_ports[cid] = port_id
            else:
                port_chauffeurs[port_id] = []
        
        return port_chauffeurs, chauffeur_ports


class ContractorMappingLoader:
    """
    Loads contractor mappings from database.
    
    REPLACES: load_contractor_mappings() [lines 655-719]
    """
    
    def __init__(self, db_connector: DatabaseConnector):
        """
        Initialize contractor mapping loader.
        
        Args:
            db_connector: Database connector instance
        """
        self.db_connector = db_connector
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self) -> Dict[str, Dict]:
        """
        Load contractor mappings for chauffeurs and vehicles from database.
        
        REPLACES: load_contractor_mappings() [lines 655-719]
        
        Returns:
            Dictionary containing:
            - chauffeur_contractors: {chauffeur_id: contractor_id}
            - vehicle_contractors: {vehicle_id: contractor_id}
        """
        chauffeur_contractors = {}
        vehicle_contractors = {}
        
        try:
            chauffeur_contractors = self._load_chauffeur_contractors()
            vehicle_contractors = self._load_vehicle_contractors()
            
            logger.debug(f"Loaded {len(chauffeur_contractors)} chauffeur-contractor mappings")
            logger.debug(f"Loaded {len(vehicle_contractors)} vehicle-contractor mappings")

            
        except Exception as e:
            logger.error(f"Error loading contractor mappings: {str(e)}")
        
        return {
            'chauffeur_contractors': chauffeur_contractors,
            'vehicle_contractors': vehicle_contractors
        }
    
    def _load_chauffeur_contractors(self) -> Dict[int, int]:
        """Load chauffeur-contractor mappings."""
        chauffeur_contractors = {}
        
        query = f"""
            SELECT ID, RefContractor
            FROM [{DB_NAME}].[dbo].[Chauffeur]
            WHERE IsActive = 1 AND RefContractor IS NOT NULL
        """
        
        rows = self.db_connector.execute_query(query)
        
        for _, row in rows.iterrows():
            chauffeur_id = int(row['ID'])
            contractor_id = int(row['RefContractor'])
            chauffeur_contractors[chauffeur_id] = contractor_id
        
        return chauffeur_contractors
    
    def _load_vehicle_contractors(self) -> Dict[int, int]:
        """Load vehicle-contractor mappings."""
        vehicle_contractors = {}
        
        query = f"""
            SELECT ID, RefContractor
            FROM [{DB_NAME}].[dbo].[Vehicle]
            WHERE IsActive = 1 AND RefContractor IS NOT NULL
        """
        
        rows = self.db_connector.execute_query(query)
        
        for _, row in rows.iterrows():
            vehicle_id = int(row['ID'])
            contractor_id = int(row['RefContractor'])
            vehicle_contractors[vehicle_id] = contractor_id
        
        return vehicle_contractors


class VehicleChauffeurPairCreator:
    """
    Creates vehicle-chauffeur pairs with contractor filtering.
    
    REPLACES: create_vehicle_chauffeur_pairs() [lines 721-781]
    """
    
    def __init__(self, contractor_loader: ContractorMappingLoader):
        """
        Initialize pair creator.
        
        Args:
            contractor_loader: ContractorMappingLoader instance
        """
        self.contractor_loader = contractor_loader
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_pairs(self, bookings_df: dd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Create unique vehicle-chauffeur pairs from bookings data.
        
        REPLACES: create_vehicle_chauffeur_pairs() [lines 721-781]
        
        Args:
            bookings_df: Booking DataFrame
            
        Returns:
            Tuple of (pairs_df, pair_to_id, id_to_pair)
        """
        logger.info("Step 1.1: Creating Vehicle-Chauffeur Pairs")
        
        # Load contractor mappings
        logger.debug("Loading contractor mappings...")
        contractor_mappings = self.contractor_loader.load()
        chauffeur_contractors = contractor_mappings['chauffeur_contractors']
        vehicle_contractors = contractor_mappings['vehicle_contractors']
        
        # Extract unique pairs
        vehicle_chauffeur_pairs = bookings_df[['RefVehicle', 'RefChauffeur']].drop_duplicates()
        vehicle_chauffeur_pairs = vehicle_chauffeur_pairs.dropna()
        vehicle_chauffeur_pairs_pd = vehicle_chauffeur_pairs.compute()
        
        # Filter by contractor constraint
        logger.debug("Filtering pairs by contractor constraint...")
        original_count = len(vehicle_chauffeur_pairs_pd)
        
        valid_pairs = self._filter_by_contractor(
            vehicle_chauffeur_pairs_pd,
            vehicle_contractors,
            chauffeur_contractors
        )
        
        vehicle_chauffeur_pairs_pd = pd.DataFrame(valid_pairs)
        filtered_count = len(vehicle_chauffeur_pairs_pd)
        
        logger.info(f"Contractor filtering: {original_count} pairs -> {filtered_count} valid pairs")
        logger.debug(f"Removed {original_count - filtered_count} cross-contractor pairs")
        
        # Create pair IDs
        vehicle_chauffeur_pairs_pd['pair_id'] = (
            vehicle_chauffeur_pairs_pd['RefVehicle'].astype(str) + '_' +
            vehicle_chauffeur_pairs_pd['RefChauffeur'].astype(str)
        )
        
        # Create mappings
        unique_pairs = vehicle_chauffeur_pairs_pd['pair_id'].unique()
        pair_to_id = {pair: idx for idx, pair in enumerate(unique_pairs)}
        id_to_pair = {idx: pair for pair, idx in pair_to_id.items()}
        
        logger.info(f"Total unique vehicle-chauffeur pairs: {len(vehicle_chauffeur_pairs_pd)}")
        logger.debug(f"Sample pairs: {vehicle_chauffeur_pairs_pd['pair_id'].head().tolist()}")
        
        
        return vehicle_chauffeur_pairs_pd, pair_to_id, id_to_pair
    
    def _filter_by_contractor(
        self,
        pairs_df: pd.DataFrame,
        vehicle_contractors: Dict[int, int],
        chauffeur_contractors: Dict[int, int]
    ) -> List[pd.Series]:
        """Filter pairs to only same-contractor pairs."""
        valid_pairs = []
        
        for _, row in pairs_df.iterrows():
            vehicle_id = int(row['RefVehicle'])
            chauffeur_id = int(row['RefChauffeur'])
            
            vehicle_contractor = vehicle_contractors.get(vehicle_id)
            chauffeur_contractor = chauffeur_contractors.get(chauffeur_id)
            
            if (vehicle_contractor is not None and
                chauffeur_contractor is not None and
                vehicle_contractor == chauffeur_contractor):
                valid_pairs.append(row)
        
        return valid_pairs


class InteractionCreator:
    """
    Creates client-pair interactions from booking data.
    
    REPLACES: create_client_pair_interactions() [lines 364-503]
    """
    
    def __init__(self):
        """Initialize interaction creator."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_client_pair_interactions(
        self,
        bookings_enhanced: dd.DataFrame,
        vehicle_chauffeur_pairs: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create interaction data between clients and vehicle-chauffeur pairs.
        
        REPLACES: create_client_pair_interactions() [lines 364-503]
        
        Args:
            bookings_enhanced: Enhanced booking DataFrame
            vehicle_chauffeur_pairs: Vehicle-chauffeur pairs DataFrame
            
        Returns:
            Tuple of (interactions_df, interaction_stats_df)
        """
        logger.info("Step 1.3: Creating Client-Pair Interactions")

        # Both operands are plain pandas by this point (engineer_features no
        # longer wraps back into dask). Guard with hasattr(.., 'compute'),
        # matching the pattern already used in metadata_enrichment.py and
        # data_filtering.py, in case a caller still passes a dask frame.
        if hasattr(bookings_enhanced, 'compute'):
            bookings_enhanced = bookings_enhanced.compute()
        if hasattr(vehicle_chauffeur_pairs, 'compute'):
            vehicle_chauffeur_pairs = vehicle_chauffeur_pairs.compute()

        # Direct pandas merge: no dask task graph, no partition bookkeeping,
        # no extra copies from converting back and forth — this previously
        # went through a single-partition dask merge + repartition retry that
        # was a recurring source of MemoryErrors despite plenty of free RAM.
        logger.debug("Merging bookings with pairs...")
        bookings_with_pairs = bookings_enhanced.merge(
            vehicle_chauffeur_pairs[['RefVehicle', 'RefChauffeur', 'pair_id']],
            on=['RefVehicle', 'RefChauffeur'],
            how='left'
        )

        # Select columns
        available_columns = self._get_available_columns(bookings_with_pairs)
        interactions_pd = bookings_with_pairs[available_columns]

        # Drop null critical columns
        for col in ['RefClient', 'pair_id']:
            if col in interactions_pd.columns:
                interactions_pd = interactions_pd.dropna(subset=[col])

        logger.debug(f"Selected columns: {available_columns}")
        gc.collect()

        # Aggregate statistics
        interaction_stats = self._aggregate_interactions(interactions_pd)
        
        # Calculate derived features
        interaction_stats = self._calculate_derived_features(interaction_stats, interactions_pd)
        
        self._log_summary(interactions_pd, interaction_stats)
        
        return interactions_pd, interaction_stats
    
    def _get_available_columns(self, df: dd.DataFrame) -> List[str]:
        """Get available columns for interactions."""
        base_columns = ['ID', 'RefClient', 'pair_id', 'RefVehicle', 'RefChauffeur']
        optional_columns = [
            'BookingDate', 'AppointmentDate', 'PriceComputed', 'QuoteToClient',
            'NoOfPax', 'NoOfBags', 'TotalDistance', 'DurationToStartPoint',
            'lead_time_days', 'is_weekend', 'is_business_hours', 'season',
            'booking_hour', 'appointment_hour', 'RefBookingSource', 'RefRegion'
        ]
        return base_columns + [col for col in optional_columns if col in df.columns]
    
    def _aggregate_interactions(self, interactions_pd: pd.DataFrame) -> pd.DataFrame:
        """Aggregate interaction statistics."""
        agg_dict = {'ID': 'count'}
        
        column_aggs = {
            'BookingDate': ['min', 'max'],
            'PriceComputed': ['mean', 'std', 'sum'],
            'QuoteToClient': ['mean'],
            'NoOfPax': ['mean', 'max'],
            'NoOfBags': ['mean', 'max'],
            'TotalDistance': ['mean', 'sum'],
            'lead_time_days': ['mean'],
            'is_weekend': ['mean'],
            'is_business_hours': ['mean']
        }
        
        for col, aggs in column_aggs.items():
            if col in interactions_pd.columns:
                agg_dict[col] = aggs
        
        interaction_stats = interactions_pd.groupby(['RefClient', 'pair_id']).agg(agg_dict).reset_index()
        
        # Flatten column names
        new_columns = self._get_flattened_columns(agg_dict)
        interaction_stats.columns = new_columns[:len(interaction_stats.columns)]
        
        return interaction_stats
    
    def _get_flattened_columns(self, agg_dict: Dict) -> List[str]:
        """Get flattened column names from aggregation dict."""
        new_columns = ['client_id', 'pair_id', 'booking_frequency']
        
        column_name_map = {
            ('BookingDate', 'min'): 'first_booking',
            ('BookingDate', 'max'): 'last_booking',
            ('PriceComputed', 'mean'): 'avg_price',
            ('PriceComputed', 'std'): 'price_std',
            ('PriceComputed', 'sum'): 'total_spent',
            ('QuoteToClient', 'mean'): 'avg_quote',
            ('NoOfPax', 'mean'): 'avg_pax',
            ('NoOfPax', 'max'): 'max_pax',
            ('NoOfBags', 'mean'): 'avg_bags',
            ('NoOfBags', 'max'): 'max_bags',
            ('TotalDistance', 'mean'): 'avg_distance',
            ('TotalDistance', 'sum'): 'total_distance',
            ('lead_time_days', 'mean'): 'avg_lead_time',
            ('is_weekend', 'mean'): 'weekend_preference',
            ('is_business_hours', 'mean'): 'business_hours_preference'
        }
        
        for col, agg_funcs in agg_dict.items():
            if col == 'ID':
                continue
            if isinstance(agg_funcs, list):
                for func in agg_funcs:
                    name = column_name_map.get((col, func), f'{col}_{func}')
                    new_columns.append(name)
        
        return new_columns
    
    def _calculate_derived_features(
        self,
        interaction_stats: pd.DataFrame,
        interactions_pd: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate derived features."""
        # Days since last booking
        if 'last_booking' in interaction_stats.columns:
            valid_bookings = interactions_pd['BookingDate'].dropna()
            
            if len(valid_bookings) > 0:
                max_date = valid_bookings.max()
                valid_last = interaction_stats['last_booking'].notna()
                
                interaction_stats['days_since_last_booking'] = np.nan
                if valid_last.sum() > 0:
                    interaction_stats.loc[valid_last, 'days_since_last_booking'] = (
                        max_date - interaction_stats.loc[valid_last, 'last_booking']
                    ).dt.days.clip(0, 36500)
                    
                    logger.info(f"  Days since last booking calculated for {valid_last.sum()}/{len(interaction_stats)} pairs")
            else:
                interaction_stats['days_since_last_booking'] = np.nan
        
        # Customer value
        if 'total_spent' in interaction_stats.columns:
            interaction_stats['customer_value'] = (
                interaction_stats['total_spent'] * interaction_stats['booking_frequency']
            )
        
        # Loyalty score
        if 'days_since_last_booking' in interaction_stats.columns:
            valid_days = interaction_stats['days_since_last_booking'].notna()
            interaction_stats['loyalty_score'] = 0.0
            
            if valid_days.sum() > 0:
                interaction_stats.loc[valid_days, 'loyalty_score'] = (
                    interaction_stats.loc[valid_days, 'booking_frequency'] /
                    (1 + interaction_stats.loc[valid_days, 'days_since_last_booking'] / 365)
                )
        
        return interaction_stats
    
    def _log_summary(self, interactions_pd: pd.DataFrame, interaction_stats: pd.DataFrame) -> None:
        """Log interaction summary."""
        logger.info("Client-pair interactions created!")
        logger.info(f"Total interactions: {len(interactions_pd)}")
        logger.info(f"Unique clients: {interactions_pd['RefClient'].nunique()}")
        logger.info(f"Unique pairs: {interactions_pd['pair_id'].nunique()}")
        logger.info(f"Client-pair combinations: {len(interaction_stats)}")
        logger.debug(f"Avg bookings per combo: {interaction_stats['booking_frequency'].mean():.2f}")
class DataPreparationPipeline(BasePipeline):
    """
    Orchestrates the complete data preparation process.
    
    REPLACES: execute_step1() [lines 506-547]
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data preparation pipeline.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize components
        self.db_connector = DatabaseConnector(DB_CONNECTION_STRING)
        
        self.booking_loader = BookingDataLoader(config)
        self.date_engineer = DateFeatureEngineer(config)
        self.contractor_loader = ContractorMappingLoader(self.db_connector)
        self.pair_creator = VehicleChauffeurPairCreator(self.contractor_loader)
        self.interaction_creator = InteractionCreator()
    
    def execute(self, input_data: Optional[dd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute all Step 1 operations.
        
        REPLACES: execute_step1() [lines 506-547]
        
        Args:
            input_data: Optional pre-loaded booking DataFrame
            
        Returns:
            Dictionary containing step 1 outputs
        """
        self._log_start("Step 1: Data Preparation & Feature Engineering")
        
        try:
            # Load data if not provided
            if input_data is None:
                bookings_df = self.booking_loader.load()
            else:
                bookings_df = input_data
            
            # Step 1.1: Create pairs
            vehicle_chauffeur_pairs, pair_to_id, id_to_pair = self.pair_creator.create_pairs(bookings_df)
            
            # Step 1.2: Engineer date features
            bookings_enhanced = self.date_engineer.engineer_features(bookings_df)
            
            # Step 1.3: Create interactions
            interactions, interaction_stats = self.interaction_creator.create_client_pair_interactions(
                bookings_enhanced, vehicle_chauffeur_pairs
            )
            
            step1_outputs = {
                'vehicle_chauffeur_pairs': vehicle_chauffeur_pairs,
                'pair_to_id': pair_to_id,
                'id_to_pair': id_to_pair,
                'bookings_enhanced': bookings_enhanced,
                'interactions': interactions,
                'interaction_stats': interaction_stats
            }
            
            self._log_summary(step1_outputs)
            self._log_complete("Step 1")
            
            return step1_outputs
            
        except Exception as e:
            self.logger.error(f"Error in Step 1: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.db_connector.dispose()
    
    def _log_summary(self, outputs: Dict[str, Any]) -> None:
        """Log step 1 summary."""
        logger.info("Step 1 completed successfully!")
        logger.info(
            "Data summary | Vehicle-chauffeur pairs: %d | Total interactions: %d | "
            "Unique clients: %d | Client-pair combinations: %d",
            len(outputs['vehicle_chauffeur_pairs']),
            len(outputs['interactions']),
            outputs['interactions']['RefClient'].nunique(),
            len(outputs['interaction_stats'])
        )
        logger.info("Ready for Step 2: Enrichment with Metadata")

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================
# These functions maintain backward compatibility with existing code

def load_booking_data(filepath: Optional[str] = None) -> dd.DataFrame:
    """
    Backward compatible wrapper for BookingDataLoader.
    
    WRAPS: BookingDataLoader.load()
    """
    config = {'booking_data_path': filepath or BOOKING_DATA_PATH}
    loader = BookingDataLoader(config, filepath)
    return loader.load()


def safe_date_diff_days(date1, date2, default=np.nan, clip_range=(0, 36500)):
    """
    Backward compatible wrapper for DateCalculator.safe_date_diff_days().
    
    WRAPS: DateCalculator.safe_date_diff_days()
    """
    return DateCalculator.safe_date_diff_days(date1, date2, default, clip_range)


def safe_date_conversion(series, column_name, min_date=MIN_VALID_DATE, max_date=MAX_VALID_DATE):
    """
    Backward compatible wrapper for DateFeatureEngineer._safe_date_conversion().
    
    WRAPS: DateFeatureEngineer._safe_date_conversion()
    """
    engineer = DateFeatureEngineer()
    engineer.MIN_REASONABLE_DATE = min_date
    engineer.MAX_REASONABLE_DATE = max_date
    return engineer._safe_date_conversion(series, column_name)


def safe_timedelta_calculation(date1, date2, unit='days', max_value=36500):
    """
    Backward compatible wrapper for DateCalculator.safe_timedelta_calculation().
    
    WRAPS: DateCalculator.safe_timedelta_calculation()
    """
    return DateCalculator.safe_timedelta_calculation(date1, date2, unit, max_value)


def engineer_date_features(bookings_df):
    """
    Backward compatible wrapper for DateFeatureEngineer.engineer_features().
    
    WRAPS: DateFeatureEngineer.engineer_features()
    """
    engineer = DateFeatureEngineer()
    return engineer.engineer_features(bookings_df)


def create_client_pair_interactions(bookings_enhanced, vehicle_chauffeur_pairs):
    """
    Backward compatible wrapper for InteractionCreator.create_client_pair_interactions().
    
    WRAPS: InteractionCreator.create_client_pair_interactions()
    """
    creator = InteractionCreator()
    return creator.create_client_pair_interactions(bookings_enhanced, vehicle_chauffeur_pairs)


def load_port_mappings():
    """
    Backward compatible wrapper for PortMappingLoader.load().
    
    WRAPS: PortMappingLoader.load()
    """
    db_connector = DatabaseConnector(DB_CONNECTION_STRING)
    loader = PortMappingLoader(db_connector)
    result = loader.load()
    db_connector.dispose()
    return result


def load_contractor_mappings():
    """
    Backward compatible wrapper for ContractorMappingLoader.load().
    
    WRAPS: ContractorMappingLoader.load()
    """
    db_connector = DatabaseConnector(DB_CONNECTION_STRING)
    loader = ContractorMappingLoader(db_connector)
    result = loader.load()
    db_connector.dispose()
    return result


def create_vehicle_chauffeur_pairs(bookings_df):
    """
    Backward compatible wrapper for VehicleChauffeurPairCreator.create_pairs().
    
    WRAPS: VehicleChauffeurPairCreator.create_pairs()
    """
    db_connector = DatabaseConnector(DB_CONNECTION_STRING)
    contractor_loader = ContractorMappingLoader(db_connector)
    pair_creator = VehicleChauffeurPairCreator(contractor_loader)
    result = pair_creator.create_pairs(bookings_df)
    db_connector.dispose()
    return result


def execute_step1(bookings_df):
    """
    Backward compatible wrapper for DataPreparationPipeline.execute().
    
    WRAPS: DataPreparationPipeline.execute()
    """
    config = {'booking_data_path': BOOKING_DATA_PATH}
    pipeline = DataPreparationPipeline(config)
    return pipeline.execute(bookings_df)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    logger.info("Data Preparation Module (Refactored) - Ready")
    logger.debug(
        "Classes: BookingDataLoader, DateFeatureEngineer, VehicleChauffeurPairCreator, "
        "InteractionCreator, DataPreparationPipeline | "
        "Functions: load_booking_data(), execute_step1()"
    )
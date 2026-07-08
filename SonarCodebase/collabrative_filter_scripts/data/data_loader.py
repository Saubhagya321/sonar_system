import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime

from config.config import Config

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass


class DataLoader:
    """Load and prepare datasets for the recommendation system"""
    
    def __init__(self, config: Config = None, strict_mode: bool = True):
        """
        Args:
            config: Configuration object
            strict_mode: If True, raises errors for missing critical files.
                        If False, uses empty DataFrames (legacy behavior)
        """
        self.config = config or Config()
        self.strict_mode = strict_mode
        self.missing_files = []
        self.loaded_datasets = {}
        self.validation_warnings = []
    
    def _check_file_exists(self, filepath: Path, description: str, critical: bool = False) -> bool:
        """
        Check if file exists and log warning if not
        
        Args:
            filepath: Path to check
            description: Human readable description
            critical: If True and strict_mode=True, will cause system to fail
        """
        if filepath.exists():
            return True
        else:
            log_level = logging.ERROR if critical else logging.WARNING
            logger.log(log_level, f"{' CRITICAL' if critical else ''} Missing {description}: {filepath}")
            self.missing_files.append({
                'filepath': str(filepath),
                'description': description,
                'critical': critical
            })
            return False
    
    def _validate_dataframe(self, df: pd.DataFrame, name: str, 
                           min_rows: int = 1, 
                           required_columns: list = None) -> bool:
        """
        Validate DataFrame meets minimum requirements
        
        Args:
            df: DataFrame to validate
            name: Dataset name for logging
            min_rows: Minimum number of rows required
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        if df is None:
            self.validation_warnings.append(f"{name}: DataFrame is None")
            return False
        
        if len(df) < min_rows:
            self.validation_warnings.append(
                f"{name}: Only {len(df)} rows (minimum {min_rows} required)"
            )
            return False
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                self.validation_warnings.append(
                    f"{name}: Missing required columns: {missing_cols}"
                )
                return False
        
        return True
    
    def _safe_int_conversion(self, series: pd.Series, default: int = 0) -> pd.Series:
        """Safely convert series to int, replacing errors with default"""
        return pd.to_numeric(series, errors='coerce').fillna(default).astype(int)
    
    def _safe_float_conversion(self, series: pd.Series, default: float = 0.0) -> pd.Series:
        """Safely convert series to float, replacing errors with default"""
        return pd.to_numeric(series, errors='coerce').fillna(default)
    
    def load_booking_data(self) -> Optional[pd.DataFrame]:
        """
        Load booking data from parquet file
        This is CRITICAL - system cannot function without it
        """
        if not self._check_file_exists(self.config.BOOKING_DATA_PATH, "booking data", critical=True):
            if self.strict_mode:
                raise DataValidationError(
                    f"CRITICAL: Booking data not found at {self.config.BOOKING_DATA_PATH}. "
                    "System cannot operate without booking data."
                )
            return None
        
        try:
            logger.info(f"Loading booking data from {self.config.BOOKING_DATA_PATH}")
            
            # Load with Dask for large files
            booking_dask = dd.read_parquet(
                str(self.config.BOOKING_DATA_PATH), 
                engine="pyarrow"
            )
            
            # Select required columns
            columns = [
                'ID', 'RefClient', 'RefChauffeur', 'RefVehicle',
                'BookingDate', 'AppointmentDate', 'PriceComputed', 'QuoteToClient',
                'NoOfPax', 'NoOfBags', 'TotalDistance', 'DurationToStartPoint',
                'DateCompleted', 'BookingStatusDate', 'CreatedOn',
                'RefVehicleTypeRequest', 'LevyAmount', 'AirportPickupFee',
                'BaseContractorCost', 'ContractorPayAmount'
            ]
            
            booking_dask = booking_dask[columns]
            
            # Convert to pandas for processing
            logger.info("Converting Dask DataFrame to Pandas...")
            booking_df = booking_dask.compute()
            
            # Validate
            if not self._validate_dataframe(
                booking_df, 
                "booking_data", 
                min_rows=100,
                required_columns=['ID', 'RefClient', 'RefChauffeur', 'RefVehicle']
            ):
                raise DataValidationError(
                    "Booking data validation failed. Check logs for details."
                )
            
            logger.info(f" Loaded {len(booking_df):,} booking records")
            return booking_df
            
        except Exception as e:
            logger.error(f" Failed to load booking data: {e}")
            if self.strict_mode:
                raise DataValidationError(f"Failed to load critical booking data: {e}")
            return None
    
    def load_vehicle_data(self) -> pd.DataFrame:
        """Load vehicle data - IMPORTANT but not critical"""
        if not self._check_file_exists(self.config.VEHICLE_DATA_PATH, "vehicle data", critical=False):
            logger.warning(" Vehicle data missing - filtering by vehicle attributes will not work")
            return self._create_empty_vehicle_df()
        
        try:
            logger.info(f"Loading vehicle data from {self.config.VEHICLE_DATA_PATH}")
            
            columns = [
                'ID', 'RefVehicleType', 'Year', 'Code', 'Colour', 'Make', 'Model',
                'RefChauffeurCurrent', 'RefChauffeurDefault', 'RefPreviousSuburb',
                'LeaseCost', 'LeaseDuration_AvgImputed', 'LeaseDuration_TillToday',
                'LeaseIsActive', 'InsuranceDaysTillExpiry', 'InsuranceIsActive',
                'RWCDaysTillExpiry', 'RWCIsActive', 'PublicLiabilityDaysTillExpiry',
                'PublicLiabilityIsActive'
            ]
            
            vehicle_df = pd.read_csv(self.config.VEHICLE_DATA_PATH)[columns]
            
            # Safe type conversions
            vehicle_df['ID'] = self._safe_int_conversion(vehicle_df['ID'])
            vehicle_df['RefVehicleType'] = self._safe_int_conversion(vehicle_df['RefVehicleType'])
            vehicle_df['Year'] = self._safe_int_conversion(vehicle_df['Year'], default=2020)
            
            # Validate
            if not self._validate_dataframe(vehicle_df, "vehicle_data", min_rows=10):
                logger.warning(" Vehicle data has fewer than 10 records")
            
            logger.info(f" Loaded {len(vehicle_df):,} vehicle records")
            return vehicle_df
            
        except Exception as e:
            logger.error(f" Failed to load vehicle data: {e}")
            logger.warning(" Continuing with empty vehicle data - some features will not work")
            return self._create_empty_vehicle_df()
    
    def _create_empty_vehicle_df(self) -> pd.DataFrame:
        """Create empty vehicle dataframe with correct schema"""
        return pd.DataFrame(columns=[
            'ID', 'RefVehicleType', 'Year', 'Code', 'Colour', 'Make', 'Model',
            'RefChauffeurCurrent', 'RefChauffeurDefault', 'RefPreviousSuburb',
            'LeaseCost', 'LeaseDuration_AvgImputed', 'LeaseDuration_TillToday',
            'LeaseIsActive', 'InsuranceDaysTillExpiry', 'InsuranceIsActive',
            'RWCDaysTillExpiry', 'RWCIsActive', 'PublicLiabilityDaysTillExpiry',
            'PublicLiabilityIsActive'
        ])
    
    def load_vehicle_type_data(self) -> pd.DataFrame:
        """Load vehicle type data - IMPORTANT for filtering"""
        if not self._check_file_exists(self.config.VEHICLE_TYPE_PATH, "vehicle type data", critical=False):
            logger.warning(" Vehicle type data missing - capacity/level filtering will not work")
            return self._create_empty_vehicle_type_df()
        
        try:
            logger.info(f"Loading vehicle type data from {self.config.VEHICLE_TYPE_PATH}")
            
            columns = [
                'ID', 'SeatCapacity', 'VehicleLevel', 'Code', 'Description',
                'DisplayName', 'TollVehicleType', 'IsLevyApplied'
            ]
            
            vehicle_type_df = pd.read_csv(self.config.VEHICLE_TYPE_PATH)[columns]
            
            # Safe type conversions
            vehicle_type_df['ID'] = self._safe_int_conversion(vehicle_type_df['ID'])
            vehicle_type_df['SeatCapacity'] = self._safe_int_conversion(vehicle_type_df['SeatCapacity'], default=4)
            
            # Validate
            if not self._validate_dataframe(vehicle_type_df, "vehicle_type_data", min_rows=3):
                logger.warning(" Vehicle type data has fewer than 3 records")
            
            logger.info(f" Loaded {len(vehicle_type_df):,} vehicle type records")
            return vehicle_type_df
            
        except Exception as e:
            logger.error(f" Failed to load vehicle type data: {e}")
            logger.warning(" Continuing with empty vehicle type data")
            return self._create_empty_vehicle_type_df()
    
    def _create_empty_vehicle_type_df(self) -> pd.DataFrame:
        """Create empty vehicle type dataframe"""
        return pd.DataFrame(columns=[
            'ID', 'SeatCapacity', 'VehicleLevel', 'Code', 'Description',
            'DisplayName', 'TollVehicleType', 'IsLevyApplied'
        ])
    
    def load_person_charge_rate_data(self) -> pd.DataFrame:
        """Load person charge rate data - OPTIONAL"""
        if not self._check_file_exists(self.config.PERSON_CHARGE_RATE_PATH, "person charge rate data"):
            return pd.DataFrame(columns=['RefVehicleType', 'PersonRate1', 'PersonRate2', 'PersonRate3', 'PersonRate4'])
        
        try:
            logger.info(f"Loading person charge rate data from {self.config.PERSON_CHARGE_RATE_PATH}")
            
            columns = ['RefVehicleType', 'PersonRate1', 'PersonRate2', 'PersonRate3', 'PersonRate4']
            person_charge_rate_df = pd.read_csv(self.config.PERSON_CHARGE_RATE_PATH)[columns]
            
            logger.info(f" Loaded {len(person_charge_rate_df):,} person charge rate records")
            return person_charge_rate_df
            
        except Exception as e:
            logger.error(f" Failed to load person charge rate data: {e}")
            return pd.DataFrame(columns=['RefVehicleType', 'PersonRate1', 'PersonRate2', 'PersonRate3', 'PersonRate4'])
    
    def load_chauffeur_data(self) -> pd.DataFrame:
        """Load chauffeur data - IMPORTANT for filtering"""
        if not self._check_file_exists(self.config.CHAUFFEUR_DATA_PATH, "chauffeur data", critical=False):
            logger.warning(" Chauffeur data missing - chauffeur attribute filtering will not work")
            return self._create_empty_chauffeur_df()
        
        try:
            logger.info(f"Loading chauffeur data from {self.config.CHAUFFEUR_DATA_PATH}")
            
            columns = [
                'ID', 'employee_duration_days', 'Code', 'RefSuburb', 'RefChauffeurStatus',
                'CommissionRate', 'RedZoneAuthorised', 'DriverTraining', 'BabySeatTraining',
                'days_to_license_expiry', 'license_expired', 'age', 'chauffeur_status_duration',
                'days_since_police_check', 'days_since_wwc', 'days_since_medical_cert'
            ]
            
            chauffeur_df = pd.read_csv(self.config.CHAUFFEUR_DATA_PATH)[columns]
            
            # Safe type conversions
            chauffeur_df['ID'] = self._safe_int_conversion(chauffeur_df['ID'])
            chauffeur_df['employee_duration_days'] = self._safe_int_conversion(chauffeur_df['employee_duration_days'], default=0)
            chauffeur_df['days_to_license_expiry'] = self._safe_int_conversion(chauffeur_df['days_to_license_expiry'], default=365)
            chauffeur_df['age'] = self._safe_int_conversion(chauffeur_df['age'], default=30)
            
            # Validate
            if not self._validate_dataframe(chauffeur_df, "chauffeur_data", min_rows=10):
                logger.warning(" Chauffeur data has fewer than 10 records")
            
            logger.info(f" Loaded {len(chauffeur_df):,} chauffeur records")
            return chauffeur_df
            
        except Exception as e:
            logger.error(f" Failed to load chauffeur data: {e}")
            logger.warning(" Continuing with empty chauffeur data")
            return self._create_empty_chauffeur_df()
    
    def _create_empty_chauffeur_df(self) -> pd.DataFrame:
        """Create empty chauffeur dataframe"""
        return pd.DataFrame(columns=[
            'ID', 'employee_duration_days', 'Code', 'RefSuburb', 'RefChauffeurStatus',
            'CommissionRate', 'RedZoneAuthorised', 'DriverTraining', 'BabySeatTraining',
            'days_to_license_expiry', 'license_expired', 'age', 'chauffeur_status_duration',
            'days_since_police_check', 'days_since_wwc', 'days_since_medical_cert'
        ])
    
    def load_client_data(self) -> pd.DataFrame:
        """Load client data - OPTIONAL"""
        if not self._check_file_exists(self.config.CLIENT_DATA_PATH, "client data"):
            return self._create_empty_client_df()
        
        try:
            logger.info(f"Loading client data from {self.config.CLIENT_DATA_PATH}")
            
            columns = [
                'ID', 'Code', 'Description', 'StartDate', 'LastBookingDate', 'IsActive',
                'RefPort', 'RefSuburb', 'PostCode', 'RefMarketSector', 'RefVehicleType',
                'VehicleTypeIds', 'RefSensitivityLevel', 'ChauffeurInstructions',
                'ClientSpecialInstruction', 'IsLevyApplied', 'RefPaymentType',
                'CommissionReserve', 'PayVariance', 'RefDiscount', 'IsControlledAccount'
            ]
            
            client_df = pd.read_csv(self.config.CLIENT_DATA_PATH)[columns]
            
            # Safe type conversions
            client_df['ID'] = self._safe_int_conversion(client_df['ID'])
            
            logger.info(f" Loaded {len(client_df):,} client records")
            return client_df
            
        except Exception as e:
            logger.error(f" Failed to load client data: {e}")
            return self._create_empty_client_df()
    
    def _create_empty_client_df(self) -> pd.DataFrame:
        """Create empty client dataframe"""
        return pd.DataFrame(columns=[
            'ID', 'Code', 'Description', 'StartDate', 'LastBookingDate', 'IsActive',
            'RefPort', 'RefSuburb', 'PostCode', 'RefMarketSector', 'RefVehicleType',
            'VehicleTypeIds', 'RefSensitivityLevel', 'ChauffeurInstructions',
            'ClientSpecialInstruction', 'IsLevyApplied', 'RefPaymentType',
            'CommissionReserve', 'PayVariance', 'RefDiscount', 'IsControlledAccount'
        ])
    
    def load_booking_rating_data(self) -> pd.DataFrame:
        """Load booking rating data - OPTIONAL"""
        if not self._check_file_exists(self.config.BOOKING_RATING_PATH, "booking rating data"):
            return pd.DataFrame(columns=['ID', 'RefBooking', 'RatingStars', 'RefRatingAspect', 'RefChauffeur', 'RefVehicle'])
        
        try:
            logger.info(f"Loading booking rating data from {self.config.BOOKING_RATING_PATH}")
            
            columns = ['ID', 'RefBooking', 'RatingStars', 'RefRatingAspect', 'RefChauffeur', 'RefVehicle']
            booking_rating_df = pd.read_csv(self.config.BOOKING_RATING_PATH)[columns]
            
            logger.info(f" Loaded {len(booking_rating_df):,} booking rating records")
            return booking_rating_df
            
        except Exception as e:
            logger.error(f" Failed to load booking rating data: {e}")
            return pd.DataFrame(columns=['ID', 'RefBooking', 'RatingStars', 'RefRatingAspect', 'RefChauffeur', 'RefVehicle'])
    
    def load_chauffeur_status_data(self) -> pd.DataFrame:
        """Load chauffeur status data - IMPORTANT for availability"""
        if not self._check_file_exists(self.config.CHAUFFEUR_STATUS_PATH, "chauffeur status data"):
            logger.warning(" Chauffeur status data missing - availability filtering will not work")
            return pd.DataFrame(columns=['RefChauffeur', 'ChauffeurStatus', 'RefVehicle'])
        
        try:
            logger.info(f"Loading chauffeur status data from {self.config.CHAUFFEUR_STATUS_PATH}")
            
            columns = ['RefChauffeur', 'ChauffeurStatus', 'RefVehicle']
            chauffeur_status_df = pd.read_csv(self.config.CHAUFFEUR_STATUS_PATH)[columns]
            
            logger.info(f" Loaded {len(chauffeur_status_df):,} chauffeur status records")
            return chauffeur_status_df
            
        except Exception as e:
            logger.error(f" Failed to load chauffeur status data: {e}")
            return pd.DataFrame(columns=['RefChauffeur', 'ChauffeurStatus', 'RefVehicle'])
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets at once - PRODUCTION READY VERSION
        FIXES: P1-2 Explicit error handling instead of silent failures
        """
        logger.info(" Loading all datasets...")
        
        # Reset tracking
        self.missing_files = []
        self.validation_warnings = []
        self.loaded_datasets = {}
        
        # Load all datasets
        datasets = {
            'booking': self.load_booking_data(),
            'vehicle': self.load_vehicle_data(),
            'vehicle_type': self.load_vehicle_type_data(),
            'person_charge_rate': self.load_person_charge_rate_data(),
            'chauffeur': self.load_chauffeur_data(),
            'client': self.load_client_data(),
            'booking_rating': self.load_booking_rating_data(),
            'chauffeur_status': self.load_chauffeur_status_data()
        }
        
        # Check for critical failures
        critical_missing = [f for f in self.missing_files if f['critical']]
        
        if critical_missing and self.strict_mode:
            error_msg = "CRITICAL datasets missing:\n"
            for f in critical_missing:
                error_msg += f"  - {f['description']}: {f['filepath']}\n"
            raise DataValidationError(error_msg)
        
        # Report on loaded data
        logger.info(f"\n{'='*80}")
        logger.info(" DATASET LOADING SUMMARY")
        logger.info(f"{'='*80}")
        
        for name, df in datasets.items():
            if df is not None and len(df) > 0:
                logger.info(f"    {name:20s}: {len(df):,} records")
            else:
                logger.warning(f"     {name:20s}: 0 records (EMPTY!)")
        
        # Report warnings
        if self.validation_warnings:
            logger.warning(f"\n  VALIDATION WARNINGS ({len(self.validation_warnings)}):")
            for warning in self.validation_warnings:
                logger.warning(f"   - {warning}")
        
        # Report missing files
        if self.missing_files:
            non_critical = [f for f in self.missing_files if not f['critical']]
            if non_critical:
                logger.warning(f"\n  NON-CRITICAL FILES MISSING ({len(non_critical)}):")
                for f in non_critical:
                    logger.warning(f"   - {f['description']}: {f['filepath']}")
                logger.warning("   System will continue with reduced functionality.\n")
        
        logger.info(f"{'='*80}\n")
        
        return datasets
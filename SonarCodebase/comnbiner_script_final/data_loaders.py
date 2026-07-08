
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sqlalchemy import create_engine, text
from config import DB_CONNECTION_STRING, BOOKING_COLUMNS, COMPLETED_STATUSES,TRANSELITE_DB

from config import MODEL_CONFIGS, GRID_SEARCH_PATHS, REQUEST_PATHS, CLIENT_TYPE_FILE, CLIENT_TYPE_MAPPING

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading data from all 4 recommendation models
    """
    # Add this method to the DataLoader class in data_loaders.py
    @staticmethod
    
    def load_port_mappings():
        """
        Load port mappings for vehicles and chauffeurs from database
        Returns dict with mappings:
        {
            'port_vehicles': {port_id: [list of vehicle_ids]},
            'port_chauffeurs': {port_id: [list of chauffeur_ids]},
            'vehicle_ports': {vehicle_id: port_id},
            'chauffeur_ports': {chauffeur_id: port_id}
        }
        """
        
        port_vehicles = {}
        port_chauffeurs = {}
        vehicle_ports = {}
        chauffeur_ports = {}
        
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(DB_CONNECTION_STRING)
            
            with engine.connect() as conn:
                # Load Port-Vehicle Mapping
                vehicle_query = text(f"""
                    SELECT 
                        P.ID AS PortID,
                        COUNT(V.ID) AS VehicleCount,
                        STRING_AGG(CAST(V.ID AS VARCHAR(20)), ',') AS VehicleIDs
                    FROM [{TRANSELITE_DB}].[dbo].[Port] P
                    LEFT JOIN [{TRANSELITE_DB}].[dbo].[Contractor] C 
                        ON C.RefPort = P.ID
                    LEFT JOIN [{TRANSELITE_DB}].[dbo].[Vehicle] V 
                        ON V.RefContractor = C.ID
                    GROUP BY P.ID
                """)
                
                vehicle_result = conn.execute(vehicle_query)
                vehicle_rows = vehicle_result.fetchall()
                
                for row in vehicle_rows:
                    port_id = int(row.PortID)
                    vehicle_ids_str = row.VehicleIDs
                    
                    if vehicle_ids_str:
                        # Split comma-separated IDs and convert to integers
                        vehicle_ids = [int(vid.strip()) for vid in vehicle_ids_str.split(',') if vid.strip()]
                        port_vehicles[port_id] = vehicle_ids
                        
                        # Create reverse mapping
                        for vid in vehicle_ids:
                            vehicle_ports[vid] = port_id
                    else:
                        port_vehicles[port_id] = []
                
            
                chauffeur_query = text(f"""
                    SELECT 
                        RefPort,
                        count([RefChauffeur]) as countofchauffeurs,
                        STRING_AGG(CAST(RefChauffeur AS VARCHAR(20)), ',') AS ChauffeurList
                    FROM [{TRANSELITE_DB}].[dbo].[PortPreferredChauffeur]
                    GROUP BY RefPort
                """)
                
                chauffeur_result = conn.execute(chauffeur_query)
                chauffeur_rows = chauffeur_result.fetchall()
                
                for row in chauffeur_rows:
                    port_id = int(row.RefPort)
                    chauffeur_ids_str = row.ChauffeurList
                    
                    if chauffeur_ids_str:
                        # Split comma-separated IDs and convert to integers
                        chauffeur_ids = [int(cid.strip()) for cid in chauffeur_ids_str.split(',') if cid.strip()]
                        port_chauffeurs[port_id] = chauffeur_ids
                        
                        # Create reverse mapping
                        for cid in chauffeur_ids:
                            chauffeur_ports[cid] = port_id
                    else:
                        port_chauffeurs[port_id] = []
                
                logger.info(f"Loaded {len(port_chauffeurs)} ports with chauffeur mappings")
                logger.info(f"Total chauffeurs mapped: {len(chauffeur_ports)}")
                
            engine.dispose()
            
            return {
                'port_vehicles': port_vehicles,
                'port_chauffeurs': port_chauffeurs,
                'vehicle_ports': vehicle_ports,
                'chauffeur_ports': chauffeur_ports
            }
            
        except Exception as e:
            logger.error(f"Error loading port mappings: {str(e)}")
            return {
                'port_vehicles': {},
                'port_chauffeurs': {},
                'vehicle_ports': {},
                'chauffeur_ports': {}
            }


    @staticmethod
    def load_sensitivity_levels():
        """
        Load sensitivity levels for clients, chauffeurs, and vehicles
        Returns dict with mappings:
        {
            'clients': {client_id: sensitivity_level},
            'chauffeurs': {chauffeur_id: sensitivity_level},
            'vehicles': {vehicle_id: sensitivity_level}
        }
        """
        logger.info("Loading sensitivity levels...")
        
        try:
            # ===== CHANGE 1: Add file existence check for client =====
            if not Path('client_final.csv').exists():
                logger.warning("client_final.csv not found")
                client_sensitivity = {}
            else:
                client_df = pd.read_csv('client_final.csv')
                
                # Handle the unnamed index column if present
                if 'ID' in client_df.columns and 'RefSensitivityLevel' in client_df.columns:
                    client_sensitivity = {}
                    for _, row in client_df.iterrows():
                        if pd.notna(row['ID']) and pd.notna(row['RefSensitivityLevel']):
                            try:
                                client_id = int(row['ID'])
                                sensitivity = int(row['RefSensitivityLevel'])
                                client_sensitivity[client_id] = sensitivity
                            except (ValueError, TypeError):
                                continue
                    
                    logger.info(f"      Loaded {len(client_sensitivity)} client sensitivity levels")
                else:
                    logger.error("  Missing ID or RefSensitivityLevel columns in client_final.csv")
                    client_sensitivity = {}
            
            # ===== CHANGE 2: Add file existence check for chauffeur =====
            if not Path('chauff_final.csv').exists():
                logger.warning("chauff_final.csv not found")
                chauffeur_sensitivity = {}
            else:
                chauffeur_df = pd.read_csv('chauff_final.csv')
                
                if 'ID' in chauffeur_df.columns and 'RefSensitivityLevel' in chauffeur_df.columns:
                    chauffeur_sensitivity = {}
                    for _, row in chauffeur_df.iterrows():
                        if pd.notna(row['ID']) and pd.notna(row['RefSensitivityLevel']):
                            try:
                                chauffeur_id = int(row['ID'])
                                sensitivity = int(row['RefSensitivityLevel'])
                                chauffeur_sensitivity[chauffeur_id] = sensitivity
                            except (ValueError, TypeError):
                                continue
                    
                    logger.info(f"      Loaded {len(chauffeur_sensitivity)} chauffeur sensitivity levels")
                else:
                    logger.error("  Missing ID or RefSensitivityLevel columns in chauff_final.csv")
                    chauffeur_sensitivity = {}
            
            # ===== CHANGE 3: Add file existence check for vehicle =====
            if not Path('vehicle_final.csv').exists():
                logger.warning("vehicle_final.csv not found")
                vehicle_sensitivity = {}
            else:
                vehicle_df = pd.read_csv('vehicle_final.csv')
                
                if 'ID' in vehicle_df.columns and 'RefSensitivityLevel' in vehicle_df.columns:
                    vehicle_sensitivity = {}
                    for _, row in vehicle_df.iterrows():
                        if pd.notna(row['ID']) and pd.notna(row['RefSensitivityLevel']):
                            try:
                                vehicle_id = int(row['ID'])
                                sensitivity = int(row['RefSensitivityLevel'])
                                vehicle_sensitivity[vehicle_id] = sensitivity
                            except (ValueError, TypeError):
                                continue
                    
                    logger.info(f"      Loaded {len(vehicle_sensitivity)} vehicle sensitivity levels")
                else:
                    logger.error("  Missing ID or RefSensitivityLevel columns in vehicle_final.csv")
                    vehicle_sensitivity = {}
            
            return {
                'clients': client_sensitivity,
                'chauffeurs': chauffeur_sensitivity,
                'vehicles': vehicle_sensitivity
            }
            
        except FileNotFoundError as e:
            logger.error(f"  Sensitivity level file not found: {str(e)}")
            return {
                'clients': {},
                'chauffeurs': {},
                'vehicles': {}
            }
        except Exception as e:
            logger.error(f"  Error loading sensitivity levels: {str(e)}")
            return {
                'clients': {},
                'chauffeurs': {},
                'vehicles': {}
            }
        
    @staticmethod
    def load_lightgcn_gridsearch(file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load LightGCN recommendations from CSV for grid search
        NEW COLUMNS: original_client_id, original_vehicle_id, original_chauffeur_id, vehicle_idx, chauffeur_idx,
                     rank, combined_score, cv_score, cc_score, vc_score, recommendation_type, similar_clients_used, client_node_id
        """
        if file_path is None:
            file_path = GRID_SEARCH_PATHS['lightgcn']
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"    Loaded LightGCN grid search data: {len(df)} rows")
            
            # Use original IDs for pair_id creation
            vehicle_col = 'original_vehicle_id'
            chauffeur_col = 'original_chauffeur_id'
            client_col = 'original_client_id'
            
            # Verify columns exist
            required_cols = [vehicle_col, chauffeur_col, client_col]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                return pd.DataFrame()
            
            # Convert IDs: float → int → str (removes .0)
            for col in [vehicle_col, chauffeur_col, client_col]:
                # Track original count
                original_count = len(df)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=[col])
                # Log dropped rows
                dropped = original_count - len(df)
                if dropped > 0:
                    logger.warning(f"  Dropped {dropped} rows due to invalid {col}")
                df[col] = df[col].astype(int).astype(str)
            
            # Create pair_id (chauffeur_vehicle format)
            df['pair_id'] = df[chauffeur_col] + '_' + df[vehicle_col]
            
            # Normalize scores (0-1 range)
            score_cols = ['cv_score', 'cc_score', 'vc_score', 'combined_score']
            for col in score_cols:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[f'{col}_normalized'] = 0.5
            
            logger.info(f"  Sample pair_ids: {df['pair_id'].head(3).tolist()}")
            logger.info(f"  Unique clients: {df[client_col].nunique()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"    LightGCN file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"    Error loading LightGCN data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def load_collaborative_gridsearch(file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load Collaborative Filtering recommendations from CSV for grid search
        NEW COLUMNS: client_id, chauffeur_vehicle_pair, predicted_rating, confidence, rank
        CHANGE: Use predicted_rating * confidence as combined score
        """
        if file_path is None:
            file_path = GRID_SEARCH_PATHS['collaborative']
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"    Loaded Collaborative CF grid search data: {len(df)} rows")
            
            # Validate required columns
            required_cols = ['client_id', 'chauffeur_vehicle_pair', 'predicted_rating', 'confidence', 'rank']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in Collaborative data: {missing_cols}")
            
            # Convert client_id to int
            df['client_id'] = pd.to_numeric(df['client_id'], errors='coerce')
            df = df.dropna(subset=['client_id'])
            df['client_id'] = df['client_id'].astype(int)
            
            # Ensure pair_id format matches
            df['pair_id'] = df['chauffeur_vehicle_pair'].astype(str)
            
            # NEW: Create combined score = predicted_rating * confidence
            df['cf_combined_score'] = df['predicted_rating'] * df['confidence']
            
            # Normalize scores
            df['predicted_rating_normalized'] = (
                (df['predicted_rating'] - df['predicted_rating'].min()) / 
                (df['predicted_rating'].max() - df['predicted_rating'].min() + 1e-10)
            )
            
            # Normalize combined score
            df['cf_combined_score_normalized'] = (
                (df['cf_combined_score'] - df['cf_combined_score'].min()) / 
                (df['cf_combined_score'].max() - df['cf_combined_score'].min() + 1e-10)
            )
            
            logger.info(f"  Sample pair_ids: {df['pair_id'].head(3).tolist()}")
            logger.info(f"  Unique clients: {df['client_id'].nunique()}")
            logger.info(f"  CF combined score range: [{df['cf_combined_score'].min():.3f}, {df['cf_combined_score'].max():.3f}]")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"    Collaborative CF file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"    Error loading Collaborative CF data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def load_booking_history(client_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load actual booking history from SQL Server database
        """
        logger.info("Loading booking history from SQL Server...")
        
        try:
            engine = create_engine(DB_CONNECTION_STRING)
            
            # Build query
            from sqlalchemy import text

            # Build base query
            booking_cols = ', '.join(BOOKING_COLUMNS)
            completed_statuses_placeholder = ','.join([':status_' + str(i) for i in range(len(COMPLETED_STATUSES))])

            # Create parameters dict
            params = {}
            for i, status in enumerate(COMPLETED_STATUSES):
                params[f'status_{i}'] = status

            if client_ids:
                # Validate client_ids are integers
                validated_client_ids = [int(cid) for cid in client_ids if str(cid).isdigit()]
                if not validated_client_ids:
                    logger.warning("No valid client IDs provided")
                    return pd.DataFrame()

                # SQL Server caps query parameters at ~2100 total, so a single IN(...)
                # clause with thousands of client IDs fails with "COUNT field incorrect
                # or syntax error" — chunk the client IDs across separate queries instead.
                CLIENT_ID_CHUNK_SIZE = 2000
                chunks = []
                with engine.connect() as conn:
                    for start in range(0, len(validated_client_ids), CLIENT_ID_CHUNK_SIZE):
                        chunk_ids = validated_client_ids[start:start + CLIENT_ID_CHUNK_SIZE]
                        client_placeholder = ','.join([':client_' + str(i) for i in range(len(chunk_ids))])
                        chunk_params = dict(params)
                        for i, cid in enumerate(chunk_ids):
                            chunk_params[f'client_{i}'] = cid

                        query = text(f"""
                            SELECT {booking_cols} FROM Booking
                            WHERE RefClient IN ({client_placeholder})
                            AND RefBookingStatus IN ({completed_statuses_placeholder})
                        """)
                        chunks.append(pd.read_sql(query, conn, params=chunk_params))

                booking_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                query = text(f"""
                    SELECT {booking_cols} FROM Booking
                    WHERE RefBookingStatus IN ({completed_statuses_placeholder})
                """)

                # Load data with parameterized query
                with engine.connect() as conn:
                    booking_df = pd.read_sql(query, conn, params=params)

            engine.dispose()
            
            # Convert IDs: float → int → str
            booking_df['RefChauffeur'] = pd.to_numeric(booking_df['RefChauffeur'], errors='coerce')
            booking_df['RefVehicle'] = pd.to_numeric(booking_df['RefVehicle'], errors='coerce')
            
            booking_df = booking_df.dropna(subset=['RefChauffeur', 'RefVehicle'])
            
            booking_df['RefChauffeur'] = booking_df['RefChauffeur'].astype(int).astype(str)
            booking_df['RefVehicle'] = booking_df['RefVehicle'].astype(int).astype(str)
            booking_df['RefClient'] = booking_df['RefClient'].astype(int)
            
            # Create pair_id in chauffeur_vehicle format
            booking_df['pair_id'] = booking_df['RefChauffeur'] + '_' + booking_df['RefVehicle']
            
            logger.info(f"    Loaded {len(booking_df):,} completed bookings")
            logger.info(f"  Unique clients: {booking_df['RefClient'].nunique()}")
            logger.info(f"  Unique pairs: {booking_df['pair_id'].nunique()}")
            
            return booking_df
            
        except Exception as e:
            logger.error(f"    Error loading booking history: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def load_lightgcn_request(client_id: int, file_path: Optional[Path] = None) -> Dict:
        """
        Load LightGCN recommendations from JSON for real-time request
        """
        if file_path is None:
            file_path = Path(str(REQUEST_PATHS['lightgcn_json']).format(client_id=client_id))
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"    Loaded LightGCN request data for client {client_id}")
            
            if 'pairs' in data:
                df = pd.DataFrame(data['pairs'])
                df['model'] = 'lightgcn'
                
                # Convert IDs
                df['vehicle_id'] = pd.to_numeric(df['vehicle_id'], errors='coerce')
                df['chauffeur_id'] = pd.to_numeric(df['chauffeur_id'], errors='coerce')
                df = df.dropna(subset=['vehicle_id', 'chauffeur_id'])
                
                df['vehicle_id'] = df['vehicle_id'].astype(int).astype(str)
                df['chauffeur_id'] = df['chauffeur_id'].astype(int).astype(str)
                
                df['pair_id'] = df['chauffeur_id'] + '_' + df['vehicle_id']
                
                # Calculate average GCN score if available
                if all(col in df.columns for col in ['cv_score', 'cc_score', 'vc_score']):
                    df['gcn_avg_score'] = (df['cv_score'] + df['cc_score'] + df['vc_score']) / 3
                
                return {
                    'metadata': {k: v for k, v in data.items() if k != 'pairs'},
                    'recommendations': df
                }
            
            return {'metadata': data, 'recommendations': pd.DataFrame()}
            
        except FileNotFoundError:
            logger.error(f"    LightGCN request file not found: {file_path}")
            return {'metadata': {}, 'recommendations': pd.DataFrame()}
        except Exception as e:
            logger.error(f"    Error loading LightGCN request data: {str(e)}")
            return {'metadata': {}, 'recommendations': pd.DataFrame()}
    
    @staticmethod
    def load_collaborative_request(client_id: int, file_path: Optional[Path] = None) -> Dict:
        """
        Load Collaborative Filtering recommendations from JSON for real-time request
        """
        if file_path is None:
            file_path = Path(str(REQUEST_PATHS['collaborative_json']).format(client_id=client_id))
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"    Loaded Collaborative CF request data for client {client_id}")
            
            if 'recommendations' in data:
                df = pd.DataFrame(data['recommendations'])
                df['model'] = 'collaborative'
                
                # Handle pair_id
                if 'pair_id' in df.columns:
                    df['pair_id'] = df['pair_id'].astype(str)
                elif 'vehicle_id' in df.columns and 'chauffeur_id' in df.columns:
                    df['vehicle_id'] = pd.to_numeric(df['vehicle_id'], errors='coerce')
                    df['chauffeur_id'] = pd.to_numeric(df['chauffeur_id'], errors='coerce')
                    df = df.dropna(subset=['vehicle_id', 'chauffeur_id'])
                    
                    df['vehicle_id'] = df['vehicle_id'].astype(int).astype(str)
                    df['chauffeur_id'] = df['chauffeur_id'].astype(int).astype(str)
                    df['pair_id'] = df['chauffeur_id'] + '_' + df['vehicle_id']
                
                return {
                    'metadata': {k: v for k, v in data.items() if k != 'recommendations'},
                    'recommendations': df
                }
            
            return {'metadata': data, 'recommendations': pd.DataFrame()}
            
        except FileNotFoundError:
            logger.error(f"    Collaborative CF request file not found: {file_path}")
            return {'metadata': {}, 'recommendations': pd.DataFrame()}
        except Exception as e:
            logger.error(f"    Error loading Collaborative CF request data: {str(e)}")
            return {'metadata': {}, 'recommendations': pd.DataFrame()}
    
    @staticmethod
    def load_item_similarity(client_id: int, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load Item-Item Similarity recommendations from CSV
        """
        if file_path is None:
            file_path = Path(str(REQUEST_PATHS['item_item_csv']).format(client_id=client_id))
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"    Loaded Item-Item similarity data for client {client_id}: {len(df)} rows")
            
            required_cols = ['RefChauffeur', 'RefVehicle', 'similarity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in Item-Item similarity: {missing_cols}")
                return pd.DataFrame()
            
            # Convert IDs
            df['RefChauffeur'] = pd.to_numeric(df['RefChauffeur'], errors='coerce')
            df['RefVehicle'] = pd.to_numeric(df['RefVehicle'], errors='coerce')
            df = df.dropna(subset=['RefChauffeur', 'RefVehicle'])
            
            df['RefChauffeur'] = df['RefChauffeur'].astype(int).astype(str)
            df['RefVehicle'] = df['RefVehicle'].astype(int).astype(str)
            
            df['pair_id'] = df['RefChauffeur'] + '_' + df['RefVehicle']
            df['model'] = 'item_similarity'
            
            return df
            
        except FileNotFoundError:
            logger.warning(f"    Item-Item similarity file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"    Error loading Item-Item similarity data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def load_user_item_similarity(client_id: int, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load User-Item Context Similarity recommendations from CSV
        """
        if file_path is None:
            file_path = Path(str(REQUEST_PATHS['user_item_csv']).format(client_id=client_id))
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"    Loaded User-Item similarity data for client {client_id}: {len(df)} rows")
            
            required_cols = ['RefVehicle', 'RefChauffeur', 'similarity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in User-Item similarity: {missing_cols}")
                return pd.DataFrame()
            
            # Convert IDs
            df['RefVehicle'] = pd.to_numeric(df['RefVehicle'], errors='coerce')
            df['RefChauffeur'] = pd.to_numeric(df['RefChauffeur'], errors='coerce')
            df = df.dropna(subset=['RefVehicle', 'RefChauffeur'])
            
            df['RefVehicle'] = df['RefVehicle'].astype(int).astype(str)
            df['RefChauffeur'] = df['RefChauffeur'].astype(int).astype(str)
            
            # Create pair_id in CHAUFFEUR_VEHICLE format
            df['pair_id'] = df['RefChauffeur'] + '_' + df['RefVehicle']
            df['model'] = 'user_similarity'
            
            return df
            
        except FileNotFoundError:
            logger.warning(f"    User-Item similarity file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"    Error loading User-Item similarity data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def load_client_types(file_path: Optional[Path] = None) -> Dict[int, str]:
        """
        Load client types from client_final.csv
        """
        if file_path is None:
            file_path = CLIENT_TYPE_FILE
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"    Loaded client data: {len(df)} clients")
            
            if 'ID' not in df.columns or 'RefClientType' not in df.columns:
                logger.error("Required columns not found in client_final.csv")
                return {}
            
            client_types = {}
            
            for _, row in df.iterrows():
                client_id = row['ID']
                ref_client_type = row['RefClientType']
                
                if pd.isna(client_id) or pd.isna(ref_client_type):
                    continue
                
                try:
                    client_id = int(client_id)
                    ref_client_type = int(ref_client_type)
                except (ValueError, TypeError):
                    continue
                
                client_types[client_id] = CLIENT_TYPE_MAPPING.get(ref_client_type, 'UNKNOWN')
            
            return client_types
            
        except FileNotFoundError:
            logger.error(f"    Client file not found: {file_path}")
            return {}
        except Exception as e:
            logger.error(f"    Error loading client types: {str(e)}")
            return {}
    
    @staticmethod
    def load_all_request_data(client_id: int) -> Dict[str, pd.DataFrame]:
        """
        Load all 4 model outputs for a single client request
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading all recommendation data for client {client_id}")
        logger.info(f"{'='*60}")
        
        lightgcn = DataLoader.load_lightgcn_request(client_id)
        collaborative = DataLoader.load_collaborative_request(client_id)
        item_sim = DataLoader.load_item_similarity(client_id)
        user_sim = DataLoader.load_user_item_similarity(client_id)
        
        result = {
            'lightgcn': lightgcn['recommendations'] if isinstance(lightgcn, dict) else pd.DataFrame(),
            'collaborative': collaborative['recommendations'] if isinstance(collaborative, dict) else pd.DataFrame(),
            'item_similarity': item_sim,
            'user_similarity': user_sim,
            'metadata': {
                'lightgcn': lightgcn.get('metadata', {}),
                'collaborative': collaborative.get('metadata', {})
            }
        }
        
        logger.info("\nData Loading Summary:")
        logger.info(f"  LightGCN:        {len(result['lightgcn'])} recommendations")
        logger.info(f"  Collaborative:   {len(result['collaborative'])} recommendations")
        logger.info(f"  Item Similarity: {len(result['item_similarity'])} pairs")
        logger.info(f"  User Similarity: {len(result['user_similarity'])} pairs")
        logger.info(f"{'='*60}\n")
        
        return result
    
    


    @staticmethod
    def load_all_gridsearch_data(lightgcn_csv=None, collab_csv=None) -> Dict[str, pd.DataFrame]:
        """
        Load all model outputs for grid search (monthly training).

        Args:
            lightgcn_csv:  path to LightGCN recommendations CSV.
                           When None falls back to GRID_SEARCH_PATHS['lightgcn'].
            collab_csv:    path to CF recommendations CSV.
                           When None falls back to GRID_SEARCH_PATHS['collaborative'].
        """
        logger.info(f"\n{'='*60}")
        logger.info("Loading all grid search data")
        logger.info(f"{'='*60}")

        lightgcn = DataLoader.load_lightgcn_gridsearch(file_path=lightgcn_csv)
        collaborative = DataLoader.load_collaborative_gridsearch(file_path=collab_csv)
        
        result = {
            'lightgcn': lightgcn,
            'collaborative': collaborative
        }
        
        logger.info("\nGrid Search Data Summary:")
        logger.info(f"  LightGCN:      {len(result['lightgcn'])} recommendations")
        logger.info(f"  Collaborative: {len(result['collaborative'])} recommendations")
        
        if not lightgcn.empty:
            n_clients_gcn = lightgcn['original_client_id'].nunique()
            logger.info(f"  Unique clients in LightGCN: {n_clients_gcn}")
        
        if not collaborative.empty:
            n_clients_cf = collaborative['client_id'].nunique()
            logger.info(f"  Unique clients in Collaborative: {n_clients_cf}")
        
        logger.info(f"{'='*60}\n")
        
        return result
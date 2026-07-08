

import warnings
from typing import Dict, Any, Optional, Tuple, Set
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from base_classes import BaseProcessor, BasePipeline

class EdgeTypeCreator(BaseProcessor):
    """
    Creates different edge types for the tripartite graph.
    
    REPLACES: create_three_edge_types() [lines 20-110]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize edge type creator."""
        super().__init__(config)
    
    def process(self, data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process data to create edge types."""
        return self.create_all(
            data['bookings_enhanced'],
            data['vehicle_features']
        )
    
    def create_all(
        self,
        bookings_enhanced: pd.DataFrame,
        vehicle_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create three types of edges for the tripartite graph.
        
        REPLACES: create_three_edge_types() [lines 20-110]
        
        Args:
            bookings_enhanced: Enhanced bookings DataFrame
            vehicle_features: Vehicle features DataFrame
            
        Returns:
            Tuple of (cv_interactions, cc_interactions)
        """
        self.logger.debug("\n=== Step 3.1: Creating Three Edge Types ===")
        
        # Convert to pandas if dask
        if hasattr(bookings_enhanced, 'compute'):
            self.logger.debug("Computing bookings data...")
            bookings_pd = bookings_enhanced.compute()
        else:
            bookings_pd = bookings_enhanced
        
        self.logger.debug(f"Processing {len(bookings_pd):,} bookings")
        
        # Merge vehicle type information
        self.logger.debug("Merging vehicle type information...")
        bookings_with_vtype = bookings_pd.merge(
            vehicle_features[['ID', 'RefVehicleType']],
            left_on='RefVehicle',
            right_on='ID',
            how='left',
            suffixes=('', '_veh')
        )
        
        bookings_with_vtype = bookings_with_vtype.drop(columns=['ID_veh'], errors='ignore')
        
        self.logger.debug(f"Bookings with vehicle type: {len(bookings_with_vtype):,}")
        self.logger.debug(f"RefVehicleType null count: {bookings_with_vtype['RefVehicleType'].isna().sum()}")
        
        # Create Client-Vehicle interactions
        cv_interactions = self._create_cv_interactions(bookings_with_vtype)
        
        # Create Client-Chauffeur interactions
        cc_interactions = self._create_cc_interactions(bookings_pd)
        
        return cv_interactions, cc_interactions
    
    def _create_cv_interactions(self, bookings_df: pd.DataFrame) -> pd.DataFrame:
        """Create Client-Vehicle interaction edges."""
        self.logger.debug("Creating Client-Vehicle interactions...")
        
        cv_interactions = bookings_df.groupby(['RefClient', 'RefVehicle']).agg({
            'ID': 'count',
            'RefVehicleType': 'first',
            'BookingDate': ['min', 'max'],
            'PriceComputed': ['sum', 'mean'],
            'DateCompleted': lambda x: x.notna().sum(),
            'TotalDistance': 'mean',
            'NoOfPax': 'mean'
        }).reset_index()
        
        cv_interactions.columns = [
            'client_id', 'vehicle_id', 'frequency', 'vehicle_type',
            'first_booking', 'last_booking',
            'total_spent', 'avg_price',
            'completed', 'avg_distance', 'avg_pax'
        ]
        
        cv_interactions['completion_rate'] = (
            cv_interactions['completed'] / cv_interactions['frequency']
        ).fillna(0)
        
        self.logger.debug(f"  Client-Vehicle edges: {len(cv_interactions):,}")
        self.logger.debug(f"  Unique clients: {cv_interactions['client_id'].nunique():,}")
        self.logger.debug(f"  Unique vehicles: {cv_interactions['vehicle_id'].nunique():,}")
        self.logger.debug(f"  Vehicle type null: {cv_interactions['vehicle_type'].isna().sum()}")
        
        return cv_interactions
    
    def _create_cc_interactions(self, bookings_df: pd.DataFrame) -> pd.DataFrame:
        """Create Client-Chauffeur interaction edges."""
        self.logger.debug("Creating Client-Chauffeur interactions...")
        
        cc_interactions = bookings_df.groupby(['RefClient', 'RefChauffeur']).agg({
            'ID': 'count',
            'BookingDate': ['min', 'max'],
            'PriceComputed': ['sum', 'mean'],
            'DateCompleted': lambda x: x.notna().sum(),
            'TotalDistance': 'mean',
            'NoOfPax': 'mean'
        }).reset_index()
        
        cc_interactions.columns = [
            'client_id', 'chauffeur_id', 'frequency',
            'first_booking', 'last_booking',
            'total_spent', 'avg_price',
            'completed', 'avg_distance', 'avg_pax'
        ]
        
        cc_interactions['completion_rate'] = (
            cc_interactions['completed'] / cc_interactions['frequency']
        ).fillna(0)
        
        self.logger.debug(f"  Client-Chauffeur edges: {len(cc_interactions):,}")
        self.logger.debug(f"  Unique clients: {cc_interactions['client_id'].nunique():,}")
        self.logger.debug(f"  Unique chauffeurs: {cc_interactions['chauffeur_id'].nunique():,}")
        
        return cc_interactions


class EdgePatternAnalyzer:
    """
    Analyzes interaction patterns for each edge type.
    
    REPLACES: analyze_edge_patterns() [lines 112-156]
    """
    
    def __init__(self):
        """Initialize edge pattern analyzer."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(
        self,
        cv_interactions: pd.DataFrame,
        cc_interactions: pd.DataFrame,
        vc_cooccurrence: pd.DataFrame
    ) -> Tuple[int, int, int]:
        """
        Analyze interaction patterns and return recommended thresholds.
        
        REPLACES: analyze_edge_patterns() [lines 112-156]
        
        Args:
            cv_interactions: Client-Vehicle interactions
            cc_interactions: Client-Chauffeur interactions
            vc_cooccurrence: Vehicle-Chauffeur co-occurrences
            
        Returns:
            Tuple of (cv_threshold, cc_threshold, vc_threshold)
        """
        self.logger.info("\n=== Step 3.2: Analyzing Edge Patterns ===")
        
        # Client-Vehicle analysis
        self.logger.debug(f"\nClient-Vehicle edge statistics:")
        self.logger.debug(f"- Total edges: {len(cv_interactions):,}")
        self.logger.debug(f"- Mean frequency: {cv_interactions['frequency'].mean():.2f}")
        self.logger.debug(f"- Median frequency: {cv_interactions['frequency'].median():.2f}")
        self.logger.debug(f"- Edges with 1 interaction: {(cv_interactions['frequency'] == 1).sum():,}")
        self.logger.debug(f"- Edges with 2-5 interactions: {((cv_interactions['frequency'] >= 2) & (cv_interactions['frequency'] <= 5)).sum():,}")
        self.logger.debug(f"- Edges with 6+ interactions: {(cv_interactions['frequency'] >= 6).sum():,}")
        
        # Client-Chauffeur analysis
        self.logger.debug(f"\nClient-Chauffeur edge statistics:")
        self.logger.debug(f"- Total edges: {len(cc_interactions):,}")
        self.logger.debug(f"- Mean frequency: {cc_interactions['frequency'].mean():.2f}")
        self.logger.debug(f"- Median frequency: {cc_interactions['frequency'].median():.2f}")
        self.logger.debug(f"- Edges with 1 interaction: {(cc_interactions['frequency'] == 1).sum():,}")
        self.logger.debug(f"- Edges with 2-5 interactions: {((cc_interactions['frequency'] >= 2) & (cc_interactions['frequency'] <= 5)).sum():,}")
        self.logger.debug(f"- Edges with 6+ interactions: {(cc_interactions['frequency'] >= 6).sum():,}")
        
        # Vehicle-Chauffeur analysis
        self.logger.debug(f"\nVehicle-Chauffeur co-occurrence statistics:")
        self.logger.debug(f"- Total pairs: {len(vc_cooccurrence):,}")
        self.logger.debug(f"- Mean times together: {vc_cooccurrence['times_together'].mean():.2f}")
        self.logger.debug(f"- Median times together: {vc_cooccurrence['times_together'].median():.2f}")
        self.logger.debug(f"- Pairs with 1-2 trips: {(vc_cooccurrence['times_together'] <= 2).sum():,}")
        self.logger.debug(f"- Pairs with 3-10 trips: {((vc_cooccurrence['times_together'] >= 3) & (vc_cooccurrence['times_together'] <= 10)).sum():,}")
        self.logger.debug(f"- Pairs with 11+ trips: {(vc_cooccurrence['times_together'] >= 11).sum():,}")
        
        # Calculate recommended thresholds
        cv_threshold = max(2, int(cv_interactions['frequency'].quantile(0.25)))
        cc_threshold = max(2, int(cc_interactions['frequency'].quantile(0.25)))
        vc_threshold = max(3, int(vc_cooccurrence['times_together'].quantile(0.30)))
        
        self.logger.debug(f"\nRecommended filtering thresholds (25-30th percentile):")
        self.logger.debug(f"- Client-Vehicle minimum: {cv_threshold}")
        self.logger.debug(f"- Client-Chauffeur minimum: {cc_threshold}")
        self.logger.debug(f"- Vehicle-Chauffeur minimum: {vc_threshold}")
        
        return cv_threshold, cc_threshold, vc_threshold


class EdgeFilter:
    """
    Filters edges based on thresholds.
    
    REPLACES: filter_three_edge_types() [lines 158-330]
    
    FIXED: Uses UNION (not INTERSECTION) for valid nodes
    FIXED: Includes vc_filtered in final valid sets
    FIXED: Uses 10 max iterations (not 5)
    """
    
    def __init__(
        self,
        min_cv: Optional[int] = None,
        min_cc: Optional[int] = None,
        min_vc: Optional[int] = None
    ):
        """
        Initialize edge filter.
        
        Args:
            min_cv: Minimum Client-Vehicle interactions
            min_cc: Minimum Client-Chauffeur interactions
            min_vc: Minimum Vehicle-Chauffeur co-occurrences
        """
        self.min_cv = min_cv
        self.min_cc = min_cc
        self.min_vc = min_vc
        self.logger = logging.getLogger(self.__class__.__name__)
        self.analyzer = EdgePatternAnalyzer()
    
    def filter(
        self,
        cv_interactions: pd.DataFrame,
        cc_interactions: pd.DataFrame,
        vc_cooccurrence: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Set, Set, Set]:
        """
        Filter each edge type independently with iterative refinement.
        
        REPLACES: filter_three_edge_types() [lines 158-330]
        
        FIXED: Uses UNION for valid nodes (node valid if in ANY edge type)
        FIXED: Includes vc_filtered in final valid vehicle/chauffeur sets
        FIXED: Uses 10 max iterations
        
        Args:
            cv_interactions: Client-Vehicle interactions
            cc_interactions: Client-Chauffeur interactions
            vc_cooccurrence: Vehicle-Chauffeur co-occurrences
            
        Returns:
            Tuple of (cv_filtered, cc_filtered, vc_filtered, 
                     valid_clients, valid_vehicles, valid_chauffeurs)
        """
        self.logger.info("\n=== Step 3.3: Filtering Three Edge Types ===")
        
        # Auto-determine thresholds if not provided
        if self.min_cv is None or self.min_cc is None or self.min_vc is None:
            auto_cv, auto_cc, auto_vc = self.analyzer.analyze(
                cv_interactions, cc_interactions, vc_cooccurrence
            )
            self.min_cv = self.min_cv or auto_cv
            self.min_cc = self.min_cc or auto_cc
            self.min_vc = self.min_vc or auto_vc
        
        self.logger.debug(f"\nApplying filters:")
        self.logger.debug(f"- Client-Vehicle minimum interactions: {self.min_cv}")
        self.logger.debug(f"- Client-Chauffeur minimum interactions: {self.min_cc}")
        self.logger.debug(f"- Vehicle-Chauffeur minimum co-occurrences: {self.min_vc}")
        
        # Initial statistics
        self.logger.debug(f"\nOriginal edge counts:")
        self.logger.debug(f"- C-V edges: {len(cv_interactions):,}")
        self.logger.debug(f"- C-C edges: {len(cc_interactions):,}")
        self.logger.debug(f"- V-C edges: {len(vc_cooccurrence):,}")
        
        original_clients = set(cv_interactions['client_id']) | set(cc_interactions['client_id'])
        original_vehicles = set(cv_interactions['vehicle_id']) | set(vc_cooccurrence['RefVehicle'])
        original_chauffeurs = set(cc_interactions['chauffeur_id']) | set(vc_cooccurrence['RefChauffeur'])
        
        self.logger.debug(f"Original node counts:")
        self.logger.debug(f"- Clients: {len(original_clients):,}")
        self.logger.debug(f"- Vehicles: {len(original_vehicles):,}")
        self.logger.debug(f"- Chauffeurs: {len(original_chauffeurs):,}")
        
        # Iterative filtering
        iteration = 0
        max_iterations = 10  # FIXED: Was 5 in broken version
        prev_total_edges = 0
        
        cv_filtered = cv_interactions.copy()
        cc_filtered = cc_interactions.copy()
        vc_filtered = vc_cooccurrence.copy()
        
        while iteration < max_iterations:
            iteration += 1
            current_total = len(cv_filtered) + len(cc_filtered) + len(vc_filtered)
            
            if current_total == prev_total_edges:
                self.logger.debug(f"Converged after {iteration} iterations")
                break
            
            prev_total_edges = current_total
            
            # Filter by frequency thresholds
            cv_filtered = cv_filtered[cv_filtered['frequency'] >= self.min_cv]
            cc_filtered = cc_filtered[cc_filtered['frequency'] >= self.min_cc]
            vc_filtered = vc_filtered[vc_filtered['times_together'] >= self.min_vc]
            
            # FIXED: Get valid nodes using UNION (node valid if in ANY edge type)
            valid_clients = set(cv_filtered['client_id']) | set(cc_filtered['client_id'])
            valid_vehicles = set(cv_filtered['vehicle_id']) | set(vc_filtered['RefVehicle'])
            valid_chauffeurs = set(cc_filtered['chauffeur_id']) | set(vc_filtered['RefChauffeur'])
            
            # Filter edges to only include valid nodes
            cv_filtered = cv_filtered[
                cv_filtered['client_id'].isin(valid_clients) &
                cv_filtered['vehicle_id'].isin(valid_vehicles)
            ]
            
            cc_filtered = cc_filtered[
                cc_filtered['client_id'].isin(valid_clients) &
                cc_filtered['chauffeur_id'].isin(valid_chauffeurs)
            ]
            
            vc_filtered = vc_filtered[
                vc_filtered['RefVehicle'].isin(valid_vehicles) &
                vc_filtered['RefChauffeur'].isin(valid_chauffeurs)
            ]
            
            self.logger.debug(f"  Iteration {iteration}: C-V={len(cv_filtered):,}, C-C={len(cc_filtered):,}, V-C={len(vc_filtered):,}")
        
        # FIXED: Final counts using UNION and including vc_filtered
        final_clients = set(cv_filtered['client_id']) | set(cc_filtered['client_id'])
        final_vehicles = set(cv_filtered['vehicle_id']) | set(vc_filtered['RefVehicle'])
        final_chauffeurs = set(cc_filtered['chauffeur_id']) | set(vc_filtered['RefChauffeur'])
        
        self.logger.debug(f"\nFiltered edge counts:")
        self.logger.debug(f"- C-V edges: {len(cv_filtered):,} ({(1-len(cv_filtered)/len(cv_interactions))*100:.1f}% reduction)")
        self.logger.debug(f"- C-C edges: {len(cc_filtered):,} ({(1-len(cc_filtered)/len(cc_interactions))*100:.1f}% reduction)")
        self.logger.debug(f"- V-C edges: {len(vc_filtered):,} ({(1-len(vc_filtered)/len(vc_cooccurrence))*100:.1f}% reduction)")
        
        self.logger.debug(f"\nFiltered node counts:")
        self.logger.debug(f"- Clients: {len(final_clients):,} ({(1-len(final_clients)/len(original_clients))*100:.1f}% reduction)")
        self.logger.debug(f"- Vehicles: {len(final_vehicles):,} ({(1-len(final_vehicles)/len(original_vehicles))*100:.1f}% reduction)")
        self.logger.debug(f"- Chauffeurs: {len(final_chauffeurs):,} ({(1-len(final_chauffeurs)/len(original_chauffeurs))*100:.1f}% reduction)")
        
        return cv_filtered, cc_filtered, vc_filtered, final_clients, final_vehicles, final_chauffeurs


class NodeMappingCreator:
    """
    Creates sequential node ID mappings for the graph.
    
    REPLACES: create_three_node_mappings()
    
    FIXED: Added validation assertions
    """
    
    def __init__(self):
        """Initialize node mapping creator."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_mappings(
        self,
        valid_clients: Set,
        valid_vehicles: Set,
        valid_chauffeurs: Set
    ) -> Dict[str, Any]:
        """
        Create sequential node ID mappings.
        
        REPLACES: create_three_node_mappings()
        
        FIXED: Added validation assertions like original
        
        Args:
            valid_clients: Set of valid client IDs
            valid_vehicles: Set of valid vehicle IDs
            valid_chauffeurs: Set of valid chauffeur IDs
            
        Returns:
            Dictionary with mappings and counts
        """
        self.logger.info("\n=== Step 3.4: Creating Node ID Mappings ===")
        
        # Sort for consistent ordering
        sorted_clients = sorted(valid_clients)
        sorted_vehicles = sorted(valid_vehicles)
        sorted_chauffeurs = sorted(valid_chauffeurs)
        
        # Client IDs: 0 to n_clients-1
        client_to_id = {c: idx for idx, c in enumerate(sorted_clients)}
        n_clients = len(client_to_id)
        
        # Vehicle IDs: n_clients to n_clients+n_vehicles-1
        vehicle_to_id = {v: idx + n_clients for idx, v in enumerate(sorted_vehicles)}
        n_vehicles = len(sorted_vehicles)
        
        # Chauffeur IDs: n_clients+n_vehicles to n_clients+n_vehicles+n_chauffeurs-1
        chauffeur_to_id = {ch: idx + n_clients + n_vehicles 
                          for idx, ch in enumerate(sorted_chauffeurs)}
        n_chauffeurs = len(sorted_chauffeurs)
        
        # Reverse mappings
        id_to_client = {v: k for k, v in client_to_id.items()}
        id_to_vehicle = {v: k for k, v in vehicle_to_id.items()}
        id_to_chauffeur = {v: k for k, v in chauffeur_to_id.items()}
        
        n_total_nodes = n_clients + n_vehicles + n_chauffeurs
        
        self.logger.debug(f"Node ID ranges:")
        self.logger.debug(f"- Clients: 0 to {n_clients - 1} ({n_clients} nodes)")
        self.logger.debug(f"- Vehicles: {n_clients} to {n_clients + n_vehicles - 1} ({n_vehicles} nodes)")
        self.logger.debug(f"- Chauffeurs: {n_clients + n_vehicles} to {n_total_nodes - 1} ({n_chauffeurs} nodes)")
        self.logger.debug(f"- Total nodes: {n_total_nodes}")
        
        # FIXED: Add validation assertions like original
        assert len(client_to_id) == len(id_to_client) == n_clients
        assert len(vehicle_to_id) == len(id_to_vehicle) == n_vehicles
        assert len(chauffeur_to_id) == len(id_to_chauffeur) == n_chauffeurs
        
        if n_clients > 0:
            assert max(client_to_id.values()) == n_clients - 1
        if n_vehicles > 0:
            assert min(vehicle_to_id.values()) == n_clients
            assert max(vehicle_to_id.values()) == n_clients + n_vehicles - 1
        if n_chauffeurs > 0:
            assert min(chauffeur_to_id.values()) == n_clients + n_vehicles
        
        self.logger.info("ID mappings validated successfully!")
        
        return {
            'client_to_id': client_to_id,
            'vehicle_to_id': vehicle_to_id,
            'chauffeur_to_id': chauffeur_to_id,
            'id_to_client': id_to_client,
            'id_to_vehicle': id_to_vehicle,
            'id_to_chauffeur': id_to_chauffeur,
            'n_clients': n_clients,
            'n_vehicles': n_vehicles,
            'n_chauffeurs': n_chauffeurs,
            'n_total_nodes': n_total_nodes
        }


class EdgeMapper:
    """
    Applies ID mappings to edges.
    
    REPLACES: apply_id_mappings_to_edges()
    
    FIXED: Added dropna, warning messages, and int casting
    """
    
    def __init__(self):
        """Initialize edge mapper."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def apply_mappings(
        self,
        cv_filtered: pd.DataFrame,
        cc_filtered: pd.DataFrame,
        vc_filtered: pd.DataFrame,
        mappings: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply ID mappings to all edge types.
        
        REPLACES: apply_id_mappings_to_edges()
        
        FIXED: Added dropna, warning messages, and int casting like original
        
        Args:
            cv_filtered: Filtered Client-Vehicle edges
            cc_filtered: Filtered Client-Chauffeur edges
            vc_filtered: Filtered Vehicle-Chauffeur edges
            mappings: Node ID mappings
            
        Returns:
            Tuple of mapped DataFrames
        """
        self.logger.info("\n=== Step 3.5: Applying ID Mappings to Edges ===")
        
        client_to_id = mappings['client_to_id']
        vehicle_to_id = mappings['vehicle_to_id']
        chauffeur_to_id = mappings['chauffeur_to_id']
        
        # Map Client-Vehicle edges
        cv_mapped = cv_filtered.copy()
        cv_mapped['client_node_id'] = cv_mapped['client_id'].map(client_to_id)
        cv_mapped['vehicle_node_id'] = cv_mapped['vehicle_id'].map(vehicle_to_id)
        
        # FIXED: Remove any failed mappings and warn
        before = len(cv_mapped)
        cv_mapped = cv_mapped.dropna(subset=['client_node_id', 'vehicle_node_id'])
        after = len(cv_mapped)
        if before != after:
            self.logger.debug(f"Warning: {before - after} C-V edges lost during mapping")
        
        # FIXED: Cast to int
        cv_mapped['client_node_id'] = cv_mapped['client_node_id'].astype(int)
        cv_mapped['vehicle_node_id'] = cv_mapped['vehicle_node_id'].astype(int)
        
        self.logger.debug(f"Client-Vehicle edges mapped: {len(cv_mapped):,}")
        
        # Map Client-Chauffeur edges
        cc_mapped = cc_filtered.copy()
        cc_mapped['client_node_id'] = cc_mapped['client_id'].map(client_to_id)
        cc_mapped['chauffeur_node_id'] = cc_mapped['chauffeur_id'].map(chauffeur_to_id)
        
        # FIXED: Remove any failed mappings and warn
        before = len(cc_mapped)
        cc_mapped = cc_mapped.dropna(subset=['client_node_id', 'chauffeur_node_id'])
        after = len(cc_mapped)
        if before != after:
            self.logger.debug(f"Warning: {before - after} C-C edges lost during mapping")
        
        # FIXED: Cast to int
        cc_mapped['client_node_id'] = cc_mapped['client_node_id'].astype(int)
        cc_mapped['chauffeur_node_id'] = cc_mapped['chauffeur_node_id'].astype(int)
        
        self.logger.debug(f"Client-Chauffeur edges mapped: {len(cc_mapped):,}")
        
        # Map Vehicle-Chauffeur edges
        vc_mapped = vc_filtered.copy()
        vc_mapped['vehicle_node_id'] = vc_mapped['RefVehicle'].map(vehicle_to_id)
        vc_mapped['chauffeur_node_id'] = vc_mapped['RefChauffeur'].map(chauffeur_to_id)
        
        # FIXED: Remove any failed mappings and warn
        before = len(vc_mapped)
        vc_mapped = vc_mapped.dropna(subset=['vehicle_node_id', 'chauffeur_node_id'])
        after = len(vc_mapped)
        if before != after:
            self.logger.debug(f"Warning: {before - after} V-C edges lost during mapping")
        
        # FIXED: Cast to int
        vc_mapped['vehicle_node_id'] = vc_mapped['vehicle_node_id'].astype(int)
        vc_mapped['chauffeur_node_id'] = vc_mapped['chauffeur_node_id'].astype(int)
        
        self.logger.debug(f"Vehicle-Chauffeur edges mapped: {len(vc_mapped):,}")
        
        return cv_mapped, cc_mapped, vc_mapped


class TrainTestSplitter:
    """
    Creates train/test splits for graph data.
    
    REPLACES: create_three_way_splits()
    
    FIXED: Added fallback for missing last_booking column
    FIXED: Added random seed for reproducibility
    FIXED: Added data leakage checks
    """
    
    def __init__(self, split_ratio: float = 0.8, method: str = 'temporal'):
        """
        Initialize train/test splitter.
        
        Args:
            split_ratio: Ratio of training data
            method: Split method ('temporal' or 'random')
        """
        self.split_ratio = split_ratio
        self.method = method
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def split(
        self,
        cv_mapped: pd.DataFrame,
        cc_mapped: pd.DataFrame,
        vc_mapped: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Create train/test splits for all edge types.
        
        REPLACES: create_three_way_splits()
        
        FIXED: Added fallback for missing last_booking
        FIXED: Added random seed for reproducibility
        FIXED: Added data leakage checks
        
        Args:
            cv_mapped: Mapped Client-Vehicle edges
            cc_mapped: Mapped Client-Chauffeur edges
            vc_mapped: Mapped Vehicle-Chauffeur edges
            
        Returns:
            Dictionary with train/test splits
        """
        self.logger.info(f"\n=== Step 3.6: Creating Data Splits ({self.method}) ===")
        
        if self.method == 'temporal':
            self.logger.debug("\n  Using PROPER temporal split (by date cutoff, not edge count)")
            
            # Client-Vehicle edges
            self.logger.debug("\nSplitting Client-Vehicle edges...")
            cv_train, cv_test = self._temporal_split_with_fallback(cv_mapped, 'last_booking')
            
            # Client-Chauffeur edges
            self.logger.debug("\nSplitting Client-Chauffeur edges...")
            cc_train, cc_test = self._temporal_split_with_fallback(cc_mapped, 'last_booking')
        else:
            # Random split
            self.logger.debug("\nUsing random split...")
            cv_train, cv_test = self._random_split(cv_mapped)
            cc_train, cc_test = self._random_split(cc_mapped)
            
            self.logger.debug(f"  C-V: Train {len(cv_train):,}, Test {len(cv_test):,}")
            self.logger.debug(f"  C-C: Train {len(cc_train):,}, Test {len(cc_test):,}")
        
        # Vehicle-Chauffeur: ALL to training (no temporal split needed)
        self.logger.debug("\nVehicle-Chauffeur edges...")
        vc_train = vc_mapped.copy()
        vc_test = pd.DataFrame()
        self.logger.debug(f"  All {len(vc_train):,} edges used for training (no test split)")
        self.logger.debug("  Rationale: V-C co-occurrence helps model learn, not for evaluation")
        
        # FIXED: Data leakage checks
        self._check_data_leakage(cv_train, cv_test, cc_train, cc_test)
        
        # Summary statistics
        total_train_edges = len(cv_train) + len(cc_train) + len(vc_train)
        total_test_edges = len(cv_test) + len(cc_test)
        
        self.logger.debug(f"\n{'='*60}")
        self.logger.debug("SPLIT SUMMARY")
        self.logger.debug(f"{'='*60}")
        self.logger.debug(f"Training edges: {total_train_edges:,}")
        self.logger.debug(f"  C-V: {len(cv_train):,}")
        self.logger.debug(f"  C-C: {len(cc_train):,}")
        self.logger.debug(f"  V-C: {len(vc_train):,}")
        
        self.logger.debug(f"\nTest edges: {total_test_edges:,}")
        self.logger.debug(f"  C-V: {len(cv_test):,}")
        self.logger.debug(f"  C-C: {len(cc_test):,}")
        self.logger.debug(f"  V-C: 0 (not evaluated)")
        
        # Node coverage
        train_clients = set(cv_train['client_node_id']) | set(cc_train['client_node_id'])
        train_vehicles = set(cv_train['vehicle_node_id']) | set(vc_train['vehicle_node_id'])
        train_chauffeurs = set(cc_train['chauffeur_node_id']) | set(vc_train['chauffeur_node_id'])
        
        self.logger.debug(f"\nTraining node coverage:")
        self.logger.debug(f"  Clients: {len(train_clients):,}")
        self.logger.debug(f"  Vehicles: {len(train_vehicles):,}")
        self.logger.debug(f"  Chauffeurs: {len(train_chauffeurs):,}")
        
        return {
            'cv_train': cv_train,
            'cv_test': cv_test,
            'cc_train': cc_train,
            'cc_test': cc_test,
            'vc_train': vc_train,
            'vc_test': vc_test
        }
    
    def _temporal_split_with_fallback(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by date using quantile cutoff, with fallback to random split.
        
        FIXED: Added fallback for missing date column
        """
        if date_column in df.columns:
            cutoff = df[date_column].quantile(self.split_ratio)
            train = df[df[date_column] <= cutoff].copy()
            test = df[df[date_column] > cutoff].copy()
            
            self.logger.debug(f"  Cutoff date: {cutoff}")
            if hasattr(cutoff, 'date'):
                self.logger.debug(f"  Train: {len(train):,} edges (last booking ≤ {cutoff.date()})")
                self.logger.debug(f"  Test: {len(test):,} edges (last booking > {cutoff.date()})")
            else:
                self.logger.debug(f"  Train: {len(train):,}, Test: {len(test):,}")
            
            return train, test
        else:
            # FIXED: Fallback to random split with warning
            self.logger.debug(f"  WARNING: No '{date_column}' column, falling back to random split")
            return self._random_split(df)
    
    def _random_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Random split with seed for reproducibility.
        
        FIXED: Added random_state for reproducibility
        """
        train = df.sample(frac=self.split_ratio, random_state=42)
        test = df.drop(train.index)
        self.logger.debug(f"  Train: {len(train):,}, Test: {len(test):,}")
        return train, test
    
    def _check_data_leakage(
        self,
        cv_train: pd.DataFrame,
        cv_test: pd.DataFrame,
        cc_train: pd.DataFrame,
        cc_test: pd.DataFrame
    ) -> None:
        """
        Check for temporal data leakage.
        
        FIXED: Added data leakage checks from original
        """
        self.logger.debug("\n🔍 Checking for data leakage...")
        
        if self.method == 'temporal':
            # Check C-V
            if 'last_booking' in cv_train.columns and len(cv_test) > 0:
                cv_train_max = cv_train['last_booking'].max()
                cv_test_min = cv_test['last_booking'].min()
                
                if hasattr(cv_train_max, 'date'):
                    self.logger.debug(f"\n  C-V: Train max date = {cv_train_max.date()}")
                    self.logger.debug(f"       Test min date = {cv_test_min.date()}")
                
                if cv_train_max <= cv_test_min:
                    self.logger.debug("    No temporal leakage in C-V split")
                else:
                    overlap = (cv_train['last_booking'] > cv_test_min).sum()
                    self.logger.debug(f"    WARNING: {overlap} C-V train edges overlap with test period!")
            
            # Check C-C
            if 'last_booking' in cc_train.columns and len(cc_test) > 0:
                cc_train_max = cc_train['last_booking'].max()
                cc_test_min = cc_test['last_booking'].min()
                
                if hasattr(cc_train_max, 'date'):
                    self.logger.debug(f"\n  C-C: Train max date = {cc_train_max.date()}")
                    self.logger.debug(f"       Test min date = {cc_test_min.date()}")
                
                if cc_train_max <= cc_test_min:
                    self.logger.debug("    No temporal leakage in C-C split")
                else:
                    overlap = (cc_train['last_booking'] > cc_test_min).sum()
                    self.logger.debug(f"    WARNING: {overlap} C-C train edges overlap with test period!")


class GraphValidator:
    """
    Validates graph structure and quality.
    
    REPLACES: validate_three_way_graph()
    
    FIXED: Uses coverage-based quality score (like original)
    FIXED: Returns all coverage metrics
    """
    
    def __init__(self):
        """Initialize graph validator."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(
        self,
        cv_mapped: pd.DataFrame,
        cc_mapped: pd.DataFrame,
        vc_mapped: pd.DataFrame,
        mappings: Dict[str, Any],
        vehicle_features: Optional[pd.DataFrame] = None,
        chauffeur_features: Optional[pd.DataFrame] = None,
        client_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Validate graph structure and return quality metrics.
        
        REPLACES: validate_three_way_graph()
        
        FIXED: Uses coverage-based quality score (like original)
        FIXED: Returns all coverage metrics
        
        Args:
            cv_mapped: Mapped Client-Vehicle edges
            cc_mapped: Mapped Client-Chauffeur edges
            vc_mapped: Mapped Vehicle-Chauffeur edges
            mappings: Node ID mappings
            vehicle_features: Optional vehicle features
            chauffeur_features: Optional chauffeur features
            client_features: Optional client features
            
        Returns:
            Dictionary with validation results including all coverage metrics
        """
        self.logger.info("\n=== Step 3.7: Graph Validation ===")
        
        validation = {}
        
        # Check node coverage
        cv_clients = set(cv_mapped['client_node_id'])
        cc_clients = set(cc_mapped['client_node_id'])
        all_clients_in_edges = cv_clients | cc_clients
        
        cv_vehicles = set(cv_mapped['vehicle_node_id'])
        vc_vehicles = set(vc_mapped['vehicle_node_id'])
        all_vehicles_in_edges = cv_vehicles | vc_vehicles
        
        cc_chauffeurs = set(cc_mapped['chauffeur_node_id'])
        vc_chauffeurs = set(vc_mapped['chauffeur_node_id'])
        all_chauffeurs_in_edges = cc_chauffeurs | vc_chauffeurs
        
        self.logger.debug(f"Node coverage in edges:")
        self.logger.debug(f"- Clients: {len(all_clients_in_edges)}/{mappings['n_clients']} ({len(all_clients_in_edges)/mappings['n_clients']*100:.1f}%)")
        self.logger.debug(f"- Vehicles: {len(all_vehicles_in_edges)}/{mappings['n_vehicles']} ({len(all_vehicles_in_edges)/mappings['n_vehicles']*100:.1f}%)")
        self.logger.debug(f"- Chauffeurs: {len(all_chauffeurs_in_edges)}/{mappings['n_chauffeurs']} ({len(all_chauffeurs_in_edges)/mappings['n_chauffeurs']*100:.1f}%)")
        
        # FIXED: Store coverage metrics like original
        validation['client_coverage'] = len(all_clients_in_edges) / mappings['n_clients']
        validation['vehicle_coverage'] = len(all_vehicles_in_edges) / mappings['n_vehicles']
        validation['chauffeur_coverage'] = len(all_chauffeurs_in_edges) / mappings['n_chauffeurs']
        
        # Feature coverage
        if vehicle_features is not None:
            vehicle_ids_with_features = set(vehicle_features['ID'].values)
            valid_vehicles_original = set(mappings['id_to_vehicle'].values())
            feature_coverage_v = len(vehicle_ids_with_features & valid_vehicles_original) / len(valid_vehicles_original)
            self.logger.debug(f"\nVehicle feature coverage: {feature_coverage_v*100:.1f}%")
            validation['vehicle_feature_coverage'] = feature_coverage_v
        
        if chauffeur_features is not None:
            chauffeur_ids_with_features = set(chauffeur_features['ID'].values)
            valid_chauffeurs_original = set(mappings['id_to_chauffeur'].values())
            feature_coverage_c = len(chauffeur_ids_with_features & valid_chauffeurs_original) / len(valid_chauffeurs_original)
            self.logger.debug(f"Chauffeur feature coverage: {feature_coverage_c*100:.1f}%")
            validation['chauffeur_feature_coverage'] = feature_coverage_c
        
        if client_features is not None:
            client_ids_with_features = set(client_features['ID'].values)
            valid_clients_original = set(mappings['id_to_client'].values())
            feature_coverage_cl = len(client_ids_with_features & valid_clients_original) / len(valid_clients_original)
            self.logger.debug(f"Client feature coverage: {feature_coverage_cl*100:.1f}%")
            validation['client_feature_coverage'] = feature_coverage_cl
        
        # Graph connectivity check
        self.logger.debug(f"\nGraph connectivity:")
        self.logger.debug(f"- C-V edges: {len(cv_mapped):,}")
        self.logger.debug(f"- C-C edges: {len(cc_mapped):,}")
        self.logger.debug(f"- V-C edges: {len(vc_mapped):,}")
        self.logger.debug(f"- Total edges: {len(cv_mapped) + len(cc_mapped) + len(vc_mapped):,}")
        
        # Density calculations
        cv_density = len(cv_mapped) / (mappings['n_clients'] * mappings['n_vehicles'])
        cc_density = len(cc_mapped) / (mappings['n_clients'] * mappings['n_chauffeurs'])
        vc_density = len(vc_mapped) / (mappings['n_vehicles'] * mappings['n_chauffeurs'])
        
        self.logger.debug(f"\nEdge densities:")
        self.logger.debug(f"- C-V density: {cv_density:.6f} ({cv_density*100:.4f}%)")
        self.logger.debug(f"- C-C density: {cc_density:.6f} ({cc_density*100:.4f}%)")
        self.logger.debug(f"- V-C density: {vc_density:.6f} ({vc_density*100:.4f}%)")
        
        validation['cv_density'] = cv_density
        validation['cc_density'] = cc_density
        validation['vc_density'] = vc_density
        
        # FIXED: Quality score based on coverage (like original)
        quality_score = (
            validation['client_coverage'] * 0.2 +
            validation['vehicle_coverage'] * 0.2 +
            validation['chauffeur_coverage'] * 0.2 +
            validation.get('vehicle_feature_coverage', 0) * 0.15 +
            validation.get('chauffeur_feature_coverage', 0) * 0.15 +
            validation.get('client_feature_coverage', 0) * 0.10
        )
        
        validation['overall_quality_score'] = quality_score
        
        self.logger.debug(f"\nOverall graph quality score: {quality_score:.3f}/1.000")
        
        if quality_score >= 0.85:
            self.logger.info(" Excellent graph quality - ready for model training")
        elif quality_score >= 0.70:
            self.logger.debug(" Good graph quality - suitable for model training")
        elif quality_score >= 0.55:
            self.logger.debug(" Moderate graph quality - consider additional filtering")
        else:
            self.logger.debug(" Low graph quality - significant issues detected")
        
        return validation


class DataFilteringPipeline(BasePipeline):
    """
    Orchestrates the data filtering process.
    
    REPLACES: execute_step3_three_way() [lines 627-730]
    
    FIXED: summary_stats includes all train/test count keys
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data filtering pipeline.
        
        Args:
            config: Configuration with min_cv, min_cc, min_vc, split_ratio, split_method
        """
        super().__init__(config or {})
        
        self.edge_creator = EdgeTypeCreator(config)
        self.edge_filter = EdgeFilter(
            min_cv=self.config.get('min_cv'),
            min_cc=self.config.get('min_cc'),
            min_vc=self.config.get('min_vc')
        )
        self.mapping_creator = NodeMappingCreator()
        self.edge_mapper = EdgeMapper()
        self.splitter = TrainTestSplitter(
            split_ratio=self.config.get('split_ratio', 0.8),
            method=self.config.get('split_method', 'temporal')
        )
        self.validator = GraphValidator()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data filtering pipeline.
        
        REPLACES: execute_step3_three_way() [lines 627-730]
        
        FIXED: summary_stats includes all train/test count keys
        
        Args:
            input_data: Step 2 results
            
        Returns:
            Dictionary containing step 3 outputs
        """
        self._log_start("Step 3: Data Filtering for Three-Way Graph")
        
        try:
            # Get data from step 2
            bookings_enhanced = input_data['bookings_enhanced']
            vc_cooccurrence = input_data['vc_cooccurrence']
            vehicle_features = input_data.get('vehicle_features')
            chauffeur_features = input_data.get('chauffeur_features')
            client_features = input_data.get('client_features')
            
            # Step 3.1: Create edge types
            cv_interactions, cc_interactions = self.edge_creator.create_all(
                bookings_enhanced, vehicle_features
            )
            
            # Step 3.2 & 3.3: Filter edges
            cv_filtered, cc_filtered, vc_filtered, valid_clients, valid_vehicles, valid_chauffeurs = \
                self.edge_filter.filter(cv_interactions, cc_interactions, vc_cooccurrence)
            
            # Step 3.4: Create mappings
            mappings = self.mapping_creator.create_mappings(
                valid_clients, valid_vehicles, valid_chauffeurs
            )
            
            # Step 3.5: Apply mappings
            cv_mapped, cc_mapped, vc_mapped = self.edge_mapper.apply_mappings(
                cv_filtered, cc_filtered, vc_filtered, mappings
            )
            
            # Step 3.6: Create splits
            splits = self.splitter.split(cv_mapped, cc_mapped, vc_mapped)
            
            # Step 3.7: Validate
            validation = self.validator.validate(
                cv_mapped, cc_mapped, vc_mapped, mappings,
                vehicle_features, chauffeur_features, client_features
            )
            
            # FIXED: Prepare summary statistics with all train/test count keys
            summary_stats = {
                'n_clients': mappings['n_clients'],
                'n_vehicles': mappings['n_vehicles'],
                'n_chauffeurs': mappings['n_chauffeurs'],
                'n_total_nodes': mappings['n_total_nodes'],
                'n_cv_edges': len(cv_mapped),
                'n_cc_edges': len(cc_mapped),
                'n_vc_edges': len(vc_mapped),
                'n_total_edges': len(cv_mapped) + len(cc_mapped) + len(vc_mapped),
                # FIXED: Added missing train/test count keys
                'n_train_cv': len(splits['cv_train']),
                'n_test_cv': len(splits['cv_test']),
                'n_train_cc': len(splits['cc_train']),
                'n_test_cc': len(splits['cc_test']),
                'n_train_vc': len(splits['vc_train']),
                'cv_density': validation['cv_density'],
                'cc_density': validation['cc_density'],
                'vc_density': validation['vc_density']
            }
            
            # Combine outputs
            step3_outputs = {
                **input_data,
                'cv_interactions': cv_interactions,
                'cc_interactions': cc_interactions,
                'cv_filtered': cv_filtered,
                'cc_filtered': cc_filtered,
                'vc_filtered': vc_filtered,
                'cv_mapped': cv_mapped,
                'cc_mapped': cc_mapped,
                'vc_mapped': vc_mapped,
                'mappings': mappings,
                'splits': splits,
                'validation': validation,
                'summary_stats': summary_stats
            }
            
            self._log_summary(step3_outputs)
            self._log_complete("Step 3")
            
            return step3_outputs
            
        except Exception as e:
            self.logger.error(f"Error in Step 3: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_summary(self, outputs: Dict[str, Any]) -> None:
        """Log step 3 summary."""
        mappings = outputs['mappings']
        splits = outputs['splits']
        validation = outputs['validation']
        
        self.logger.info("\n" + "=" * 60)
        self.logger.debug("Step 3 completed successfully!")
        self.logger.debug("\nThree-Way Graph Summary:")
        self.logger.debug(f"\nNodes:")
        self.logger.debug(f"  Clients: {mappings['n_clients']:,} (IDs: 0-{mappings['n_clients']-1})")
        self.logger.debug(f"  Vehicles: {mappings['n_vehicles']:,} (IDs: {mappings['n_clients']}-{mappings['n_clients']+mappings['n_vehicles']-1})")
        self.logger.debug(f"  Chauffeurs: {mappings['n_chauffeurs']:,} (IDs: {mappings['n_clients']+mappings['n_vehicles']}-{mappings['n_total_nodes']-1})")
        self.logger.debug(f"  Total: {mappings['n_total_nodes']:,}")
        
        self.logger.debug(f"\nEdges:")
        self.logger.debug(f"  Client-Vehicle: {len(outputs['cv_mapped']):,}")
        self.logger.debug(f"  Client-Chauffeur: {len(outputs['cc_mapped']):,}")
        self.logger.debug(f"  Vehicle-Chauffeur: {len(outputs['vc_mapped']):,}")
        self.logger.debug(f"  Total: {outputs['summary_stats']['n_total_edges']:,}")
        
        self.logger.debug(f"\nTraining/Test Split:")
        self.logger.debug(f"  C-V Train: {len(splits['cv_train']):,}, Test: {len(splits['cv_test']):,}")
        self.logger.debug(f"  C-C Train: {len(splits['cc_train']):,}, Test: {len(splits['cc_test']):,}")
        self.logger.debug(f"  V-C Train: {len(splits['vc_train']):,} (no test split)")
        
        self.logger.debug(f"\nGraph Quality Score: {validation['overall_quality_score']:.3f}/1.000")
        self.logger.info("\nReady for Step 4: Graph Construction & Matrix Creation!")


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def create_three_edge_types(bookings_enhanced, vehicle_features):
    """Backward compatible wrapper for EdgeTypeCreator.create_all()."""
    creator = EdgeTypeCreator()
    return creator.create_all(bookings_enhanced, vehicle_features)


def analyze_edge_patterns(cv_interactions, cc_interactions, vc_cooccurrence):
    """Backward compatible wrapper for EdgePatternAnalyzer.analyze()."""
    analyzer = EdgePatternAnalyzer()
    return analyzer.analyze(cv_interactions, cc_interactions, vc_cooccurrence)


def filter_three_edge_types(cv_interactions, cc_interactions, vc_cooccurrence,
                            min_cv=None, min_cc=None, min_vc=None):
    """Backward compatible wrapper for EdgeFilter.filter()."""
    edge_filter = EdgeFilter(min_cv, min_cc, min_vc)
    return edge_filter.filter(cv_interactions, cc_interactions, vc_cooccurrence)


def create_three_node_mappings(valid_clients, valid_vehicles, valid_chauffeurs):
    """Backward compatible wrapper for NodeMappingCreator.create_mappings()."""
    creator = NodeMappingCreator()
    return creator.create_mappings(valid_clients, valid_vehicles, valid_chauffeurs)


def apply_id_mappings_to_edges(cv_filtered, cc_filtered, vc_filtered, mappings):
    """Backward compatible wrapper for EdgeMapper.apply_mappings()."""
    mapper = EdgeMapper()
    return mapper.apply_mappings(cv_filtered, cc_filtered, vc_filtered, mappings)


def create_three_way_splits(cv_mapped, cc_mapped, vc_mapped, split_ratio=0.8, split_method='temporal'):
    """Backward compatible wrapper for TrainTestSplitter.split()."""
    splitter = TrainTestSplitter(split_ratio, split_method)
    return splitter.split(cv_mapped, cc_mapped, vc_mapped)


def validate_three_way_graph(cv_mapped, cc_mapped, vc_mapped, mappings,
                             vehicle_features=None, chauffeur_features=None, client_features=None):
    """Backward compatible wrapper for GraphValidator.validate()."""
    validator = GraphValidator()
    return validator.validate(cv_mapped, cc_mapped, vc_mapped, mappings,
                              vehicle_features, chauffeur_features, client_features)


def execute_step3_three_way(step2_results, min_cv=None, min_cc=None, min_vc=None,
                            split_ratio=0.8, split_method='temporal'):
    """Backward compatible wrapper for DataFilteringPipeline.execute()."""
    config = {
        'min_cv': min_cv,
        'min_cc': min_cc,
        'min_vc': min_vc,
        'split_ratio': split_ratio,
        'split_method': split_method
    }
    pipeline = DataFilteringPipeline(config)
    return pipeline.execute(step2_results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    logger.debug("Data Filtering Module (Refactored - Corrected) - Ready")
    logger.debug("\nClasses available:")
    logger.debug("  - EdgeTypeCreator")
    logger.debug("  - EdgePatternAnalyzer")
    logger.debug("  - EdgeFilter")
    logger.debug("  - NodeMappingCreator")
    logger.debug("  - EdgeMapper")
    logger.debug("  - TrainTestSplitter")
    logger.debug("  - GraphValidator")
    logger.debug("  - DataFilteringPipeline")
    logger.debug("\nBackward compatible functions:")
    logger.debug("  - create_three_edge_types()")
    logger.debug("  - analyze_edge_patterns()")
    logger.debug("  - filter_three_edge_types()")
    logger.debug("  - create_three_node_mappings()")
    logger.debug("  - apply_id_mappings_to_edges()")
    logger.debug("  - create_three_way_splits()")
    logger.debug("  - validate_three_way_graph()")
    logger.debug("  - execute_step3_three_way()")
    logger.debug("\n" + "="*80)
    logger.debug("DEMONSTRATING TEMPORAL SPLIT (Quantile-Based Date Cutoff)")
    logger.debug("="*80)
    
    # Create sample data
    np.random.seed(42)
    n_edges = 1000
    
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=int(x)) for x in np.random.exponential(180, n_edges)]
    
    sample_df = pd.DataFrame({
        'client_node_id': np.random.randint(0, 100, n_edges),
        'vehicle_node_id': np.random.randint(100, 200, n_edges),
        'last_booking': dates,
        'frequency': np.random.randint(1, 20, n_edges)
    })
    
    # Demonstrate the quantile-based temporal split
    logger.debug("\n  Quantile-Based Temporal Split (80th percentile date cutoff):")
    cutoff = sample_df['last_booking'].quantile(0.8)
    train_split = sample_df[sample_df['last_booking'] <= cutoff]
    test_split = sample_df[sample_df['last_booking'] > cutoff]
    
    logger.debug(f"   Cutoff date: {cutoff.date()}")
    logger.debug(f"   Train: {len(train_split)} edges (all bookings ≤ {cutoff.date()})")
    logger.debug(f"   Train date range: {train_split['last_booking'].min().date()} to {train_split['last_booking'].max().date()}")
    logger.debug(f"   Test: {len(test_split)} edges (all bookings > {cutoff.date()})")
    logger.debug(f"   Test date range: {test_split['last_booking'].min().date()} to {test_split['last_booking'].max().date()}")
    
    # Verify no temporal leakage
    overlap = (train_split['last_booking'] > test_split['last_booking'].min()).sum()
    if overlap == 0:
        logger.debug(f"     NO LEAKAGE: All train edges are strictly before all test edges!")
    else:
        logger.debug(f"     LEAKAGE: {overlap} train edges overlap with test period")
    
    logger.debug("\n This is the approach used in create_three_way_splits()")
    logger.debug("="*80)
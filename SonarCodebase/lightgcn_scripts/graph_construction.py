import warnings
from typing import Dict, Any, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

import json

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from base_classes import BasePipeline
from config import GRAPH_OUTPUT_DIR, GRAPH_OUTPUT_PREFIX


class EdgeWeightCalculator:
    """
    Calculates edge weights for different edge types.
    
    REPLACES: calculate_cv_edge_weights(), calculate_cc_edge_weights(), calculate_vc_edge_weights()
    """
    
    def __init__(self):
        """Initialize edge weight calculator."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_cv_weights(self, cv_train: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate edge weights for Client-Vehicle interactions.
        
        REPLACES: calculate_cv_edge_weights()
        
        Args:
            cv_train: Client-Vehicle training edges
            
        Returns:
            DataFrame with added weight column
        """
        self.logger.info("\n=== Step 4.1: Client-Vehicle Edge Weights ===")
        
        self.logger.debug(f"Calculating weights for {len(cv_train):,} C-V edges")
        
        # Check for valid dates
        valid_dates = cv_train['last_booking'].notna()
        
        if valid_dates.sum() == 0:
            self.logger.debug("  WARNING: No valid last_booking dates")
            cv_train['recency_score'] = 0.5
        else:
            max_date = cv_train.loc[valid_dates, 'last_booking'].max()
            
            cv_train['days_since_last'] = np.nan
            cv_train['recency_score'] = 0.5
            
            cv_train.loc[valid_dates, 'days_since_last'] = (
                max_date - cv_train.loc[valid_dates, 'last_booking']
            ).dt.days.clip(0, 36500)
            
            cv_train.loc[valid_dates, 'recency_score'] = np.exp(
                -cv_train.loc[valid_dates, 'days_since_last'] / 365
            )
            
            self.logger.debug(f"  Valid dates: {valid_dates.sum()}/{len(cv_train)}")
        
        # Component 1: Frequency
        max_freq = cv_train['frequency'].max()
        cv_train['freq_score'] = cv_train['frequency'] / max_freq if max_freq > 0 else 0.5
        
        # Component 3: Monetary
        max_spent = cv_train['total_spent'].max()
        cv_train['monetary_score'] = cv_train['total_spent'] / max_spent if max_spent > 0 else 0.5
        
        # Component 4: Reliability
        cv_train['reliability_score'] = cv_train['completion_rate'].fillna(0.5)
        
        # Combined weight
        cv_train['weight'] = (
            0.35 * cv_train['freq_score'] +
            0.25 * cv_train['recency_score'] +
            0.20 * cv_train['monetary_score'] +
            0.20 * cv_train['reliability_score']
        ).clip(0.1, 10.0)
        
        self.logger.debug(f"C-V weight statistics:")
        self.logger.debug(f"  Min: {cv_train['weight'].min():.3f}")
        self.logger.debug(f"  Max: {cv_train['weight'].max():.3f}")
        self.logger.debug(f"  Mean: {cv_train['weight'].mean():.3f}")
        
        return cv_train
    
    def calculate_cc_weights(self, cc_train: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate edge weights for Client-Chauffeur interactions.
        
        REPLACES: calculate_cc_edge_weights()
        
        Args:
            cc_train: Client-Chauffeur training edges
            
        Returns:
            DataFrame with added weight column
        """
        self.logger.info("\n=== Step 4.2: Client-Chauffeur Edge Weights ===")
        
        self.logger.debug(f"Calculating weights for {len(cc_train):,} C-C edges")
        
        
        # Safe date handling
        valid_dates = cc_train['last_booking'].notna()
        
        if valid_dates.sum() == 0:
            self.logger.warning("  WARNING: No valid last_booking dates")
            cc_train['recency_score'] = 0.5
        else:
            max_date = cc_train.loc[valid_dates, 'last_booking'].max()
            
            # Initialize
            cc_train['days_since_last'] = np.nan
            cc_train['recency_score'] = 0.5
            
            # Compute recency safely
            cc_train.loc[valid_dates, 'days_since_last'] = (
                max_date - cc_train.loc[valid_dates, 'last_booking']
            ).dt.days.clip(0, 36500)
            
            cc_train.loc[valid_dates, 'recency_score'] = np.exp(
                -cc_train.loc[valid_dates, 'days_since_last'] / 365
            )
            
            self.logger.debug(f"  Valid dates: {valid_dates.sum()}/{len(cc_train)}")
        
        # Component 1: Frequency (normalized)
        max_freq = cc_train['frequency'].max()
        cc_train['freq_score'] = cc_train['frequency'] / max_freq if max_freq > 0 else 0.5
        
        # Component 2: Monetary value (normalized)
        max_spent = cc_train['total_spent'].max()
        cc_train['monetary_score'] = cc_train['total_spent'] / max_spent if max_spent > 0 else 0.5
        
        # Component 3: Completion rate (reliability)
        cc_train['reliability_score'] = cc_train['completion_rate'].fillna(0.5)
        
        # Combined weight
        cc_train['weight'] = (
            0.35 * cc_train['freq_score'] +
            0.25 * cc_train['recency_score'] +
            0.20 * cc_train['monetary_score'] +
            0.20 * cc_train['reliability_score']
        ).clip(0.1, 10.0)
        
        self.logger.debug(f"C-C weight statistics:")
        self.logger.debug(f"  Min: {cc_train['weight'].min():.3f}")
        self.logger.debug(f"  Max: {cc_train['weight'].max():.3f}")
        self.logger.debug(f"  Mean: {cc_train['weight'].mean():.3f}")
        self.logger.debug(f"  Median: {cc_train['weight'].median():.3f}")
        
        return cc_train
    
    def calculate_vc_weights(self, vc_train: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate edge weights for Vehicle-Chauffeur co-occurrence.
        
        REPLACES: calculate_vc_edge_weights()
        
        Args:
            vc_train: Vehicle-Chauffeur training edges
            
        Returns:
            DataFrame with added weight column
        """
        self.logger.info("\n=== Step 4.3: Vehicle-Chauffeur Edge Weights ===")
        
        self.logger.debug(f"Calculating weights for {len(vc_train):,} V-C edges")
        
        # Component 1: Co-occurrence frequency (normalized)
        max_times = vc_train['times_together'].max()
        vc_train['freq_score'] = vc_train['times_together'] / max_times
        
        # Component 2: Compatibility score (already calculated in Step 2)
        vc_train['compat_score'] = vc_train['compatibility_score']
        
        # Component 3: Co-occurrence quality (from Step 2)
        vc_train['quality_score'] = vc_train.get('cooccurrence_quality', 0.5)
        
        # Combined weight (V-C edges are about compatibility, not booking behavior)
        vc_train['weight'] = (
            0.40 * vc_train['freq_score'] +
            0.35 * vc_train['compat_score'] +
            0.25 * vc_train['quality_score']
        ).clip(0.1, 10.0)
        
        self.logger.debug(f"V-C weight statistics:")
        self.logger.debug(f"  Min: {vc_train['weight'].min():.3f}")
        self.logger.debug(f"  Max: {vc_train['weight'].max():.3f}")
        self.logger.debug(f"  Mean: {vc_train['weight'].mean():.3f}")
        self.logger.debug(f"  Median: {vc_train['weight'].median():.3f}")
        
        return vc_train


class EdgeListBuilder:
    """
    Builds edge lists for the heterogeneous graph.
    
    REPLACES: create_edge_lists()
    """
    
    def __init__(self):
        """Initialize edge list builder."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build(
        self,
        cv_train: pd.DataFrame,
        cc_train: pd.DataFrame,
        vc_train: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create edge lists for all three edge types.
        
        REPLACES: create_edge_lists()
        
        Args:
            cv_train: Client-Vehicle edges with weights
            cc_train: Client-Chauffeur edges with weights
            vc_train: Vehicle-Chauffeur edges with weights
            
        Returns:
            Tuple of (all_edges, cv_edges, cc_edges, vc_edges_forward, vc_edges_backward)
        """
        self.logger.info("\n=== Step 4.4: Creating Edge Lists ===")
        
        # Client-Vehicle edges
        cv_edges = cv_train[['client_node_id', 'vehicle_node_id', 'weight']].copy()
        cv_edges.columns = ['source', 'target', 'weight']
        cv_edges['edge_type'] = 'cv'
        self.logger.debug(f"C-V edges: {len(cv_edges):,}")
        
        # Client-Chauffeur edges
        cc_edges = cc_train[['client_node_id', 'chauffeur_node_id', 'weight']].copy()
        cc_edges.columns = ['source', 'target', 'weight']
        cc_edges['edge_type'] = 'cc'
        self.logger.debug(f"C-C edges: {len(cc_edges):,}")
        
        # Vehicle-Chauffeur edges (undirected, so add both directions)
        vc_edges_forward = vc_train[['vehicle_node_id', 'chauffeur_node_id', 'weight']].copy()
        vc_edges_forward.columns = ['source', 'target', 'weight']
        vc_edges_forward['edge_type'] = 'vc'
        
        vc_edges_backward = vc_train[['chauffeur_node_id', 'vehicle_node_id', 'weight']].copy()
        vc_edges_backward.columns = ['source', 'target', 'weight']
        vc_edges_backward['edge_type'] = 'vc'
        
        self.logger.debug(f"V-C edges: {len(vc_train):,} (bidirectional: {len(vc_edges_forward) + len(vc_edges_backward):,})")
        
        # Combine all edges
        all_edges = pd.concat([
            cv_edges,
            cc_edges,
            vc_edges_forward,
            vc_edges_backward
        ], ignore_index=True)
        
        self.logger.debug(f"\nTotal edges (with V-C bidirectional): {len(all_edges):,}")
        self.logger.debug(f"Edge type distribution:")
        self.logger.debug(all_edges['edge_type'].value_counts().to_dict())
        
        return all_edges, cv_edges, cc_edges, vc_edges_forward, vc_edges_backward


class AdjacencyMatrixBuilder:
    """
    Builds adjacency matrices for the heterogeneous graph.
    
    REPLACES: create_heterogeneous_adjacency_matrix(), create_separate_interaction_matrices()
    
    FIXED: Uses row+col degree (symmetric), NO self-loops, returns COO matrix (float32)
    """
    
    def __init__(self):
        """Initialize adjacency matrix builder."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_normalized(
        self,
        all_edges: pd.DataFrame,
        n_total_nodes: int
    ) -> coo_matrix:
        """
        Create normalized adjacency matrix for heterogeneous graph.
        
        REPLACES: create_heterogeneous_adjacency_matrix()
        
        FIXED: Uses row+col degree (symmetric normalization)
        FIXED: NO self-loops added
        FIXED: Returns COO matrix (float32)
        
        Args:
            all_edges: All edges combined
            n_total_nodes: Total number of nodes
            
        Returns:
            Normalized adjacency matrix as COO matrix
        """
        self.logger.info("\n=== Step 4.5: Creating Heterogeneous Adjacency Matrix ===")
        
        self.logger.debug(f"Building adjacency matrix for {n_total_nodes:,} nodes")
        
        # Extract edge data with explicit dtypes
        sources = all_edges['source'].values.astype(np.int32)
        targets = all_edges['target'].values.astype(np.int32)
        weights = all_edges['weight'].values.astype(np.float32)
        
        # Create adjacency matrix (directed edges)
        adj_matrix = coo_matrix(
            (weights, (sources, targets)),
            shape=(n_total_nodes, n_total_nodes),
            dtype=np.float32
        )
        
        self.logger.debug(f"Adjacency matrix shape: {adj_matrix.shape}")
        self.logger.debug(f"Non-zero entries: {adj_matrix.nnz:,}")
        self.logger.debug(f"Sparsity: {(1 - adj_matrix.nnz / (n_total_nodes ** 2)) * 100:.4f}%")
        
        # FIXED: Normalize adjacency matrix using row+col degree (symmetric)
        # D^(-1/2) * A * D^(-1/2)
        adj_coo = adj_matrix.tocoo()
        
        # FIXED: Calculate degree matrix using BOTH row and column sums
        degrees = np.array(adj_matrix.sum(axis=1)).flatten() + \
                  np.array(adj_matrix.sum(axis=0)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        
        # D^(-1/2)
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        # Normalize
        normalized_weights = (
            d_inv_sqrt[adj_coo.row] * 
            adj_coo.data * 
            d_inv_sqrt[adj_coo.col]
        )
        
        # FIXED: Return COO matrix (float32), NO self-loops
        normalized_adj = coo_matrix(
            (normalized_weights, (adj_coo.row, adj_coo.col)),
            shape=adj_matrix.shape,
            dtype=np.float32
        )
        
        self.logger.info("Adjacency matrix normalized successfully")
        
        return normalized_adj
    
    def build_separate(
        self,
        cv_edges: pd.DataFrame,
        cc_edges: pd.DataFrame,
        vc_edges_forward: pd.DataFrame,
        n_clients: int,
        n_vehicles: int,
        n_chauffeurs: int,
        client_offset: int,
        vehicle_offset: int,
        chauffeur_offset: int
    ) -> Dict[str, csr_matrix]:
        """
        Create separate interaction matrices for each edge type.
        
        REPLACES: create_separate_interaction_matrices()
        
        FIXED: Added explicit dtypes (int32, float32)
        
        Args:
            cv_edges: Client-Vehicle edges
            cc_edges: Client-Chauffeur edges
            vc_edges_forward: Vehicle-Chauffeur edges (forward direction)
            n_clients: Number of clients
            n_vehicles: Number of vehicles
            n_chauffeurs: Number of chauffeurs
            client_offset: Offset for client IDs (0)
            vehicle_offset: Offset for vehicle IDs
            chauffeur_offset: Offset for chauffeur IDs
            
        Returns:
            Dictionary with separate interaction matrices
        """
        self.logger.info("\n=== Step 4.6: Creating Separate Interaction Matrices ===")
        
        # FIXED: Client-Vehicle matrix with explicit dtypes
        cv_sources = cv_edges['source'].values.astype(np.int32)  # Already client IDs
        cv_targets = cv_edges['target'].values.astype(np.int32) - vehicle_offset  # Convert to 0-based vehicle IDs
        cv_weights = cv_edges['weight'].values.astype(np.float32)
        
        cv_matrix = coo_matrix(
            (cv_weights, (cv_sources, cv_targets)),
            shape=(n_clients, n_vehicles),
            dtype=np.float32
        ).tocsr()
        
        self.logger.debug(f"C-V matrix: {cv_matrix.shape}, nnz={cv_matrix.nnz:,}")
        
        # FIXED: Client-Chauffeur matrix with explicit dtypes
        cc_sources = cc_edges['source'].values.astype(np.int32)  # Already client IDs
        cc_targets = cc_edges['target'].values.astype(np.int32) - chauffeur_offset  # Convert to 0-based chauffeur IDs
        cc_weights = cc_edges['weight'].values.astype(np.float32)
        
        cc_matrix = coo_matrix(
            (cc_weights, (cc_sources, cc_targets)),
            shape=(n_clients, n_chauffeurs),
            dtype=np.float32
        ).tocsr()
        
        self.logger.debug(f"C-C matrix: {cc_matrix.shape}, nnz={cc_matrix.nnz:,}")
        
        # FIXED: Vehicle-Chauffeur matrix with explicit dtypes
        vc_sources = vc_edges_forward['source'].values.astype(np.int32) - vehicle_offset
        vc_targets = vc_edges_forward['target'].values.astype(np.int32) - chauffeur_offset
        vc_weights = vc_edges_forward['weight'].values.astype(np.float32)
        
        vc_matrix = coo_matrix(
            (vc_weights, (vc_sources, vc_targets)),
            shape=(n_vehicles, n_chauffeurs),
            dtype=np.float32
        ).tocsr()
        
        self.logger.debug(f"V-C matrix: {vc_matrix.shape}, nnz={vc_matrix.nnz:,}")
        
        matrices = {
            'cv_matrix': cv_matrix,
            'cc_matrix': cc_matrix,
            'vc_matrix': vc_matrix
        }
        
        return matrices


class PyTorchDataConverter:
    """
    Converts graph data to PyTorch Geometric format.
    
    REPLACES: create_pytorch_geometric_data()
    
    FIXED: Derives edge_index from normalized_adj (COO format)
    FIXED: Returns all original keys including num_nodes, ranges, etc.
    """
    
    def __init__(self):
        """Initialize PyTorch data converter."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def convert(
        self,
        all_edges: pd.DataFrame,
        normalized_adj: coo_matrix,
        n_total_nodes: int,
        n_clients: int,
        n_vehicles: int,
        n_chauffeurs: int
    ) -> Dict[str, Any]:
        """
        Create PyTorch Geometric compatible data format.
        
        REPLACES: create_pytorch_geometric_data()
        
        FIXED: Derives edge_index FROM normalized_adj (COO format)
        FIXED: Returns all original keys (num_nodes, n_clients, ranges, etc.)
        
        Args:
            all_edges: All edges combined
            normalized_adj: Normalized adjacency matrix (COO format)
            n_total_nodes: Total number of nodes
            n_clients: Number of clients
            n_vehicles: Number of vehicles
            n_chauffeurs: Number of chauffeurs
            
        Returns:
            Dictionary with PyTorch Geometric compatible data
        """
        self.logger.info("\n=== Step 4.7: Creating PyTorch Geometric Data ===")
        
        # FIXED: Derive edge_index FROM normalized_adj (COO format)
        adj_coo = normalized_adj.tocoo()
        edge_index = np.vstack([adj_coo.row, adj_coo.col])
        edge_weights = adj_coo.data
        
        self.logger.debug(f"PyTorch Geometric edge_index shape: {edge_index.shape}")
        self.logger.debug(f"Edge weights shape: {edge_weights.shape}")
        
        # Edge type encoding
        edge_types = []
        for _, row in all_edges.iterrows():
            if row['edge_type'] == 'cv':
                edge_types.append(0)
            elif row['edge_type'] == 'cc':
                edge_types.append(1)
            else:  # vc
                edge_types.append(2)
        
        edge_types = np.array(edge_types, dtype=np.int32)
        
        # Node type encoding (for heterogeneous GNN)
        node_types = np.zeros(n_total_nodes, dtype=np.int32)
        node_types[:n_clients] = 0  # Clients
        node_types[n_clients:n_clients + n_vehicles] = 1  # Vehicles
        node_types[n_clients + n_vehicles:] = 2  # Chauffeurs
        
        # FIXED: Return all original keys
        pytorch_data = {
            'edge_index': edge_index.astype(np.int64),
            'edge_weights': edge_weights.astype(np.float32),
            'edge_types': edge_types,
            'node_types': node_types,
            'num_nodes': n_total_nodes,  # FIXED: Use 'num_nodes' not 'n_nodes'
            'n_clients': n_clients,
            'n_vehicles': n_vehicles,
            'n_chauffeurs': n_chauffeurs,
            'client_range': (0, n_clients),
            'vehicle_range': (n_clients, n_clients + n_vehicles),
            'chauffeur_range': (n_clients + n_vehicles, n_total_nodes)
        }
        
        self.logger.info("PyTorch Geometric data created successfully")
        self.logger.debug(f"  Nodes: {n_total_nodes:,}")
        self.logger.debug(f"  Edges: {len(edge_index[0]):,}")
        self.logger.debug(f"  Edge types: {len(np.unique(edge_types))}")
        self.logger.debug(f"  Node types: {len(np.unique(node_types))}")
        return pytorch_data


class GraphAnalyzer:
    """
    Analyzes properties of the heterogeneous graph.
    
    REPLACES: analyze_graph_properties()
    
    FIXED: Returns all original keys (client_degree_mean, cv_density, total_edges, etc.)
    FIXED: Uses COO format for matrix access
    """
    
    def __init__(self):
        """Initialize graph analyzer."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze(
        self,
        all_edges: pd.DataFrame,
        normalized_adj: coo_matrix,
        separate_matrices: Dict[str, csr_matrix],
        n_clients: int,
        n_vehicles: int,
        n_chauffeurs: int
    ) -> Dict[str, Any]:
        """
        Analyze properties of the heterogeneous graph.
        
        REPLACES: analyze_graph_properties()
        
        FIXED: Returns all original keys
        FIXED: Uses COO format for memory calculation
        
        Args:
            all_edges: All edges combined
            normalized_adj: Normalized adjacency matrix
            separate_matrices: Separate interaction matrices
            n_clients: Number of clients
            n_vehicles: Number of vehicles
            n_chauffeurs: Number of chauffeurs
            
        Returns:
            Dictionary with graph analysis results
        """
        self.logger.info("\n=== Step 4.8: Graph Analysis ===")
        
        # Node degree analysis by type
        adj_csr = normalized_adj.tocsr()
        degrees = np.array(adj_csr.sum(axis=1)).flatten()
        
        client_degrees = degrees[:n_clients]
        vehicle_degrees = degrees[n_clients:n_clients + n_vehicles]
        chauffeur_degrees = degrees[n_clients + n_vehicles:]
        
        self.logger.debug(f"Node degree statistics:")
        self.logger.debug(f"  Clients: mean={client_degrees.mean():.2f}, median={np.median(client_degrees):.2f}")
        self.logger.debug(f"  Vehicles: mean={vehicle_degrees.mean():.2f}, median={np.median(vehicle_degrees):.2f}")
        self.logger.debug(f"  Chauffeurs: mean={chauffeur_degrees.mean():.2f}, median={np.median(chauffeur_degrees):.2f}")
      
        edge_type_counts = all_edges['edge_type'].value_counts()
        self.logger.debug(f"\nEdge type distribution:")

        for edge_type, count in edge_type_counts.items():
             self.logger.debug(f"  {edge_type}: {count:,}")
        
        # Density by edge type
        cv_density = separate_matrices['cv_matrix'].nnz / (n_clients * n_vehicles)
        cc_density = separate_matrices['cc_matrix'].nnz / (n_clients * n_chauffeurs)
        vc_density = separate_matrices['vc_matrix'].nnz / (n_vehicles * n_chauffeurs)
        
        self.logger.debug(f"\nDensity by edge type:")
        self.logger.debug(f"  C-V: {cv_density:.6f} ({cv_density*100:.4f}%)")
        self.logger.debug(f"  C-C: {cc_density:.6f} ({cc_density*100:.4f}%)")
        self.logger.debug(f"  V-C: {vc_density:.6f} ({vc_density*100:.4f}%)")
        
        adj_coo = normalized_adj.tocoo()
        memory_mb = (
            adj_coo.data.nbytes + 
            adj_coo.row.nbytes + 
            adj_coo.col.nbytes
        ) / 1024**2
        
        self.logger.debug(f"\nMemory usage:")
        self.logger.debug(f"  Adjacency matrix: {memory_mb:.2f} MB")
        self.logger.debug(f"  Sparsity: {(1 - normalized_adj.nnz / (normalized_adj.shape[0]**2)) * 100:.4f}%")
        
        # FIXED: Return all original keys
        analysis = {
            'client_degree_mean': float(client_degrees.mean()),
            'vehicle_degree_mean': float(vehicle_degrees.mean()),
            'chauffeur_degree_mean': float(chauffeur_degrees.mean()),
            'cv_density': float(cv_density),
            'cc_density': float(cc_density),
            'vc_density': float(vc_density),
            'total_edges': len(all_edges),
            'memory_mb': float(memory_mb),
            'sparsity': float(1 - normalized_adj.nnz / (normalized_adj.shape[0]**2))
        }
        
        return analysis


class GraphDataSaver:
    """
    Saves graph data for model training.
    
    REPLACES: save_heterogeneous_graph_data()
    """
    
    def __init__(self, output_dir=None, output_prefix=None):
        """
        Initialize graph data saver.
        
        Args:
            output_dir: Output directory path
            output_prefix: Prefix for output files
        """
        self.output_dir = output_dir or GRAPH_OUTPUT_DIR
        self.output_prefix = output_prefix or GRAPH_OUTPUT_PREFIX
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save(
        self,
        all_edges: pd.DataFrame,
        normalized_adj: coo_matrix,
        separate_matrices: Dict[str, csr_matrix],
        pytorch_data: Dict[str, Any],
        analysis: Dict[str, Any],
        step3_results: Dict[str, Any],
        output_prefix: Optional[str] = None
    ) -> bool:
        """
        Save all graph data for model training.
        
        REPLACES: save_heterogeneous_graph_data()
        
        Args:
            all_edges: All edges combined
            normalized_adj: Normalized adjacency matrix
            separate_matrices: Separate interaction matrices
            pytorch_data: PyTorch Geometric compatible data
            analysis: Graph analysis results
            step3_results: Results from Step 3
            output_prefix: Optional prefix for output files
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"\n=== Step 4.9: Saving Graph Data ===")
        
        try:
            if output_prefix is None:
                output_prefix = self.output_prefix
            
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save edge list
            edges_file = self.output_dir / f"{output_prefix}_edges.csv"
            all_edges.to_csv(edges_file, index=False)
            self.logger.debug(f" Edge list saved to {edges_file}")
            
            # Save adjacency matrix
            adj_file = self.output_dir / f"{output_prefix}_adj_matrix.npz"
            adj_coo = normalized_adj.tocoo()
            np.savez_compressed(
                adj_file,
                row=adj_coo.row,
                col=adj_coo.col,
                data=adj_coo.data,
                shape=adj_coo.shape
            )
            self.logger.debug(f" Adjacency matrix saved to {adj_file}")
            
            # Save separate interaction matrices
            matrices_file = self.output_dir / f"{output_prefix}_interaction_matrices.npz"
            np.savez_compressed(
                matrices_file,
                cv_data=separate_matrices['cv_matrix'].data,
                cv_indices=separate_matrices['cv_matrix'].indices,
                cv_indptr=separate_matrices['cv_matrix'].indptr,
                cv_shape=separate_matrices['cv_matrix'].shape,
                cc_data=separate_matrices['cc_matrix'].data,
                cc_indices=separate_matrices['cc_matrix'].indices,
                cc_indptr=separate_matrices['cc_matrix'].indptr,
                cc_shape=separate_matrices['cc_matrix'].shape,
                vc_data=separate_matrices['vc_matrix'].data,
                vc_indices=separate_matrices['vc_matrix'].indices,
                vc_indptr=separate_matrices['vc_matrix'].indptr,
                vc_shape=separate_matrices['vc_matrix'].shape
            )
            self.logger.debug(f" Interaction matrices saved to {matrices_file}")
            
            # Save PyTorch data
            pytorch_file = self.output_dir / f"{output_prefix}_pytorch.npz"
            np.savez_compressed(pytorch_file, **pytorch_data)
            self.logger.debug(f" PyTorch data saved to {pytorch_file}")
            
            # Save mappings
            mappings = step3_results['mappings']
            mappings_file = self.output_dir / f"{output_prefix}_mappings.npz"
            np.savez_compressed(
                mappings_file,
                client_ids=np.array(list(mappings['client_to_id'].keys())),
                client_indices=np.array(list(mappings['client_to_id'].values())),
                vehicle_ids=np.array(list(mappings['vehicle_to_id'].keys())),
                vehicle_indices=np.array(list(mappings['vehicle_to_id'].values())),
                chauffeur_ids=np.array(list(mappings['chauffeur_to_id'].keys())),
                chauffeur_indices=np.array(list(mappings['chauffeur_to_id'].values())),
                n_clients=mappings['n_clients'],
                n_vehicles=mappings['n_vehicles'],
                n_chauffeurs=mappings['n_chauffeurs']
            )
            self.logger.debug(f" ID mappings saved to {mappings_file}")
            
            # Save analysis
            analysis_file = self.output_dir / f"{output_prefix}_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            self.logger.debug(f" Analysis saved to {analysis_file}")
            
            self.logger.debug(f"\nAll data saved with prefix: {output_prefix}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            return False


class GraphConstructionPipeline(BasePipeline):
    """
    Orchestrates the graph construction process.
    
    REPLACES: execute_step4_three_way()
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize graph construction pipeline.
        
        Args:
            config: Configuration with save_data, output_prefix
        """
        super().__init__(config or {})
        
        self.weight_calculator = EdgeWeightCalculator()
        self.edge_list_builder = EdgeListBuilder()
        self.adj_matrix_builder = AdjacencyMatrixBuilder()
        self.pytorch_converter = PyTorchDataConverter()
        self.analyzer = GraphAnalyzer()
        self.saver = GraphDataSaver()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute graph construction pipeline.
        
        REPLACES: execute_step4_three_way()
        
        Args:
            input_data: Step 3 results
            
        Returns:
            Dictionary containing step 4 outputs
        """
        self._log_start("Step 4: Heterogeneous Graph Construction")
        
        try:
            # Get data from Step 3
            splits = input_data['splits']
            mappings = input_data['mappings']
            
            cv_train = splits['cv_train'].copy()
            cc_train = splits['cc_train'].copy()
            vc_train = splits['vc_train'].copy()
            
            n_clients = mappings['n_clients']
            n_vehicles = mappings['n_vehicles']
            n_chauffeurs = mappings['n_chauffeurs']
            n_total_nodes = mappings['n_total_nodes']
            
            client_offset = 0
            vehicle_offset = n_clients
            chauffeur_offset = n_clients + n_vehicles
            
            self.logger.debug(f"Processing heterogeneous graph:")
            self.logger.debug(f"  Nodes: {n_total_nodes:,} (C:{n_clients:,}, V:{n_vehicles:,}, Ch:{n_chauffeurs:,})")
            self.logger.debug(f"  Training edges: C-V:{len(cv_train):,}, C-C:{len(cc_train):,}, V-C:{len(vc_train):,}")
            
            
            # Step 4.1-4.3: Calculate edge weights
            cv_train = self.weight_calculator.calculate_cv_weights(cv_train)
            cc_train = self.weight_calculator.calculate_cc_weights(cc_train)
            vc_train = self.weight_calculator.calculate_vc_weights(vc_train)
            
            # Step 4.4: Create edge lists
            all_edges, cv_edges, cc_edges, vc_fwd, vc_bwd = self.edge_list_builder.build(
                cv_train, cc_train, vc_train
            )
            
            # Step 4.5: Create heterogeneous adjacency matrix
            normalized_adj = self.adj_matrix_builder.build_normalized(all_edges, n_total_nodes)
            
            # Step 4.6: Create separate interaction matrices
            separate_matrices = self.adj_matrix_builder.build_separate(
                cv_edges, cc_edges, vc_fwd,
                n_clients, n_vehicles, n_chauffeurs,
                client_offset, vehicle_offset, chauffeur_offset
            )
            
            # Step 4.7: Create PyTorch Geometric data
            pytorch_data = self.pytorch_converter.convert(
                all_edges, normalized_adj, n_total_nodes,
                n_clients, n_vehicles, n_chauffeurs
            )
            
            # Step 4.8: Analyze graph properties
            analysis = self.analyzer.analyze(
                all_edges, normalized_adj, separate_matrices,
                n_clients, n_vehicles, n_chauffeurs
            )
            
            # Step 4.9: Save data
            save_data = self.config.get('save_data', True)
            output_prefix = self.config.get('output_prefix', 'heterogeneous_lightgcn')
            
            if save_data:
                save_success = self.saver.save(
                    all_edges, normalized_adj, separate_matrices,
                    pytorch_data, analysis, input_data, output_prefix
                )
            else:
                save_success = True
            
            # Combine outputs
            step4_outputs = {
                **input_data,
                'all_edges': all_edges,
                'cv_edges_weighted': cv_train,
                'cc_edges_weighted': cc_train,
                'vc_edges_weighted': vc_train,
                'normalized_adj': normalized_adj,
                'separate_matrices': separate_matrices,
                'pytorch_data': pytorch_data,
                'analysis': analysis,
                'save_success': save_success
            }
            
            self._log_summary(step4_outputs, analysis, save_data, output_prefix)
            self._log_complete("Step 4")
            
            return step4_outputs
            
        except Exception as e:
            self.logger.error(f"Error in Step 4: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_summary(
        self,
        outputs: Dict[str, Any],
        analysis: Dict[str, Any],
        save_data: bool,
        output_prefix: str
    ) -> None:
        """Log step 4 summary."""
        mappings = outputs['mappings']
        all_edges = outputs['all_edges']
        normalized_adj = outputs['normalized_adj']
        
        self.logger.debug("\n" + "=" * 60)
        self.logger.info("Step 4 completed successfully!")
        self.logger.debug("\nHeterogeneous Graph Summary:")
        self.logger.debug(f"  Total nodes: {mappings['n_total_nodes']:,}")
        self.logger.debug(f"  Total edges: {len(all_edges):,}")
        self.logger.debug(f"  Adjacency matrix: {normalized_adj.shape}")
        self.logger.debug(f"  Memory usage: {analysis['memory_mb']:.2f} MB")
        self.logger.debug(f"  Sparsity: {analysis['sparsity']*100:.4f}%")
        if save_data:
            self.logger.debug(f"  Data saved: {output_prefix}_*")
        self.logger.debug("\nReady for LightGCN Model Training!")


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def calculate_cv_edge_weights(cv_train):
    """Backward compatible wrapper for EdgeWeightCalculator.calculate_cv_weights()."""
    calculator = EdgeWeightCalculator()
    return calculator.calculate_cv_weights(cv_train)


def calculate_cc_edge_weights(cc_train):
    """Backward compatible wrapper for EdgeWeightCalculator.calculate_cc_weights()."""
    calculator = EdgeWeightCalculator()
    return calculator.calculate_cc_weights(cc_train)


def calculate_vc_edge_weights(vc_train):
    """Backward compatible wrapper for EdgeWeightCalculator.calculate_vc_weights()."""
    calculator = EdgeWeightCalculator()
    return calculator.calculate_vc_weights(vc_train)


def create_edge_lists(cv_train, cc_train, vc_train):
    """Backward compatible wrapper for EdgeListBuilder.build()."""
    builder = EdgeListBuilder()
    return builder.build(cv_train, cc_train, vc_train)


def create_heterogeneous_adjacency_matrix(all_edges, n_total_nodes):
    """Backward compatible wrapper for AdjacencyMatrixBuilder.build_normalized()."""
    builder = AdjacencyMatrixBuilder()
    return builder.build_normalized(all_edges, n_total_nodes)


def create_separate_interaction_matrices(cv_edges, cc_edges, vc_edges_forward,
                                         n_clients, n_vehicles, n_chauffeurs,
                                         client_offset, vehicle_offset, chauffeur_offset):
    """Backward compatible wrapper for AdjacencyMatrixBuilder.build_separate()."""
    builder = AdjacencyMatrixBuilder()
    return builder.build_separate(
        cv_edges, cc_edges, vc_edges_forward,
        n_clients, n_vehicles, n_chauffeurs,
        client_offset, vehicle_offset, chauffeur_offset
    )


def create_pytorch_geometric_data(all_edges, normalized_adj, n_total_nodes,
                                   n_clients, n_vehicles, n_chauffeurs):
    """Backward compatible wrapper for PyTorchDataConverter.convert()."""
    converter = PyTorchDataConverter()
    return converter.convert(
        all_edges, normalized_adj, n_total_nodes,
        n_clients, n_vehicles, n_chauffeurs
    )


def analyze_graph_properties(all_edges, normalized_adj, separate_matrices,
                              n_clients, n_vehicles, n_chauffeurs):
    """Backward compatible wrapper for GraphAnalyzer.analyze()."""
    analyzer = GraphAnalyzer()
    return analyzer.analyze(
        all_edges, normalized_adj, separate_matrices,
        n_clients, n_vehicles, n_chauffeurs
    )


def save_heterogeneous_graph_data(all_edges, normalized_adj, separate_matrices,
                                   pytorch_data, analysis, step3_results,
                                   output_prefix=None):
    """Backward compatible wrapper for GraphDataSaver.save()."""
    saver = GraphDataSaver()
    return saver.save(
        all_edges, normalized_adj, separate_matrices,
        pytorch_data, analysis, step3_results, output_prefix
    )


def execute_step4_three_way(step3_results, save_data=True,
                            output_prefix="heterogeneous_lightgcn"):
    """Backward compatible wrapper for GraphConstructionPipeline.execute()."""
    config = {
        'save_data': save_data,
        'output_prefix': output_prefix
    }
    pipeline = GraphConstructionPipeline(config)
    return pipeline.execute(step3_results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("Graph Construction Module (Refactored - Corrected) - Ready")
    logger.debug("\nClasses available:")
    logger.debug("  - EdgeWeightCalculator")
    logger.debug("  - EdgeListBuilder")
    logger.debug("  - AdjacencyMatrixBuilder")
    logger.debug("  - PyTorchDataConverter")
    logger.debug("  - GraphAnalyzer")
    logger.debug("  - GraphDataSaver")
    logger.info("  - GraphConstructionPipeline")
    logger.debug("\nBackward compatible functions:")
    logger.debug("  - calculate_cv_edge_weights()")
    logger.debug("  - calculate_cc_edge_weights()")
    logger.debug("  - calculate_vc_edge_weights()")
    logger.debug("  - create_edge_lists()")
    logger.debug("  - create_heterogeneous_adjacency_matrix()")
    logger.debug("  - create_separate_interaction_matrices()")
    logger.debug("  - create_pytorch_geometric_data()")
    logger.debug("  - analyze_graph_properties()")
    logger.debug("  - save_heterogeneous_graph_data()")
    logger.debug("  - execute_step4_three_way()")
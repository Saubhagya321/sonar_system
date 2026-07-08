
import pandas as pd
import numpy as np
from itertools import product
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import random

from config import GRID_SEARCH_CONFIG, WEIGHT_FILES, CLIENT_TYPES
from data_loaders import DataLoader
from fusion_models import RecommendationFusion, NDCGCalculator

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    """
    Optimize fusion weights using grid search and NDCG validation
    """
    
    def __init__(self, validation_k: int = 10):
        """
        Initialize optimizer
        """
        self.validation_k = validation_k
        self.fusion = RecommendationFusion(quiet=True)
        self.ndcg_calc = NDCGCalculator()
    
    def prepare_validation_data(self, 
                                lightgcn_df: pd.DataFrame,
                                collaborative_df: pd.DataFrame,
                                client_types: Dict[int, str],
                                ground_truth_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Prepare validation data split by client type with sampling
        """
        logger.info("Preparing validation data...")
        
        clients_with_ground_truth = set(ground_truth_df['client_id'].unique())
        logger.info(f"Clients with ground truth bookings: {len(clients_with_ground_truth)}")
        
        validation_data = {}
        for client_type in set(client_types.values()):
            validation_data[client_type] = {'clients': [], 'data': []}
        
        # Group both tables by client_id ONCE up front (instead of re-scanning
        # the full table per client below) — turns an O(n_clients * n_rows)
        # loop into a single O(n_rows) pass per table. original_client_id is
        # stored as a string (see data_loaders.py), so cast to int here too,
        # both to key the groups consistently and to match collaborative_df's
        # int client_id (previously these were mixed types in the same set).
        gcn_by_client: Dict[int, pd.DataFrame] = {}
        if not lightgcn_df.empty and 'original_client_id' in lightgcn_df.columns:
            gcn_by_client = {
                int(cid): grp for cid, grp in lightgcn_df.groupby('original_client_id')
            }

        cf_by_client: Dict[int, pd.DataFrame] = {}
        if not collaborative_df.empty and 'client_id' in collaborative_df.columns:
            cf_by_client = {
                int(cid): grp for cid, grp in collaborative_df.groupby('client_id')
            }

        all_clients = set(gcn_by_client.keys()) | set(cf_by_client.keys())
        all_clients = all_clients.intersection(clients_with_ground_truth)

        logger.info(f"Total unique clients with both recommendations and ground truth: {len(all_clients)}")

        for client_id in all_clients:
            client_type = client_types.get(client_id, 'UNKNOWN')

            if client_type not in validation_data:
                validation_data[client_type] = {'clients': [], 'data': []}

            client_gcn = gcn_by_client.get(client_id, pd.DataFrame()).copy()
            client_cf = cf_by_client.get(client_id, pd.DataFrame()).copy()

            if len(client_gcn) < 5 and len(client_cf) < 5:
                continue
            
            validation_data[client_type]['clients'].append(client_id)
            validation_data[client_type]['data'].append({
                'client_id': client_id,
                'lightgcn': client_gcn,
                'collaborative': client_cf
            })
        
        # Sampling
        max_clients_per_type = GRID_SEARCH_CONFIG.get('max_clients_per_type', {
            'CORPORATE': 1200,
            'GLOBAL CASH': 400,
            'GLOBAL CREDIT': 200,
            'UNKNOWN': 100
        })
        
        random.seed(42)
        
        logger.info("\nClient sampling:")
        for client_type, data in validation_data.items():
            n_clients = len(data['data'])
            max_clients = max_clients_per_type.get(client_type, 100)
            
            if n_clients > max_clients:
                logger.info(f"  {client_type}: Sampling {max_clients} from {n_clients} clients")
                sampled_data = random.sample(data['data'], max_clients)
                data['data'] = sampled_data
                data['clients'] = [d['client_id'] for d in sampled_data]
            else:
                logger.info(f"  {client_type}: Using all {n_clients} clients")
        
        logger.info("\nFinal validation set:")
        total_clients = 0
        for client_type, data in validation_data.items():
            n = len(data['clients'])
            logger.info(f"  {client_type}: {n} clients")
            total_clients += n
        logger.info(f"  TOTAL: {total_clients} clients")
        
        return validation_data
    
    def evaluate_weights(self,
                        weights: Dict[str, float],
                        validation_data: List[Dict],
                        gt_by_client: Dict[int, pd.DataFrame]) -> float:
        """
        Evaluate a weight configuration on validation data
        """
        ndcg_scores = []
        empty_gt = pd.DataFrame()

        for client_data in validation_data:
            client_id = client_data['client_id']

            recommendations = {
                'lightgcn': client_data['lightgcn'],
                'collaborative': client_data['collaborative'],
                'item_similarity': pd.DataFrame(),
                'user_similarity': pd.DataFrame()
            }

            if recommendations['lightgcn'].empty and recommendations['collaborative'].empty:
                continue

            try:
                fused = self.fusion.fuse_recommendations(recommendations, weights)

                if fused.empty:
                    continue

                client_gt = gt_by_client.get(client_id, empty_gt)

                ndcg = self.ndcg_calc.compute_ndcg_for_client(
                    fused.head(self.validation_k),
                    client_gt,
                    k=self.validation_k
                )

                ndcg_scores.append(ndcg)

            except Exception as e:
                logger.debug(f"Error evaluating client {client_id}: {str(e)}")
                continue

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def grid_search(self,
                   validation_data: List[Dict],
                   ground_truth_df: pd.DataFrame,
                   alpha_ranges: Dict[str, List[float]]) -> Tuple[Dict[str, float], float]:
        """
        Perform grid search to find optimal weights
        """
        logger.info(f"Starting grid search with {len(validation_data)} clients...")

        # Group ground truth by client_id ONCE instead of re-scanning the full
        # table inside evaluate_weights on every (client, weight-combo) pair —
        # same fix as prepare_validation_data's groupby, applied here since
        # this loop calls evaluate_weights once per combo (up to 72), each of
        # which previously filtered the full table once per client.
        gt_by_client: Dict[int, pd.DataFrame] = {
            int(cid): grp for cid, grp in ground_truth_df.groupby('client_id')
        }

        param_names = list(alpha_ranges.keys())
        param_values = list(alpha_ranges.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} weight combinations...")

        best_weights = None
        best_score = -1

        for combo in tqdm(combinations, desc="Grid Search"):
            weights = dict(zip(param_names, combo))

            score = self.evaluate_weights(weights, validation_data, gt_by_client)

            if score > best_score:
                best_score = score
                best_weights = weights.copy()
        
        logger.info(f"    Best NDCG@{self.validation_k}: {best_score:.4f}")
        logger.info(f"    Best weights: {best_weights}")
        
        return best_weights, best_score
    
    def optimize_all_client_types(self,
                                  lightgcn_df: pd.DataFrame,
                                  collaborative_df: pd.DataFrame,
                                  ground_truth_df: pd.DataFrame,
                                  client_types: Dict[int, str]) -> Dict[str, Dict]:
        """
        Optimize weights for all client types
        """
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZING WEIGHTS FOR ALL CLIENT TYPES")
        logger.info("="*60 + "\n")
        
        validation_data = self.prepare_validation_data(
            lightgcn_df, collaborative_df, client_types, ground_truth_df
        )
        
        alpha_ranges = GRID_SEARCH_CONFIG['alpha_ranges']
        
        optimal_weights = {}
        
        client_type_list = [ct for ct in validation_data.keys() 
                           if ct != 'UNKNOWN' and validation_data[ct]['data']]
        
        for client_type in client_type_list:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing for {client_type}")
            logger.info(f"{'='*60}")
            
            type_data = validation_data[client_type]['data']
            
            if not type_data:
                logger.warning(f"No data for {client_type}, skipping...")
                continue
            
            best_weights, best_score = self.grid_search(
                type_data,
                ground_truth_df,
                alpha_ranges
            )
            
            optimal_weights[client_type] = {
                'weights': best_weights,
                'validation_ndcg': best_score,
                'n_clients': len(type_data)
            }
        
        # Compute UNKNOWN weights
        logger.info(f"\n{'='*60}")
        logger.info("Computing UNKNOWN weights (average of all types)")
        logger.info(f"{'='*60}")
        
        unknown_weights = {}
        for param in alpha_ranges.keys():
            values = [
                optimal_weights[ct]['weights'][param]
                for ct in optimal_weights.keys()
                if param in optimal_weights[ct]['weights']
            ]
            unknown_weights[param] = np.mean(values) if values else 0.2
        
        optimal_weights['UNKNOWN'] = {
            'weights': unknown_weights,
            'validation_ndcg': None,
            'n_clients': 0
        }
        
        logger.info(f"    UNKNOWN weights: {unknown_weights}")
        
        return optimal_weights
    
    def save_weights(self, optimal_weights: Dict[str, Dict], weights_dir=None):
        """
        Save optimal weights to JSON files.

        Args:
            optimal_weights: mapping of client_type → weight dict.
            weights_dir:     directory to write weight files.  When None the
                             default WEIGHT_FILES paths from config are used
                             (original behaviour — writes to combiner/weights/).
        """
        logger.info("\n" + "="*60)
        logger.info("SAVING OPTIMAL WEIGHTS")
        logger.info("="*60)

        for client_type, data in optimal_weights.items():
            file_key = client_type.replace(' ', '_')

            if weights_dir is not None:
                file_path = Path(weights_dir) / f"weights_{file_key}.json"
            else:
                file_path = WEIGHT_FILES.get(file_key, WEIGHT_FILES.get('UNKNOWN'))

            if file_path is None:
                from config import WEIGHTS_DIR
                file_path = WEIGHTS_DIR / f"weights_{file_key}.json"
            
            save_data = {
                'client_type': client_type,
                'weights': data['weights'],
                'validation_ndcg': data.get('validation_ndcg'),
                'n_clients': data.get('n_clients', 0),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"    Saved {client_type} weights to {file_path}")
        
        logger.info("\n" + "="*60)
        logger.info("WEIGHT OPTIMIZATION COMPLETE")
        logger.info("="*60 + "\n")
    
    def load_weights(self, client_type: str) -> Dict[str, float]:
        """Load saved weights for a client type"""
        file_key = client_type.replace(' ', '_')
        file_path = WEIGHT_FILES.get(file_key, WEIGHT_FILES.get('UNKNOWN'))
        
        if file_path is None:
            from config import WEIGHTS_DIR
            file_path = WEIGHTS_DIR / f"weights_{file_key}.json"
        
        try:
            if not Path(file_path).exists():
                logger.warning(f"    Weight file not found at {file_path}, using defaults")
                return self._get_default_weights()
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'weights' not in data:
                logger.warning(f"    No 'weights' key in {file_path}, using defaults")
                return self._get_default_weights()
            
            logger.info(f"    Loaded weights for {client_type} from {file_path}")
            return data['weights']
            
        except Exception as e:
            logger.warning(f"    Error loading weights for {client_type}: {str(e)}")
            return self._get_default_weights()

    def _get_default_weights(self) -> Dict[str, float]:
        """Return default weights"""
        return {
            'alpha1_rrf': 0.3,
            'alpha2_gcn': 0.3,
            'alpha3_item_sim': 0.1,
            'alpha4_user_sim': 0.1,
            'alpha5_confidence': 0.1
        }


def create_ground_truth_from_bookings(lightgcn_df: pd.DataFrame,
                                     collaborative_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ground truth from actual booking history
    """
    logger.info("Creating ground truth from actual booking history...")
    
    client_ids = set()
    
    if not lightgcn_df.empty and 'original_client_id' in lightgcn_df.columns:
        client_ids.update(lightgcn_df['original_client_id'].unique())
    
    if not collaborative_df.empty and 'client_id' in collaborative_df.columns:
        client_ids.update(collaborative_df['client_id'].unique())
    
    if not client_ids:
        logger.warning("No clients found in recommendation data")
        return pd.DataFrame()
    
    logger.info(f"Loading bookings for {len(client_ids)} clients...")
    
    booking_df = DataLoader.load_booking_history(client_ids=list(client_ids))
    
    if booking_df.empty:
        logger.error("No booking history loaded!")
        return pd.DataFrame()
    
    ground_truth = booking_df[['RefClient', 'pair_id']].copy()
    ground_truth.rename(columns={'RefClient': 'client_id'}, inplace=True)
    ground_truth['interaction'] = 1
    
    ground_truth = ground_truth.drop_duplicates(subset=['client_id', 'pair_id'])
    
    logger.info(f"    Created ground truth with {len(ground_truth):,} client-pair interactions")
    logger.info(f"  Unique clients: {ground_truth['client_id'].nunique()}")
    logger.info(f"  Unique pairs: {ground_truth['pair_id'].nunique()}")
    
    return ground_truth


def run_monthly_optimization(weights_dir=None, lightgcn_csv=None, collab_csv=None):
    """
    Run monthly weight optimization pipeline.

    Args:
        weights_dir:    directory to write the optimised weight JSON files.
                        Defaults to the standard combiner/weights/ path.
        lightgcn_csv:   path to the LightGCN recommendations CSV to use for
                        grid search.  When None, falls back to the static
                        comnbiner_script_final/data/grid_search/ file (legacy).
        collab_csv:     path to the CF recommendations CSV to use for grid
                        search.  When None, falls back to the static file.
    """
    logger.info("\n" + "="*60)
    logger.info("MONTHLY WEIGHT OPTIMIZATION PIPELINE")
    logger.info("="*60 + "\n")

    data = DataLoader.load_all_gridsearch_data(
        lightgcn_csv=lightgcn_csv,
        collab_csv=collab_csv,
    )
    lightgcn_df = data['lightgcn']
    collaborative_df = data['collaborative']
    
    if lightgcn_df.empty and collaborative_df.empty:
        logger.error("No data loaded! Please check your data files.")
        return
    
    ground_truth_df = create_ground_truth_from_bookings(lightgcn_df, collaborative_df)
    
    if ground_truth_df.empty:
        logger.error("No ground truth data! Cannot optimize weights.")
        return
    
    logger.info("Loading client types from client_final.csv...")
    client_types = DataLoader.load_client_types()
    
    if not client_types:
        logger.warning("No client types loaded! Using default UNKNOWN for all clients.")
        client_types = {}
        
        if not lightgcn_df.empty and 'original_client_id' in lightgcn_df.columns:
            for client_id in lightgcn_df['original_client_id'].unique():
                client_types[client_id] = 'UNKNOWN'
        
        if not collaborative_df.empty and 'client_id' in collaborative_df.columns:
            for client_id in collaborative_df['client_id'].unique():
                if client_id not in client_types:
                    client_types[client_id] = 'UNKNOWN'
    
    logger.info(f"Client type distribution: {pd.Series(client_types).value_counts().to_dict()}")
    
    optimizer = GridSearchOptimizer(validation_k=10)
    
    optimal_weights = optimizer.optimize_all_client_types(
        lightgcn_df,
        collaborative_df,
        ground_truth_df,
        client_types
    )
    
    optimizer.save_weights(optimal_weights, weights_dir=weights_dir)
    
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*60 + "\n")
    
    logger.info("\n    OPTIMIZATION SUMMARY")
    logger.info("="*60)
    for client_type, data in optimal_weights.items():
        if data['validation_ndcg'] is not None:
            logger.info(f"\n{client_type}:")
            logger.info(f"  NDCG@10: {data['validation_ndcg']:.4f}")
            logger.info(f"  Clients: {data['n_clients']}")
            logger.info(f"  Weights: {data['weights']}")
    logger.info("\n" + "="*60 + "\n")


if __name__ == "__main__":
    run_monthly_optimization()
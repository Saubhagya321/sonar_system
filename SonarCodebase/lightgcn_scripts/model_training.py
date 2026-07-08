

import warnings
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from base_classes import BaseModel, BasePipeline
from config import MODEL_PATHS


# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class TorchConverter:
    """
    Utility class for converting data to PyTorch tensors.
    
    REPLACES: sparse_to_torch() [lines 29-37]
    """
    
    @staticmethod
    def sparse_to_torch(sparse_matrix, device) -> torch.sparse.FloatTensor:
        """
        Convert scipy sparse matrix to torch sparse tensor.
        
        REPLACES: sparse_to_torch() [lines 29-37]
        """
        coo = sparse_matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = coo.shape
        
        sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
        return sparse_tensor.to(device)


class HeterogeneousLightGCN(nn.Module):
    """
    LightGCN for heterogeneous graph with 3 node types and 3 edge types.
    Clients <-> Vehicles, Clients <-> Chauffeurs, Vehicles <-> Chauffeurs
    
    UNCHANGED from original - already a class
    """
    
    def __init__(
        self,
        n_clients: int,
        n_vehicles: int,
        n_chauffeurs: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        super(HeterogeneousLightGCN, self).__init__()
        
        self.n_clients = n_clients
        self.n_vehicles = n_vehicles
        self.n_chauffeurs = n_chauffeurs
        self.n_total_nodes = n_clients + n_vehicles + n_chauffeurs
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.is_trained = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Learnable embeddings for all nodes
        self.client_embedding = nn.Embedding(n_clients, embedding_dim)
        self.vehicle_embedding = nn.Embedding(n_vehicles, embedding_dim)
        self.chauffeur_embedding = nn.Embedding(n_chauffeurs, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.client_embedding.weight)
        nn.init.xavier_uniform_(self.vehicle_embedding.weight)
        nn.init.xavier_uniform_(self.chauffeur_embedding.weight)
        
        # Add batch norm if requested
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(embedding_dim) for _ in range(n_layers)
            ])
        
        self.logger.info(f"Model initialized:")
        self.logger.debug(f"  Embedding dim: {embedding_dim}")
        self.logger.debug(f"  Layers: {n_layers}")
        self.logger.debug(f"  Dropout: {dropout}")
        self.logger.debug(f"  Total parameters: {self.count_parameters():,}")
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_all_embeddings(self) -> torch.Tensor:
        client_emb = self.client_embedding.weight
        vehicle_emb = self.vehicle_embedding.weight
        chauffeur_emb = self.chauffeur_embedding.weight
        return torch.cat([client_emb, vehicle_emb, chauffeur_emb], dim=0)
    
    def forward(self, adj_matrix_sparse: torch.sparse.FloatTensor) -> torch.Tensor:
        all_embeddings = self.get_all_embeddings()
        embeddings_list = [all_embeddings]
        
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix_sparse, all_embeddings)
            if self.use_batch_norm:
                all_embeddings = self.batch_norms[layer](all_embeddings)
            if self.training:
                all_embeddings = F.dropout(all_embeddings, p=self.dropout)
            embeddings_list.append(all_embeddings)
        
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        return final_embeddings
    
    def get_client_embeddings(self, final_embeddings: torch.Tensor) -> torch.Tensor:
        return final_embeddings[:self.n_clients]
    
    def get_vehicle_embeddings(self, final_embeddings: torch.Tensor) -> torch.Tensor:
        return final_embeddings[self.n_clients:self.n_clients + self.n_vehicles]
    
    def get_chauffeur_embeddings(self, final_embeddings: torch.Tensor) -> torch.Tensor:
        return final_embeddings[self.n_clients + self.n_vehicles:]
    
    def predict_cv(self, client_ids, vehicle_ids, final_embeddings) -> torch.Tensor:
        client_emb = final_embeddings[client_ids]
        vehicle_emb = final_embeddings[vehicle_ids]
        return torch.sum(client_emb * vehicle_emb, dim=1)
    
    def predict_cc(self, client_ids, chauffeur_ids, final_embeddings) -> torch.Tensor:
        client_emb = final_embeddings[client_ids]
        chauffeur_emb = final_embeddings[chauffeur_ids]
        return torch.sum(client_emb * chauffeur_emb, dim=1)
    
    def bpr_loss(self, pos_scores, neg_scores) -> torch.Tensor:
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    
    def l2_regularization(self, *embeddings) -> torch.Tensor:
        reg_loss = 0
        for emb in embeddings:
            reg_loss += torch.sum(emb ** 2)
        return reg_loss
    
    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, weights_only=True))
        self.is_trained = True
        self.logger.info(f"Model loaded from {path}")


class GraphDataLoader:
    """
    Loads graph data from Step 4 results for training.
    
    REPLACES: load_graph_data() [lines 172-213]
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self, step4_results: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("\n=== Loading Graph Data ===")
        
        splits = step4_results['splits']
        cv_train = splits['cv_train']
        cc_train = splits['cc_train']
        cv_test = splits['cv_test']
        cc_test = splits['cc_test']
        vc_train = splits['vc_train']
        vc_test = splits['vc_test']
        
        mappings = step4_results['mappings']
        n_clients = mappings['n_clients']
        n_vehicles = mappings['n_vehicles']
        n_chauffeurs = mappings['n_chauffeurs']
        
        normalized_adj = step4_results['normalized_adj']
        
        self.logger.info(f"Loaded data:")
        self.logger.debug(f"  Clients: {n_clients:,}")
        self.logger.debug(f"  Vehicles: {n_vehicles:,}")
        self.logger.debug(f"  Chauffeurs: {n_chauffeurs:,}")
        self.logger.debug(f"  C-V Train: {len(cv_train):,}, Test: {len(cv_test):,}")
        self.logger.debug(f"  C-C Train: {len(cc_train):,}, Test: {len(cc_test):,}")
        self.logger.debug(f"  V-C Train: {len(vc_train):,}, Test: {len(vc_test):,}")
        
        
        return {
            'cv_train': cv_train, 'cc_train': cc_train,
            'cv_test': cv_test, 'cc_test': cc_test,
            'vc_train': vc_train, 'vc_test': vc_test,
            'n_clients': n_clients, 'n_vehicles': n_vehicles,
            'n_chauffeurs': n_chauffeurs, 'normalized_adj': normalized_adj,
            'mappings': mappings
        }


class NegativeSampler:
    """
    Creates negative samplers for BPR training.
    
    REPLACES: create_negative_sampler(), create_vc_negative_sampler()
    
    FIXED: Returns callable functions (like original) instead of dicts
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_cv_cc_sampler(self, train_interactions: pd.DataFrame, n_items: int, 
                              n_negatives: int = 4) -> Callable:
        """
        Create negative sampler for C-V or C-C interactions.
        Returns a callable function that samples negatives.
        
        FIXED: Returns callable (like original), stores 0-based item IDs
        """
        user_items = {}
        
        # Determine columns
        if 'vehicle_node_id' in train_interactions.columns:
            user_col, item_col = 'client_node_id', 'vehicle_node_id'
        else:
            user_col, item_col = 'client_node_id', 'chauffeur_node_id'
        
        # Build user interaction dict with 0-based item IDs
        for _, row in train_interactions.iterrows():
            user_id = row[user_col]
            item_id = row[item_col]  # This is already 0-based for the item type
            if user_id not in user_items:
                user_items[user_id] = set()
            user_items[user_id].add(item_id)
        
        def sample_negatives(user_id: int, n_samples: int = None) -> List[int]:
            """Sample negative items for a user"""
            if n_samples is None:
                n_samples = n_negatives
            user_positives = user_items.get(user_id, set())
            negatives = []
            
            while len(negatives) < n_samples:
                neg_item = np.random.randint(0, n_items)
                if neg_item not in user_positives:
                    negatives.append(neg_item)
            
            return negatives
        
        return sample_negatives
    
    def create_vc_sampler(self, vc_train: pd.DataFrame, n_chauffeurs: int,
                          chauffeur_offset: int, n_negatives: int = 4) -> Callable:
        """
        Create negative sampler for V-C interactions.
        Returns a callable function that samples negatives.
        
        FIXED: Stores 0-based chauffeur IDs (subtracts offset like original)
        """
        vehicle_chauffeurs = {}
        
        for _, row in vc_train.iterrows():
            vehicle_id = row['vehicle_node_id']  # Keep as global ID (key)
            # FIXED: Convert chauffeur to 0-based for sampling comparison
            chauffeur_0based = row['chauffeur_node_id'] - chauffeur_offset
            if vehicle_id not in vehicle_chauffeurs:
                vehicle_chauffeurs[vehicle_id] = set()
            vehicle_chauffeurs[vehicle_id].add(chauffeur_0based)
        
        def sample_negatives(vehicle_id_global: int, n_samples: int = None) -> List[int]:
            """Sample negative chauffeurs for a vehicle"""
            if n_samples is None:
                n_samples = n_negatives
            vehicle_positives = vehicle_chauffeurs.get(vehicle_id_global, set())
            negatives = []
            
            while len(negatives) < n_samples:
                neg_chauffeur_0based = np.random.randint(0, n_chauffeurs)
                if neg_chauffeur_0based not in vehicle_positives:
                    negatives.append(neg_chauffeur_0based)
            
            return negatives
        
        return sample_negatives


class ModelTrainer:
    """
    Handles model training logic.
    
    REPLACES: train_epoch() [lines 290-403]
    
    FIXED: Matches original structure - processes edge types separately,
           uses all negatives, correct offset handling
    """
    
    def __init__(self, device=device):
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train_epoch(self, model, adj_sparse, cv_train, cc_train, vc_train,
                    cv_neg_sampler, cc_neg_sampler, vc_neg_sampler,
                    n_vehicles, n_chauffeurs, vehicle_offset, chauffeur_offset,
                    optimizer, reg_weight=1e-5, batch_size=2048,
                    cv_weight=1.0, cc_weight=1.0, vc_weight=1.0, n_negatives=4) -> float:
        """
        Train for one epoch - FIXED: Uses all negative samples like original
        """
        model.train()
        
        # Forward pass to get embeddings
        final_embeddings = model(adj_sparse)
        
        total_loss = 0
        n_batches = 0
        
        # Get training data as writable numpy arrays (.values can return a
        # read-only view when the source columns are pyarrow-backed, which
        # breaks the in-place shuffle below).
        cv_train_data = cv_train[['client_node_id', 'vehicle_node_id']].to_numpy(copy=True)
        cc_train_data = cc_train[['client_node_id', 'chauffeur_node_id']].to_numpy(copy=True)
        vc_train_data = vc_train[['vehicle_node_id', 'chauffeur_node_id']].to_numpy(copy=True)
        
        # Shuffle training data
        np.random.shuffle(cv_train_data)
        np.random.shuffle(cc_train_data)
        np.random.shuffle(vc_train_data)
        
        # ================================================================
        # Train on C-V edges - FIXED: Uses all negatives
        # ================================================================
        for start_idx in range(0, len(cv_train_data), batch_size):
            end_idx = min(start_idx + batch_size, len(cv_train_data))
            batch = cv_train_data[start_idx:end_idx]
            
            # FIXED: Expand to use ALL negative samples
            client_ids_expanded = []
            pos_vehicle_ids_expanded = []
            neg_vehicle_ids = []
            
            for client_id, pos_vehicle in batch:
                negs = cv_neg_sampler(client_id, n_samples=n_negatives)
                for neg in negs:  # USE ALL NEGATIVES
                    client_ids_expanded.append(client_id)
                    pos_vehicle_ids_expanded.append(pos_vehicle)
                    neg_vehicle_ids.append(neg + vehicle_offset)
            
            client_ids = torch.LongTensor(client_ids_expanded).to(self.device)
            pos_vehicle_ids = torch.LongTensor(pos_vehicle_ids_expanded).to(self.device)
            neg_vehicle_ids_t = torch.LongTensor(neg_vehicle_ids).to(self.device)
            
            # Get scores
            pos_scores = model.predict_cv(client_ids, pos_vehicle_ids, final_embeddings)
            neg_scores = model.predict_cv(client_ids, neg_vehicle_ids_t, final_embeddings)
            
            # BPR loss
            loss = model.bpr_loss(pos_scores, neg_scores)
            
            # L2 regularization
            client_emb = final_embeddings[client_ids]
            pos_vehicle_emb = final_embeddings[pos_vehicle_ids]
            neg_vehicle_emb = final_embeddings[neg_vehicle_ids_t]
            reg_loss = model.l2_regularization(client_emb, pos_vehicle_emb, neg_vehicle_emb)
            
            total_batch_loss = cv_weight * loss + reg_weight * reg_loss
            
            # Backward
            optimizer.zero_grad()
            total_batch_loss.backward(retain_graph=True)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            n_batches += 1
        
        # ================================================================
        # Train on C-C edges - FIXED: Uses all negatives
        # ================================================================
        for start_idx in range(0, len(cc_train_data), batch_size):
            end_idx = min(start_idx + batch_size, len(cc_train_data))
            batch = cc_train_data[start_idx:end_idx]
            
            # FIXED: Expand to use ALL negative samples
            client_ids_expanded = []
            pos_chauffeur_ids_expanded = []
            neg_chauffeur_ids = []
            
            for client_id, pos_chauffeur in batch:
                negs = cc_neg_sampler(client_id, n_samples=n_negatives)
                for neg in negs:  # USE ALL NEGATIVES
                    client_ids_expanded.append(client_id)
                    pos_chauffeur_ids_expanded.append(pos_chauffeur)
                    neg_chauffeur_ids.append(neg + chauffeur_offset)
            
            client_ids = torch.LongTensor(client_ids_expanded).to(self.device)
            pos_chauffeur_ids = torch.LongTensor(pos_chauffeur_ids_expanded).to(self.device)
            neg_chauffeur_ids_t = torch.LongTensor(neg_chauffeur_ids).to(self.device)
            
            # Get scores
            pos_scores = model.predict_cc(client_ids, pos_chauffeur_ids, final_embeddings)
            neg_scores = model.predict_cc(client_ids, neg_chauffeur_ids_t, final_embeddings)
            
            # BPR loss
            loss = model.bpr_loss(pos_scores, neg_scores)
            
            # L2 regularization
            client_emb = final_embeddings[client_ids]
            pos_chauffeur_emb = final_embeddings[pos_chauffeur_ids]
            neg_chauffeur_emb = final_embeddings[neg_chauffeur_ids_t]
            reg_loss = model.l2_regularization(client_emb, pos_chauffeur_emb, neg_chauffeur_emb)
            
            total_batch_loss = cc_weight * loss + reg_weight * reg_loss
            
            # Backward
            optimizer.zero_grad()
            total_batch_loss.backward(retain_graph=True)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            n_batches += 1
        
        # ================================================================
        # Train on V-C edges - FIXED: Uses all negatives
        # ================================================================
        for start_idx in range(0, len(vc_train_data), batch_size):
            end_idx = min(start_idx + batch_size, len(vc_train_data))
            batch = vc_train_data[start_idx:end_idx]
            
            # FIXED: Expand to use ALL negative samples
            vehicle_ids_expanded = []
            pos_chauffeur_ids_expanded = []
            neg_chauffeur_ids = []
            
            for vehicle_id, pos_chauffeur in batch:
                negs = vc_neg_sampler(vehicle_id, n_samples=n_negatives)
                for neg in negs:  # USE ALL NEGATIVES
                    vehicle_ids_expanded.append(vehicle_id)
                    pos_chauffeur_ids_expanded.append(pos_chauffeur)
                    neg_chauffeur_ids.append(neg + chauffeur_offset)
            
            vehicle_ids = torch.LongTensor(vehicle_ids_expanded).to(self.device)
            pos_chauffeur_ids = torch.LongTensor(pos_chauffeur_ids_expanded).to(self.device)
            neg_chauffeur_ids_t = torch.LongTensor(neg_chauffeur_ids).to(self.device)
            
            # Get scores
            vehicle_emb = final_embeddings[vehicle_ids]
            pos_chauffeur_emb = final_embeddings[pos_chauffeur_ids]
            neg_chauffeur_emb = final_embeddings[neg_chauffeur_ids_t]
            
            pos_scores = torch.sum(vehicle_emb * pos_chauffeur_emb, dim=1)
            neg_scores = torch.sum(vehicle_emb * neg_chauffeur_emb, dim=1)
            
            # BPR loss
            loss = model.bpr_loss(pos_scores, neg_scores)
            
            # L2 regularization
            reg_loss = model.l2_regularization(vehicle_emb, pos_chauffeur_emb, neg_chauffeur_emb)
            
            total_batch_loss = vc_weight * loss + reg_weight * reg_loss
            
            # Backward
            optimizer.zero_grad()
            total_batch_loss.backward(retain_graph=True)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        return avg_loss


class ModelEvaluator:
    """
    Handles model evaluation and metrics computation.
    
    REPLACES: evaluate_model() [lines 406-500]
    
    FIXED: Correct column names, correct offset handling, correct ID comparisons
    """
    
    def __init__(self, device=device):
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @torch.no_grad()
    def evaluate(self, model, adj_sparse, test_data, train_data,
                 item_offset, n_items, top_k=[10, 20, 50], interaction_type='cv'):
        """
        Evaluate model on test set.
        
        FIXED: Uses correct column names (_node_id suffix)
        FIXED: Correct offset handling for train/test dicts
        FIXED: Correct user embedding lookup for VC
        """
        model.eval()
        
        # Get final embeddings
        final_embeddings = model(adj_sparse)
        
        # FIXED: Use correct column names with _node_id suffix
        if interaction_type == 'cv':
            user_col = 'client_node_id'
            item_col = 'vehicle_node_id'
        elif interaction_type == 'cc':
            user_col = 'client_node_id'
            item_col = 'chauffeur_node_id'
        elif interaction_type == 'vc':
            user_col = 'vehicle_node_id'
            item_col = 'chauffeur_node_id'
        else:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")
        
        # Build train interaction dict for filtering
        # FIXED: Store 0-based item IDs (subtract offset)
        train_dict = {}
        for _, row in train_data.iterrows():
            user = row[user_col]
            item = row[item_col] - item_offset  # Convert to 0-based
            if user not in train_dict:
                train_dict[user] = set()
            train_dict[user].add(item)
        
        # Build test ground truth
        # FIXED: Store 0-based item IDs (subtract offset)
        test_dict = {}
        for _, row in test_data.iterrows():
            user = row[user_col]
            item = row[item_col] - item_offset  # Convert to 0-based
            if user not in test_dict:
                test_dict[user] = []
            test_dict[user].append(item)
        
        # Evaluate
        recalls = {k: [] for k in top_k}
        ndcgs = {k: [] for k in top_k}
        
        for user_id in test_dict.keys():
            # FIXED: Get user embedding - user_id is already global for VC
            # No need to add extra offset
            user_emb = final_embeddings[user_id].unsqueeze(0)
            
            # Get all item embeddings
            item_start = item_offset
            item_end = item_offset + n_items
            item_embs = final_embeddings[item_start:item_end]
            
            # Compute scores
            scores = torch.matmul(user_emb, item_embs.T).squeeze()
            
            # Filter out training items (already 0-based in train_dict)
            if user_id in train_dict:
                train_items = list(train_dict[user_id])
                scores[train_items] = -float('inf')
            
            # Get top-K recommendations
            max_k = max(top_k)
            actual_k = min(max_k, len(scores))
            _, top_indices = torch.topk(scores, actual_k)
            top_indices = top_indices.cpu().numpy()
            
            # Ground truth (already 0-based in test_dict)
            ground_truth = set(test_dict[user_id])
            
            # Calculate metrics for each K
            for k in top_k:
                k_actual = min(k, len(top_indices))
                top_k_items = set(top_indices[:k_actual])
                
                # Recall@K
                hits = len(top_k_items & ground_truth)
                recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0
                recalls[k].append(recall)
                
                # NDCG@K
                dcg = 0
                for idx, item in enumerate(top_indices[:k_actual]):
                    if item in ground_truth:
                        dcg += 1 / np.log2(idx + 2)
                
                idcg = sum([1 / np.log2(i + 2) for i in range(min(len(ground_truth), k_actual))])
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs[k].append(ndcg)
        
        # Average metrics
        results = {}
        for k in top_k:
            results[f'Recall@{k}'] = np.mean(recalls[k])
            results[f'NDCG@{k}'] = np.mean(ndcgs[k])
        
        return results


class RecommendationGenerator:
    """
    Generates recommendations from trained model.
    
    REPLACES: generate_recommendations() [lines 751-788]
    """
    
    def __init__(self, device=device):
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @torch.no_grad()
    def generate_basic(self, model, adj_sparse, client_ids, vehicle_offset,
                       chauffeur_offset, n_vehicles, n_chauffeurs, top_k=10):
        """
        Generate top-K vehicle and chauffeur recommendations for clients
        """
        model.eval()
        final_embeddings = model(adj_sparse)
        
        recommendations = []
        for client_id in client_ids:
            # Get client embedding
            client_emb = final_embeddings[client_id].unsqueeze(0)
            
            # Vehicle recommendations
            vehicle_embs = final_embeddings[vehicle_offset:vehicle_offset + n_vehicles]
            vehicle_scores = torch.matmul(client_emb, vehicle_embs.T).squeeze()
            top_vehicles = torch.topk(vehicle_scores, min(top_k, n_vehicles)).indices.cpu().numpy()
            
            # Chauffeur recommendations
            chauffeur_embs = final_embeddings[chauffeur_offset:chauffeur_offset + n_chauffeurs]
            chauffeur_scores = torch.matmul(client_emb, chauffeur_embs.T).squeeze()
            top_chauffeurs = torch.topk(chauffeur_scores, min(top_k, n_chauffeurs)).indices.cpu().numpy()
            
            recommendations.append({
                'client_node_id': client_id,
                'top_vehicles': top_vehicles.tolist(),
                'top_chauffeurs': top_chauffeurs.tolist(),
                'vehicle_scores': vehicle_scores[top_vehicles].cpu().numpy().tolist(),
                'chauffeur_scores': chauffeur_scores[top_chauffeurs].cpu().numpy().tolist()
            })
        
        return recommendations


class ModelTrainingPipeline(BasePipeline):
    """
    Orchestrates the model training process.
    
    REPLACES: train_heterogeneous_lightgcn(), execute_step5_training()
    
    FIXED: Correct defaults (n_negatives=4, vc_weight=1.0)
    """
    
    DEFAULT_CONFIG = {
        'embedding_dim': 64,
        'n_layers': 3,
        'learning_rate': 0.001,
        'reg_weight': 1e-5,
        'n_epochs': 100,
        'batch_size': 2048,
        'patience': 10,
        'eval_every': 5,
        'dropout': 0.1,
        'use_batch_norm': False,
        'n_negatives': 4,  # FIXED: Was 1, should be 4
        'cv_weight': 1.0,
        'cc_weight': 1.0,
        'vc_weight': 1.0,  # FIXED: Was 0.5, should be 1.0
        'use_scheduler': True,
        'scheduler_type': 'cosine',
        'warmup_epochs': 5
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or self.DEFAULT_CONFIG)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = device
        self.data_loader = GraphDataLoader()
        self.trainer = ModelTrainer(device)
        self.evaluator = ModelEvaluator(device)
        self.negative_sampler = NegativeSampler()
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self._log_start("Step 5: Model Training")
        
        self.logger.debug(f"\nHyperparameters:")
        self.logger.debug(f"  Embedding dim: {self.config['embedding_dim']}")
        self.logger.debug(f"  Layers: {self.config['n_layers']}")
        self.logger.debug(f"  Learning rate: {self.config['learning_rate']}")
        self.logger.debug(f"  Reg weight: {self.config['reg_weight']}")
        self.logger.debug(f"  Epochs: {self.config['n_epochs']}")
        self.logger.debug(f"  Batch size: {self.config['batch_size']}")
        self.logger.debug(f"  Patience: {self.config['patience']}")
        self.logger.debug(f"  N negatives: {self.config['n_negatives']}")
        
        try:
            # Load data
            data = self.data_loader.load(input_data)
            
            # Initialize model
            model = HeterogeneousLightGCN(
                n_clients=data['n_clients'],
                n_vehicles=data['n_vehicles'],
                n_chauffeurs=data['n_chauffeurs'],
                embedding_dim=self.config['embedding_dim'],
                n_layers=self.config['n_layers'],
                dropout=self.config['dropout'],
                use_batch_norm=self.config['use_batch_norm']
            ).to(self.device)
            
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            
            # Scheduler
            scheduler = None
            if self.config.get('use_scheduler', False):
                scheduler = self._create_scheduler(optimizer)
            
            # Convert adjacency matrix to torch sparse
            adj_sparse = TorchConverter.sparse_to_torch(data['normalized_adj'], self.device)
            
            # Calculate offsets
            vehicle_offset = data['n_clients']
            chauffeur_offset = data['n_clients'] + data['n_vehicles']
            
            # Create negative samplers (returns callable functions)
            cv_neg_sampler = self.negative_sampler.create_cv_cc_sampler(
                data['cv_train'], data['n_vehicles'], self.config['n_negatives'])
            cc_neg_sampler = self.negative_sampler.create_cv_cc_sampler(
                data['cc_train'], data['n_chauffeurs'], self.config['n_negatives'])
            vc_neg_sampler = self.negative_sampler.create_vc_sampler(
                data['vc_train'], data['n_chauffeurs'], chauffeur_offset, self.config['n_negatives'])
            
            # Training loop
            best_recall = 0
            patience_counter = 0
            training_history = []
            current_recall = 0
            
            self.logger.info("\nStarting training...")
            for epoch in range(1, self.config['n_epochs'] + 1):
                # Train
                train_loss = self.trainer.train_epoch(
                    model, adj_sparse,
                    data['cv_train'], data['cc_train'], data['vc_train'],
                    cv_neg_sampler, cc_neg_sampler, vc_neg_sampler,
                    data['n_vehicles'], data['n_chauffeurs'],
                    vehicle_offset, chauffeur_offset,
                    optimizer, self.config['reg_weight'], self.config['batch_size'],
                    self.config['cv_weight'], self.config['cc_weight'], self.config['vc_weight'],
                    self.config['n_negatives']
                )
                
                # Step scheduler (for non-plateau schedulers)
                if scheduler and self.config['scheduler_type'] != 'reduce_on_plateau':
                    scheduler.step()
                
                # Evaluate
                if epoch % self.config['eval_every'] == 0 or epoch == 1:
                    metrics, current_recall = self._evaluate_and_log(
                        model, adj_sparse, data, epoch, train_loss, vehicle_offset, chauffeur_offset)
                    
                    training_history.append({'epoch': epoch, 'train_loss': train_loss, **metrics})
                    
                    # Early stopping based on C-V Recall@20
                    if current_recall > best_recall:
                        best_recall = current_recall
                        patience_counter = 0
                        torch.save(model.state_dict(), MODEL_PATHS['training_checkpoint'])
                        self.logger.debug(f"  New best model saved (Recall@20: {best_recall:.4f})")
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config['patience']:
                            self.logger.debug(f"\nEarly stopping at epoch {epoch}")
                            break
                    
                    # Step plateau scheduler with metric
                    if scheduler and self.config['scheduler_type'] == 'reduce_on_plateau':
                        scheduler.step(current_recall)
                else:
                    self.logger.debug(f"Epoch {epoch}/{self.config['n_epochs']} - Loss: {train_loss:.4f}")
            
            # Load best model
            model.load_state_dict(torch.load(MODEL_PATHS['training_checkpoint'], weights_only=True))
            model.is_trained = True
            
            # Preserve original adjacency matrix
            data['normalized_adj'] = input_data['normalized_adj']
            
            # Save training history
            history_df = pd.DataFrame(training_history)
            history_df.to_csv(MODEL_PATHS['training_history'], index=False)
            self.logger.debug(f"\nTraining history saved to {MODEL_PATHS['training_history']}")
            
            # Prepare outputs
            step5_outputs = {
                **input_data,
                'model': model,
                'training_history': training_history,
                'data': data
            }
            
            self._log_summary(best_recall)
            self._log_complete("Step 5")
            
            return step5_outputs
            
        except Exception as e:
            self.logger.error(f"Error in Step 5: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler_type', 'cosine')
        lr = self.config['learning_rate']
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=lr * 0.01
            )
        elif scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.5
            )
        return None
    
    def _evaluate_and_log(self, model, adj_sparse, data, epoch, train_loss, vehicle_offset, chauffeur_offset):
        """Evaluate model and log metrics"""
        self.logger.debug(f"\nEpoch {epoch}/{self.config['n_epochs']}")
        self.logger.debug(f"  Train Loss: {train_loss:.4f}")
        
        all_metrics = {}
        
        # Evaluate on C-V test set
        cv_metrics = self.evaluator.evaluate(
            model, adj_sparse,
            data['cv_test'], data['cv_train'],
            vehicle_offset, data['n_vehicles'],
            top_k=[10, 20, 50],
            interaction_type='cv'
        )
        self.logger.debug(f"  C-V Metrics:")
        for metric, value in cv_metrics.items():
            self.logger.debug(f"    {metric}: {value:.4f}")
        all_metrics.update({f'cv_{k}': v for k, v in cv_metrics.items()})
        
        # Evaluate on C-C test set
        cc_metrics = self.evaluator.evaluate(
            model, adj_sparse,
            data['cc_test'], data['cc_train'],
            chauffeur_offset, data['n_chauffeurs'],
            top_k=[10, 20, 50],
            interaction_type='cc'
        )
        self.logger.debug(f"  C-C Metrics:")
        for metric, value in cc_metrics.items():
            self.logger.debug(f"    {metric}: {value:.4f}")
        all_metrics.update({f'cc_{k}': v for k, v in cc_metrics.items()})
        
        # Evaluate on V-C test set
        vc_metrics = self.evaluator.evaluate(
            model, adj_sparse,
            data['vc_test'], data['vc_train'],
            chauffeur_offset, data['n_chauffeurs'],
            top_k=[10, 20, 50, 100],
            interaction_type='vc'
        )
       
        all_metrics.update({f'vc_{k}': v for k, v in vc_metrics.items()})
        
        return all_metrics, cv_metrics['Recall@20']
    
    def _log_summary(self, best_recall: float) -> None:
        """Log training summary"""
        self.logger.debug("\n" + "=" * 60)
        self.logger.info("Step 5 completed successfully!")
        self.logger.debug(f"\nBest Recall@20: {best_recall:.4f}")
        self.logger.debug(f"Training history saved to: {MODEL_PATHS['training_history']}")
        self.logger.info("\nReady for Step 6: Recommendation Generation!")



# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def sparse_to_torch(sparse_matrix, device):
    """Convert scipy sparse matrix to torch sparse tensor"""
    return TorchConverter.sparse_to_torch(sparse_matrix, device)


def load_graph_data(step4_results):
    """Load graph data from Step 4 results"""
    loader = GraphDataLoader()
    return loader.load(step4_results)


def create_negative_sampler(train_interactions, n_items, n_negatives=4):
    """Create negative sampler for training"""
    sampler = NegativeSampler()
    return sampler.create_cv_cc_sampler(train_interactions, n_items, n_negatives)


def create_vc_negative_sampler(vc_train, n_chauffeurs, chauffeur_offset, n_negatives=4):
    """Create negative sampler for V-C interactions"""
    sampler = NegativeSampler()
    return sampler.create_vc_sampler(vc_train, n_chauffeurs, chauffeur_offset, n_negatives)


def train_epoch(model, adj_sparse, cv_train, cc_train, vc_train,
                cv_neg_sampler, cc_neg_sampler, vc_neg_sampler,
                n_vehicles, n_chauffeurs, vehicle_offset, chauffeur_offset,
                optimizer, reg_weight=1e-5, batch_size=2048,
                cv_weight=1.0, cc_weight=1.0, vc_weight=1.0, n_negatives=4):
    """Train for one epoch"""
    trainer = ModelTrainer(device)
    return trainer.train_epoch(
        model, adj_sparse, cv_train, cc_train, vc_train,
        cv_neg_sampler, cc_neg_sampler, vc_neg_sampler,
        n_vehicles, n_chauffeurs, vehicle_offset, chauffeur_offset,
        optimizer, reg_weight, batch_size, cv_weight, cc_weight, vc_weight, n_negatives
    )


@torch.no_grad()
def evaluate_model(model, adj_sparse, test_data, train_data, item_offset, n_items, 
                   top_k=[10, 20, 50], interaction_type='cv'):
    """Evaluate model on test set"""
    evaluator = ModelEvaluator(device)
    return evaluator.evaluate(model, adj_sparse, test_data, train_data, item_offset, n_items, top_k, interaction_type)


def train_heterogeneous_lightgcn(step4_results, embedding_dim=64, n_layers=3,
                                  learning_rate=0.001, reg_weight=1e-5,
                                  n_epochs=100, batch_size=2048, patience=10,
                                  eval_every=5, use_scheduler=True,
                                  scheduler_type='cosine', warmup_epochs=5,
                                  dropout=0.1, n_negatives=4, use_batch_norm=False,
                                  cv_weight=1.0, cc_weight=1.0, vc_weight=1.0):
    """Train heterogeneous LightGCN model"""
    config = {
        'embedding_dim': embedding_dim,
        'n_layers': n_layers,
        'learning_rate': learning_rate,
        'reg_weight': reg_weight,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'patience': patience,
        'eval_every': eval_every,
        'use_scheduler': use_scheduler,
        'scheduler_type': scheduler_type,
        'warmup_epochs': warmup_epochs,
        'dropout': dropout,
        'n_negatives': n_negatives,
        'use_batch_norm': use_batch_norm,
        'cv_weight': cv_weight,
        'cc_weight': cc_weight,
        'vc_weight': vc_weight
    }
    
    pipeline = ModelTrainingPipeline(config)
    results = pipeline.execute(step4_results)
    
    if results is None:
        return None, [], None
    
    return results['model'], results['training_history'], results['data']


@torch.no_grad()
def generate_recommendations(model, adj_sparse, client_ids, vehicle_offset, chauffeur_offset,
                             n_vehicles, n_chauffeurs, top_k=10):
    """Generate top-K vehicle and chauffeur recommendations for clients"""
    generator = RecommendationGenerator(device)
    return generator.generate_basic(
        model, adj_sparse, client_ids, vehicle_offset, chauffeur_offset,
        n_vehicles, n_chauffeurs, top_k
    )


def execute_step5_training(step4_results, config=None):
    """Execute Step 5: Model training"""
    pipeline = ModelTrainingPipeline(config)
    return pipeline.execute(step4_results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("Model Training Module (Refactored - Corrected) - Ready")
    logger.debug("\nClasses available:")
    logger.debug("  - TorchConverter")
    logger.debug("  - HeterogeneousLightGCN")
    logger.debug("  - GraphDataLoader")
    logger.debug("  - NegativeSampler")
    logger.debug("  - ModelTrainer")
    logger.debug("  - ModelEvaluator")
    logger.debug("  - RecommendationGenerator")
    logger.info("  - ModelTrainingPipeline")
    logger.debug("\nBackward compatible functions:")
    logger.debug("  - sparse_to_torch()")
    logger.debug("  - load_graph_data()")
    logger.debug("  - create_negative_sampler()")
    logger.debug("  - create_vc_negative_sampler()")
    logger.debug("  - train_epoch()")
    logger.debug("  - evaluate_model()")
    logger.debug("  - train_heterogeneous_lightgcn()")
    logger.debug("  - generate_recommendations()")
    logger.debug("  - execute_step5_training()")

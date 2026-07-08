
import warnings
from typing import Dict, Any, Optional, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from base_classes import BaseRecommender
from model_training import sparse_to_torch, device
from config import RECOMMENDATION_FILES


class ComprehensiveRecommender(BaseRecommender):
    """
    Generates comprehensive recommendations using all three edge types.
    
    REPLACES: generate_comprehensive_recommendations() [lines 23-186]
    """
    
    def __init__(self, device=device):
        """
        Initialize comprehensive recommender.
        
        Args:
            device: PyTorch device
        """
        super().__init__()
        self.device = device
    
    @torch.no_grad()
    def recommend(
        self,
        model,
        adj_sparse: torch.sparse.FloatTensor,
        client_ids: Optional[List[int]],
        vehicle_ids: Optional[List[int]],
        chauffeur_ids: Optional[List[int]],
        vehicle_offset: int,
        chauffeur_offset: int,
        n_vehicles: int,
        n_chauffeurs: int,
        top_k: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate comprehensive recommendations using all three edge types.
        
        REPLACES: generate_comprehensive_recommendations() [lines 23-186]
        
        Args:
            model: Trained LightGCN model
            adj_sparse: Sparse adjacency matrix
            client_ids: List of client node IDs
            vehicle_ids: List of vehicle indices
            chauffeur_ids: List of chauffeur indices
            vehicle_offset: Offset for vehicle nodes
            chauffeur_offset: Offset for chauffeur nodes
            n_vehicles: Number of vehicles
            n_chauffeurs: Number of chauffeurs
            top_k: Number of recommendations
            
        Returns:
            Dictionary with client, vehicle, and chauffeur recommendations
        """
        model.eval()
        
        # Get embeddings
        final_embeddings = model(adj_sparse)
        
        recommendations = {
            'client_recommendations': [],
            'vehicle_recommendations': [],
            'chauffeur_recommendations': []
        }
        
        # Client Recommendations (C-V, C-C)
        if client_ids is not None:
            recommendations['client_recommendations'] = self._generate_client_recommendations(
                final_embeddings, client_ids, vehicle_offset, chauffeur_offset,
                n_vehicles, n_chauffeurs, top_k
            )
        
        # Vehicle Recommendations (V-C, reverse C-V)
        if vehicle_ids is not None:
            recommendations['vehicle_recommendations'] = self._generate_vehicle_recommendations(
                final_embeddings, vehicle_ids, vehicle_offset, chauffeur_offset,
                n_vehicles, n_chauffeurs, top_k
            )
        
        # Chauffeur Recommendations (reverse C-C, reverse V-C)
        if chauffeur_ids is not None:
            recommendations['chauffeur_recommendations'] = self._generate_chauffeur_recommendations(
                final_embeddings, chauffeur_ids, vehicle_offset, chauffeur_offset,
                n_vehicles, n_chauffeurs, top_k
            )
        
        return recommendations
    
    def _generate_client_recommendations(
        self,
        final_embeddings: torch.Tensor,
        client_ids: List[int],
        vehicle_offset: int,
        chauffeur_offset: int,
        n_vehicles: int,
        n_chauffeurs: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for clients."""
        client_recs = []
        
        for client_id in client_ids:
            client_emb = final_embeddings[client_id].unsqueeze(0)
            
            # Vehicle recommendations (C-V)
            vehicle_embs = final_embeddings[vehicle_offset:vehicle_offset + n_vehicles]
            vehicle_scores = torch.matmul(client_emb, vehicle_embs.T).squeeze()
            top_vehicles_idx = torch.topk(vehicle_scores, min(top_k, len(vehicle_scores))).indices
            top_vehicles = top_vehicles_idx.cpu().numpy()
            top_vehicle_scores = vehicle_scores[top_vehicles_idx].cpu().numpy()
            
            # Chauffeur recommendations (C-C)
            chauffeur_embs = final_embeddings[chauffeur_offset:chauffeur_offset + n_chauffeurs]
            chauffeur_scores = torch.matmul(client_emb, chauffeur_embs.T).squeeze()
            top_chauffeurs_idx = torch.topk(chauffeur_scores, min(top_k, len(chauffeur_scores))).indices
            top_chauffeurs = top_chauffeurs_idx.cpu().numpy()
            top_chauffeur_scores = chauffeur_scores[top_chauffeurs_idx].cpu().numpy()
            
            # Vehicle-Chauffeur pair recommendations
            vc_pairs = self._generate_vc_pairs_for_client(
                final_embeddings, client_emb, vehicle_scores, chauffeur_scores,
                chauffeur_embs, top_vehicles, vehicle_offset, chauffeur_offset, top_k
            )
            
            client_recs.append({
                'client_node_id': int(client_id),
                'top_vehicles': top_vehicles.tolist(),
                'vehicle_scores': top_vehicle_scores.tolist(),
                'top_chauffeurs': top_chauffeurs.tolist(),
                'chauffeur_scores': top_chauffeur_scores.tolist(),
                'top_vc_pairs': vc_pairs[:top_k]
            })
        
        return client_recs
    
    def _generate_vc_pairs_for_client(
        self,
        final_embeddings: torch.Tensor,
        client_emb: torch.Tensor,
        vehicle_scores: torch.Tensor,
        chauffeur_scores: torch.Tensor,
        chauffeur_embs: torch.Tensor,
        top_vehicles: np.ndarray,
        vehicle_offset: int,
        chauffeur_offset: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Generate V-C pairs for a specific client."""
        vc_pairs = []
        
        n_vehicles_to_consider = min(20, len(top_vehicles))
        n_chauffeurs_per_vehicle = max(5, (top_k // n_vehicles_to_consider) + 1)
        
        for vehicle_idx in top_vehicles[:n_vehicles_to_consider]:
            vehicle_node_id = vehicle_offset + vehicle_idx
            vehicle_emb = final_embeddings[vehicle_node_id].unsqueeze(0)
            
            # Score chauffeurs based on C-C and V-C compatibility
            vc_scores = (
                0.4 * chauffeur_scores +
                0.4 * torch.matmul(vehicle_emb, chauffeur_embs.T).squeeze()
            )
            
            top_vc_chauffeurs_idx = torch.topk(
                vc_scores, min(n_chauffeurs_per_vehicle, len(vc_scores))
            ).indices
            
            for chauffeur_idx in top_vc_chauffeurs_idx.cpu().numpy():
                embedding_score = float(vc_scores[chauffeur_idx].cpu().numpy())
                
                vc_pairs.append({
                    'vehicle_idx': int(vehicle_idx),
                    'chauffeur_idx': int(chauffeur_idx),
                    'combined_score': embedding_score,
                    'cv_score': float(vehicle_scores[vehicle_idx].cpu().numpy()),
                    'cc_score': float(chauffeur_scores[chauffeur_idx].cpu().numpy()),
                    'vc_score': float(torch.matmul(
                        vehicle_emb,
                        final_embeddings[chauffeur_offset + chauffeur_idx]
                    ).squeeze().cpu().numpy())
                })
            
            if len(vc_pairs) >= top_k * 1.5:
                break
        
        vc_pairs.sort(key=lambda x: x['combined_score'], reverse=True)
        return vc_pairs
    
    def _generate_vehicle_recommendations(
        self,
        final_embeddings: torch.Tensor,
        vehicle_ids: List[int],
        vehicle_offset: int,
        chauffeur_offset: int,
        n_vehicles: int,
        n_chauffeurs: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for vehicles."""
        vehicle_recs = []
        
        for vehicle_idx in vehicle_ids:
            vehicle_node_id = vehicle_offset + vehicle_idx
            vehicle_emb = final_embeddings[vehicle_node_id].unsqueeze(0)
            
            # Chauffeur recommendations (V-C)
            chauffeur_embs = final_embeddings[chauffeur_offset:chauffeur_offset + n_chauffeurs]
            chauffeur_scores = torch.matmul(vehicle_emb, chauffeur_embs.T).squeeze()
            top_chauffeurs_idx = torch.topk(chauffeur_scores, min(top_k, len(chauffeur_scores))).indices
            top_chauffeurs = top_chauffeurs_idx.cpu().numpy()
            top_chauffeur_scores = chauffeur_scores[top_chauffeurs_idx].cpu().numpy()
            
            # Client recommendations (reverse C-V)
            client_embs = final_embeddings[:vehicle_offset]
            client_scores = torch.matmul(vehicle_emb, client_embs.T).squeeze()
            top_clients_idx = torch.topk(client_scores, min(top_k, len(client_scores))).indices
            top_clients = top_clients_idx.cpu().numpy()
            top_client_scores = client_scores[top_clients_idx].cpu().numpy()
            
            vehicle_recs.append({
                'vehicle_idx': int(vehicle_idx),
                'vehicle_node_id': int(vehicle_node_id),
                'top_chauffeurs': top_chauffeurs.tolist(),
                'chauffeur_scores': top_chauffeur_scores.tolist(),
                'top_clients': top_clients.tolist(),
                'client_scores': top_client_scores.tolist()
            })
        
        return vehicle_recs
    
    def _generate_chauffeur_recommendations(
        self,
        final_embeddings: torch.Tensor,
        chauffeur_ids: List[int],
        vehicle_offset: int,
        chauffeur_offset: int,
        n_vehicles: int,
        n_chauffeurs: int,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for chauffeurs."""
        chauffeur_recs = []
        
        for chauffeur_idx in chauffeur_ids:
            chauffeur_node_id = chauffeur_offset + chauffeur_idx
            chauffeur_emb = final_embeddings[chauffeur_node_id].unsqueeze(0)
            
            # Client recommendations (reverse C-C)
            client_embs = final_embeddings[:vehicle_offset]
            client_scores = torch.matmul(chauffeur_emb, client_embs.T).squeeze()
            top_clients_idx = torch.topk(client_scores, min(top_k, len(client_scores))).indices
            top_clients = top_clients_idx.cpu().numpy()
            top_client_scores = client_scores[top_clients_idx].cpu().numpy()
            
            # Vehicle recommendations (reverse V-C)
            vehicle_embs = final_embeddings[vehicle_offset:vehicle_offset + n_vehicles]
            vehicle_scores = torch.matmul(chauffeur_emb, vehicle_embs.T).squeeze()
            top_vehicles_idx = torch.topk(vehicle_scores, min(top_k, len(vehicle_scores))).indices
            top_vehicles = top_vehicles_idx.cpu().numpy()
            top_vehicle_scores = vehicle_scores[top_vehicles_idx].cpu().numpy()
            
            chauffeur_recs.append({
                'chauffeur_idx': int(chauffeur_idx),
                'chauffeur_node_id': int(chauffeur_node_id),
                'top_clients': top_clients.tolist(),
                'client_scores': top_client_scores.tolist(),
                'top_vehicles': top_vehicles.tolist(),
                'vehicle_scores': top_vehicle_scores.tolist()
            })
        
        return chauffeur_recs


class VCPairRecommender(BaseRecommender):
    """
    Generates global vehicle-chauffeur pair recommendations.
    
    REPLACES: generate_vc_pair_recommendations() [lines 188-234]
    """
    
    def __init__(self, device=device):
        """Initialize V-C pair recommender."""
        super().__init__()
        self.device = device
    
    @torch.no_grad()
    def recommend(
        self,
        model,
        adj_sparse: torch.sparse.FloatTensor,
        vehicle_offset: int,
        chauffeur_offset: int,
        n_vehicles: int,
        n_chauffeurs: int,
        top_k: int = 100,
        min_score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate global vehicle-chauffeur pair recommendations.
        
        REPLACES: generate_vc_pair_recommendations() [lines 188-234]
        
        Args:
            model: Trained LightGCN model
            adj_sparse: Sparse adjacency matrix
            vehicle_offset: Offset for vehicle nodes
            chauffeur_offset: Offset for chauffeur nodes
            n_vehicles: Number of vehicles
            n_chauffeurs: Number of chauffeurs
            top_k: Number of recommendations
            min_score_threshold: Minimum score threshold
            
        Returns:
            List of V-C pair recommendations
        """
        model.eval()
        
        final_embeddings = model(adj_sparse)
        vehicle_embs = final_embeddings[vehicle_offset:vehicle_offset + n_vehicles]
        chauffeur_embs = final_embeddings[chauffeur_offset:chauffeur_offset + n_chauffeurs]
        
        # Compute all V-C scores
        all_scores = torch.matmul(vehicle_embs, chauffeur_embs.T)
        
        # Flatten and get top-K pairs
        flat_scores = all_scores.flatten()
        top_k_actual = min(top_k, len(flat_scores))
        top_values, top_indices = torch.topk(flat_scores, top_k_actual)
        
        # Convert flat indices to (vehicle, chauffeur) pairs
        vehicle_indices = (top_indices // n_chauffeurs).cpu().numpy()
        chauffeur_indices = (top_indices % n_chauffeurs).cpu().numpy()
        scores = top_values.cpu().numpy()
        
        # Filter by threshold if provided
        if min_score_threshold is not None:
            mask = scores >= min_score_threshold
            vehicle_indices = vehicle_indices[mask]
            chauffeur_indices = chauffeur_indices[mask]
            scores = scores[mask]
        
        vc_pairs = []
        for v_idx, c_idx, score in zip(vehicle_indices, chauffeur_indices, scores):
            vc_pairs.append({
                'vehicle_idx': int(v_idx),
                'chauffeur_idx': int(c_idx),
                'compatibility_score': float(score)
            })
        
        return vc_pairs


class ClientRecommender:
    """
    Gets recommendations for specific clients.
    
    REPLACES: get_client_recommendations() [lines 236-258]
    """
    
    def __init__(self, device=device):
        """Initialize client recommender."""
        self.device = device
        self.comprehensive_recommender = ComprehensiveRecommender(device)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_recommendations(
        self,
        step5_results: Dict[str, Any],
        client_ids: List[int],
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for specific clients.
        
        REPLACES: get_client_recommendations() [lines 236-258]
        
        Args:
            step5_results: Model training results
            client_ids: List of client node IDs
            top_k: Number of recommendations
            
        Returns:
            List of client recommendations
        """
        model = step5_results['model']
        data = step5_results['data']
        
        adj_sparse = sparse_to_torch(data['normalized_adj'], self.device)
        vehicle_offset = data['n_clients']
        chauffeur_offset = data['n_clients'] + data['n_vehicles']
        
        recs = self.comprehensive_recommender.recommend(
            model, adj_sparse,
            client_ids=client_ids,
            vehicle_ids=None,
            chauffeur_ids=None,
            vehicle_offset=vehicle_offset,
            chauffeur_offset=chauffeur_offset,
            n_vehicles=data['n_vehicles'],
            n_chauffeurs=data['n_chauffeurs'],
            top_k=top_k
        )
        
        return recs['client_recommendations']


class VehicleRecommender:
    """
    Gets recommendations for specific vehicles.
    
    REPLACES: get_vehicle_recommendations() [lines 260-282]
    """
    
    def __init__(self, device=device):
        """Initialize vehicle recommender."""
        self.device = device
        self.comprehensive_recommender = ComprehensiveRecommender(device)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_recommendations(
        self,
        step5_results: Dict[str, Any],
        vehicle_indices: List[int],
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get chauffeur and client recommendations for specific vehicles.
        
        REPLACES: get_vehicle_recommendations() [lines 260-282]
        
        Args:
            step5_results: Model training results
            vehicle_indices: List of vehicle indices
            top_k: Number of recommendations
            
        Returns:
            List of vehicle recommendations
        """
        model = step5_results['model']
        data = step5_results['data']
        
        adj_sparse = sparse_to_torch(data['normalized_adj'], self.device)
        vehicle_offset = data['n_clients']
        chauffeur_offset = data['n_clients'] + data['n_vehicles']
        
        recs = self.comprehensive_recommender.recommend(
            model, adj_sparse,
            client_ids=None,
            vehicle_ids=vehicle_indices,
            chauffeur_ids=None,
            vehicle_offset=vehicle_offset,
            chauffeur_offset=chauffeur_offset,
            n_vehicles=data['n_vehicles'],
            n_chauffeurs=data['n_chauffeurs'],
            top_k=top_k
        )
        
        return recs['vehicle_recommendations']


class OptimalPairsGenerator:
    """
    Gets optimal vehicle-chauffeur pairings globally.
    
    REPLACES: get_optimal_vc_pairs() [lines 284-299]
    """
    
    def __init__(self, device=device):
        """Initialize optimal pairs generator."""
        self.device = device
        self.vc_pair_recommender = VCPairRecommender(device)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_optimal_pairs(
        self,
        step5_results: Dict[str, Any],
        top_k: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get optimal vehicle-chauffeur pairings globally.
        
        REPLACES: get_optimal_vc_pairs() [lines 284-299]
        
        Args:
            step5_results: Model training results
            top_k: Number of pairs
            
        Returns:
            List of optimal V-C pairs
        """
        model = step5_results['model']
        data = step5_results['data']
        
        adj_sparse = sparse_to_torch(data['normalized_adj'], self.device)
        vehicle_offset = data['n_clients']
        chauffeur_offset = data['n_clients'] + data['n_vehicles']
        
        return self.vc_pair_recommender.recommend(
            model, adj_sparse,
            vehicle_offset, chauffeur_offset,
            data['n_vehicles'], data['n_chauffeurs'],
            top_k=top_k
        )


class SimilarityFinder:
    """
    Finds similar clients based on features.
    
    REPLACES: find_similar_clients() [lines 306-343]
    """
    
    def __init__(self):
        """Initialize similarity finder."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def find_similar_clients(
        self,
        target_client_id: int,
        client_feature_matrix: pd.DataFrame,
        clients_in_model: Set[int],
        available_features: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find K most similar clients that exist in the model.
        
        REPLACES: find_similar_clients() [lines 306-343]
        
        Args:
            target_client_id: Target client ID
            client_feature_matrix: DataFrame with client features
            clients_in_model: Set of clients in the model
            available_features: List of feature columns
            top_k: Number of similar clients
            
        Returns:
            List of (client_id, similarity_score) tuples
        """
        # Get target client features
        target_features = client_feature_matrix[
            client_feature_matrix['ID'] == target_client_id
        ][available_features].values
        
        if len(target_features) == 0:
            return []
        
        # Get features for clients in model
        model_clients_df = client_feature_matrix[
            client_feature_matrix['ID'].isin(clients_in_model)
        ]
        
        if len(model_clients_df) == 0:
            return []
        
        model_client_ids = model_clients_df['ID'].values
        model_features = model_clients_df[available_features].values
        
        # Calculate cosine similarity
        similarities = cosine_similarity(target_features, model_features)[0]
        
        # Get top-K similar clients
        top_k_actual = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k_actual:][::-1]
        
        similar_clients = [
            (model_client_ids[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return similar_clients


class FallbackRecommender(BaseRecommender):
    """
    Generates fallback recommendations for missing clients.
    
    REPLACES: generate_fallback_recommendations() [lines 350-562]
    """
    
    def __init__(self):
        """Initialize fallback recommender."""
        super().__init__()
        self.similarity_finder = SimilarityFinder()
    
    def recommend(
        self,
        missing_clients: Set[int],
        client_feature_matrix: pd.DataFrame,
        clients_in_model: Set[int],
        available_features: List[str],
        cv_recs_df: pd.DataFrame,
        cc_recs_df: pd.DataFrame,
        vc_pair_recs_df: pd.DataFrame,
        mappings: Dict[str, Any],
        top_k: int = 100,
        n_similar: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate recommendations for missing clients using similar clients.
        
        REPLACES: generate_fallback_recommendations() [lines 350-562]
        
        Args:
            missing_clients: Set of missing client IDs
            client_feature_matrix: DataFrame with client features
            clients_in_model: Set of clients in the model
            available_features: List of feature columns
            cv_recs_df: Client-Vehicle recommendations DataFrame
            cc_recs_df: Client-Chauffeur recommendations DataFrame
            vc_pair_recs_df: VC pair recommendations DataFrame
            mappings: ID mappings
            top_k: Number of recommendations
            n_similar: Number of similar clients to use
            
        Returns:
            Tuple of (fallback_cv_df, fallback_cc_df, fallback_vc_pairs_df)
        """
        # Get reverse ID mappings
        id_to_vehicle = mappings['id_to_vehicle']
        id_to_chauffeur = mappings['id_to_chauffeur']
        id_to_client = mappings['id_to_client']
        
        fallback_cv_list = []
        fallback_cc_list = []
        fallback_vc_pairs_list = []
        
        missing_clients_list = list(missing_clients)
        
        self.logger.debug(f"  Processing {len(missing_clients_list):,} missing clients...")
        
        for idx, target_client_id in enumerate(missing_clients_list):
            if (idx + 1) % 100 == 0:
                 self.logger.debug(f"    Progress: {idx + 1}/{len(missing_clients_list)}")
            
            # Find similar clients
            similar_clients = self.similarity_finder.find_similar_clients(
                target_client_id, client_feature_matrix, clients_in_model,
                available_features, top_k=n_similar
            )
            
            if len(similar_clients) == 0:
                # Popularity-based fallback
                self._add_popularity_fallback(
                    target_client_id, cv_recs_df, cc_recs_df, vc_pair_recs_df,
                    mappings, id_to_vehicle, id_to_chauffeur,
                    fallback_cv_list, fallback_cc_list, fallback_vc_pairs_list, top_k
                )
                continue
            
            # Similarity-based recommendations
            self._add_similarity_recommendations(
                target_client_id, similar_clients, mappings,
                cv_recs_df, cc_recs_df, vc_pair_recs_df,
                id_to_vehicle, id_to_chauffeur, id_to_client,
                fallback_cv_list, fallback_cc_list, fallback_vc_pairs_list, top_k
            )
        
        fallback_cv_df = pd.DataFrame(fallback_cv_list)
        fallback_cc_df = pd.DataFrame(fallback_cc_list)
        fallback_vc_pairs_df = pd.DataFrame(fallback_vc_pairs_list)
        
        return fallback_cv_df, fallback_cc_df, fallback_vc_pairs_df
    
    def _add_popularity_fallback(
        self,
        target_client_id: int,
        cv_recs_df: pd.DataFrame,
        cc_recs_df: pd.DataFrame,
        vc_pair_recs_df: pd.DataFrame,
        mappings: Dict[str, Any],
        id_to_vehicle: Dict[int, int],
        id_to_chauffeur: Dict[int, int],
        fallback_cv_list: List[Dict],
        fallback_cc_list: List[Dict],
        fallback_vc_pairs_list: List[Dict],
        top_k: int
    ) -> None:
        """Add popularity-based fallback recommendations."""
        top_vehicles = cv_recs_df.groupby('vehicle_idx')['score'].mean().nlargest(top_k)
        top_chauffeurs = cc_recs_df.groupby('chauffeur_idx')['score'].mean().nlargest(top_k)
        top_vc_pairs = vc_pair_recs_df.groupby(
            ['vehicle_idx', 'chauffeur_idx']
        )['combined_score'].mean().nlargest(top_k)
        
        # Add vehicle recommendations
        for rank, (vehicle_idx, score) in enumerate(top_vehicles.items(), 1):
            vehicle_node_id = vehicle_idx + mappings['n_clients']
            original_vehicle_id = id_to_vehicle.get(vehicle_node_id, None)
            
            fallback_cv_list.append({
                'original_client_id': target_client_id,
                'vehicle_idx': vehicle_idx,
                'original_vehicle_id': original_vehicle_id,
                'rank': rank,
                'score': float(score) * 0.5,
                'recommendation_type': 'popularity',
                'similar_clients_used': 0
            })
        
        # Add chauffeur recommendations
        for rank, (chauffeur_idx, score) in enumerate(top_chauffeurs.items(), 1):
            chauffeur_node_id = chauffeur_idx + mappings['n_clients'] + mappings['n_vehicles']
            original_chauffeur_id = id_to_chauffeur.get(chauffeur_node_id, None)
            
            fallback_cc_list.append({
                'original_client_id': target_client_id,
                'chauffeur_idx': chauffeur_idx,
                'original_chauffeur_id': original_chauffeur_id,
                'rank': rank,
                'score': float(score) * 0.5,
                'recommendation_type': 'popularity',
                'similar_clients_used': 0
            })
        
        # Add VC pair recommendations
        for rank, ((vehicle_idx, chauffeur_idx), score) in enumerate(top_vc_pairs.items(), 1):
            vehicle_node_id = vehicle_idx + mappings['n_clients']
            chauffeur_node_id = chauffeur_idx + mappings['n_clients'] + mappings['n_vehicles']
            
            original_vehicle_id = id_to_vehicle.get(vehicle_node_id, None)
            original_chauffeur_id = id_to_chauffeur.get(chauffeur_node_id, None)
            
            fallback_vc_pairs_list.append({
                'original_client_id': target_client_id,
                'vehicle_idx': vehicle_idx,
                'chauffeur_idx': chauffeur_idx,
                'original_vehicle_id': original_vehicle_id,
                'original_chauffeur_id': original_chauffeur_id,
                'rank': rank,
                'combined_score': float(score) * 0.5,
                'recommendation_type': 'popularity',
                'similar_clients_used': 0
            })
    
    def _add_similarity_recommendations(
        self,
        target_client_id: int,
        similar_clients: List[Tuple[int, float]],
        mappings: Dict[str, Any],
        cv_recs_df: pd.DataFrame,
        cc_recs_df: pd.DataFrame,
        vc_pair_recs_df: pd.DataFrame,
        id_to_vehicle: Dict[int, int],
        id_to_chauffeur: Dict[int, int],
        id_to_client: Dict[int, int],
        fallback_cv_list: List[Dict],
        fallback_cc_list: List[Dict],
        fallback_vc_pairs_list: List[Dict],
        top_k: int
    ) -> None:
        """Add similarity-based recommendations."""
        similar_client_node_ids = [
            mappings['client_to_id'][client_id]
            for client_id, _ in similar_clients
        ]
        
        # Vehicle Recommendations
        similar_cv_recs = cv_recs_df[
            cv_recs_df['client_node_id'].isin(similar_client_node_ids)
        ]
        
        vehicle_scores = {}
        for _, row in similar_cv_recs.iterrows():
            vehicle_idx = row['vehicle_idx']
            client_node_id = row['client_node_id']
            score = row['score']
            
            original_client_id = id_to_client[client_node_id]
            weight = next((w for cid, w in similar_clients if cid == original_client_id), 0)
            
            if vehicle_idx not in vehicle_scores:
                vehicle_scores[vehicle_idx] = 0
            vehicle_scores[vehicle_idx] += score * weight
        
        sorted_vehicles = sorted(vehicle_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for rank, (vehicle_idx, score) in enumerate(sorted_vehicles, 1):
            vehicle_node_id = vehicle_idx + mappings['n_clients']
            original_vehicle_id = id_to_vehicle.get(vehicle_node_id, None)
            
            fallback_cv_list.append({
                'original_client_id': target_client_id,
                'vehicle_idx': vehicle_idx,
                'original_vehicle_id': original_vehicle_id,
                'rank': rank,
                'score': float(score),
                'recommendation_type': 'similarity_based',
                'similar_clients_used': len(similar_clients)
            })
        
        # Chauffeur Recommendations
        similar_cc_recs = cc_recs_df[
            cc_recs_df['client_node_id'].isin(similar_client_node_ids)
        ]
        
        chauffeur_scores = {}
        for _, row in similar_cc_recs.iterrows():
            chauffeur_idx = row['chauffeur_idx']
            client_node_id = row['client_node_id']
            score = row['score']
            
            original_client_id = id_to_client[client_node_id]
            weight = next((w for cid, w in similar_clients if cid == original_client_id), 0)
            
            if chauffeur_idx not in chauffeur_scores:
                chauffeur_scores[chauffeur_idx] = 0
            chauffeur_scores[chauffeur_idx] += score * weight
        
        sorted_chauffeurs = sorted(chauffeur_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for rank, (chauffeur_idx, score) in enumerate(sorted_chauffeurs, 1):
            chauffeur_node_id = chauffeur_idx + mappings['n_clients'] + mappings['n_vehicles']
            original_chauffeur_id = id_to_chauffeur.get(chauffeur_node_id, None)
            
            fallback_cc_list.append({
                'original_client_id': target_client_id,
                'chauffeur_idx': chauffeur_idx,
                'original_chauffeur_id': original_chauffeur_id,
                'rank': rank,
                'score': float(score),
                'recommendation_type': 'similarity_based',
                'similar_clients_used': len(similar_clients)
            })
        
        # VC Pair Recommendations
        similar_vc_recs = vc_pair_recs_df[
            vc_pair_recs_df['client_node_id'].isin(similar_client_node_ids)
        ]
        
        vc_pair_scores = {}
        for _, row in similar_vc_recs.iterrows():
            vc_pair = (row['vehicle_idx'], row['chauffeur_idx'])
            client_node_id = row['client_node_id']
            score = row['combined_score']
            
            original_client_id = id_to_client[client_node_id]
            weight = next((w for cid, w in similar_clients if cid == original_client_id), 0)
            
            if vc_pair not in vc_pair_scores:
                vc_pair_scores[vc_pair] = 0
            vc_pair_scores[vc_pair] += score * weight
        
        sorted_vc_pairs = sorted(vc_pair_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for rank, ((vehicle_idx, chauffeur_idx), score) in enumerate(sorted_vc_pairs, 1):
            vehicle_node_id = vehicle_idx + mappings['n_clients']
            chauffeur_node_id = chauffeur_idx + mappings['n_clients'] + mappings['n_vehicles']
            
            original_vehicle_id = id_to_vehicle.get(vehicle_node_id, None)
            original_chauffeur_id = id_to_chauffeur.get(chauffeur_node_id, None)
            
            fallback_vc_pairs_list.append({
                'original_client_id': target_client_id,
                'vehicle_idx': vehicle_idx,
                'chauffeur_idx': chauffeur_idx,
                'original_vehicle_id': original_vehicle_id,
                'original_chauffeur_id': original_chauffeur_id,
                'rank': rank,
                'combined_score': float(score),
                'recommendation_type': 'similarity_based',
                'similar_clients_used': len(similar_clients)
            })


class RecommendationPipeline:
    """
    Complete recommendation generation pipeline.
    
    REPLACES: generate_all_recommendations() [lines 569-911]
    """
    
    def __init__(self, device=device):
        """
        Initialize recommendation pipeline.
        
        Args:
            device: PyTorch device
        """
        self.device = device
        self.comprehensive_recommender = ComprehensiveRecommender(device)
        self.fallback_recommender = FallbackRecommender()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(
        self,
        model,
        step2_results: Dict[str, Any],
        step3_results: Dict[str, Any],
        step4_results: Dict[str, Any],
        data: Dict[str, Any],
        adj_sparse: torch.sparse.FloatTensor,
        top_k: int = 100,
        n_similar: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute complete recommendation generation pipeline.
        
        REPLACES: generate_all_recommendations() [lines 569-911]
        
        Args:
            model: Trained LightGCN model
            step2_results: Metadata enrichment results
            step3_results: Data filtering results
            step4_results: Graph construction results
            data: Graph data
            adj_sparse: Sparse adjacency matrix
            top_k: Number of recommendations
            n_similar: Number of similar clients for fallback
            
        Returns:
            Dictionary with complete recommendations
        """
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("GENERATING RECOMMENDATIONS FOR ALL CLIENTS")
        self.logger.debug("=" * 80)
        
        # Get parameters
        all_client_ids = list(range(data['n_clients']))
        vehicle_offset = data['n_clients']
        chauffeur_offset = data['n_clients'] + data['n_vehicles']
        
        self.logger.info(f"Generating recommendations for {len(all_client_ids):,} clients...")
        
        # Step 1: LightGCN Recommendations
        recommendations = self.comprehensive_recommender.recommend(
            model, adj_sparse,
            client_ids=all_client_ids,
            vehicle_ids=None,
            chauffeur_ids=None,
            vehicle_offset=vehicle_offset,
            chauffeur_offset=chauffeur_offset,
            n_vehicles=data['n_vehicles'],
            n_chauffeurs=data['n_chauffeurs'],
            top_k=top_k
        )
        
        self.logger.debug(f" Generated recommendations for {len(recommendations['client_recommendations']):,} clients")
        
        # Convert to DataFrames
        cv_recs_df, cc_recs_df, vc_pair_recs_df = self._convert_to_dataframes(recommendations)
        
        # Step 2: Identify Missing Clients
        self.logger.debug("\n" + "=" * 80)
        self.logger.debug("GENERATING FALLBACK RECOMMENDATIONS FOR MISSING CLIENTS")
        self.logger.debug("=" * 80)
        
        all_original_clients = set(step2_results['client_features']['ID'].values)
        clients_in_model = set(step3_results['mappings']['client_to_id'].keys())
        missing_clients = all_original_clients - clients_in_model
        
        self.logger.debug(f"  Total original clients: {len(all_original_clients):,}")
        self.logger.debug(f"  Clients with embeddings: {len(clients_in_model):,}")
        self.logger.debug(f"  Missing clients (need fallback): {len(missing_clients):,}")
        
        # Step 3: Generate Fallback Recommendations
        if len(missing_clients) > 0:
            fallback_cv_df, fallback_cc_df, fallback_vc_pairs_df = self._generate_fallback(
                missing_clients, step2_results, step3_results,
                cv_recs_df, cc_recs_df, vc_pair_recs_df, top_k, n_similar
            )
        else:
            fallback_cv_df = pd.DataFrame()
            fallback_cc_df = pd.DataFrame()
            fallback_vc_pairs_df = pd.DataFrame()
        
        # Step 4: Map to Original IDs and Combine
        cv_recs_complete, cc_recs_complete, vc_pair_recs_complete = self._map_and_combine(
            cv_recs_df, cc_recs_df, vc_pair_recs_df,
            fallback_cv_df, fallback_cc_df, fallback_vc_pairs_df,
            step3_results['mappings']
        )
        
        # Step 5: Data Quality Checks
        self._run_quality_checks(cv_recs_complete, cc_recs_complete, vc_pair_recs_complete)
        
        # Step 6: Save Recommendations
        self._save_recommendations(cv_recs_complete, cc_recs_complete, vc_pair_recs_complete)
        
        self.logger.debug("\n" + "=" * 80)
        self.logger.info(" RECOMMENDATION GENERATION COMPLETE")
        self.logger.debug("=" * 80)
        
        return {
            'cv_recommendations': cv_recs_complete,
            'cc_recommendations': cc_recs_complete,
            'vc_pair_recommendations': vc_pair_recs_complete
        }
    
    def _convert_to_dataframes(
        self,
        recommendations: Dict[str, List]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert recommendations to DataFrames."""
        self.logger.debug("\nConverting to DataFrames...")
        
        # Vehicle recommendations
        cv_recs = []
        for rec in recommendations['client_recommendations']:
            for i, (vehicle_idx, score) in enumerate(zip(rec['top_vehicles'], rec['vehicle_scores'])):
                cv_recs.append({
                    'client_node_id': rec['client_node_id'],
                    'vehicle_idx': vehicle_idx,
                    'rank': i + 1,
                    'score': score
                })
        cv_recs_df = pd.DataFrame(cv_recs)
        
        # Chauffeur recommendations
        cc_recs = []
        for rec in recommendations['client_recommendations']:
            for i, (chauffeur_idx, score) in enumerate(zip(rec['top_chauffeurs'], rec['chauffeur_scores'])):
                cc_recs.append({
                    'client_node_id': rec['client_node_id'],
                    'chauffeur_idx': chauffeur_idx,
                    'rank': i + 1,
                    'score': score
                })
        cc_recs_df = pd.DataFrame(cc_recs)
        
        # VC pair recommendations
        vc_pair_recs = []
        for rec in recommendations['client_recommendations']:
            for i, pair in enumerate(rec['top_vc_pairs']):
                vc_pair_recs.append({
                    'client_node_id': rec['client_node_id'],
                    'vehicle_idx': pair['vehicle_idx'],
                    'chauffeur_idx': pair['chauffeur_idx'],
                    'rank': i + 1,
                    'combined_score': pair['combined_score'],
                    'cv_score': pair['cv_score'],
                    'cc_score': pair['cc_score'],
                    'vc_score': pair['vc_score']
                })
        vc_pair_recs_df = pd.DataFrame(vc_pair_recs)
        
        return cv_recs_df, cc_recs_df, vc_pair_recs_df
    
    def _generate_fallback(
        self,
        missing_clients: Set[int],
        step2_results: Dict[str, Any],
        step3_results: Dict[str, Any],
        cv_recs_df: pd.DataFrame,
        cc_recs_df: pd.DataFrame,
        vc_pair_recs_df: pd.DataFrame,
        top_k: int,
        n_similar: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate fallback recommendations for missing clients."""
        self.logger.debug("\nPreparing client features for similarity matching...")
        
        # Feature preparation
        similarity_features = [
            'RefRegion', 'RefMarketSector', 'RefVehicleType',
            'RefSensitivityLevel', 'tenure_days', 'total_bookings',
            'avg_booking_price', 'lifetime_value'
        ]
        
        client_features = step2_results['client_features']
        available_features = [f for f in similarity_features if f in client_features.columns]
        
        client_feature_matrix = client_features[['ID'] + available_features].copy()
        
        # Handle missing values
        for col in available_features:
            if client_feature_matrix[col].dtype == 'object':
                client_feature_matrix[col] = client_feature_matrix[col].fillna('Unknown')
            else:
                median_val = client_feature_matrix[col].median()
                fill_val = median_val if pd.notna(median_val) else 0.0
                client_feature_matrix[col] = client_feature_matrix[col].fillna(fill_val)
        
        # Encode categorical
        categorical_cols = [
            col for col in available_features
            if client_feature_matrix[col].dtype == 'object'
        ]
        
        for col in categorical_cols:
            le = LabelEncoder()
            client_feature_matrix[col] = client_feature_matrix[col].fillna('Unknown')
            client_feature_matrix[col] = le.fit_transform(client_feature_matrix[col].astype(str))
        
        # Normalize numerical
        numerical_cols = [col for col in available_features if col not in categorical_cols]
        
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            client_feature_matrix[numerical_cols] = scaler.fit_transform(
                client_feature_matrix[numerical_cols]
            )
            client_feature_matrix[available_features] = client_feature_matrix[available_features].fillna(0)
        
        clients_in_model = set(step3_results['mappings']['client_to_id'].keys())
        
        # Generate fallback recommendations
        fallback_cv_df, fallback_cc_df, fallback_vc_pairs_df = self.fallback_recommender.recommend(
            missing_clients,
            client_feature_matrix,
            clients_in_model,
            available_features,
            cv_recs_df,
            cc_recs_df,
            vc_pair_recs_df,
            step3_results['mappings'],
            top_k=top_k,
            n_similar=n_similar
        )
        
        self.logger.debug(f"   Fallback recommendations generated:")
        self.logger.debug(f"    - Vehicle: {len(fallback_cv_df):,}")
        self.logger.debug(f"    - Chauffeur: {len(fallback_cc_df):,}")
        self.logger.debug(f"    - VC Pairs: {len(fallback_vc_pairs_df):,}")
        
        return fallback_cv_df, fallback_cc_df, fallback_vc_pairs_df
    
    def _map_and_combine(
        self,
        cv_recs_df: pd.DataFrame,
        cc_recs_df: pd.DataFrame,
        vc_pair_recs_df: pd.DataFrame,
        fallback_cv_df: pd.DataFrame,
        fallback_cc_df: pd.DataFrame,
        fallback_vc_pairs_df: pd.DataFrame,
        mappings: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Map node IDs to original IDs and combine with fallback."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.debug("MAPPING ALL NODE IDs TO ORIGINAL DATABASE IDs")
        self.logger.debug("=" * 80)
        
        # Get reverse mappings
        id_to_client = mappings['id_to_client']
        id_to_vehicle = mappings['id_to_vehicle']
        id_to_chauffeur = mappings['id_to_chauffeur']
        
        # Create direct index-to-original-ID mappings
        self.logger.debug("\n1. Creating direct index mappings...")
        sorted_original_vehicles = sorted(id_to_vehicle.values())
        sorted_original_chauffeurs = sorted(id_to_chauffeur.values())
        
        self.logger.debug(f"  Vehicle IDs range: {min(sorted_original_vehicles)} - {max(sorted_original_vehicles)}")
        self.logger.debug(f"  Chauffeur IDs range: {min(sorted_original_chauffeurs)} - {max(sorted_original_chauffeurs)}")
        
        # Map LightGCN recommendations
        self.logger.debug("\n2. Mapping LightGCN recommendations...")
        
        cv_recs_df['original_client_id'] = cv_recs_df['client_node_id'].map(id_to_client)
        cc_recs_df['original_client_id'] = cc_recs_df['client_node_id'].map(id_to_client)
        vc_pair_recs_df['original_client_id'] = vc_pair_recs_df['client_node_id'].map(id_to_client)
        
        # Map vehicle indices using sorted list
        self.logger.debug("  Mapping vehicle indices to database IDs...")
        cv_recs_df['original_vehicle_id'] = cv_recs_df['vehicle_idx'].apply(
            lambda idx: sorted_original_vehicles[idx] if 0 <= idx < len(sorted_original_vehicles) else None
        )
        vc_pair_recs_df['original_vehicle_id'] = vc_pair_recs_df['vehicle_idx'].apply(
            lambda idx: sorted_original_vehicles[idx] if 0 <= idx < len(sorted_original_vehicles) else None
        )
        
        # Map chauffeur indices using sorted list
        self.logger.debug("  Mapping chauffeur indices to database IDs...")
        cc_recs_df['original_chauffeur_id'] = cc_recs_df['chauffeur_idx'].apply(
            lambda idx: sorted_original_chauffeurs[idx] if 0 <= idx < len(sorted_original_chauffeurs) else None
        )
        vc_pair_recs_df['original_chauffeur_id'] = vc_pair_recs_df['chauffeur_idx'].apply(
            lambda idx: sorted_original_chauffeurs[idx] if 0 <= idx < len(sorted_original_chauffeurs) else None
        )
        
        # Verify mappings
        self.logger.debug(f"\n  Verification:")
        self.logger.debug(f"    Client mapping: {cv_recs_df['original_client_id'].notna().sum()}/{len(cv_recs_df)} ")
        self.logger.debug(f"    Vehicle mapping: {cv_recs_df['original_vehicle_id'].notna().sum()}/{len(cv_recs_df)} ")
        self.logger.debug(f"    Chauffeur mapping: {cc_recs_df['original_chauffeur_id'].notna().sum()}/{len(cc_recs_df)} ")
        
        # Check for failed mappings
        failed_vehicles = cv_recs_df['original_vehicle_id'].isna().sum()
        failed_chauffeurs = cc_recs_df['original_chauffeur_id'].isna().sum()
        
        if failed_vehicles > 0:
            self.logger.warning(f"      WARNING: {failed_vehicles} vehicles failed to map")
        if failed_chauffeurs > 0:
            self.logger.warning(f"      WARNING: {failed_chauffeurs} chauffeurs failed to map")
        
        # Clean up failed mappings
        if failed_vehicles > 0 or failed_chauffeurs > 0:
            self.logger.error("\n  Cleaning up failed mappings...")
            initial_cv = len(cv_recs_df)
            initial_cc = len(cc_recs_df)
            initial_vc = len(vc_pair_recs_df)
            
            cv_recs_df = cv_recs_df.dropna(subset=['original_vehicle_id', 'original_client_id'])
            cc_recs_df = cc_recs_df.dropna(subset=['original_chauffeur_id', 'original_client_id'])
            vc_pair_recs_df = vc_pair_recs_df.dropna(
                subset=['original_vehicle_id', 'original_chauffeur_id', 'original_client_id']
            )
            
            self.logger.debug(f"    CV: {initial_cv} → {len(cv_recs_df)} ({initial_cv - len(cv_recs_df)} dropped)")
            self.logger.debug(f"    CC: {initial_cc} → {len(cc_recs_df)} ({initial_cc - len(cc_recs_df)} dropped)")
            self.logger.debug(f"    VC: {initial_vc} → {len(vc_pair_recs_df)} ({initial_vc - len(vc_pair_recs_df)} dropped)")
        
        # Add metadata
        cv_recs_df['recommendation_type'] = 'lightgcn'
        cc_recs_df['recommendation_type'] = 'lightgcn'
        vc_pair_recs_df['recommendation_type'] = 'lightgcn'
        
        cv_recs_df['similar_clients_used'] = 0
        cc_recs_df['similar_clients_used'] = 0
        vc_pair_recs_df['similar_clients_used'] = 0
        
        # Combine with fallback
        self.logger.debug("\n3. Combining with fallback recommendations...")
        
        cv_recs_complete = pd.concat([cv_recs_df, fallback_cv_df], ignore_index=True)
        cc_recs_complete = pd.concat([cc_recs_df, fallback_cc_df], ignore_index=True)
        vc_pair_recs_complete = pd.concat([vc_pair_recs_df, fallback_vc_pairs_df], ignore_index=True)
        
        self.logger.debug(f"  Combined recommendations:")
        self.logger.info(f"    - Vehicle: {len(cv_recs_complete):,} (LightGCN: {len(cv_recs_df):,}, Fallback: {len(fallback_cv_df):,})")
        self.logger.info(f"    - Chauffeur: {len(cc_recs_complete):,} (LightGCN: {len(cc_recs_df):,}, Fallback: {len(fallback_cc_df):,})")
        self.logger.info(f"    - VC Pairs: {len(vc_pair_recs_complete):,} (LightGCN: {len(vc_pair_recs_df):,}, Fallback: {len(fallback_vc_pairs_df):,})")
        
        return cv_recs_complete, cc_recs_complete, vc_pair_recs_complete
    
    def _run_quality_checks(
        self,
        cv_recs_complete: pd.DataFrame,
        cc_recs_complete: pd.DataFrame,
        vc_pair_recs_complete: pd.DataFrame
    ) -> None:
        """Run data quality checks."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.debug("DATA QUALITY CHECKS")
        self.logger.debug("=" * 80)
        
        cv_missing = cv_recs_complete['original_vehicle_id'].isna().sum()
        cc_missing = cc_recs_complete['original_chauffeur_id'].isna().sum()
        vc_missing_v = vc_pair_recs_complete['original_vehicle_id'].isna().sum()
        vc_missing_c = vc_pair_recs_complete['original_chauffeur_id'].isna().sum()
        
        if cv_missing > 0:
             self.logger.warning(f"    Warning: {cv_missing} vehicle recommendations missing original IDs")
        else:
             self.logger.debug(f"   All vehicle recommendations have original IDs")
        
        if cc_missing > 0:
             self.logger.warning(f"    Warning: {cc_missing} chauffeur recommendations missing original IDs")
        else:
             self.logger.debug(f"   All chauffeur recommendations have original IDs")
        
        if vc_missing_v > 0 or vc_missing_c > 0:
             self.logger.warning(f"    Warning: VC pairs missing IDs (vehicles: {vc_missing_v}, chauffeurs: {vc_missing_c})")
        else:
             self.logger.debug(f"   All VC pair recommendations have original IDs")
    
    def _save_recommendations(
        self,
        cv_recs_complete: pd.DataFrame,
        cc_recs_complete: pd.DataFrame,
        vc_pair_recs_complete: pd.DataFrame
    ) -> None:
        """Save recommendations to CSV files."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("SAVING RECOMMENDATIONS")
        self.logger.debug("=" * 80)
        
        # Reorder columns
        cv_cols = [
            'original_client_id', 'original_vehicle_id', 'vehicle_idx', 'rank', 'score',
            'recommendation_type', 'similar_clients_used', 'client_node_id'
        ]
        cv_cols = [c for c in cv_cols if c in cv_recs_complete.columns]
        cv_recs_complete = cv_recs_complete[cv_cols]
        
        cc_cols = [
            'original_client_id', 'original_chauffeur_id', 'chauffeur_idx', 'rank', 'score',
            'recommendation_type', 'similar_clients_used', 'client_node_id'
        ]
        cc_cols = [c for c in cc_cols if c in cc_recs_complete.columns]
        cc_recs_complete = cc_recs_complete[cc_cols]
        
        vc_cols = [
            'original_client_id', 'original_vehicle_id', 'original_chauffeur_id',
            'vehicle_idx', 'chauffeur_idx', 'rank', 'combined_score', 'cv_score', 'cc_score', 'vc_score',
            'recommendation_type', 'similar_clients_used', 'client_node_id'
        ]
        vc_cols = [c for c in vc_cols if c in vc_pair_recs_complete.columns]
        vc_pair_recs_complete = vc_pair_recs_complete[vc_cols]
        
        # Save to CSV
        cv_recs_complete.to_csv(RECOMMENDATION_FILES['vehicle'], index=False)
        cc_recs_complete.to_csv(RECOMMENDATION_FILES['chauffeur'], index=False)
        vc_pair_recs_complete.to_csv(RECOMMENDATION_FILES['vc_pair'], index=False)
        
        self.logger.debug(f"   Recommendations saved to:")
        self.logger.debug(f"    - {RECOMMENDATION_FILES['vehicle']}")
        self.logger.debug(f"    - {RECOMMENDATION_FILES['chauffeur']}")
        self.logger.debug(f"    - {RECOMMENDATION_FILES['vc_pair']}")


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

@torch.no_grad()
def generate_comprehensive_recommendations(
    model, adj_sparse,
    client_ids, vehicle_ids, chauffeur_ids,
    vehicle_offset, chauffeur_offset,
    n_vehicles, n_chauffeurs,
    top_k=100
):
    """
    Backward compatible wrapper for ComprehensiveRecommender.recommend().
    
    WRAPS: ComprehensiveRecommender.recommend()
    """
    recommender = ComprehensiveRecommender()
    return recommender.recommend(
        model, adj_sparse,
        client_ids, vehicle_ids, chauffeur_ids,
        vehicle_offset, chauffeur_offset,
        n_vehicles, n_chauffeurs,
        top_k
    )


@torch.no_grad()
def generate_vc_pair_recommendations(
    model, adj_sparse,
    vehicle_offset, chauffeur_offset,
    n_vehicles, n_chauffeurs,
    top_k=100,
    min_score_threshold=None
):
    """
    Backward compatible wrapper for VCPairRecommender.recommend().
    
    WRAPS: VCPairRecommender.recommend()
    """
    recommender = VCPairRecommender()
    return recommender.recommend(
        model, adj_sparse,
        vehicle_offset, chauffeur_offset,
        n_vehicles, n_chauffeurs,
        top_k, min_score_threshold
    )


def get_client_recommendations(step5_results, client_ids, top_k=100):
    """
    Backward compatible wrapper for ClientRecommender.get_recommendations().
    
    WRAPS: ClientRecommender.get_recommendations()
    """
    recommender = ClientRecommender()
    return recommender.get_recommendations(step5_results, client_ids, top_k)


def get_vehicle_recommendations(step5_results, vehicle_indices, top_k=100):
    """
    Backward compatible wrapper for VehicleRecommender.get_recommendations().
    
    WRAPS: VehicleRecommender.get_recommendations()
    """
    recommender = VehicleRecommender()
    return recommender.get_recommendations(step5_results, vehicle_indices, top_k)


def get_optimal_vc_pairs(step5_results, top_k=100):
    """
    Backward compatible wrapper for OptimalPairsGenerator.get_optimal_pairs().
    
    WRAPS: OptimalPairsGenerator.get_optimal_pairs()
    """
    generator = OptimalPairsGenerator()
    return generator.get_optimal_pairs(step5_results, top_k)


def find_similar_clients(
    target_client_id, client_feature_matrix, clients_in_model,
    available_features, top_k=5
):
    """
    Backward compatible wrapper for SimilarityFinder.find_similar_clients().
    
    WRAPS: SimilarityFinder.find_similar_clients()
    """
    finder = SimilarityFinder()
    return finder.find_similar_clients(
        target_client_id, client_feature_matrix, clients_in_model,
        available_features, top_k
    )


def generate_fallback_recommendations(
    missing_clients, client_feature_matrix,
    clients_in_model, available_features,
    cv_recs_df, cc_recs_df, vc_pair_recs_df,
    mappings, top_k=100, n_similar=5
):
    """
    Backward compatible wrapper for FallbackRecommender.recommend().
    
    WRAPS: FallbackRecommender.recommend()
    """
    recommender = FallbackRecommender()
    return recommender.recommend(
        missing_clients, client_feature_matrix,
        clients_in_model, available_features,
        cv_recs_df, cc_recs_df, vc_pair_recs_df,
        mappings, top_k, n_similar
    )


def generate_all_recommendations(
    model, step2_results, step3_results,
    step4_results, data, adj_sparse,
    top_k=100, n_similar=5
):
    """
    Backward compatible wrapper for RecommendationPipeline.execute().
    
    WRAPS: RecommendationPipeline.execute()
    """
    pipeline = RecommendationPipeline()
    return pipeline.execute(
        model, step2_results, step3_results,
        step4_results, data, adj_sparse,
        top_k, n_similar
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.debug("Recommendation Generator Module (Refactored) - Ready")
    logger.debug("\nClasses available:")
    logger.debug("  - ComprehensiveRecommender")
    logger.debug("  - ComprehensiveRecommender")
    logger.debug("  - VCPairRecommender")
    logger.debug("  - ClientRecommender")
    logger.debug("  - VehicleRecommender")
    logger.debug("  - OptimalPairsGenerator")
    logger.debug("  - SimilarityFinder")
    logger.debug("  - FallbackRecommender")
    logger.debug("  - RecommendationPipeline")
    logger.debug("\nBackward compatible functions:")
    logger.debug("  - generate_comprehensive_recommendations()")
    logger.debug("  - generate_vc_pair_recommendations()")
    logger.debug("  - get_client_recommendations()")
    logger.debug("  - get_vehicle_recommendations()")
    logger.debug("  - get_optimal_vc_pairs()")
    logger.debug("  - find_similar_clients()")
    logger.debug("  - generate_fallback_recommendations()")
    logger.debug("  - generate_all_recommendations()")

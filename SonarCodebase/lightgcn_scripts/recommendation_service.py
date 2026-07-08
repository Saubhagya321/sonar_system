"""
Recommendation Service Module
Production-ready service for managing the recommendation engine
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pandas as pd

from model_training import sparse_to_torch, device
from config import MODEL_PATHS, PIPELINE_PATHS, setup_logging
from complete_recommendation_api import (
    EnhancedBookingRequest,
    CompleteRecommendationEngine
)

logger = logging.getLogger(__name__)


# ============================================================================
# SERVICE CLASS
# ============================================================================

class RecommendationService:
    """High-level service for managing the recommendation engine"""
    
    def __init__(self, model_path: str = None,
                 step2_path: str = None,
                 step3_path: str = None,
                 step4_path: str = None):
        logger.info("=" * 80)
        logger.info("INITIALIZING RECOMMENDATION SERVICE")
        logger.info("=" * 80)
       
        if model_path is None:
            model_path = str(MODEL_PATHS['best_model'])
        if step2_path is None:
            step2_path = str(PIPELINE_PATHS['step2_results'])
        if step3_path is None:
            step3_path = str(PIPELINE_PATHS['step3_results'])
        if step4_path is None:
            step4_path = str(PIPELINE_PATHS['step4_results'])
        
        self.engine = None
        self.model = None
        self.pipeline_results = None
        self.str = "Engine not initialized"
        logger.info("1. Loading trained model...")
        self._load_model(model_path)
        
        logger.info("2. Loading pipeline data...")
        if step2_path and step3_path and step4_path:
            self._load_pipeline_from_files(step2_path, step3_path, step4_path)
        else:
            logger.info("   No pipeline files provided")
        
        logger.info("=" * 80)
        logger.info("SERVICE READY!")
        logger.info("=" * 80)

    def _load_model(self, model_path: str):
        """Load the trained model from checkpoint"""
        try:
            # weights_only=False: checkpoint's 'history'/'config' dicts contain
            # numpy scalars that PyTorch 2.6+'s safe unpickler rejects by default.
            # Safe here since this checkpoint is produced locally by our own
            # training pipeline (flask_api_server.py already does the same).
            checkpoint = torch.load(model_path, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        from model_training import HeterogeneousLightGCN
        
        data_info = checkpoint['data_info']
        config = checkpoint['config']
        
        self.model = HeterogeneousLightGCN(
            n_clients=data_info['n_clients'],
            n_vehicles=data_info['n_vehicles'],
            n_chauffeurs=data_info['n_chauffeurs'],
            embedding_dim=config['embedding_dim'],
            n_layers=config['n_layers'],
            dropout=config.get('dropout', 0.1),
            use_batch_norm=config.get('use_batch_norm', False)
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("   Model loaded from %s", model_path)
    
    def _load_pipeline_from_files(self, step2_path: str, step3_path: str, step4_path: str):
        """Load pipeline results from pickle files"""
        import pickle
        
        try:
            with open(step2_path, 'rb') as f:
                step2_results = pickle.load(f)
            
            with open(step3_path, 'rb') as f:
                step3_results = pickle.load(f)
            
            with open(step4_path, 'rb') as f:
                step4_results = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Pipeline file not found: {e}")
        
        logger.info("   Pipeline results loaded")
        
        self._initialize_engine(step2_results, step3_results, step4_results)
    
    def initialize_from_pipeline_run(self, pipeline_results: Dict):
        """Initialize from a fresh pipeline run"""
        self._initialize_engine(
            pipeline_results['step2_results'],
            pipeline_results['step3_results'],
            pipeline_results['step4_results']
        )
    
    def _initialize_engine(self, step2_results, step3_results, step4_results):
        """Initialize the recommendation engine with pipeline data"""
        from model_training import load_graph_data
        
        data = load_graph_data(step4_results)
        adj_sparse = sparse_to_torch(data['normalized_adj'], device)
        
        self.engine = CompleteRecommendationEngine(
            model=self.model,
            step2_results=step2_results,
            step3_results=step3_results,
            step4_results=step4_results,
            data=data,
            adj_sparse=adj_sparse
        )
        logger.info("   Recommendation engine initialized")
    
    def recommend_from_json(self, json_str: str) -> Dict:
        """Get recommendations from JSON string request"""
        if self.engine is None:
            raise RuntimeError(self.str)
        
        try:
            request = EnhancedBookingRequest.from_json(json_str)
            response = self.engine.recommend(request)
            return self._format_response(response)
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON: {e}', 'status': 'failed'}
        except ValueError as e:
            return {'error': f'Validation error: {e}', 'status': 'failed'}
        except Exception as e:
            logger.exception("Error processing recommendation request")
            return {'error': str(e), 'status': 'failed'}
    
    def recommend_from_dict(self, request_dict: Dict) -> Dict:
        """Get recommendations from dictionary request"""
        json_str = json.dumps(request_dict)
        return self.recommend_from_json(json_str)
    
    def _format_response(self, response) -> Dict:
        """Format the response object into a dictionary"""
        return {
            'status': 'success',
            'request_id': response.request_id,
            'client_id': response.client_id,
            'method': response.recommendation_method,
            'recommendations': {
                'vehicles': response.vehicles,
                'chauffeurs': response.chauffeurs,
                'pairs': response.pairs
            },
            'metadata': {
                'total_candidates_vehicles': response.total_candidates_vehicles,
                'total_candidates_chauffeurs': response.total_candidates_chauffeurs,
                'filters_applied': response.filters_applied,
                'warnings': response.warnings
            }
        }
    
    # ========================================================================
    # TEST & UTILITY FUNCTIONS
    # ========================================================================
    
    def get_available_ids(self) -> Optional[Dict]:
        """Get all available IDs for testing"""
        if self.engine is None:
            logger.warning(self.str)
            return None
        
        logger.info("=" * 80)
        logger.info("AVAILABLE IDS FOR TESTING")
        logger.info("=" * 80)
        
        clients = list(self.engine.client_id_to_node.keys())
        vehicles = list(self.engine.vehicle_id_to_idx.keys())
        chauffeurs = list(self.engine.chauffeur_id_to_idx.keys())
        
        logger.info("Clients: %d total (sample: %s)", len(clients), clients[:5])
        logger.info("Vehicles: %d total (sample: %s)", len(vehicles), vehicles[:5])
        logger.info("Chauffeurs: %d total (sample: %s)", len(chauffeurs), chauffeurs[:5])
        
        return {
            'clients': clients,
            'vehicles': vehicles,
            'chauffeurs': chauffeurs
        }
    
    def show_request_options(self):
        """Show all available request parameters and their meanings"""
        logger.info("=" * 80)
        logger.info("BOOKING REQUEST OPTIONS")
        logger.info("=" * 80)
        
        logger.info("REQUIRED FIELDS:")
        logger.info("  client_id (str)      - Client identifier")
        logger.info("  no_of_pax (int)      - Number of passengers (1-20)")
        
        logger.info("OPTIONAL CONTEXT (improves recommendations):")
        logger.info("  ref_vehicle_type (str)      - Preferred vehicle type ID")
        logger.info("  appointment_date (str)      - Trip date (YYYY-MM-DD format)")
        logger.info("  pickup_suburb (str)         - Pickup location")
        logger.info("  dropoff_suburb (str)        - Dropoff location")
        logger.info("  estimated_distance (float)  - Trip distance in km")
        logger.info("  estimated_duration (int)    - Trip duration in minutes")
        
        logger.info("PREFERENCE FILTERS:")
        logger.info("  preferred_chauffeur_ids (list)  - Specific chauffeur IDs (ONLY these)")
        logger.info("  excluded_vehicle_ids (list)     - Vehicle IDs to exclude")
        logger.info("  excluded_chauffeur_ids (list)   - Chauffeur IDs to exclude")
        logger.info("  require_baby_seat (bool)        - Must have baby seat training")
        logger.info("  require_red_zone_auth (bool)    - Must have airport access")
        logger.info("  min_vehicle_level (int)         - Minimum luxury level (1-5)")
        logger.info("  max_price (float)               - Budget constraint")
        
        logger.info("RECOMMENDATION SETTINGS:")
        logger.info("  top_k_vehicles (int)    - Number of vehicle recommendations (default: 20)")
        logger.info("  top_k_chauffeurs (int)  - Number of chauffeur recommendations (default: 20)")
        logger.info("  top_k_pairs (int)       - Number of V-C pairs (default: 10)")
        
        logger.info("=" * 80)
    
    def get_vehicle_types(self) -> Optional[List]:
        """Get available vehicle types"""
        if self.engine is None:
            logger.warning(self.str)
            return None
        
        vehicle_types = self.engine.vehicle_features['RefVehicleType'].dropna().unique()
        logger.info("Available Vehicle Types (%d):", len(vehicle_types))
        for vt in vehicle_types[:20]:
            logger.info("  - %s", vt)
        if len(vehicle_types) > 20:
            logger.info("  ... and %d more", len(vehicle_types) - 20)
        
        return list(vehicle_types)
    
    def test_with_random_client(self) -> Optional[Dict]:
        """Test with a random client from the model"""
        if self.engine is None:
            logger.warning(self.str)
            return None
        
        clients = list(self.engine.client_id_to_node.keys())
        if not clients:
            logger.warning("No clients available")
            return None
        
        import random
        client_id = random.choice(clients)
        
        logger.info("=" * 80)
        logger.info("TESTING WITH RANDOM CLIENT: %s", client_id)
        logger.info("=" * 80)
        
        request = {
            "client_id": client_id,
            "no_of_pax": 2,
            "top_k_vehicles": 5,
            "top_k_chauffeurs": 3,
            "top_k_pairs": 3
        }
        
        logger.info("Request: %s", json.dumps(request, indent=2))
        
        response = self.recommend_from_dict(request)
        self._log_response_summary(response)
        
        return response
    
    def test_with_filters(self) -> Optional[Dict]:
        """Test with various filters"""
        if self.engine is None:
            logger.warning(self.str)
            return None
        
        clients = list(self.engine.client_id_to_node.keys())
        if not clients:
            logger.warning("No clients available")
            return None
        
        import random
        client_id = random.choice(clients)
        
        logger.info("=" * 80)
        logger.info("TESTING WITH FILTERS: %s", client_id)
        logger.info("=" * 80)
        
        request = {
            "client_id": client_id,
            "no_of_pax": 4,
            "min_vehicle_level": 3,
            "require_red_zone_auth": True,
            "top_k_vehicles": 10,
            "top_k_chauffeurs": 5,
            "top_k_pairs": 5
        }
        
        logger.info("Request: %s", json.dumps(request, indent=2))
        
        response = self.recommend_from_dict(request)
        self._log_response_summary(response)
        
        return response
    
    def test_cold_start(self) -> Dict:
        """Test with a client NOT in the model"""
        logger.info("=" * 80)
        logger.info("TESTING COLD START (NEW CLIENT)")
        logger.info("=" * 80)
        
        request = {
            "client_id": "NEW_CLIENT_TEST_999",
            "no_of_pax": 2,
            "top_k_vehicles": 5,
            "top_k_chauffeurs": 3,
            "top_k_pairs": 3
        }
        
        logger.info("Request: %s", json.dumps(request, indent=2))
        
        response = self.recommend_from_dict(request)
        self._log_response_summary(response)
        
        return response
    
    def _log_response_summary(self, response: Dict):
        """Log a summary of the response"""
        if response['status'] != 'success':
            logger.error("Error: %s", response.get('error'))
            return
        
        logger.info("Success!")
        logger.info("  Method: %s", response['method'])
        logger.info("  Filters: %s", 
                   ', '.join(response['metadata']['filters_applied']) if response['metadata']['filters_applied'] else 'None')
        
        if response['metadata']['warnings']:
            logger.warning("  Warnings: %s", ', '.join(response['metadata']['warnings']))
        
        logger.info("Top 3 Vehicles:")
        for i, v in enumerate(response['recommendations']['vehicles'][:3], 1):
            logger.info("  %d. %s %s (Score: %.3f, Seats: %s)",
                       i, v.get('make', 'N/A'), v.get('model', 'N/A'),
                       v['score'], v.get('seat_capacity', 'N/A'))
        
        logger.info("Top 3 Chauffeurs:")
        for i, c in enumerate(response['recommendations']['chauffeurs'][:3], 1):
            logger.info("  %d. %s (Score: %.3f, Exp: %s/4)",
                       i, c['chauffeur_id'], c['score'], c.get('experience_score', 'N/A'))
        
        logger.info("Top 3 Pairs:")
        for i, p in enumerate(response['recommendations']['pairs'][:3], 1):
            logger.info("  %d. %s %s + Chauffeur (Score: %.3f)",
                       i, p.get('vehicle_make', 'N/A'), p.get('vehicle_model', 'N/A'),
                       p['combined_score'])
    
    def run_comprehensive_tests(self) -> List:
        """Run all tests"""
        logger.info("=" * 80)
        logger.info("RUNNING COMPREHENSIVE TESTS")
        logger.info("=" * 80)
        
        tests = [
            ("Random Client", self.test_with_random_client),
            ("With Filters", self.test_with_filters),
            ("Cold Start", self.test_cold_start)
        ]
        
        results = []
        
        for name, test_func in tests:
            try:
                logger.info("")
                logger.info("=" * 80)
                logger.info("TEST: %s", name)
                logger.info("=" * 80)
                result = test_func()
                results.append((name, "PASS", result))
            except Exception as e:
                logger.error("Test failed: %s", e)
                results.append((name, "FAIL", str(e)))
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        
        for name, status, _ in results:
            symbol = "✓" if status == "PASS" else "✗"
            logger.info("%s %s: %s", symbol, name, status)
        
        return results
    
    def save_pipeline_data(self, output_dir: str = 'pipeline_data'):
        """Save pipeline data for future loading"""
        import pickle
        
        Path(output_dir).mkdir(exist_ok=True)
        
        if hasattr(self, 'step2_results'):
            with open(f"{output_dir}/step2_results.pkl", 'wb') as f:
                pickle.dump(self.step2_results, f)
        
        if hasattr(self, 'step3_results'):
            with open(f"{output_dir}/step3_results.pkl", 'wb') as f:
                pickle.dump(self.step3_results, f)
        
        if hasattr(self, 'step4_results'):
            with open(f"{output_dir}/step4_results.pkl", 'wb') as f:
                pickle.dump(self.step4_results, f)
        
        logger.info("Pipeline data saved to %s/", output_dir)


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_cli(service: RecommendationService):
    """Interactive command-line interface"""
    logger.info("INTERACTIVE RECOMMENDATION CLI")
    logger.info("OPTIONS: 1-Show IDs, 2-Request options, 3-Random client, "
                "4-Filters, 5-Cold start, 6-All tests, 7-Custom JSON, 8-Vehicle types, q-Quit")
    
    while True:
        try:
            choice = input("\nSelect option: ").strip().lower()
            
            if choice == 'q':
                logger.info("Goodbye!")
                break
            
            if choice == '1':
                service.get_available_ids()
            elif choice == '2':
                service.show_request_options()
            elif choice == '3':
                service.test_with_random_client()
            elif choice == '4':
                service.test_with_filters()
            elif choice == '5':
                service.test_cold_start()
            elif choice == '6':
                service.run_comprehensive_tests()
            elif choice == '7':
                logger.info("Enter JSON request (or press Enter for example):")
                json_input = input().strip()
                if not json_input:
                    clients = list(service.engine.client_id_to_node.keys())
                    if clients:
                        json_input = json.dumps({
                            "client_id": clients[0],
                            "no_of_pax": 2
                        })
                response = service.recommend_from_json(json_input)
                service._log_response_summary(response)
            elif choice == '8':
                service.get_vehicle_types()
            else:
                logger.warning("Invalid option")
        
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error("Error: %s", e)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # Setup logging for CLI usage
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Recommendation Service')
    parser.add_argument('--mode', choices=['test', 'interactive', 'demo'], 
                       default='test', help='Run mode')
    parser.add_argument('--model-path', type=str, 
                       default=str(MODEL_PATHS['best_model']),
                       help='Path to model')
    parser.add_argument('--pipeline-dir', type=str, 
                       default='pipeline_data',
                       help='Directory with pipeline data')
    
    args = parser.parse_args()
    
    # Initialize service
    logger.info("Initializing service...")
    try:
        service = RecommendationService(
            model_path=args.model_path,
            step2_path=f"{args.pipeline_dir}/step2_results.pkl",
            step3_path=f"{args.pipeline_dir}/step3_results.pkl",
            step4_path=f"{args.pipeline_dir}/step4_results.pkl"
        )
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        logger.error("Please ensure the pipeline has been run and files exist")
        exit(1)
    except Exception as e:
        logger.exception("Failed to initialize service")
        exit(1)
    
    if args.mode == 'test':
        logger.info("RUNNING TESTS")
        service.run_comprehensive_tests()
    
    elif args.mode == 'interactive':
        interactive_cli(service)
    
    elif args.mode == 'demo':
        logger.info("Showing available options...")
        service.show_request_options()
        logger.info("Getting available IDs...")
        service.get_available_ids()
        logger.info("Running test...")
        service.test_with_random_client()
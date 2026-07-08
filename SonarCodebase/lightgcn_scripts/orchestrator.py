

import warnings
import pickle
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
import torch
import pandas as pd

from base_classes import BasePipeline

# Import refactored pipeline classes
from data_preparation_and_loading import (
    BookingDataLoader,
    DataPreparationPipeline,
    load_booking_data,
    execute_step1
)
from metadata_enrichment import (
    MetadataLoader,
    MetadataEnrichmentPipeline,
    load_metadata_tables,
    execute_step2_revised
)
from data_filtering import (
    DataFilteringPipeline,
    execute_step3_three_way
)
from graph_construction import (
    GraphConstructionPipeline,
    execute_step4_three_way
)
from model_training import (
    HeterogeneousLightGCN,
    ModelTrainingPipeline,
    GraphDataLoader,
    TorchConverter,
    train_heterogeneous_lightgcn,
    load_graph_data
)
from recommendation_genrator import generate_all_recommendations

from config import (
    BOOKING_DATA_PATH,
    MODEL_PATHS,
    GRAPH_OUTPUT_PREFIX,
    PIPELINE_PATHS,
    PIPELINE_DIR,
    BEST_CONFIG,
    setup_logging
)

warnings.filterwarnings('ignore')
setup_logging()



# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model save path
MODEL_SAVE_PATH = MODEL_PATHS['best_model']


class PipelineConfig:
    """
    Configuration holder for the LightGCN pipeline.
    
    NEW CLASS - Centralizes pipeline configuration
    """
    
    DEFAULT_MODEL_CONFIG = BEST_CONFIG
    
    DEFAULT_FILTERING_CONFIG = {
        'min_cv': 2,
        'min_cc': 2,
        'min_vc': 3,
        'split_ratio': 0.8,
        'split_method': 'temporal'
    }
    
    DEFAULT_GRAPH_CONFIG = {
        'save_data': True,
        'output_prefix': GRAPH_OUTPUT_PREFIX
    }
    
    DEFAULT_RECOMMENDATION_CONFIG = {
        'top_k': 100,
        'n_similar': 5
    }
    
    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        filtering_config: Optional[Dict[str, Any]] = None,
        graph_config: Optional[Dict[str, Any]] = None,
        recommendation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline configuration.
        
        Args:
            model_config: Model training configuration
            filtering_config: Data filtering configuration
            graph_config: Graph construction configuration
            recommendation_config: Recommendation generation configuration
        """
        self.model = {**self.DEFAULT_MODEL_CONFIG, **(model_config or {})}
        self.filtering = {**self.DEFAULT_FILTERING_CONFIG, **(filtering_config or {})}
        self.graph = {**self.DEFAULT_GRAPH_CONFIG, **(graph_config or {})}
        self.recommendation = {**self.DEFAULT_RECOMMENDATION_CONFIG, **(recommendation_config or {})}


class ModelLoader:
    """
    Loads pre-trained models.
    
    REPLACES: load_trained_model() [lines 285-316]
    """
    
    def __init__(self, device=device):
        """
        Initialize model loader.
        
        Args:
            device: PyTorch device
        """
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self, model_path: str = MODEL_SAVE_PATH) -> tuple:
        """
        Load a pre-trained model without running the full pipeline.
        
        REPLACES: load_trained_model() [lines 285-316]
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Tuple of (model, config, data_info)
        """
        self.logger.debug(f"Loading model from: {model_path}")

        # weights_only=False: checkpoint's 'history'/'config' dicts contain
        # numpy scalars that PyTorch 2.6+'s safe unpickler rejects by default.
        # Safe here since this checkpoint is produced locally by our own
        # training pipeline (flask_api_server.py already does the same).
        checkpoint = torch.load(model_path, weights_only=False)
        
        data_info = checkpoint['data_info']
        config = checkpoint['config']
        
        model = HeterogeneousLightGCN(
            n_clients=data_info['n_clients'],
            n_vehicles=data_info['n_vehicles'],
            n_chauffeurs=data_info['n_chauffeurs'],
            embedding_dim=config['embedding_dim'],
            n_layers=config['n_layers'],
            dropout=config.get('dropout', 0.1),
            use_batch_norm=config.get('use_batch_norm', False)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.logger.info("Model loaded successfully")

        return model, config, data_info


class PipelineDataSaver:
    """
    Saves pipeline intermediate results for API use.
    
    NEW CLASS - Extracted from run_complete_pipeline()
    """
    
    def __init__(self, output_dir: Path = PIPELINE_DIR):
        """
        Initialize pipeline data saver.
        
        Args:
            output_dir: Directory to save pipeline data
        """
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save(
        self,
        step2_results: Dict[str, Any],
        step3_results: Dict[str, Any],
        step4_results: Dict[str, Any]
    ) -> bool:
        """
        Save pipeline step results for API use.
        
        Args:
            step2_results: Metadata enrichment results
            step3_results: Data filtering results
            step4_results: Graph construction results
            
        Returns:
            True if successful
        """
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("SAVING PIPELINE DATA FOR API")
        self.logger.debug("=" * 80)
        
        try:
            # Create directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save step results
            self.logger.info("Saving step2_results.pkl...")
            with open(PIPELINE_PATHS['step2_results'], 'wb') as f:
                pickle.dump(step2_results, f)
            
            self.logger.info("Saving step3_results.pkl...")
            with open(PIPELINE_PATHS['step3_results'], 'wb') as f:
                pickle.dump(step3_results, f)
            
            self.logger.info("Saving step4_results.pkl...")
            with open(PIPELINE_PATHS['step4_results'], 'wb') as f:
                pickle.dump(step4_results, f)
            
            self.logger.info(f"Pipeline data saved to {self.output_dir}/")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline data: {e!r}", exc_info=True)
            return False


class LightGCNPipeline(BasePipeline):
    """
    Main orchestrator for the complete LightGCN recommendation pipeline.
    
    REPLACES: run_complete_pipeline() [lines 49-278]
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        booking_data_path: str = BOOKING_DATA_PATH,
        model_save_path: str = MODEL_SAVE_PATH,
        skip_training: bool = False,
        load_existing_model: bool = False
    ):
        """
        Initialize LightGCN pipeline.
        
        Args:
            config: Pipeline configuration
            booking_data_path: Path to booking parquet file
            model_save_path: Where to save/load the model
            skip_training: If True, skip training and use existing model
            load_existing_model: If True, load model from model_save_path
        """
        super().__init__(config.model if config else BEST_CONFIG)
        
        self.pipeline_config = config or PipelineConfig()
        self.booking_data_path = booking_data_path
        self.model_save_path = model_save_path
        self.skip_training = skip_training
        self.load_existing_model = load_existing_model
        self.device = device
        
        # Initialize components
        self.data_saver = PipelineDataSaver()
        self.model_loader = ModelLoader(device)
    
    def execute(self, input_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Execute the complete LightGCN pipeline.
        
        REPLACES: run_complete_pipeline() [lines 49-278]
        
        Args:
            input_data: Optional pre-loaded data (not used, included for interface)
            
        Returns:
            Dictionary with all results and recommendations
        """
        return self.run()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete LightGCN recommendation pipeline.
        
        REPLACES: run_complete_pipeline() [lines 49-278]
        
        Returns:
            Dictionary with all results and recommendations
        """
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("LIGHTGCN RECOMMENDATION SYSTEM - COMPLETE PIPELINE")
        self.logger.debug("=" * 80)
        
        # STEP 1: Data Preparation
        step1_results = self._require_results(self._run_step1(), "Step 1: Data Preparation")

        # STEP 2: Metadata Enrichment
        step2_results = self._require_results(self._run_step2(step1_results), "Step 2: Metadata Enrichment")

        # STEP 3: Data Filtering
        step3_results = self._require_results(self._run_step3(step2_results), "Step 3: Data Filtering")

        # STEP 4: Graph Construction
        step4_results = self._require_results(self._run_step4(step3_results), "Step 4: Graph Construction")
        
        # Save pipeline data for API
        self.data_saver.save(step2_results, step3_results, step4_results)
        
        # STEP 5: Model Training
        model, history, data = self._run_step5(step4_results)
        
        # STEP 6: Generate Recommendations
        all_recommendations = self._run_step6(
            model, step2_results, step3_results, step4_results, data
        )
        
        # PIPELINE COMPLETE
        self._log_final_summary(all_recommendations)
        
        return {
            'step1_results': step1_results,
            'step2_results': step2_results,
            'step3_results': step3_results,
            'step4_results': step4_results,
            'model': model,
            'training_history': history,
            'data': data,
            'recommendations': all_recommendations
        }
    
    def _require_results(self, results: Optional[Dict[str, Any]], step_name: str) -> Dict[str, Any]:
        """Abort the pipeline instead of letting a failed step's None cascade downstream."""
        if results is None:
            raise RuntimeError(
                f"{step_name} failed (returned None) — see the traceback logged above "
                f"for the underlying error. Aborting pipeline."
            )
        return results

    def _run_step1(self) -> Dict[str, Any]:
        """Execute Step 1: Data Preparation."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("STEP 1: DATA PREPARATION")
        self.logger.debug("=" * 80)
        
        config = {'booking_data_path': self.booking_data_path}
        pipeline = DataPreparationPipeline(config)
        return pipeline.execute()
    
    def _run_step2(self, step1_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Step 2: Metadata Enrichment."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("STEP 2: METADATA ENRICHMENT")
        self.logger.debug("=" * 80)
        
        pipeline = MetadataEnrichmentPipeline()
        return pipeline.execute(step1_results)
    
    def _run_step3(self, step2_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Step 3: Data Filtering."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("STEP 3: DATA FILTERING")
        self.logger.debug("=" * 80)
        
        pipeline = DataFilteringPipeline(self.pipeline_config.filtering)
        return pipeline.execute(step2_results)
    
    def _run_step4(self, step3_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Step 4: Graph Construction."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("STEP 4: GRAPH CONSTRUCTION")
        self.logger.debug("=" * 80)
        
        pipeline = GraphConstructionPipeline(self.pipeline_config.graph)
        return pipeline.execute(step3_results)
    
    def _run_step5(self, step4_results: Dict[str, Any]) -> tuple:
        """Execute Step 5: Model Training."""
        if not self.skip_training:
            self.logger.debug("\n" + "=" * 80)
            self.logger.info("STEP 5: MODEL TRAINING")
            self.logger.debug("=" * 80)
            
            pipeline = ModelTrainingPipeline(self.pipeline_config.model)
            results = self._require_results(pipeline.execute(step4_results), "Step 5: Model Training")

            model = results['model']
            history = results['training_history']
            data = results['data']
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': self.pipeline_config.model,
                'history': history,
                'data_info': {
                    'n_clients': data['n_clients'],
                    'n_vehicles': data['n_vehicles'],
                    'n_chauffeurs': data['n_chauffeurs']
                }
            }, self.model_save_path)
            
            self.logger.debug(f"\nModel saved to: {self.model_save_path}")
            
        else:
            self.logger.debug("\nSkipping training, loading existing model...")
            
            model, config, data_info = self.model_loader.load(self.model_save_path)
            
            data_loader = GraphDataLoader()
            data = data_loader.load(step4_results)
            
            history = []
        
        return model, history, data
    
    def _run_step6(
        self,
        model: HeterogeneousLightGCN,
        step2_results: Dict[str, Any],
        step3_results: Dict[str, Any],
        step4_results: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Step 6: Generate Recommendations."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("STEP 6: GENERATING RECOMMENDATIONS")
        self.logger.debug("=" * 80)
        
        # Convert adjacency matrix to torch sparse
        normalized_adj = data['normalized_adj']
        adj_sparse = TorchConverter.sparse_to_torch(normalized_adj, self.device)
        
        # Generate all recommendations
        all_recommendations = generate_all_recommendations(
            model=model,
            step2_results=step2_results,
            step3_results=step3_results,
            step4_results=step4_results,
            data=data,
            adj_sparse=adj_sparse,
            top_k=self.pipeline_config.recommendation['top_k'],
            n_similar=self.pipeline_config.recommendation['n_similar']
        )
        
        return all_recommendations
    
    def _log_final_summary(self, recommendations: Dict[str, Any]) -> None:
        """Log final pipeline summary."""
        self.logger.debug("\n" + "=" * 80)
        self.logger.debug("=" * 80)
        self.logger.info("PIPELINE COMPLETE!")
        self.logger.debug("=" * 80)
        
        cv_df = recommendations.get('cv_recommendations')
        if cv_df is not None and not cv_df.empty:
            total_clients = cv_df['client_id'].nunique() if 'client_id' in cv_df.columns else len(cv_df)
            self.logger.debug(f"  Total clients with recommendations: {total_clients:,}")
        self.logger.debug(f"\nFiles saved:")
        self.logger.debug(f"  Model: {self.model_save_path}")
        self.logger.info(f"  Graph data: {self.pipeline_config.graph['output_prefix']}_*")
        self.logger.info(f"  Recommendations: client_*_recommendations_COMPLETE.csv")



class RecommendationOnlyPipeline:
    """
    Generates recommendations using a pre-trained model and existing pipeline results.
    
    REPLACES: generate_recommendations_only() [lines 319-360]
    """
    
    def __init__(self, model_path: str = MODEL_SAVE_PATH, device=device):
        """
        Initialize recommendation-only pipeline.
        
        Args:
            model_path: Path to saved model
            device: PyTorch device
        """
        self.model_path = model_path
        self.device = device
        self.model_loader = ModelLoader(device)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(
        self,
        step2_results: Dict[str, Any],
        step3_results: Dict[str, Any],
        step4_results: Dict[str, Any],
        top_k: int = 100,
        n_similar: int = 5
    ) -> Dict[str, Any]:
        """
        Generate recommendations using a pre-trained model.
        
        REPLACES: generate_recommendations_only() [lines 319-360]
        
        Args:
            step2_results: Metadata enrichment results
            step3_results: Data filtering results
            step4_results: Graph construction results
            top_k: Number of recommendations
            n_similar: Number of similar items for fallback
            
        Returns:
            Dictionary of recommendations
        """
        self.logger.debug("\n" + "=" * 80)
        self.logger.info("GENERATING RECOMMENDATIONS (USING PRE-TRAINED MODEL)")
        self.logger.debug("=" * 80)
        
        # Load model
        model, config, data_info = self.model_loader.load(self.model_path)
        
        # Load graph data
        data_loader = GraphDataLoader()
        data = data_loader.load(step4_results)
        
        # Convert adjacency matrix
        adj_sparse = TorchConverter.sparse_to_torch(data['normalized_adj'], self.device)
        
        # Generate recommendations
        all_recommendations = generate_all_recommendations(
            model=model,
            step2_results=step2_results,
            step3_results=step3_results,
            step4_results=step4_results,
            data=data,
            adj_sparse=adj_sparse,
            top_k=top_k,
            n_similar=n_similar
        )
        
        self.logger.info("\nRecommendations generated successfully!")
        
        return all_recommendations


class PipelineTester:
    """
    Tests pipeline components.
    
    REPLACES: quick_test() [lines 363-394]
    """
    
    def __init__(self):
        """Initialize pipeline tester."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def quick_test(self) -> bool:
        """
        Quick test to verify all modules are working.
        
        REPLACES: quick_test() [lines 363-394]
        
        Returns:
            True if all tests pass
        """
        self.logger.debug("\n" + "=" * 80)
        self.logger.debug("RUNNING QUICK TEST")
        self.logger.debug("=" * 80)
        
        try:
            # Test imports
            self.logger.debug("\n1. Testing imports...")
            from data_preparation_and_loading import load_booking_data
            from metadata_enrichment import load_metadata_tables
            from data_filtering import create_three_edge_types
            from graph_construction import create_edge_lists
            from model_training import HeterogeneousLightGCN
            from recommendation_genrator import generate_comprehensive_recommendations
            self.logger.debug("   All imports successful")
            
            # Test data loading
            self.logger.debug("\n2. Testing data loading...")
            booking_df = load_booking_data()
            self.logger.info(f"   Loaded {len(booking_df)} booking records")
            
            metadata = load_metadata_tables()
            self.logger.info(f"   Loaded {len(metadata)} metadata tables")
            
            self.logger.info("\nQuick test passed!")
            self.logger.info("All modules are ready to use.")
            return True
            
        except Exception as e:
            self.logger.error(f"\nTest failed: {e}")
            import traceback
            traceback.print_exc()
            return False


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def run_complete_pipeline(
    booking_data_path=BOOKING_DATA_PATH,
    model_config=BEST_CONFIG,
    model_save_path=MODEL_SAVE_PATH,
    graph_output_prefix=GRAPH_OUTPUT_PREFIX,
    skip_training=False,
    load_existing_model=False
):
    """
    Backward compatible wrapper for LightGCNPipeline.run().
    
    WRAPS: LightGCNPipeline.run()
    """
    config = PipelineConfig(
        model_config=model_config,
        graph_config={'save_data': True, 'output_prefix': graph_output_prefix}
    )
    
    pipeline = LightGCNPipeline(
        config=config,
        booking_data_path=booking_data_path,
        model_save_path=model_save_path,
        skip_training=skip_training,
        load_existing_model=load_existing_model
    )
    
    return pipeline.run()


def load_trained_model(model_path=MODEL_SAVE_PATH):
    """
    Backward compatible wrapper for ModelLoader.load().
    
    WRAPS: ModelLoader.load()
    """
    loader = ModelLoader(device)
    return loader.load(model_path)


def generate_recommendations_only(
    model_path=MODEL_SAVE_PATH,
    step2_results=None,
    step3_results=None,
    step4_results=None
):
    """
    Backward compatible wrapper for RecommendationOnlyPipeline.run().
    
    WRAPS: RecommendationOnlyPipeline.run()
    """
    if step2_results is None or step3_results is None or step4_results is None:
        raise ValueError("Must provide step2_results, step3_results, and step4_results")
    
    pipeline = RecommendationOnlyPipeline(model_path, device)
    return pipeline.run(step2_results, step3_results, step4_results)


def quick_test():
    """
    Backward compatible wrapper for PipelineTester.quick_test().
    
    WRAPS: PipelineTester.quick_test()
    """
    tester = PipelineTester()
    return tester.quick_test()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LightGCN Recommendation System')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train_only', 'recommendations_only', 'test'],
                       help='Execution mode')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and load existing model')
    parser.add_argument('--model-path', type=str, default=MODEL_SAVE_PATH,
                       help='Path to save/load model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config JSON file')
    
    args = parser.parse_args()
    
    # Load custom config if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        logger.info(f"Loaded custom config from: {args.config}")
    else:
        custom_config = BEST_CONFIG
    
    # Execute based on mode
    if args.mode == 'test':
        quick_test()
    
    elif args.mode == 'full':
        config = PipelineConfig(model_config=custom_config)
        pipeline = LightGCNPipeline(
            config=config,
            model_save_path=args.model_path,
            skip_training=args.skip_training
        )
        results = pipeline.run()
        logger.info("\nFull pipeline completed successfully!")
    
    elif args.mode == 'train_only':
        logger.info("Training mode: Will stop after model training")
        # Implement if needed
    
    elif args.mode == 'recommendations_only':
        logger.info("Recommendations only mode")
        logger.info("Note: You need to provide step2, step3, step4 results")
        logger.info("This mode is best used programmatically, not from command line")
    
    else:
        logger.info(f"Unknown mode: {args.mode}")


# =============================================================================
# MODULE INFO
# =============================================================================

logger.info("Orchestrator Module (Refactored) - Ready")
logger.debug(
    "Classes: PipelineConfig, ModelLoader, PipelineDataSaver, LightGCNPipeline, "
    "RecommendationOnlyPipeline, PipelineTester | "
    "Functions: run_complete_pipeline(), load_trained_model(), "
    "generate_recommendations_only(), quick_test()"
)

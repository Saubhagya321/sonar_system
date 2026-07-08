import argparse
import logging
from pathlib import Path
from datetime import datetime

from config.config import Config
from data.data_loader import DataLoader
from preprocessing.data_cleaner import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer
from models.ensemble_model import EnsembleRecommender, generate_recommendations_batch
from evaluation.metrics import ModelEvaluator
from utils.helpers import (
    setup_logging, save_pickle, load_pickle, save_dataframe, load_dataframe,
    print_summary_statistics, export_recommendations_csv
)

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]


class RecommendationPipeline:
    """Complete recommendation system pipeline"""
    
    def __init__(self, config: Config = None, force_retrain: bool = False):
        self.config = config or Config()
        self.config.ensure_directories()
        self.force_retrain = force_retrain
        
        # Pipeline components
        self.data_loader = DataLoader(self.config)
        self.data_cleaner = DataCleaner(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.evaluator = ModelEvaluator()
        
        # Data storage
        self.raw_data = None
        self.cleaned_booking = None
        self.booking_with_pairs = None
        self.pair_features = None
        self.implicit_ratings = None
        self.final_ratings = None
        self.interaction_matrix = None
        self.model_ready_data = None
        self.train_data = None
        self.test_data = None
        self.ensemble_model = None
    
    def run_step_1_data_loading(self):
        """
        STEP 1: Load all data
        FROM: Initial cell + Document 1 analysis
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("="*80)
        
        self.raw_data = self.data_loader.load_all_data()
        
        logger.info(f"\n   STEP 1 COMPLETE: Loaded {len(self.raw_data)} datasets")
        return self.raw_data
    
    def run_step_2_preprocessing(self):
        """
        STEP 2: Data cleaning and feature engineering - FIXED VERSION
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
        logger.info("="*80)
        
        # NEW: Use separate cache for booking_with_pairs
        BOOKING_WITH_PAIRS_PATH = self.config.OUTPUT_DIR / "booking_with_pairs.pkl"
        PAIR_FEATURES_PATH = self.config.PAIR_FEATURES_PATH

        # Check if already processed
        if (BOOKING_WITH_PAIRS_PATH.exists() and 
            PAIR_FEATURES_PATH.exists() and
            not self.force_retrain):
            
            logger.info("  Loading preprocessed data from cache...")
            try:
                self.booking_with_pairs = load_dataframe(BOOKING_WITH_PAIRS_PATH)
                self.pair_features = load_dataframe(PAIR_FEATURES_PATH)
                
                # Validate loaded data
                if self.booking_with_pairs is None or 'ChauffeurVehiclePair' not in self.booking_with_pairs.columns:
                    logger.warning("  Cached data invalid or missing ChauffeurVehiclePair column")
                    logger.info("   Regenerating from scratch...")
                    self.force_retrain = True  # Force fresh generation
                else:
                    logger.info(f"      Loaded {len(self.booking_with_pairs):,} bookings with pairs")
                    logger.info(f"      Loaded {len(self.pair_features):,} pair features")
                    
                    # Set cleaned_booking for later use
                    self.cleaned_booking = self.booking_with_pairs
            
            except Exception as e:
                logger.error(f"     Failed to load cached data: {e}")
                logger.info("   Regenerating from scratch...")
                self.force_retrain = True
        
        # Generate fresh data if needed
        if self.force_retrain or not BOOKING_WITH_PAIRS_PATH.exists():
            # 2A: Clean booking data
            logger.info(" Cleaning booking data...")
            self.cleaned_booking = self.data_cleaner.clean_booking_data(
                self.raw_data['booking']
            )
            
            # 2B: Create pair features (includes pair ID creation with validation)
            logger.info("   Creating pair features...")
            self.booking_with_pairs, self.pair_features = self.feature_engineer.create_pair_features(
                self.cleaned_booking
            )
            
            # Save both versions
            logger.info(" Caching preprocessed data...")
            save_dataframe(self.cleaned_booking, self.config.BOOKING_CLEAN_PATH)
            save_dataframe(self.booking_with_pairs, BOOKING_WITH_PAIRS_PATH)
            save_dataframe(self.pair_features, PAIR_FEATURES_PATH)
        
        # 2C: Build implicit ratings
        logger.info(" Building implicit ratings...")
        self.implicit_ratings = self.feature_engineer.build_implicit_ratings(
            self.booking_with_pairs,
            self.pair_features
        )
        
        # 2D: Integrate explicit ratings
        logger.info(" Integrating explicit ratings...")
        self.final_ratings = self.feature_engineer.integrate_explicit_ratings(
            self.implicit_ratings,
            self.raw_data['booking_rating'],
            self.booking_with_pairs
        )
        
        # 2E: Create interaction matrix
        logger.info(" Creating interaction matrix...")
        self.interaction_matrix, self.model_ready_data = self.feature_engineer.create_interaction_matrix(
            self.final_ratings,
            min_interactions=self.config.MIN_INTERACTIONS
        )
        
        # Save processed data
        save_dataframe(self.interaction_matrix, self.config.INTERACTION_MATRIX_PATH)
        save_dataframe(self.model_ready_data, self.config.MODEL_READY_DATA_PATH)
        
        logger.info(f"\n   STEP 2 COMPLETE: Created interaction matrix "
                f"({self.interaction_matrix.shape[0]:,} × {self.interaction_matrix.shape[1]:,})")
        
        return self.interaction_matrix, self.model_ready_data
    
    def run_step_3_train_model(self):
        """
        STEP 3: Train ensemble model
        FROM: Document 4 (Step 4) - optimized ensemble
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*80)
        
        # Check if model already trained
        if self.config.TRAINED_MODEL_PATH.exists() and not self.force_retrain:
            logger.info("  Loading pre-trained model...")
            self.ensemble_model = load_pickle(self.config.TRAINED_MODEL_PATH)
            logger.info("   Loaded pre-trained ensemble model")
        else:
            # Create train/test split
            from sklearn.model_selection import train_test_split
            self.train_data, self.test_data = train_test_split(
                self.model_ready_data,
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE
            )
            
            # Create training matrix
            train_matrix = self.train_data.pivot_table(
                index='RefClient',
                columns='ChauffeurVehiclePair',
                values='final_rating',
                fill_value=0
            )
            
            # Train ensemble model
            self.ensemble_model = EnsembleRecommender(
                weights=self.config.ENSEMBLE_WEIGHTS,
                config=self.config
            )
            self.ensemble_model.fit(train_matrix, self.train_data)
            
            # Evaluate model
            evaluation_results = self.evaluator.evaluate_ensemble(
                self.ensemble_model,
                self.test_data,
                n_samples=1000
            )
            
            logger.info("\n MODEL EVALUATION RESULTS:")
            logger.info(f"   • RMSE: {evaluation_results['rmse']:.4f}")
            logger.info(f"   • MAE: {evaluation_results['mae']:.4f}")
            logger.info(f"   • Avg Confidence: {evaluation_results['avg_confidence']:.3f}")
            
            # Save trained model
            save_pickle(self.ensemble_model, self.config.TRAINED_MODEL_PATH)
        
        logger.info("\n   STEP 3 COMPLETE: Model trained and ready")
        
        return self.ensemble_model
    
    def run_step_4_generate_recommendations(self, n_recommendations: int = None, batch_size: int = None):
        """
        STEP 4: Generate recommendations for all clients
        FROM: Document 4 (Step 4) - recommendation generation
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 4: RECOMMENDATION GENERATION")
        logger.info("="*80)
        
        n_recommendations = n_recommendations or self.config.N_RECOMMENDATIONS
        batch_size = batch_size or self.config.BATCH_SIZE
        
        # Get all clients
        all_clients = list(self.interaction_matrix.index)
        
        # Generate recommendations
        recommendations, stats = generate_recommendations_batch(
            model=self.ensemble_model,
            client_list=all_clients,
            n_recommendations=n_recommendations,
            batch_size=batch_size
        )
        
        # Print summary
        print_summary_statistics(recommendations, self.model_ready_data)
        
        # Export to CSV
        output_csv = self.config.OUTPUT_DIR / f"recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # RECOMMENDATIONS_CSV_FINAL_PATH defaults to centralized_data/ for standalone runs.
        # The retraining orchestrator overrides it to staging/ so this write never
        # touches the live path during a retraining cycle.
        output_csv_final = Path(self.config.RECOMMENDATIONS_CSV_FINAL_PATH)
        export_recommendations_csv(recommendations, output_csv)
        output_csv_final.parent.mkdir(parents=True, exist_ok=True)
        export_recommendations_csv(recommendations, output_csv_final)
        
        logger.info(f"\n   STEP 4 COMPLETE: Generated recommendations for {len(recommendations):,} clients")
        
        return recommendations, stats
    
    def run_full_pipeline(self, n_recommendations: int = None, batch_size: int = None):
        """Run the complete pipeline"""
        logger.info("\n" + "="*80)
        logger.info("   STARTING COMPLETE RECOMMENDATION PIPELINE")
        logger.info("="*80)
        logger.info("Configuration:")
        logger.info(f"  • Force retrain: {self.force_retrain}")
        logger.info(f"  • Recommendations per client: {n_recommendations or self.config.N_RECOMMENDATIONS}")
        logger.info(f"  • Batch size: {batch_size or self.config.BATCH_SIZE}")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        self.run_step_1_data_loading()
        
        # Step 2: Preprocess and engineer features
        self.run_step_2_preprocessing()
        
        # Step 3: Train model
        self.run_step_3_train_model()
        
        # Step 4: Generate recommendations
        recommendations, stats = self.run_step_4_generate_recommendations(
            n_recommendations=n_recommendations,
            batch_size=batch_size
        )
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info(" PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total execution time: {total_time/60:.1f} minutes")
        logger.info(f"Recommendations generated: {len(recommendations):,}")
        logger.info("="*80)
        
        return recommendations, stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Chauffeur-Vehicle Recommendation System')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retrain the model (ignore cached models)')
    parser.add_argument('--n_recommendations', type=int, default=100,
                       help='Number of recommendations per client (default: 100)')
    parser.add_argument('--batch_size', type=int, default=200,
                       help='Batch size for processing (default: 200)')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4],
                       help='Run only a specific step (default: run all)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config = Config()
    pipeline = RecommendationPipeline(config=config, force_retrain=args.retrain)
    
    if args.step:
        # Run specific step
        if args.step == 1:
            pipeline.run_step_1_data_loading()
        elif args.step == 2:
            pipeline.run_step_1_data_loading()
            pipeline.run_step_2_preprocessing()
        elif args.step == 3:
            pipeline.run_step_1_data_loading()
            pipeline.run_step_2_preprocessing()
            pipeline.run_step_3_train_model()
        elif args.step == 4:
            # Load previously processed data
            pipeline.interaction_matrix = load_dataframe(config.INTERACTION_MATRIX_PATH)
            pipeline.model_ready_data = load_dataframe(config.MODEL_READY_DATA_PATH)
            pipeline.ensemble_model = load_pickle(config.TRAINED_MODEL_PATH)
            pipeline.run_step_4_generate_recommendations(
                n_recommendations=args.n_recommendations,
                batch_size=args.batch_size
            )
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(
            n_recommendations=args.n_recommendations,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
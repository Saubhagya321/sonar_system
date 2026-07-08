# collabrative 
"""
Model evaluation metrics and utilities
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate recommendation models"""
    
    @staticmethod
    def evaluate_model(model, test_data: pd.DataFrame, model_name: str, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Evaluate a recommendation model
        FROM: Document 3 - evaluate_model function
        
        Args:
            model: Trained model with predict_rating method
            test_data: Test dataframe with ratings
            model_name: Name of the model
            n_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f" Evaluating {model_name}...")
        
        # Sample test data for efficiency
        test_sample = test_data.sample(min(n_samples, len(test_data)), random_state=42)
        
        predictions = []
        actuals = []
        
        for _, row in test_sample.iterrows():
            user_id = row['RefClient']
            item_id = row['ChauffeurVehiclePair']
            actual_rating = row['final_rating']
            
            if hasattr(model, 'predict_rating'):
                predicted_rating = model.predict_rating(user_id, item_id)
            else:
                # For popularity model
                if hasattr(model, 'item_popularity') and item_id in model.item_popularity.index:
                    predicted_rating = model.item_popularity.loc[item_id, 'popularity_score']
                else:
                    predicted_rating = model.global_mean
            
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        logger.info(f"   Evaluated on {len(predictions):,} test cases")
        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   MAE:  {mae:.4f}")
        
        return {
            'model': model_name,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'n_predictions': len(predictions)
        }
    
    @staticmethod
    def evaluate_ensemble(model, test_data: pd.DataFrame, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Evaluate ensemble model with confidence scores
        FROM: Document 4 - evaluate_ensemble function
        
        Args:
            model: Trained ensemble model
            test_data: Test dataframe
            n_samples: Number of samples
            
        Returns:
            Dictionary with metrics including confidence
        """
        logger.info(" Evaluating Ensemble Model...")
        
        test_sample = test_data.sample(min(n_samples, len(test_data)), random_state=42)
        
        predictions = []
        actuals = []
        confidences = []
        
        for _, row in test_sample.iterrows():
            user_id = row['RefClient']
            item_id = row['ChauffeurVehiclePair']
            actual_rating = row['final_rating']
            
            pred_rating = model.predict_rating(user_id, item_id)
            confidence = model.calculate_confidence(user_id, item_id)
            
            predictions.append(pred_rating)
            actuals.append(actual_rating)
            confidences.append(confidence)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        logger.info(f"   Evaluated on {len(predictions):,} test cases")
        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   MAE:  {mae:.4f}")
        logger.info(f"   Avg Confidence: {np.mean(confidences):.3f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'n_predictions': len(predictions)
        }
    
    @staticmethod
    def compare_models(evaluation_results: list) -> pd.DataFrame:
        """
        Compare multiple model evaluation results
        
        Args:
            evaluation_results: List of evaluation result dicts
            
        Returns:
            DataFrame with comparison
        """
        results_df = pd.DataFrame(evaluation_results)
        results_df = results_df.sort_values('rmse')
        
        logger.info("\n MODEL PERFORMANCE RANKING (by RMSE):")
        logger.info("-" * 70)
        for i, row in results_df.iterrows():
            logger.info(f"{i+1}. {row['model']:<30} | RMSE: {row['rmse']:.4f} | MAE: {row['mae']:.4f}")
        
        return results_df
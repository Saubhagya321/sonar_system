#feedback file
"""
Feedback Loop System for Recommendation Combiner
================================================

This package provides a dispatcher feedback loop that learns which 
recommendation models work best for each client and dynamically adjusts 
fusion weights to improve accuracy over time.

Modules:
    - config: Configuration constants and rejection reason codes
    - store: SQL Server database operations for feedback storage
    - logger: Logs recommendation events for feedback matching
    - processor: Processes dispatcher selections and computes model credits
    - weight_learner: Updates model performance statistics
    - weight_selector: Provides personalized weights to the combiner
    - api: Flask endpoints for feedback submission

Usage:
    from feedback import WeightSelector, FeedbackLogger, feedback_bp
    
    # Initialize
    weight_selector = WeightSelector()
    feedback_logger = FeedbackLogger()
    
    # Register Flask blueprint
    app.register_blueprint(feedback_bp)
    
    # Get personalized weights
    weights = weight_selector.get_weights(client_id, segment, base_weights)
    
    # Log shown event
    feedback_logger.log_shown_event(request_id, client_id, ...)

Author: Recommendation System Team
Version: 1.0.0
"""

from .config import (
    FEEDBACK_CONFIG,
    REJECTION_REASONS,
    DB_CONFIG,
    get_db_connection_string
)

from .store import FeedbackStore

from .logger import FeedbackLogger

from .processor import FeedbackProcessor

from .weight_learner import WeightLearner

from .weight_selector import WeightSelector

from .api import feedback_bp, init_feedback_system

__all__ = [
    # Configuration
    'FEEDBACK_CONFIG',
    'REJECTION_REASONS',
    'DB_CONFIG',
    'get_db_connection_string',
    
    # Core classes
    'FeedbackStore',
    'FeedbackLogger',
    'FeedbackProcessor',
    'WeightLearner',
    'WeightSelector',
    
    # Flask blueprint
    'feedback_bp',
    'init_feedback_system'
]

__version__ = '1.0.0'
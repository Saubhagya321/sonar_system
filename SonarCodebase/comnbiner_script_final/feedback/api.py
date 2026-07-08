"""
Feedback API - Flask Endpoints for Feedback Submission
======================================================

UPDATED: Added separate /reject and /select endpoints + booking helpers

Provides REST API endpoints for:
- Submitting dispatcher selections
- Submitting rejections (NEW)
- Getting current recommendation (NEW)
- Querying learning statistics
- Monitoring system health

Register with Flask app:
    from feedback.api import feedback_bp
    app.register_blueprint(feedback_bp)

╔═══════════════════════════════════════════════════════════════════════════════╗
║                              ENDPOINTS SUMMARY                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ NEW ENDPOINTS:                                                                ║
║   POST /api/feedback/reject                  - Submit single rejection        ║
║   POST /api/feedback/select                  - Submit selection (calls SP)    ║
║   GET  /api/feedback/booking/<id>/current    - Get current recommendation     ║
║   GET  /api/feedback/booking/<id>/rejections - Get rejection history          ║
║                                                                               ║
║ EXISTING ENDPOINTS:                                                           ║
║   POST /api/feedback/selection               - Legacy (backward compat)       ║
║   POST /api/feedback/batch                   - Batch selections               ║
║   GET  /api/feedback/stats/<id>              - Client statistics              ║
║   GET  /api/feedback/weights/<id>            - Client weights                 ║
║   GET  /api/feedback/model-ranking/<id>      - Model ranking                  ║
║   GET  /api/feedback/system/stats            - System statistics              ║
║   GET  /api/feedback/system/health           - Health check                   ║
║   GET  /api/feedback/rejection-reasons       - Rejection codes                ║
║   POST /api/feedback/invalidate-cache        - Invalidate cache               ║
║   GET  /api/feedback/shown-event/<id>        - Shown event details            ║
║   GET  /api/feedback/recommendations/<id>    - Recommendations by booking     ║
║   GET  /api/feedback/batch-recommendations/<id> - Recommendations by batch    ║
║   GET  /api/feedback/recommendations-by-batch/<id> - Alt batch endpoint       ║
║   GET  /api/feedback/recent-feedback         - Recent feedback                ║
║   GET  /api/feedback/client-feedback/<id>    - Client feedback history        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                           SP CALL LOCATIONS                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ [DispatcherActivity] @Status=1 (Accept)                                       ║
║   - POST /api/feedback/select                                                 ║
║   - POST /api/feedback/selection (legacy)                                     ║
║   - POST /api/feedback/batch (each selection)                                 ║
║   → Updates AvailaibityAllocation.DispatcherStatus = 1                        ║
║   → Inserts into FinalAllocation                                              ║
║                                                                               ║
║ [DispatcherActivity] @Status=0 (Reject)                                       ║
║   - POST /api/feedback/reject                                                 ║
║   → Updates AvailaibityAllocation.DispatcherStatus = 0                        ║
║                                                                               ║
║ [SelectRecommendation] - Can be used in GET /booking/<id>/current             ║
║   → Currently using direct query instead (more control for filtering)         ║
║                                                                               ║
║ [AllocateByRecommendation] - NOT called from feedback API                     ║
║   → Called from combiner_api after batch-process                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import json
from flask import Blueprint, request, jsonify
from typing import Dict, Optional
from sqlalchemy import create_engine, text

from .store import FeedbackStore
from .logger import FeedbackLogger
from .processor import FeedbackProcessor
from .weight_learner import WeightLearner
from .weight_selector import WeightSelector
from .config import REJECTION_REASONS, DEFAULT_BASE_WEIGHTS, DB_CONNECTION_STRING, TRANSELITE_DB

import pytz
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Create Blueprint
feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/feedback')

# Module-level components (initialized via init_feedback_system)
_store: Optional[FeedbackStore] = None
_logger: Optional[FeedbackLogger] = None
_processor: Optional[FeedbackProcessor] = None
_weight_learner: Optional[WeightLearner] = None
_weight_selector: Optional[WeightSelector] = None




def init_feedback_system() -> Dict:
    """
    Initialize the feedback system components.
    
    Call this before using the feedback API endpoints.
    
    Returns:
        Dictionary with initialized component references
    
    Usage:
        from feedback.api import init_feedback_system
        
        components = init_feedback_system()
        feedback_logger = components['logger']
    """
    global _store, _logger, _processor, _weight_learner, _weight_selector
    
    _store = FeedbackStore() 
    _weight_learner = WeightLearner(_store)
    _logger = FeedbackLogger(_store)
    _processor = FeedbackProcessor(_store, _weight_learner)
    _weight_selector = WeightSelector()
    
    logger.info("Feedback system initialized")
    
    return {
        'store': _store,
        'logger': _logger,
        'processor': _processor,
        'weight_learner': _weight_learner,
        'weight_selector': _weight_selector
    }


def get_components() -> Dict:
    """
    Get feedback system components.
    
    Initializes if not already done.
    
    Returns:
        Dictionary with component references
    """
    global _store, _logger, _processor, _weight_learner, _weight_selector
    
    if _store is None:
        init_feedback_system()
    
    return {
        'store': _store,
        'logger': _logger,
        'processor': _processor,
        'weight_learner': _weight_learner,
        'weight_selector': _weight_selector
    }


# =============================================================================
# NEW ENDPOINTS - Separate Reject and Select
# =============================================================================

@feedback_bp.route('/reject', methods=['POST'])
def submit_rejection():
    """
    Submit a SINGLE rejection for a recommendation
    ---
    tags:
      - Feedback - Dispatcher Actions
    summary: Reject a recommendation and get the next one
    description: |
      Submit a single rejection for a recommendation. 
      - LLM classifies the rejection reason
      - Learning penalty is applied immediately (if learnable)
      - **CALLS [DispatcherActivity] SP @Status=0** to update AvailaibityAllocation
      - Returns the next available recommendation
      
      **SP CALLED: [DispatcherActivity] @Status=0, @BookingID**
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - request_id
            - pair_id
          properties:
            request_id:
              type: string
              description: BookingID
              example: "123"
            pair_id:
              type: string
              description: The pair being rejected (VehicleId_ChauffeurId)
              example: "2710_1287"
            reason_text:
              type: string
              description: Dispatcher's reason for rejection (free text)
              example: "Driver was late on previous bookings"
    responses:
      200:
        description: Rejection processed successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            rejected_pair:
              type: string
              example: "2710_1287"
            rejection_category:
              type: string
              description: LLM classified category
              example: "past_negative_experience"
            classification:
              type: object
              description: Full LLM classification details
            credits_applied:
              type: object
              description: Penalty credits (negative values)
              example: {"lightgcn": -0.02, "collaborative": -0.015, "user_content": -0.01, "item_content": -0.008}
            learning_applied:
              type: boolean
              description: Whether learning was applied (false for non-learnable rejections)
              example: true
            allocation_updated:
              type: boolean
              description: Whether DispatcherActivity SP succeeded
              example: true
            next_recommendation:
              type: object
              description: Next available recommendation or null if none left
              properties:
                booking_id:
                  type: integer
                  example: 123
                pair_id:
                  type: string
                  example: "3132_1541"
                vehicle_id:
                  type: string
                  example: "3132"
                chauffeur_id:
                  type: string
                  example: "1541"
                rank:
                  type: integer
                  example: 2
                score:
                  type: number
                  example: 0.89
            remaining_count:
              type: integer
              description: How many recommendations still available
              example: 19
      400:
        description: Invalid request (missing required fields)
      404:
        description: Request ID or Pair ID not found
    """
    try:
        components = get_components()
        processor = components['processor']
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        if not data.get('request_id'):
            return jsonify({
                'success': False,
                'error': 'request_id is required'
            }), 400
        
        if not data.get('pair_id'):
            return jsonify({
                'success': False,
                'error': 'pair_id is required'
            }), 400
        
        result = processor.process_rejection(data)
        
        if result['success']:
            return jsonify(result), 200
        else:
            status_code = 404 if 'not found' in result.get('error', '').lower() else 400
            return jsonify(result), status_code
            
    except Exception as e:
        logger.error(f"Error in submit_rejection: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@feedback_bp.route('/select', methods=['POST'])
def submit_select():
    """
    Submit SELECTION (acceptance) of a recommendation
    ---
    tags:
      - Feedback - Dispatcher Actions
    summary: Accept a recommendation and confirm booking
    description: |
      Submit selection/acceptance of a recommendation.
      - Calculates reward credits for models
      - Updates learning (EMA)
      - **CALLS [DispatcherActivity] SP** to confirm booking
      - Moves booking to FinalAllocation
      
      **SP CALLED: [DispatcherActivity] @Status=1, @BookingID**
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - request_id
            - selected_pair_id
          properties:
            request_id:
              type: string
              description: BookingID
              example: "123"
            selected_pair_id:
              type: string
              description: The pair being accepted (VehicleId_ChauffeurId)
              example: "3132_1541"
            overall_quality:
              type: integer
              minimum: 1
              maximum: 5
              description: Optional 1-5 star quality rating
              example: 4
    responses:
      200:
        description: Selection processed and booking confirmed
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            selected_pair_id:
              type: string
              example: "3132_1541"
            selected_rank:
              type: integer
              description: Rank of the selected recommendation
              example: 2
            overall_quality:
              type: integer
              example: 4
            model_credits:
              type: object
              description: Reward credits (positive values)
              example: {"lightgcn": 0.05, "collaborative": 0.03, "user_content": 0.02, "item_content": 0.015}
            total_rejections:
              type: integer
              description: Number of rejections before this selection
              example: 1
            learning_applied:
              type: boolean
              example: true
            allocation_confirmed:
              type: boolean
              description: Whether DispatcherActivity SP succeeded
              example: true
            client_id:
              type: integer
              example: 4415
      400:
        description: Invalid request
      404:
        description: Request ID or Pair ID not found
    """
    try:
        components = get_components()
        processor = components['processor']
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        if not data.get('request_id'):
            return jsonify({
                'success': False,
                'error': 'request_id is required'
            }), 400
        
        if not data.get('selected_pair_id'):
            return jsonify({
                'success': False,
                'error': 'selected_pair_id is required'
            }), 400
        
        quality = data.get('overall_quality')
        if quality is not None:
            if not isinstance(quality, int) or quality < 1 or quality > 5:
                return jsonify({
                    'success': False,
                    'error': 'overall_quality must be an integer between 1 and 5'
                }), 400
        
        result = processor.process_selection(data)
        
        if result['success']:
            return jsonify(result), 200
        else:
            status_code = 404 if 'not found' in result.get('error', '').lower() else 400
            return jsonify(result), status_code
            
    except Exception as e:
        logger.error(f"Error in submit_select: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@feedback_bp.route('/booking/<request_id>/current', methods=['GET'])
def get_current_recommendation(request_id: str):
    """
    Get the CURRENT recommendation to review for a booking
    ---
    tags:
      - Feedback - Dispatcher Actions
    summary: Get next available recommendation for dispatcher
    description: |
      Returns the next available (non-rejected, non-selected) recommendation
      for a booking. Used by dispatcher UI to show current option.
      
      **NO SP CALLED** - Uses direct query with filtering.
      Could optionally use [SelectRecommendation] SP.
    parameters:
      - name: request_id
        in: path
        type: string
        required: true
        description: BookingID
        example: "123"
    responses:
      200:
        description: Current recommendation details
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            booking_id:
              type: integer
              example: 123
            client_id:
              type: integer
              example: 4415
            client_type:
              type: string
              example: "CORPORATE"
            current_recommendation:
              type: object
              description: Next available recommendation or null if none left
              properties:
                pair_id:
                  type: string
                  example: "2710_1287"
                vehicle_id:
                  type: string
                  example: "2710"
                chauffeur_id:
                  type: string
                  example: "1287"
                rank:
                  type: integer
                  example: 1
                score:
                  type: number
                  example: 0.95
                gcn_score:
                  type: number
                cf_score:
                  type: number
                user_content_score:
                  type: number
                item_content_score:
                  type: number
            remaining_count:
              type: integer
              description: How many recommendations still available
              example: 20
            rejected_count:
              type: integer
              description: How many have been rejected
              example: 0
            total_recommendations:
              type: integer
              description: Total recommendations generated
              example: 20
      404:
        description: Booking not found
    """
    try:
        components = get_components()
        processor = components['processor']
        
        shown_event = components['store'].get_shown_event(request_id)
        
        if not shown_event:
            return jsonify({
                'success': False,
                'error': f'Booking {request_id} not found'
            }), 404
        
        current_rec = processor._get_next_recommendation(request_id)
        remaining = processor._count_remaining_recommendations(request_id)
        
        rejections = processor._get_rejections_for_booking(request_id)
        rejected_count = len(rejections)
        
        return jsonify({
            'success': True,
            'booking_id': int(request_id),
            'client_id': shown_event['client_id'],
            'client_type': shown_event['client_segment'],
            'current_recommendation': current_rec,
            'remaining_count': remaining,
            'rejected_count': rejected_count,
            'total_recommendations': shown_event['total_shown']
        })
        
    except Exception as e:
        logger.error(f"Error getting current recommendation: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@feedback_bp.route('/booking/<request_id>/rejections', methods=['GET'])
def get_booking_rejections(request_id: str):
    """
    Get all rejections for a booking
    ---
    tags:
      - Feedback - Dispatcher Actions
    summary: Get rejection history for a booking
    description: Returns all pairs that have been rejected for this booking with their reasons and categories.
    parameters:
      - name: request_id
        in: path
        type: string
        required: true
        description: BookingID
        example: "123"
    responses:
      200:
        description: List of rejections
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            booking_id:
              type: integer
              example: 123
            rejections:
              type: array
              items:
                type: object
                properties:
                  pair_id:
                    type: string
                    example: "2710_1287"
                  reason:
                    type: string
                    description: Formatted category weights
                    example: "chauffeur_quality:0.60,past_negative_experience:0.40"
                  category:
                    type: string
                    description: Primary category
                    example: "chauffeur_quality"
                  rejected_at:
                    type: string
                    format: date-time
                    example: "2026-01-09T10:30:00"
            total_rejected:
              type: integer
              example: 1
    """
    try:
        components = get_components()
        processor = components['processor']
        
        rejections = processor._get_rejections_for_booking(request_id)
        
        return jsonify({
            'success': True,
            'booking_id': int(request_id),
            'rejections': rejections,
            'total_rejected': len(rejections)
        })
        
    except Exception as e:
        logger.error(f"Error getting rejections: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# LEGACY ENDPOINT - Backward Compatibility
# =============================================================================

@feedback_bp.route('/selection', methods=['POST'])
def submit_selection():
    """
    Submit dispatcher selection feedback (Legacy - backward compatible)
    ---
    tags:
      - Feedback
    summary: Submit selection with optional rejections array
    description: |
      Legacy endpoint that accepts both selection AND rejections in a single call.
      For new implementations, use /reject and /select separately.
      
      **SP CALLED: [DispatcherActivity] @Status=1** (via process_selection)
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - request_id
            - selected_pair_id
          properties:
            request_id:
              type: string
              example: "req_12345"
              description: Reference to BookingID
            selected_pair_id:
              type: string
              example: "123_456"
              description: The pair that was selected
            rejections:
              type: array
              items:
                type: object
                properties:
                  pair_id:
                    type: string
                  reason_text:
                    type: string
              example: [{"pair_id": "111_222", "reason_text": "Driver was late"}]
            overall_quality:
              type: integer
              minimum: 1
              maximum: 5
              example: 4
    responses:
      200:
        description: Feedback submitted successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            model_credits:
              type: object
            selected_rank:
              type: integer
            learning_applied:
              type: boolean
            allocation_confirmed:
              type: boolean
      400:
        description: Invalid request
      404:
        description: Request ID not found
    """
    try:
        components = get_components()
        processor = components['processor']
        
        feedback = request.get_json()
        
        if not feedback:
            return jsonify({
                'success': False,
                'error': 'Request body is required'
            }), 400
        
        is_valid, error = processor.validate_feedback(feedback)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        result = processor.process_selection(feedback)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 404
            
    except Exception as e:
        logger.error(f"Error in submit_selection: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@feedback_bp.route('/batch', methods=['POST'])
def submit_batch_selections():
    """
    Submit multiple selections in batch
    ---
    tags:
      - Feedback
    summary: Process multiple selections at once
    description: |
      Submit multiple selections in a single request.
      Each selection triggers [DispatcherActivity] SP.
      
      **SP CALLED: [DispatcherActivity]** for each selection
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - selections
          properties:
            selections:
              type: array
              items:
                type: object
                properties:
                  request_id:
                    type: string
                  selected_pair_id:
                    type: string
                  overall_quality:
                    type: integer
              example: [{"request_id": "1", "selected_pair_id": "123_456"}, {"request_id": "2", "selected_pair_id": "789_012"}]
    responses:
      200:
        description: Batch processed
        schema:
          type: object
          properties:
            total:
              type: integer
              example: 10
            successful:
              type: integer
              example: 9
            failed:
              type: integer
              example: 1
            results:
              type: array
    """
    try:
        components = get_components()
        processor = components['processor']
        
        data = request.get_json()
        
        if not data or 'selections' not in data:
            return jsonify({
                'success': False,
                'error': 'selections array is required'
            }), 400
        
        selections = data['selections']
        
        if not isinstance(selections, list):
            return jsonify({
                'success': False,
                'error': 'selections must be an array'
            }), 400
        
        result = processor.process_batch_selections(selections)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in submit_batch_selections: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# STATS AND WEIGHTS ENDPOINTS
# =============================================================================

@feedback_bp.route('/stats/<int:client_id>', methods=['GET'])
def get_client_stats(client_id: int):
    """
    Get learning statistics for a client
    ---
    tags:
      - Feedback - Stats
    summary: Get client learning statistics
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Client ID
        example: 101
    responses:
      200:
        description: Client statistics retrieved successfully
        schema:
          type: object
          properties:
            client_id:
              type: integer
            summary:
              type: object
              properties:
                total_selections:
                  type: integer
                average_selected_rank:
                  type: number
                has_personalized_weights:
                  type: boolean
            learning_status:
              type: object
    """
    try:
        components = get_components()
        weight_selector = components['weight_selector']
        
        stats = weight_selector.get_stats(client_id)
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting client stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@feedback_bp.route('/weights/<int:client_id>', methods=['GET'])
def get_client_weights(client_id: int):
    """
    Get current personalized weights for a client
    ---
    tags:
      - Feedback - Stats
    summary: Get client learned weights
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Client ID
        example: 101
      - name: segment
        in: query
        type: string
        description: Client segment
        default: "UNKNOWN"
        example: "VIP"
    responses:
      200:
        description: Client weights retrieved successfully
        schema:
          type: object
          properties:
            client_id:
              type: integer
            weights:
              type: object
              properties:
                alpha1_rrf:
                  type: number
                alpha2_gcn:
                  type: number
                alpha3_item_sim:
                  type: number
                alpha4_user_sim:
                  type: number
                alpha5_confidence:
                  type: number
            source:
              type: string
              example: "client"
            observations:
              type: integer
            confidence:
              type: number
    """
    try:
        components = get_components()
        weight_selector = components['weight_selector']
        
        segment = request.args.get('segment', 'UNKNOWN')
        
        result = weight_selector.get_weights_with_source(
            client_id=client_id,
            client_segment=segment,
            base_weights=DEFAULT_BASE_WEIGHTS.copy()
        )
        
        result['client_id'] = client_id
        result['segment'] = segment
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting client weights: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@feedback_bp.route('/model-ranking/<int:client_id>', methods=['GET'])
def get_model_ranking(client_id: int):
    """
    Get models ranked by performance for a client
    ---
    tags:
      - Feedback - Stats
    summary: Get model performance ranking
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Client ID
        example: 101
    responses:
      200:
        description: Model ranking
        schema:
          type: object
          properties:
            client_id:
              type: integer
            ranking:
              type: array
              items:
                type: object
                properties:
                  model:
                    type: string
                  avg_credit:
                    type: number
                  observations:
                    type: integer
            best_model:
              type: string
    """
    try:
        components = get_components()
        weight_selector = components['weight_selector']
        
        result = weight_selector.get_model_ranking(client_id)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error getting model ranking: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================

@feedback_bp.route('/system/stats', methods=['GET'])
def get_system_stats():
    """
    Get overall system statistics
    ---
    tags:
      - Feedback System
    summary: Get system-wide statistics
    responses:
      200:
        description: System statistics retrieved successfully
        schema:
          type: object
          properties:
            total_shown_events:
              type: integer
            total_selections:
              type: integer
            unique_clients_with_feedback:
              type: integer
            average_selected_rank:
              type: number
            top_5_selection_rate:
              type: number
            global_model_performance:
              type: object
    """
    try:
        components = get_components()
        weight_selector = components['weight_selector']
        
        stats = weight_selector.get_system_stats()
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@feedback_bp.route('/system/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    ---
    tags:
      - Feedback System
    summary: Check system health
    responses:
      200:
        description: System is healthy
        schema:
          type: object
          properties:
            status:
              type: string
              example: "healthy"
            database:
              type: string
              example: "connected"
            components:
              type: array
              items:
                type: string
      503:
        description: System is unhealthy
    """
    try:
        components = get_components()
        weight_selector = components['weight_selector']
        
        is_healthy = weight_selector.is_healthy()
        
        if is_healthy:
            return jsonify({
                'status': 'healthy',
                'database': 'connected',
                'components': list(components.keys())
            }), 200
        else:
            return jsonify({
                'status': 'unhealthy',
                'database': 'error'
            }), 503
            
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


@feedback_bp.route('/rejection-reasons', methods=['GET'])
def get_rejection_reasons():
    """
    Get list of valid rejection reason codes
    ---
    tags:
      - Feedback
    summary: Get rejection reason codes
    parameters:
      - name: learnable_only
        in: query
        type: boolean
        description: If true, only return learnable reasons
        default: false
    responses:
      200:
        description: List of rejection reasons
        schema:
          type: object
    """
    learnable_only = request.args.get('learnable_only', '').lower() == 'true'
    
    if learnable_only:
        filtered = {
            k: v for k, v in REJECTION_REASONS.items() 
            if v.get('learnable', False)
        }
        return jsonify(filtered), 200
    
    return jsonify(REJECTION_REASONS), 200


@feedback_bp.route('/invalidate-cache', methods=['POST'])
def invalidate_cache():
    """
    Invalidate cached weights
    ---
    tags:
      - Feedback System
    summary: Invalidate weight cache
    parameters:
      - name: body
        in: body
        schema:
          type: object
          properties:
            client_id:
              type: integer
              description: If not provided, invalidates all
    responses:
      200:
        description: Cache invalidated
    """
    try:
        components = get_components()
        weight_selector = components['weight_selector']
        
        data = request.get_json() or {}
        client_id = data.get('client_id')
        
        weight_selector.invalidate_cache(client_id)
        
        return jsonify({
            'success': True,
            'message': f'Cache invalidated for {"client " + str(client_id) if client_id else "all clients"}'
        }), 200
        
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# =============================================================================
# RECOMMENDATION RETRIEVAL ENDPOINTS
# =============================================================================

@feedback_bp.route('/shown-event/<request_id>', methods=['GET'])
def get_shown_event(request_id: str):
    """
    Retrieve shown recommendations by request_id
    ---
    tags:
      - Feedback - Data Retrieval
    summary: Get shown event details
    description: |
      This endpoint retrieves what was shown to the dispatcher from the feedback system.
      It includes model rankings and scores for each recommendation.
    parameters:
      - name: request_id
        in: path
        type: string
        required: true
        description: The request_id (BookingID) to retrieve
        example: "123"
    responses:
      200:
        description: Shown event details with all pairs and rankings
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            request_id:
              type: string
              example: "123"
            client_id:
              type: integer
              example: 5
            client_segment:
              type: string
              example: "Premium Client"
            total_shown:
              type: integer
              example: 20
            weights_used:
              type: object
              example: {"lightgcn": 0.3, "collaborative": 0.3, "user_content": 0.2, "item_content": 0.2}
            pairs_shown:
              type: array
              description: List of all pairs shown with their ranks and scores
            shown_at:
              type: string
              format: date-time
      404:
        description: Request ID not found in feedback system
    """
    try:
        components = get_components()
        store = components['store']
        
        shown_event = store.get_shown_event(request_id)
        
        if not shown_event:
            return jsonify({
                'success': False,
                'error': f'Request ID {request_id} not found in feedback system'
            }), 404
        
        return jsonify({
            'success': True,
            **shown_event
        })
        
    except Exception as e:
        logger.error(f"Error retrieving shown event: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@feedback_bp.route('/recommendations/<request_id>', methods=['GET'])
def get_recommendations_by_request_id(request_id: str):
    """
    Retrieve recommendations from AvailabilityAllocation table by request_id (BookingID)
    ---
    tags:
      - Feedback - Data Retrieval
    summary: Get recommendations by BookingID
    parameters:
      - name: request_id
        in: path
        type: string
        required: true
        description: The BookingID to retrieve
        example: "123"
    responses:
      200:
        description: Full recommendations list from AvailabilityAllocation table
        schema:
          type: object
          properties:
            success:
              type: boolean
            request_id:
              type: string
            total_recommendations:
              type: integer
            recommendations:
              type: array
      404:
        description: No recommendations found for request_id
    """
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    BookingID,
                    ClientId,
                    ClientType,
                    PairId,
                    VehicleId,
                    ChauffeurId,
                    FinalRank,
                    FinalScore,
                    RrfScore,
                    GcnScore,
                    CfScore,
                    UserContentScore,
                    ItemContentScore,
                    WeightsUsed,
                    BatchId,
                    CreatedDate,
                    IsSelected,
                    OverallQuality,
                    SelectedAt
                FROM AvailabilityAllocation
                WHERE BookingID = :request_id
                ORDER BY FinalRank
            """)
            
            result = conn.execute(query, {'request_id': int(request_id)})
            rows = result.fetchall()
            
            if not rows:
                return jsonify({
                    'success': False,
                    'error': f'No recommendations found for request_id {request_id}'
                }), 404
            
            recommendations = []
            for row in rows:
                rec = dict(row._mapping)
                
                if rec.get('WeightsUsed'):
                    try:
                        rec['WeightsUsed'] = json.loads(rec['WeightsUsed'])
                    except (json.JSONDecodeError, TypeError):
                        rec['WeightsUsed'] = {}
                
                if rec.get('CreatedDate'):
                    rec['CreatedDate'] = rec['CreatedDate'].isoformat()
                if rec.get('SelectedAt'):
                    rec['SelectedAt'] = rec['SelectedAt'].isoformat()
                
                recommendations.append(rec)
            
            return jsonify({
                'success': True,
                'request_id': request_id,
                'total_recommendations': len(recommendations),
                'recommendations': recommendations
            })
        
    except Exception as e:
        logger.error(f"Error retrieving recommendations: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        if 'engine' in locals():
            engine.dispose()


@feedback_bp.route('/batch-recommendations/<batch_id>', methods=['GET'])
def get_recommendations_by_batch_id(batch_id: str):
    """
    Retrieve all recommendations for a given batch_id from AvailabilityAllocation
    ---
    tags:
      - Feedback - Data Retrieval
    summary: Get all recommendations in a batch
    parameters:
      - name: batch_id
        in: path
        type: string
        required: true
        description: The batch_id to retrieve
        example: "BATCH-20241204-143022"
    responses:
      200:
        description: All recommendations in the batch
        schema:
          type: object
          properties:
            success:
              type: boolean
            batch_id:
              type: string
            total_requests:
              type: integer
            total_recommendations:
              type: integer
            requests:
              type: array
      404:
        description: Batch ID not found
    """
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    BookingID,
                    ClientId,
                    ClientType,
                    PairId,
                    VehicleId,
                    ChauffeurId,
                    FinalRank,
                    FinalScore,
                    RrfScore,
                    GcnScore,
                    CfScore,
                    UserContentScore,
                    ItemContentScore,
                    WeightsUsed,
                    BatchId,
                    CreatedDate,
                    IsSelected,
                    OverallQuality,
                    SelectedAt
                FROM AvailabilityAllocation
                WHERE BatchId = :batch_id
                ORDER BY BookingID, FinalRank
            """)
            
            result = conn.execute(query, {'batch_id': batch_id})
            rows = result.fetchall()
            
            if not rows:
                return jsonify({
                    'success': False,
                    'error': f'No recommendations found for batch_id {batch_id}'
                }), 404
            
            requests_dict = {}
            for row in rows:
                rec = dict(row._mapping)
                
                if rec.get('WeightsUsed'):
                    try:
                        rec['WeightsUsed'] = json.loads(rec['WeightsUsed'])
                    except:
                        rec['WeightsUsed'] = {}
                
                if rec.get('CreatedDate'):
                    rec['CreatedDate'] = rec['CreatedDate'].isoformat()
                if rec.get('SelectedAt'):
                    rec['SelectedAt'] = rec['SelectedAt'].isoformat()
                
                req_id = rec['BookingID']
                if req_id not in requests_dict:
                    requests_dict[req_id] = []
                requests_dict[req_id].append(rec)
            
            requests = [
                {
                    'request_id': req_id,
                    'recommendation_count': len(recs),
                    'recommendations': recs
                }
                for req_id, recs in requests_dict.items()
            ]
            
            return jsonify({
                'success': True,
                'batch_id': batch_id,
                'total_requests': len(requests),
                'total_recommendations': len(rows),
                'requests': requests
            })
        
    except Exception as e:
        logger.error(f"Error retrieving batch recommendations: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        if 'engine' in locals():
            engine.dispose()


@feedback_bp.route('/recommendations-by-batch/<batch_id>', methods=['GET'])
def get_recommendations_by_batch(batch_id: str):
    """
    Get all recommendations for a batch from SQL Server (Alternative endpoint)
    ---
    tags:
      - Feedback - Data Retrieval
    summary: Get first request recommendations from batch
    parameters:
      - name: batch_id
        in: path
        type: string
        required: true
        description: The batch_id to retrieve
        example: "BATCH-20241204-143022"
    responses:
      200:
        description: Recommendations found
      404:
        description: Batch ID not found
    """
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        
        with engine.connect() as conn:
            query = text("""
                SELECT TOP 20
                    BookingID as request_id,
                    ClientId as client_id,
                    PairId as pair_id,
                    VehicleId as vehicle_id,
                    ChauffeurId as chauffeur_id,
                    FinalRank as rank_shown,
                    FinalScore as final_score,
                    RrfScore as rrf_score,
                    GcnScore as lightgcn_score,
                    CfScore as collaborative_score,
                    UserContentScore as user_content_score,
                    ItemContentScore as item_content_score
                FROM AvailabilityAllocation
                WHERE BatchId = :batch_id
                ORDER BY BookingID, FinalRank
            """)
            
            result = conn.execute(query, {'batch_id': batch_id})
            rows = result.fetchall()
            
            if not rows:
                return jsonify({
                    'error': f'No recommendations found for batch {batch_id}'
                }), 404
            
            requests = {}
            for row in rows:
                row_dict = dict(row._mapping)
                request_id = row_dict['request_id']
                
                if request_id not in requests:
                    requests[request_id] = {
                        'request_id': str(request_id),
                        'client_id': row_dict['client_id'],
                        'pairs_shown': []
                    }
                
                requests[request_id]['pairs_shown'].append(row_dict)
            
            first_request = list(requests.values())[0]
            
            return jsonify(first_request), 200
            
    except Exception as e:
        logger.error(f"Error getting batch recommendations: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        if 'engine' in locals():
            engine.dispose()


@feedback_bp.route('/recent-feedback', methods=['GET'])
def get_recent_feedback():
    """
    Get recent feedback/selections from AvailabilityAllocation
    ---
    tags:
      - Feedback - Data Retrieval
    summary: Get recent selections
    parameters:
      - name: limit
        in: query
        type: integer
        default: 20
        description: Number of records to return
    responses:
      200:
        description: Recent feedback list
        schema:
          type: array
          items:
            type: object
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        
        engine = create_engine(DB_CONNECTION_STRING)
        
        with engine.connect() as conn:
            query = text(f"""
                SELECT TOP {limit}
                    CAST(BookingID AS VARCHAR(50)) as request_id,
                    PairId as selected_pair_id,
                    FinalRank as selected_rank,
                    OverallQuality as overall_quality,
                    ClientId as client_id,
                    ClientType as client_segment,
                    SelectedAt as selected_at
                FROM AvailabilityAllocation
                WHERE IsSelected = 1
                ORDER BY SelectedAt DESC
            """)
            
            result = conn.execute(query)
            rows = result.fetchall()
            
            feedback_list = []
            for row in rows:
                item = dict(row._mapping)
                if item.get('selected_at'):
                    item['selected_at'] = item['selected_at'].isoformat()
                feedback_list.append(item)
            
            return jsonify(feedback_list), 200
            
    except Exception as e:
        logger.error(f"Error getting recent feedback: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        if 'engine' in locals():
            engine.dispose()


@feedback_bp.route('/client-feedback/<int:client_id>', methods=['GET'])
def get_client_feedback(client_id: int):
    """
    Get all feedback for a specific client from AvailabilityAllocation
    ---
    tags:
      - Feedback - Data Retrieval
    summary: Get client feedback history
    parameters:
      - name: client_id
        in: path
        type: integer
        required: true
        description: Client ID
        example: 4415
    responses:
      200:
        description: Client feedback history
        schema:
          type: object
          properties:
            client_id:
              type: integer
            total_selections:
              type: integer
            feedback:
              type: array
    """
    try:
        engine = create_engine(DB_CONNECTION_STRING)
        
        with engine.connect() as conn:
            query = text("""
                SELECT 
                    CAST(BookingID AS VARCHAR(50)) as request_id,
                    PairId as selected_pair_id,
                    FinalRank as selected_rank,
                    OverallQuality as overall_quality,
                    ClientType as client_segment,
                    LightGcnCredit as lightgcn_credit,
                    CollaborativeCredit as collaborative_credit,
                    UserContentCredit as user_content_credit,
                    ItemContentCredit as item_content_credit,
                    SelectedAt as selected_at
                FROM AvailabilityAllocation
                WHERE ClientId = :client_id
                  AND IsSelected = 1
                ORDER BY SelectedAt DESC
            """)
            
            result = conn.execute(query, {'client_id': client_id})
            rows = result.fetchall()
            
            feedback_list = []
            for row in rows:
                item = dict(row._mapping)
                if item.get('selected_at'):
                    item['selected_at'] = item['selected_at'].isoformat()
                feedback_list.append(item)
            
            return jsonify({
                'client_id': client_id,
                'total_selections': len(feedback_list),
                'feedback': feedback_list
            }), 200
            
    except Exception as e:
        logger.error(f"Error getting client feedback: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        if 'engine' in locals():
            engine.dispose()





# =============================================================================
# NEW ENDPOINT - Get Pending Bookings with Top Recommendation
# =============================================================================
# Add this to your api.py file




@feedback_bp.route('/pending-bookings', methods=['GET'])
def get_pending_bookings():
    """
    Get all pending bookings with their top recommendation and full dashboard details
    ---
    tags:
      - Feedback - Dispatcher Actions
    summary: Get all pending bookings for dispatcher review with complete details
    description: |
      Returns all bookings that haven't been allocated yet,
      with only the top-ranked recommendation that is still pending (IsSelected = 0).
      Includes complete booking details from TransElite.Q3 database.
      
      DATE FILTERING:
      Window is anchored to IST (dispatcher timezone), converted to UTC,
      and filtered on PickUpDateTimeUTC. PickUpDateTime (local) is returned
      for display only.
      
      CRITICAL LOGIC:
      If a BookingID has ANY allocation where IsSelected = 1 (Accepted), 
      that booking is completely excluded from this list.
    parameters:
      - name: days_ahead
        in: query
        type: integer
        default: 1
        description: Number of days ahead to include (0=today only, 1=today+tomorrow, etc.)
      - name: client_id
        in: query
        type: integer
        description: Filter by specific client ID (optional)
    responses:
      200:
        description: List of pending bookings with top recommendations and full details
    """
    try:
        days_ahead = request.args.get('days_ahead', 1, type=int)
        client_id_filter = request.args.get('client_id', type=int)

        if days_ahead < 0:
            days_ahead = 0

       

        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)

        # Start of today in IST → UTC
        start_of_today_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        TIMEZONE_BUFFER_HOURS = 12
        window_start_utc = start_of_today_ist.astimezone(pytz.utc) - timedelta(hours=TIMEZONE_BUFFER_HOURS)
        

        # End boundary = start of (today + days_ahead + 1) in IST → UTC
        # e.g. days_ahead=1 means include today and tomorrow fully
        window_end_ist = start_of_today_ist + timedelta(days=days_ahead + 1)
        window_end_utc = window_end_ist.astimezone(pytz.utc)

        logger.info(
            f"get_pending_bookings — "
            f"IST window: {start_of_today_ist.date()} to {window_end_ist.date()} | "
            f"UTC window: {window_start_utc} to {window_end_utc}"
        )

        engine = create_engine(DB_CONNECTION_STRING)

        with engine.connect() as conn:
            query = text(f"""
                WITH UnfulfilledBookings AS (
                    SELECT DISTINCT aa.BookingID
                    FROM AvailabilityAllocation aa
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM AvailabilityAllocation accepted
                        WHERE accepted.BookingID = aa.BookingID
                        AND accepted.IsSelected = 1
                    )
                ),
                PendingRecommendations AS (
                    SELECT
                        aa.BookingID,
                        aa.ClientId,
                        aa.ClientType,
                        aa.PairId,
                        aa.VehicleId,
                        aa.ChauffeurId,
                        aa.FinalRank,
                        aa.FinalScore,
                        aa.RrfScore,
                        aa.GcnScore,
                        aa.CfScore,
                        aa.UserContentScore,
                        aa.ItemContentScore,
                        aa.BatchId,
                        aa.CreatedDate,
                        aa.IsSelected,
                        aa.PickUpDateTime,
                        aa.PickUpDateTimeUTC,
                        ROW_NUMBER() OVER (PARTITION BY aa.BookingID ORDER BY aa.FinalRank) as RowNum
                    FROM AvailabilityAllocation aa
                    JOIN UnfulfilledBookings ub ON aa.BookingID = ub.BookingID
                    WHERE aa.IsSelected = 0         
                      AND aa.PickUpDateTimeUTC >= :window_start_utc
                      AND aa.PickUpDateTimeUTC <  :window_end_utc
                      ),
                BookingCounts AS (
                    SELECT
                        aa.BookingID,
                        COUNT(*) as TotalRecommendations
                    FROM AvailabilityAllocation aa
                    JOIN UnfulfilledBookings ub ON aa.BookingID = ub.BookingID
                    WHERE aa.IsSelected = 0
                    GROUP BY aa.BookingID
                )
                SELECT
                    r.BookingID,
                    r.ClientId,
                    r.ClientType,
                    r.PairId,
                    r.VehicleId,
                    r.ChauffeurId,
                    r.FinalRank,
                    r.FinalScore,
                    r.RrfScore,
                    r.GcnScore,
                    r.CfScore,
                    r.UserContentScore,
                    r.ItemContentScore,
                    r.BatchId,
                    r.CreatedDate,
                    bc.TotalRecommendations,
                    b.BookingNumber,
                    r.PickUpDateTime,
                    r.PickUpDateTimeUTC,
                    CONVERT(CHAR(8), r.PickUpDateTime, 108) AS PickupTime,
                    CAST(r.PickUpDateTime AS DATE) AS PickupDate,
                    p.Description AS Port,
                    bp_pickup.AddressLine AS PickupAddress,
                    bp_dropoff.AddressLine AS DropOffAddress,
                    c.AccountNo AS ClientAccountNumber,
                    c.Description AS ClientName,
                    vt_requested.Description AS RequestedVehicleType,
                    vt_assigned.Description AS AssignedVehicleType,
                    v.Code AS AssignedVehicleCode,
                    v.Make AS AssignedVehicleMake,
                    v.Model AS AssignedVehicleModel,
                    CONCAT(v.Code, ' - ', v.Make, ' ', v.Model) AS AssignedVehicle,
                    CONCAT(ch.Code, ' - ', ch.FirstName, ' ', ch.LastName) AS AssignedChauffeur,
                    ch.Code AS ChauffeurCode,
                    ch.FirstName AS ChauffeurFirstName,
                    ch.LastName AS ChauffeurLastName
                FROM PendingRecommendations r
                JOIN BookingCounts bc ON r.BookingID = bc.BookingID
                INNER JOIN [{TRANSELITE_DB}]..Booking b ON r.BookingID = b.ID
                LEFT JOIN [{TRANSELITE_DB}]..[Port] p ON b.RefPort = p.ID
                LEFT JOIN [{TRANSELITE_DB}]..BookingPoint bp_pickup
                    ON r.BookingID = bp_pickup.RefBooking
                    AND bp_pickup.RefBookingPointType = 1
                LEFT JOIN [{TRANSELITE_DB}]..BookingPoint bp_dropoff
                    ON r.BookingID = bp_dropoff.RefBooking
                    AND bp_dropoff.RefBookingPointType = 2
                LEFT JOIN [{TRANSELITE_DB}]..Client c ON c.ID = b.RefClient
                LEFT JOIN [{TRANSELITE_DB}]..VehicleType vt_requested ON b.RefVehicleTypeRequest = vt_requested.ID
                LEFT JOIN [{TRANSELITE_DB}]..Vehicle v ON v.ID = r.VehicleId
                LEFT JOIN [{TRANSELITE_DB}]..VehicleType vt_assigned ON vt_assigned.ID = v.RefVehicleType
                LEFT JOIN [{TRANSELITE_DB}]..Chauffeur ch ON ch.ID = r.ChauffeurId
                WHERE r.RowNum = 1
                """ + (" AND r.ClientId = :client_id" if client_id_filter else "") + """
                ORDER BY r.PickUpDateTimeUTC ASC
            """)

            params = {
                'window_start_utc': window_start_utc,
                'window_end_utc':   window_end_utc
            }

            if client_id_filter:
                params['client_id'] = client_id_filter

            result = conn.execute(query, params)
            rows = result.fetchall()

            bookings = []
            for row in rows:
                row_dict = dict(row._mapping)

                booking = {
                    'booking_id': row_dict['BookingID'],
                    'booking_number': row_dict['BookingNumber'],
                    'client_id': row_dict['ClientId'],
                    'client_type': row_dict['ClientType'] or 'N/A',
                    'batch_id': row_dict['BatchId'],
                    'created_date': row_dict['CreatedDate'].isoformat() if row_dict['CreatedDate'] else None,
                    'total_recommendations': row_dict['TotalRecommendations'],

                    # PickUpDateTime = local port time, for display only
                    # PickUpDateTimeUTC = UTC, used for filtering (not shown to dispatcher)
                    'appointment_datetime': row_dict['PickUpDateTime'].isoformat() if row_dict['PickUpDateTime'] else None,
                    'pickup_time': row_dict['PickupTime'],
                    'pickup_date': row_dict['PickupDate'].isoformat() if row_dict['PickupDate'] else None,
                    'port': row_dict['Port'],
                    'pickup_address': row_dict['PickupAddress'],
                    'dropoff_address': row_dict['DropOffAddress'],

                    'client_account_number': row_dict['ClientAccountNumber'],
                    'client_name': row_dict['ClientName'],

                    'requested_vehicle_type': row_dict['RequestedVehicleType'],
                    'assigned_vehicle_type': row_dict['AssignedVehicleType'],
                    'assigned_vehicle_code': row_dict['AssignedVehicleCode'],
                    'assigned_vehicle_make': row_dict['AssignedVehicleMake'],
                    'assigned_vehicle_model': row_dict['AssignedVehicleModel'],
                    'assigned_vehicle': row_dict['AssignedVehicle'],

                    'assigned_chauffeur': row_dict['AssignedChauffeur'],
                    'chauffeur_code': row_dict['ChauffeurCode'],
                    'chauffeur_first_name': row_dict['ChauffeurFirstName'],
                    'chauffeur_last_name': row_dict['ChauffeurLastName'],

                    'top_recommendation': {
                        'pair_id': row_dict['PairId'],
                        'vehicle_id': str(row_dict['VehicleId']),
                        'chauffeur_id': str(row_dict['ChauffeurId']),
                        'rank': row_dict['FinalRank'],
                        'final_score': float(row_dict['FinalScore']) if row_dict['FinalScore'] else None,
                        'rrf_score': float(row_dict['RrfScore']) if row_dict['RrfScore'] else None,
                        'gcn_score': float(row_dict['GcnScore']) if row_dict['GcnScore'] else None,
                        'cf_score': float(row_dict['CfScore']) if row_dict['CfScore'] else None,
                        'user_content_score': float(row_dict['UserContentScore']) if row_dict['UserContentScore'] else None,
                        'item_content_score': float(row_dict['ItemContentScore']) if row_dict['ItemContentScore'] else None
                    }
                }
                bookings.append(booking)

            
            return jsonify({
                'success': True,
                'total_pending': len(bookings),
                'date_range': {
                    'from_date': start_of_today_ist.date().isoformat(),
                    'to_date': window_end_ist.date().isoformat(),
                    'days_ahead': days_ahead,
                    'filter_basis': 'IST (Asia/Kolkata)',
                    'window_start_utc': window_start_utc.isoformat(),
                    'window_end_utc': window_end_utc.isoformat()
                },
                'bookings': bookings
            })

    except Exception as e:
        logger.error(f"Error getting pending bookings: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        if 'engine' in locals():
            engine.dispose()

@feedback_bp.route('/pending-bookingsold', methods=['GET'])
def get_pending_bookingsoldmain():
    """
    Get all pending bookings with their top recommendation and full dashboard details
    ---
    tags:
      - Feedback - Dispatcher Actions
    summary: Get all pending bookings for dispatcher review with complete details
    description: |
      Returns all bookings that haven't been allocated yet,
      with only the top-ranked recommendation that is still pending (IsSelected = NULL or 0).
      Includes complete booking details from TransElite.Q3 database.
      
      DATE FILTERING:
      Only returns bookings with PickUpDateTime from today to today + days_ahead.
      Default days_ahead=1 means today and tomorrow.
      
      CRITICAL LOGIC:
      If a BookingID has ANY allocation where IsSelected = 1 (Accepted), 
      that booking is completely excluded from this list (it is considered fulfilled).
    parameters:
      - name: days_ahead
        in: query
        type: integer
        default: 1
        description: Number of days ahead to include (0=today only, 1=today+tomorrow, etc.)
      - name: client_id
        in: query
        type: integer
        description: Filter by specific client ID (optional)
    responses:
      200:
        description: List of pending bookings with top recommendations and full details
        schema:
          type: object
          properties:
            success:
              type: boolean
            total_pending:
              type: integer
            date_range:
              type: object
              properties:
                from_date:
                  type: string
                to_date:
                  type: string
            bookings:
              type: array
    """
    try:
        days_ahead = request.args.get('days_ahead', 1, type=int)
        client_id_filter = request.args.get('client_id', type=int)
        
        # Ensure days_ahead is not negative
        if days_ahead < 0:
            days_ahead = 0
        
        engine = create_engine(DB_CONNECTION_STRING)
        
        with engine.connect() as conn:
            query = text(f"""
                WITH UnfulfilledBookings AS (
                    -- First, identify bookings that do NOT have an accepted allocation
                    SELECT DISTINCT aa.BookingID
                    FROM AvailabilityAllocation aa
                    WHERE NOT EXISTS (
                        SELECT 1 
                        FROM AvailabilityAllocation accepted 
                        WHERE accepted.BookingID = aa.BookingID 
                        AND accepted.IsSelected = 1
                    )
                ),
                PendingRecommendations AS (
                    SELECT 
                        aa.BookingID,
                        aa.ClientId,
                        aa.ClientType,
                        aa.PairId,
                        aa.VehicleId,
                        aa.ChauffeurId,
                        aa.FinalRank,
                        aa.FinalScore,
                        aa.RrfScore,
                        aa.GcnScore,
                        aa.CfScore,
                        aa.UserContentScore,
                        aa.ItemContentScore,
                        aa.BatchId,
                        aa.CreatedDate,
                        aa.IsSelected,
                        aa.PickUpDateTime,
                        ROW_NUMBER() OVER (PARTITION BY aa.BookingID ORDER BY aa.FinalRank) as RowNum
                    FROM AvailabilityAllocation aa
                    JOIN UnfulfilledBookings ub ON aa.BookingID = ub.BookingID
                    WHERE aa.IsSelected = 0
                      AND CAST(aa.PickUpDateTime AS DATE) >= CAST(GETDATE() AS DATE)
                      AND CAST(aa.PickUpDateTime AS DATE) <= CAST(DATEADD(DAY, :days_ahead, GETDATE()) AS DATE)
                ),
                BookingCounts AS (
                    SELECT 
                        aa.BookingID,
                        COUNT(*) as TotalRecommendations
                    FROM AvailabilityAllocation aa
                    JOIN UnfulfilledBookings ub ON aa.BookingID = ub.BookingID
                    WHERE aa.IsSelected = 0
                    GROUP BY aa.BookingID
                )
                SELECT
                    r.BookingID,
                    r.ClientId,
                    r.ClientType,
                    r.PairId,
                    r.VehicleId,
                    r.ChauffeurId,
                    r.FinalRank,
                    r.FinalScore,
                    r.RrfScore,
                    r.GcnScore,
                    r.CfScore,
                    r.UserContentScore,
                    r.ItemContentScore,
                    r.BatchId,
                    r.CreatedDate,
                    bc.TotalRecommendations,
                    -- Booking details from TransElite.Q3
                    b.BookingNumber,
                    r.PickUpDateTime,
                    CONVERT(CHAR(8), r.PickUpDateTime, 108) AS PickupTime,
                    CAST(r.PickUpDateTime AS DATE) AS PickupDate,
                    -- Port details
                    p.Description AS Port,
                    -- Pickup address (RefBookingPointType = 1)
                    bp_pickup.AddressLine AS PickupAddress,
                    -- Drop-off address (RefBookingPointType = 2)
                    bp_dropoff.AddressLine AS DropOffAddress,
                    -- Client details
                    c.AccountNo AS ClientAccountNumber,
                    c.Description AS ClientName,
                    -- Requested vehicle type
                    vt_requested.Description AS RequestedVehicleType,
                    -- Assigned vehicle type (through Vehicle.RefVehicleType)
                    vt_assigned.Description AS AssignedVehicleType,
                    -- Assigned vehicle details (Code + Make + Model)
                    v.Code AS AssignedVehicleCode,
                    v.Make AS AssignedVehicleMake,
                    v.Model AS AssignedVehicleModel,
                    CONCAT(v.Code, ' - ', v.Make, ' ', v.Model) AS AssignedVehicle,
                    -- Assigned chauffeur (using ChauffeurId)
                    CONCAT(ch.Code, ' - ', ch.FirstName, ' ', ch.LastName) AS AssignedChauffeur,
                    ch.Code AS ChauffeurCode,
                    ch.FirstName AS ChauffeurFirstName,
                    ch.LastName AS ChauffeurLastName
                FROM PendingRecommendations r
                JOIN BookingCounts bc ON r.BookingID = bc.BookingID
                -- Join with TransElite.Q3 tables
                INNER JOIN [{TRANSELITE_DB}]..Booking b ON r.BookingID = b.ID
                LEFT JOIN [{TRANSELITE_DB}]..[Port] p ON b.RefPort = p.ID
                LEFT JOIN [{TRANSELITE_DB}]..BookingPoint bp_pickup 
                    ON r.BookingID = bp_pickup.RefBooking 
                    AND bp_pickup.RefBookingPointType = 1
                LEFT JOIN [{TRANSELITE_DB}]..BookingPoint bp_dropoff 
                    ON r.BookingID = bp_dropoff.RefBooking 
                    AND bp_dropoff.RefBookingPointType = 2
                LEFT JOIN [{TRANSELITE_DB}]..Client c ON c.ID = b.RefClient
                LEFT JOIN [{TRANSELITE_DB}]..VehicleType vt_requested ON b.RefVehicleTypeRequest = vt_requested.ID
                -- Vehicle join
                LEFT JOIN [{TRANSELITE_DB}]..Vehicle v ON v.ID = r.VehicleId
                -- VehicleType through Vehicle.RefVehicleType
                LEFT JOIN [{TRANSELITE_DB}]..VehicleType vt_assigned ON vt_assigned.ID = v.RefVehicleType
                -- Chauffeur join using ChauffeurId
                LEFT JOIN [{TRANSELITE_DB}]..Chauffeur ch ON ch.ID = r.ChauffeurId
                WHERE r.RowNum = 1
                """ + (" AND r.ClientId = :client_id" if client_id_filter else "") + """
                ORDER BY r.PickUpDateTime ASC
            """)
            
            params = {'days_ahead': days_ahead}
            if client_id_filter:
                params['client_id'] = client_id_filter
            
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            bookings = []
            for row in rows:
                row_dict = dict(row._mapping)
                
                booking = {
                    'booking_id': row_dict['BookingID'],
                    'booking_number': row_dict['BookingNumber'],
                    'client_id': row_dict['ClientId'],
                    'client_type': row_dict['ClientType'] or 'N/A',
                    'batch_id': row_dict['BatchId'],
                    'created_date': row_dict['CreatedDate'].isoformat() if row_dict['CreatedDate'] else None,
                    'total_recommendations': row_dict['TotalRecommendations'],
                    
                    # Booking details
                    'appointment_datetime': row_dict['PickUpDateTime'].isoformat() if row_dict['PickUpDateTime'] else None,
                    'pickup_time': row_dict['PickupTime'],
                    'pickup_date': row_dict['PickupDate'].isoformat() if row_dict['PickupDate'] else None,
                    'port': row_dict['Port'],
                    'pickup_address': row_dict['PickupAddress'],
                    'dropoff_address': row_dict['DropOffAddress'],
                    
                    # Client details
                    'client_account_number': row_dict['ClientAccountNumber'],
                    'client_name': row_dict['ClientName'],
                    
                    # Vehicle details
                    'requested_vehicle_type': row_dict['RequestedVehicleType'],
                    'assigned_vehicle_type': row_dict['AssignedVehicleType'],
                    'assigned_vehicle_code': row_dict['AssignedVehicleCode'],
                    'assigned_vehicle_make': row_dict['AssignedVehicleMake'],
                    'assigned_vehicle_model': row_dict['AssignedVehicleModel'],
                    'assigned_vehicle': row_dict['AssignedVehicle'],
                    
                    # Chauffeur details
                    'assigned_chauffeur': row_dict['AssignedChauffeur'],
                    'chauffeur_code': row_dict['ChauffeurCode'],
                    'chauffeur_first_name': row_dict['ChauffeurFirstName'],
                    'chauffeur_last_name': row_dict['ChauffeurLastName'],
                    
                    # Recommendation details
                    'top_recommendation': {
                        'pair_id': row_dict['PairId'],
                        'vehicle_id': str(row_dict['VehicleId']),
                        'chauffeur_id': str(row_dict['ChauffeurId']),
                        'rank': row_dict['FinalRank'],
                        'final_score': float(row_dict['FinalScore']) if row_dict['FinalScore'] else None,
                        'rrf_score': float(row_dict['RrfScore']) if row_dict['RrfScore'] else None,
                        'gcn_score': float(row_dict['GcnScore']) if row_dict['GcnScore'] else None,
                        'cf_score': float(row_dict['CfScore']) if row_dict['CfScore'] else None,
                        'user_content_score': float(row_dict['UserContentScore']) if row_dict['UserContentScore'] else None,
                        'item_content_score': float(row_dict['ItemContentScore']) if row_dict['ItemContentScore'] else None
                    }
                }
                bookings.append(booking)
            
            # Calculate date range for response
            from datetime import datetime, timedelta
            today = datetime.now().date()
            end_date = today + timedelta(days=days_ahead)
            
            return jsonify({
                'success': True,
                'total_pending': len(bookings),
                'date_range': {
                    'from_date': today.isoformat(),
                    'to_date': end_date.isoformat(),
                    'days_ahead': days_ahead
                },
                'bookings': bookings
            })
        
    except Exception as e:
        logger.error(f"Error getting pending bookings: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        if 'engine' in locals():
            engine.dispose()


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@feedback_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request',
        'message': str(error)
    }), 400


@feedback_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not found',
        'message': str(error)
    }), 404


@feedback_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500
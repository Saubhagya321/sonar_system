"""
Resilience Module for Recommendation System
Circuit breaker pattern and retry logic for production reliability
"""

import time
import random
import logging
from functools import wraps
from enum import Enum
from typing import Tuple, Type, Callable, Optional
from threading import Lock

logger = logging.getLogger(__name__)


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    def __init__(self, message: str = "Circuit breaker is OPEN", last_failure: Optional[Exception] = None):
        super().__init__(message)
        self.last_failure = last_failure


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by failing fast when a service is down.
    
    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Service is failing, requests are rejected immediately
        HALF_OPEN: Testing if service has recovered
    
    Usage:
        db_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        @db_circuit
        def query_database(query):
            return execute_query(query)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
        name: str = "default"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exceptions: Exception types that trigger the breaker
            name: Identifier for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.name = name
        
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._last_failure_time: Optional[float] = None
        self._last_exception: Optional[Exception] = None
        self._lock = Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def failures(self) -> int:
        """Get current failure count"""
        return self._failures
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute(func, *args, **kwargs)
        return wrapper
    
    def _execute(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            # Check if circuit should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(
                        "Circuit '%s' transitioning to HALF_OPEN after %ds",
                        self.name, self.recovery_timeout
                    )
                    self._state = CircuitState.HALF_OPEN
                else:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' is OPEN. "
                        f"Retry after {self._time_until_retry():.1f}s",
                        self._last_exception
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def _time_until_retry(self) -> float:
        """Calculate seconds until circuit can be retried"""
        if self._last_failure_time is None:
            return 0
        elapsed = time.time() - self._last_failure_time
        return max(0, self.recovery_timeout - elapsed)
    
    def _on_success(self):
        """Handle successful execution"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("Circuit '%s' recovered, transitioning to CLOSED", self.name)
            self._failures = 0
            self._state = CircuitState.CLOSED
            self._last_exception = None
    
    def _on_failure(self, exception: Exception):
        """Handle failed execution"""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            self._last_exception = exception
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test, go back to OPEN
                logger.warning(
                    "Circuit '%s' failed during HALF_OPEN, returning to OPEN: %s",
                    self.name, exception
                )
                self._state = CircuitState.OPEN
            elif self._failures >= self.failure_threshold:
                logger.error(
                    "Circuit '%s' opened after %d failures: %s",
                    self.name, self._failures, exception
                )
                self._state = CircuitState.OPEN
            else:
                logger.warning(
                    "Circuit '%s' failure %d/%d: %s",
                    self.name, self._failures, self.failure_threshold, exception
                )
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED
            self._last_failure_time = None
            self._last_exception = None
            logger.info("Circuit '%s' manually reset", self.name)
    
    def get_status(self) -> dict:
        """Get circuit breaker status for health checks"""
        return {
            "name": self.name,
            "state": self._state.value,
            "failures": self._failures,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_until_retry": self._time_until_retry() if self._state == CircuitState.OPEN else 0
        }


# =============================================================================
# RETRY WITH EXPONENTIAL BACKOFF
# =============================================================================

class MaxRetriesExceededError(Exception):
    """Raised when maximum retries are exceeded"""
    def __init__(self, message: str, last_exception: Exception, attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (ConnectionError, TimeoutError),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
):
    """
    Decorator for retry with exponential backoff.
    
    Automatically retries failed operations with increasing delays.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential calculation
        jitter: Add randomness to prevent thundering herd
        retryable_exceptions: Exception types that trigger retry
        on_retry: Optional callback(attempt, exception, delay) on each retry
    
    Usage:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def fetch_data():
            return requests.get(url, timeout=10)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # No more retries
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.2fs...",
                        func.__name__, attempt + 1, max_retries + 1, e, delay
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    
                    time.sleep(delay)
            
            # All retries exhausted
            raise MaxRetriesExceededError(
                f"{func.__name__} failed after {max_retries + 1} attempts",
                last_exception,
                max_retries + 1
            )
        
        return wrapper
    return decorator


# =============================================================================
# TIMEOUT DECORATOR
# =============================================================================

class TimeoutError(Exception):
    """Raised when operation times out"""
    pass


def with_timeout(seconds: float):
    """
    Decorator to add timeout to a function (uses threading).
    
    Note: This is a simple implementation. For I/O operations,
    prefer using the timeout parameter of the underlying library.
    
    Args:
        seconds: Maximum execution time in seconds
    
    Usage:
        @with_timeout(30)
        def slow_operation():
            ...
    """
    import threading
    import queue
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def target():
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(result)
                except Exception as e:
                    exception_queue.put(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)
            
            if thread.is_alive():
                raise TimeoutError(
                    f"{func.__name__} timed out after {seconds}s"
                )
            
            if not exception_queue.empty():
                raise exception_queue.get()
            
            return result_queue.get()
        
        return wrapper
    return decorator


# =============================================================================
# COMBINED RESILIENCE DECORATOR
# =============================================================================

def resilient(
    circuit_name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    max_retries: int = 2,
    base_delay: float = 0.5,
    retryable_exceptions: Tuple[Type[Exception], ...] = (ConnectionError, TimeoutError)
):
    """
    Combined resilience decorator with circuit breaker and retry.
    
    Applies retry first, then circuit breaker wraps the retrying function.
    
    Usage:
        @resilient(circuit_name="database", failure_threshold=3)
        def query_db(sql):
            return execute(sql)
    """
    # Create circuit breaker for this name
    circuit = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exceptions=retryable_exceptions + (MaxRetriesExceededError,),
        name=circuit_name
    )
    
    def decorator(func: Callable) -> Callable:
        # Apply retry first (inner decorator)
        retrying_func = retry_with_backoff(
            max_retries=max_retries,
            base_delay=base_delay,
            retryable_exceptions=retryable_exceptions
        )(func)
        
        # Then wrap with circuit breaker (outer decorator)
        @wraps(func)
        @circuit
        def wrapper(*args, **kwargs):
            return retrying_func(*args, **kwargs)
        
        # Attach circuit for status checks
        wrapper.circuit = circuit
        return wrapper
    
    return decorator


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================

class CircuitBreakerRegistry:
    """
    Registry to track all circuit breakers for health monitoring.
    
    Usage:
        registry = CircuitBreakerRegistry()
        
        db_circuit = registry.get_or_create("database", failure_threshold=3)
        
        @db_circuit
        def query():
            ...
        
        # Get all statuses for health endpoint
        statuses = registry.get_all_statuses()
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._circuits = {}
        return cls._instance
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> CircuitBreaker:
        """Get existing circuit or create new one"""
        if name not in self._circuits:
            self._circuits[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exceptions=expected_exceptions,
                name=name
            )
        return self._circuits[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit by name"""
        return self._circuits.get(name)
    
    def get_all_statuses(self) -> dict:
        """Get status of all circuits for health checks"""
        return {
            name: circuit.get_status()
            for name, circuit in self._circuits.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for circuit in self._circuits.values():
            circuit.reset()


# Global registry instance
circuit_registry = CircuitBreakerRegistry()


# =============================================================================
# PRE-CONFIGURED CIRCUIT BREAKERS
# =============================================================================

# Database operations circuit breaker
db_circuit = circuit_registry.get_or_create(
    name="database",
    failure_threshold=3,
    recovery_timeout=60,
    expected_exceptions=(ConnectionError, TimeoutError, Exception)
)

# Model inference circuit breaker
model_circuit = circuit_registry.get_or_create(
    name="model_inference",
    failure_threshold=5,
    recovery_timeout=30,
    expected_exceptions=(RuntimeError, MemoryError)
)

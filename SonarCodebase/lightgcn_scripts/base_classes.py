"""
Base classes for the LightGCN Recommendation System
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import pandas as pd


class BaseDataLoader(ABC):
    """Abstract base class for data loading operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def load(self) -> Any:
        """Load data from source"""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate loaded data"""
        pass


class BaseProcessor(ABC):
    """Abstract base class for data processing operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data"""
        pass


class BaseModel(ABC):
    """Abstract base class for ML models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_trained = False
    
    @abstractmethod
    def train(self, data: Any) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass


class BasePipeline(ABC):
    """Abstract base class for pipeline orchestration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute the pipeline"""
        pass
    
    def _log_start(self, step_name: str) -> None:
        """Log pipeline step start"""
        self.logger.info("=" * 60)
        self.logger.info(step_name)
        self.logger.info("=" * 60)
    
    def _log_complete(self, step_name: str) -> None:
        """Log pipeline step completion"""
        self.logger.info("%s completed!", step_name)


class DatabaseConnector:
    """Database connection manager using SQLAlchemy with connection pooling"""
    
    def __init__(self, connection_string: str, pool_size: int = 5, max_overflow: int = 10):
        from sqlalchemy import create_engine
        self.connection_string = connection_string
        self.engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600    # Recycle connections after 1 hour
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        from sqlalchemy import text
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()
                columns = result.keys()
                return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            self.logger.error("Database query failed: %s", e)
            raise
    
    def dispose(self) -> None:
        """Dispose of database connection pool"""
        self.engine.dispose()
        self.logger.debug("Database connection pool disposed")


class BaseRecommender(ABC):
    """Abstract base class for recommendation generators"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def recommend(self, *args, **kwargs) -> Any:
        """Generate recommendations"""
        pass
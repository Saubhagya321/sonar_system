"""
Structured JSON Logging Configuration
"""

import json
import logging
import traceback
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class HybridFormatter(logging.Formatter):
    """JSON in production, human-readable in development."""
    
    def __init__(self, environment: str = 'development'):
        super().__init__()
        self.environment = environment
        self.json_formatter = JSONFormatter()
        self.text_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record: logging.LogRecord) -> str:
        if self.environment == 'production':
            return self.json_formatter.format(record)
        return self.text_formatter.format(record)


def setup_logging(environment: str = 'development', level: int = logging.INFO):
    """Configure application logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(HybridFormatter(environment))
    
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)
    
    logging.getLogger('werkzeug').setLevel(logging.WARNING)


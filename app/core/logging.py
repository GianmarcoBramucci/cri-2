"""Logging configuration for the CroceRossa Qdrant Cloud application."""

import sys
import time
import structlog
import logging
from typing import Any, Dict, Optional

from app.core.config import settings


def configure_logging() -> None:
    """Configure structlog for application logging."""
    
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info, # Prefer this for development/more info
        # structlog.processors.format_exc_info, # Use this for more compact prod logs
        structlog.processors.TimeStamper(fmt="iso", utc=True), # Use UTC for consistency
    ]
    
    # Conditional formatting for development vs production
    if settings.ENVIRONMENT == "development":
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(), # Pretty printing for dev
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks, # Add traceback to JSON
            structlog.processors.JSONRenderer(), # JSON for production
        ]
    
    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(), # Outputs to stdout
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    If name is None, returns the root logger.
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger() # Root logger if no name
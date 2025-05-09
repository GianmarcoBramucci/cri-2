"""Logging configuration for the CroceRossa Qdrant Cloud application."""

import sys
import time
import structlog
import logging  # Aggiungi questo import
from typing import Any, Dict, Optional

from app.core.config import settings


def configure_logging() -> None:
    """Configure structlog for application logging."""
    
    # Configura il livello di logging usando il modulo standard logging
    level = getattr(logging, settings.LOG_LEVEL.upper())
    
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a configured logger instance with the given name.
    
    Args:
        name: The name of the logger, typically the module name
        
    Returns:
        A bound structlog logger instance
    """
    return structlog.get_logger(name)
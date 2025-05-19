"""
Enhanced logging system for the Silent-Face-Anti-Spoofing project.

This module provides structured logging with request tracking,
configurable log levels, and formatted output to improve debugging
and monitoring capabilities.
"""

import os
import sys
import uuid
import logging
import datetime
from typing import Optional, Dict, Any, Union

# Configure default logging format
DEFAULT_LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s'
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DIR = './saved_logs'

# Ensure log directory exists
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

class RequestContext:
    """Thread-local storage for request context information."""
    _request_id = None
    
    @classmethod
    def get_request_id(cls) -> str:
        """Get the current request ID or generate a new one."""
        if cls._request_id is None:
            cls._request_id = str(uuid.uuid4())
        return cls._request_id
    
    @classmethod
    def set_request_id(cls, request_id: str) -> None:
        """Set the request ID for the current context."""
        cls._request_id = request_id
    
    @classmethod
    def clear_request_id(cls) -> None:
        """Clear the request ID from the current context."""
        cls._request_id = None


class RequestIdFilter(logging.Filter):
    """Filter that adds request_id to log records."""
    
    def filter(self, record):
        record.request_id = RequestContext.get_request_id()
        return True


def configure_logging(
    log_level: Union[int, str] = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    log_file: Optional[str] = None
) -> None:
    """
    Configure the logging system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        log_file: Optional file path to write logs to
    """
    # Convert string log level to numeric if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(RequestIdFilter())
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.addFilter(RequestIdFilter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add request ID filter if not already present
    has_request_filter = False
    for filter in logger.filters:
        if isinstance(filter, RequestIdFilter):
            has_request_filter = True
            break
    
    if not has_request_filter:
        logger.addFilter(RequestIdFilter())
    
    return logger


def log_api_request(
    logger: logging.Logger,
    endpoint: str,
    method: str,
    params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Log an API request with a new request ID.
    
    Args:
        logger: Logger instance
        endpoint: API endpoint
        method: HTTP method
        params: Request parameters
        
    Returns:
        Generated request ID
    """
    # Generate new request ID
    request_id = str(uuid.uuid4())
    RequestContext.set_request_id(request_id)
    
    # Log request
    logger.info(
        f"API Request: {method} {endpoint} - "
        f"Params: {params if params else 'None'}"
    )
    
    return request_id


def log_api_response(
    logger: logging.Logger,
    endpoint: str,
    status_code: int,
    response_time_ms: float,
    response_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an API response.
    
    Args:
        logger: Logger instance
        endpoint: API endpoint
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        response_data: Response data (will be truncated if too large)
    """
    # Truncate response data if too large
    if response_data and isinstance(response_data, dict):
        # Create a copy to avoid modifying the original
        response_data_copy = {}
        for key, value in response_data.items():
            if isinstance(value, str) and len(value) > 1000:
                response_data_copy[key] = f"{value[:1000]}... [truncated]"
            else:
                response_data_copy[key] = value
    else:
        response_data_copy = response_data
    
    # Log response
    logger.info(
        f"API Response: {endpoint} - Status: {status_code} - "
        f"Time: {response_time_ms:.2f}ms - "
        f"Data: {response_data_copy if response_data_copy else 'None'}"
    )


def log_exception(
    logger: logging.Logger,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an exception with context information.
    
    Args:
        logger: Logger instance
        exception: Exception instance
        context: Additional context information
    """
    logger.exception(
        f"Exception: {type(exception).__name__}: {str(exception)} - "
        f"Context: {context if context else 'None'}"
    )


# Configure default logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(DEFAULT_LOG_DIR, f"app_{timestamp}.log")
configure_logging(log_file=log_file)

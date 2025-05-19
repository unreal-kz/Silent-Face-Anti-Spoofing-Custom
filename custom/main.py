"""
Main entry point for the enhanced Silent-Face-Anti-Spoofing project.

This module integrates the enhancements with the original codebase,
providing improved performance, error handling, and configuration
while maintaining compatibility with the original project structure.
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import from custom modules
from custom.logging import configure_logging, get_logger
from custom.config import config
from custom.model_cache import ModelCache
from custom.api_extensions import APIExtensions
from custom.utils import ensure_directory, cleanup_old_temp_files

# Configure logging
logger = get_logger("main")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="Face Liveness Detection API (Enhanced)",
        description="Enhanced API for detecting fake/real faces in images and videos with real-time WebSocket support",
        version="1.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api_cors_origins"),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Create necessary directories
    ensure_directory(config.get("temp_dir"))
    ensure_directory(config.get("log_dir"))
    
    # Clean up old temporary files
    cleanup_old_temp_files(24)  # Clean up files older than 24 hours
    
    # Mount static files directory for serving processed videos
    app.mount("/static", StaticFiles(directory=config.get("temp_dir")), name="static")
    
    # Import original API
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from api import app as original_app
    
    # Copy routes from original app
    for route in original_app.routes:
        app.routes.append(route)
    
    # Add API extensions
    APIExtensions.add_api_extensions(app)
    
    # Initialize model cache
    model_cache = ModelCache.get_instance()
    model_cache.set_max_cache_size(config.get("model_cache_size"))
    
    # Log startup information
    logger.info(f"Enhanced API initialized with configuration:")
    logger.info(f"  Model directory: {config.get('model_dir')}")
    logger.info(f"  Device ID: {config.get('device_id')}")
    logger.info(f"  Confidence threshold: {config.get('confidence_threshold')}")
    logger.info(f"  Model cache size: {config.get('model_cache_size')}")
    
    return app

def run_server():
    """Run the API server."""
    # Configure logging
    log_file = os.path.join(config.get("log_dir"), "api.log")
    configure_logging(log_level=config.get("log_level"), log_file=log_file)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        logger.error("Exiting due to configuration errors")
        sys.exit(1)
    
    # Create app
    app = create_app()
    
    # Run server
    host = config.get("api_host")
    port = config.get("api_port")
    workers = config.get("api_workers")
    
    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    
    # Check if SSL is enabled
    if config.get("api_ssl_enabled"):
        ssl_cert = config.get("api_ssl_cert")
        ssl_key = config.get("api_ssl_key")
        
        if ssl_cert and ssl_key and os.path.exists(ssl_cert) and os.path.exists(ssl_key):
            logger.info(f"SSL enabled with certificate: {ssl_cert}")
            
            uvicorn.run(
                "custom.main:create_app",
                host=host,
                port=port,
                workers=workers,
                ssl_keyfile=ssl_key,
                ssl_certfile=ssl_cert,
                factory=True
            )
        else:
            logger.error("SSL enabled but certificate or key not found")
            sys.exit(1)
    else:
        uvicorn.run(
            "custom.main:create_app",
            host=host,
            port=port,
            workers=workers,
            factory=True
        )

if __name__ == "__main__":
    run_server()

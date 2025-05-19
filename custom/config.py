"""
Configuration management system for the Silent-Face-Anti-Spoofing project.

This module provides a flexible configuration system that supports
environment variables, configuration files, and runtime overrides
while maintaining backward compatibility with the original codebase.
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Union, List
from pathlib import Path

# Configure logging
logger = logging.getLogger("config")

class Config:
    """
    Configuration management class that supports multiple sources
    with priority order: defaults < config file < environment variables < runtime overrides.
    """
    _instance = None
    
    # Default configuration values
    _defaults = {
        # Model settings
        "model_dir": "./resources/anti_spoof_models",
        "detection_model_dir": "./resources/detection_model",
        "device_id": -1,  # -1 means CPU
        "confidence_threshold": 0.65,
        
        # API settings
        "api_host": "0.0.0.0",
        "api_port": 8000,
        "api_workers": 1,
        "api_cors_origins": ["*"],
        "api_ssl_enabled": False,
        "api_ssl_cert": None,
        "api_ssl_key": None,
        
        # Processing settings
        "temp_dir": "./temp_uploads",
        "max_file_size": 100 * 1024 * 1024,  # 100MB
        "allowed_image_types": ["image/jpeg", "image/png", "image/jpg"],
        "allowed_video_types": ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime"],
        "smoothing_window": 5,
        
        # Logging settings
        "log_level": "INFO",
        "log_dir": "./saved_logs",
        
        # Cache settings
        "model_cache_size": 10,
        
        # Performance settings
        "batch_size": 1,
    }
    
    def __init__(self):
        """Initialize configuration with default values."""
        self._config = self._defaults.copy()
        self._load_from_file()
        self._load_from_env()
        
    @classmethod
    def get_instance(cls) -> 'Config':
        """Get the singleton instance of the Config class."""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance
    
    def _load_from_file(self, config_path: Optional[str] = None) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        # Default config path
        if config_path is None:
            config_path = os.environ.get("CONFIG_FILE", "./config.json")
        
        # Check if file exists
        if not os.path.exists(config_path):
            logger.info(f"Configuration file not found at {config_path}, using defaults")
            return
        
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                
            # Update configuration with values from file
            for key, value in file_config.items():
                if key in self._config:
                    self._config[key] = value
                else:
                    logger.warning(f"Unknown configuration key in file: {key}")
                    
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Environment variable prefix
        prefix = "FACE_ANTI_SPOOF_"
        
        # Mapping of config keys to environment variable names
        env_mapping = {
            "model_dir": f"{prefix}MODEL_DIR",
            "detection_model_dir": f"{prefix}DETECTION_MODEL_DIR",
            "device_id": f"{prefix}DEVICE_ID",
            "confidence_threshold": f"{prefix}CONFIDENCE_THRESHOLD",
            "api_host": f"{prefix}API_HOST",
            "api_port": f"{prefix}API_PORT",
            "api_workers": f"{prefix}API_WORKERS",
            "api_cors_origins": f"{prefix}API_CORS_ORIGINS",
            "api_ssl_enabled": f"{prefix}API_SSL_ENABLED",
            "api_ssl_cert": f"{prefix}API_SSL_CERT",
            "api_ssl_key": f"{prefix}API_SSL_KEY",
            "temp_dir": f"{prefix}TEMP_DIR",
            "max_file_size": f"{prefix}MAX_FILE_SIZE",
            "smoothing_window": f"{prefix}SMOOTHING_WINDOW",
            "log_level": f"{prefix}LOG_LEVEL",
            "log_dir": f"{prefix}LOG_DIR",
            "model_cache_size": f"{prefix}MODEL_CACHE_SIZE",
            "batch_size": f"{prefix}BATCH_SIZE",
        }
        
        # Update configuration with values from environment variables
        for config_key, env_var in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Convert value to appropriate type
                if config_key in self._config:
                    original_type = type(self._config[config_key])
                    
                    try:
                        if original_type == bool:
                            value = value.lower() in ('true', 'yes', '1', 'y')
                        elif original_type == int:
                            value = int(value)
                        elif original_type == float:
                            value = float(value)
                        elif original_type == list:
                            # Handle comma-separated list
                            value = [item.strip() for item in value.split(',')]
                            
                        self._config[config_key] = value
                        logger.info(f"Loaded configuration from environment: {config_key}={value}")
                        
                    except Exception as e:
                        logger.error(f"Error converting environment variable {env_var}: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value at runtime.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
        logger.info(f"Runtime configuration update: {key}={value}")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.
        
        Returns:
            Dictionary of all configuration values
        """
        return self._config.copy()
    
    def validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Validate model directory
        model_dir = self.get("model_dir")
        if not os.path.isdir(model_dir):
            errors.append(f"Model directory not found: {model_dir}")
        
        # Validate detection model directory
        detection_model_dir = self.get("detection_model_dir")
        if not os.path.isdir(detection_model_dir):
            errors.append(f"Detection model directory not found: {detection_model_dir}")
        
        # Validate temp directory
        temp_dir = self.get("temp_dir")
        if not os.path.exists(temp_dir):
            try:
                os.makedirs(temp_dir, exist_ok=True)
                logger.info(f"Created temp directory: {temp_dir}")
            except Exception as e:
                errors.append(f"Failed to create temp directory {temp_dir}: {str(e)}")
        
        # Validate log directory
        log_dir = self.get("log_dir")
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Created log directory: {log_dir}")
            except Exception as e:
                errors.append(f"Failed to create log directory {log_dir}: {str(e)}")
        
        # Validate SSL configuration
        if self.get("api_ssl_enabled"):
            ssl_cert = self.get("api_ssl_cert")
            ssl_key = self.get("api_ssl_key")
            
            if not ssl_cert or not os.path.isfile(ssl_cert):
                errors.append(f"SSL certificate not found: {ssl_cert}")
            
            if not ssl_key or not os.path.isfile(ssl_key):
                errors.append(f"SSL key not found: {ssl_key}")
        
        return errors
    
    def create_default_config_file(self, path: str = "./config.json") -> None:
        """
        Create a default configuration file.
        
        Args:
            path: Path to the configuration file
        """
        try:
            with open(path, 'w') as f:
                json.dump(self._defaults, f, indent=4)
            
            logger.info(f"Created default configuration file at {path}")
            
        except Exception as e:
            logger.error(f"Error creating default configuration file at {path}: {str(e)}")


# Create a default instance
config = Config.get_instance()

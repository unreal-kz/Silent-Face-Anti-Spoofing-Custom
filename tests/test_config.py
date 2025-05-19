"""
Tests for the configuration management system.

This module tests the Config class to ensure it correctly
loads configuration from different sources and validates it.
"""

import os
import json
import unittest
import tempfile
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from custom modules
from custom.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset Config singleton before each test
        Config._instance = None
        
        # Save original environment variables
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up after each test."""
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_singleton_pattern(self):
        """Test that Config implements the singleton pattern correctly."""
        # Get two instances
        config1 = Config.get_instance()
        config2 = Config.get_instance()
        
        # Check that they are the same instance
        self.assertIs(config1, config2)
    
    def test_default_config(self):
        """Test that default configuration is loaded correctly."""
        # Get config instance
        config = Config.get_instance()
        
        # Check that default values are set
        self.assertEqual(config.get("model_dir"), "./resources/anti_spoof_models")
        self.assertEqual(config.get("device_id"), -1)
        self.assertEqual(config.get("confidence_threshold"), 0.65)
    
    def test_config_file_loading(self):
        """Test that configuration is loaded from file correctly."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write custom configuration
            json.dump({
                "device_id": 0,
                "confidence_threshold": 0.75,
                "api_port": 8080
            }, f)
            config_path = f.name
        
        try:
            # Set environment variable to point to config file
            os.environ["CONFIG_FILE"] = config_path
            
            # Reset Config singleton
            Config._instance = None
            
            # Get config instance
            config = Config.get_instance()
            
            # Check that values from file are loaded
            self.assertEqual(config.get("device_id"), 0)
            self.assertEqual(config.get("confidence_threshold"), 0.75)
            self.assertEqual(config.get("api_port"), 8080)
            
            # Check that default values are still used for other settings
            self.assertEqual(config.get("model_dir"), "./resources/anti_spoof_models")
            
        finally:
            # Clean up temporary file
            os.unlink(config_path)
    
    def test_environment_variable_loading(self):
        """Test that configuration is loaded from environment variables correctly."""
        # Set environment variables
        os.environ["FACE_ANTI_SPOOF_DEVICE_ID"] = "1"
        os.environ["FACE_ANTI_SPOOF_CONFIDENCE_THRESHOLD"] = "0.8"
        os.environ["FACE_ANTI_SPOOF_API_PORT"] = "9000"
        os.environ["FACE_ANTI_SPOOF_API_CORS_ORIGINS"] = "http://localhost:3000,http://example.com"
        os.environ["FACE_ANTI_SPOOF_API_SSL_ENABLED"] = "true"
        
        # Reset Config singleton
        Config._instance = None
        
        # Get config instance
        config = Config.get_instance()
        
        # Check that values from environment variables are loaded
        self.assertEqual(config.get("device_id"), 1)
        self.assertEqual(config.get("confidence_threshold"), 0.8)
        self.assertEqual(config.get("api_port"), 9000)
        self.assertEqual(config.get("api_cors_origins"), ["http://localhost:3000", "http://example.com"])
        self.assertEqual(config.get("api_ssl_enabled"), True)
    
    def test_runtime_config_override(self):
        """Test that configuration can be overridden at runtime."""
        # Get config instance
        config = Config.get_instance()
        
        # Override configuration at runtime
        config.set("device_id", 2)
        config.set("confidence_threshold", 0.9)
        
        # Check that values are overridden
        self.assertEqual(config.get("device_id"), 2)
        self.assertEqual(config.get("confidence_threshold"), 0.9)
    
    def test_config_validation(self):
        """Test that configuration validation works correctly."""
        # Get config instance
        config = Config.get_instance()
        
        # Set invalid values
        config.set("model_dir", "./nonexistent_directory")
        config.set("api_ssl_enabled", True)
        config.set("api_ssl_cert", "./nonexistent_cert.pem")
        config.set("api_ssl_key", "./nonexistent_key.pem")
        
        # Validate configuration
        errors = config.validate()
        
        # Check that validation errors are reported
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("Model directory not found" in error for error in errors))
        self.assertTrue(any("SSL certificate not found" in error for error in errors))
        self.assertTrue(any("SSL key not found" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

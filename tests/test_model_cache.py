"""
Tests for the model cache functionality.

This module tests the ModelCache class to ensure it correctly
caches models and improves performance.
"""

import os
import time
import unittest
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from custom modules
from custom.model_cache import ModelCache
from src.anti_spoof_predict import AntiSpoofPredict


class TestModelCache(unittest.TestCase):
    """Test cases for the ModelCache class."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear cache before each test
        ModelCache.get_instance().clear_cache()
        
        # Set up model directory
        self.model_dir = "./resources/anti_spoof_models"
        
        # Skip tests if model directory doesn't exist
        if not os.path.isdir(self.model_dir):
            self.skipTest(f"Model directory not found: {self.model_dir}")
        
        # Get list of model files
        self.model_files = [
            os.path.join(self.model_dir, f)
            for f in os.listdir(self.model_dir)
            if f.endswith('.pth')
        ]
        
        # Skip tests if no model files found
        if not self.model_files:
            self.skipTest(f"No model files found in {self.model_dir}")
        
        # Set up device
        self.device = torch.device("cpu")
    
    def test_singleton_pattern(self):
        """Test that ModelCache implements the singleton pattern correctly."""
        # Get two instances
        cache1 = ModelCache.get_instance()
        cache2 = ModelCache.get_instance()
        
        # Check that they are the same instance
        self.assertIs(cache1, cache2)
    
    def test_model_loading(self):
        """Test that models are loaded correctly."""
        # Get cache instance
        cache = ModelCache.get_instance()
        
        # Load a model
        model_path = self.model_files[0]
        model = cache.get_model(model_path, self.device)
        
        # Check that model is loaded
        self.assertIsNotNone(model)
        
        # Check that model is in eval mode
        self.assertFalse(model.training)
    
    def test_caching_performance(self):
        """Test that caching improves performance."""
        # Get cache instance
        cache = ModelCache.get_instance()
        
        # Load a model for the first time and measure time
        model_path = self.model_files[0]
        
        start_time = time.time()
        model1 = cache.get_model(model_path, self.device)
        first_load_time = time.time() - start_time
        
        # Load the same model again and measure time
        start_time = time.time()
        model2 = cache.get_model(model_path, self.device)
        second_load_time = time.time() - start_time
        
        # Check that second load is faster
        self.assertLess(second_load_time, first_load_time)
        
        # Check that both loads return the same model
        self.assertIs(model1, model2)
    
    def test_max_cache_size(self):
        """Test that cache size is limited correctly."""
        # Get cache instance
        cache = ModelCache.get_instance()
        
        # Set max cache size
        max_size = 2
        cache.set_max_cache_size(max_size)
        
        # Check that max size is set correctly
        self.assertEqual(cache.get_cache_info()["max_cache_size"], max_size)
        
        # Load more models than max size
        for i in range(min(len(self.model_files), max_size + 2)):
            model_path = self.model_files[i]
            cache.get_model(model_path, self.device)
        
        # Check that cache size is limited
        self.assertLessEqual(cache.get_cache_info()["cached_models"], max_size)


if __name__ == "__main__":
    unittest.main()

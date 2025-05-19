"""
Model caching system for the Silent-Face-Anti-Spoofing project.

This module provides a caching mechanism for loaded models to avoid
repeatedly loading the same models from disk, which improves
performance significantly for repeated inferences.
"""

import os
import torch
import logging
from collections import OrderedDict
from typing import Dict, Optional, Any

# Import from the original codebase
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.utility import get_kernel, parse_model_name

# Create a mapping of model types to their respective classes
MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_cache")

class ModelCache:
    """
    Singleton class for caching models to avoid repeated loading.
    
    This cache stores loaded models in memory to avoid the overhead
    of loading models from disk for each inference request.
    """
    _instance = None
    _models: Dict[str, Any] = {}
    _max_cache_size = 10  # Maximum number of models to keep in cache
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the ModelCache."""
        if cls._instance is None:
            cls._instance = ModelCache()
        return cls._instance
    
    def get_model(self, model_path: str, device: torch.device) -> Any:
        """
        Get a model from the cache or load it if not cached.
        
        Args:
            model_path: Path to the model file
            device: PyTorch device to load the model on
            
        Returns:
            The loaded model
        """
        if model_path not in self._models:
            if len(self._models) >= self._max_cache_size:
                # Remove the least recently used model
                self._models.pop(next(iter(self._models)))
                
            logger.info(f"Loading model from {model_path}")
            self._load_model(model_path, device)
            
        return self._models[model_path]["model"]
    
    def _load_model(self, model_path: str, device: torch.device) -> None:
        """
        Load a model from disk and cache it.
        
        Args:
            model_path: Path to the model file
            device: PyTorch device to load the model on
        """
        try:
            # Parse model name to get parameters
            model_name = os.path.basename(model_path)
            h_input, w_input, model_type, _ = parse_model_name(model_name)
            kernel_size = get_kernel(h_input, w_input)
            
            # Create model instance
            model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)
            
            # Load model weights
            state_dict = torch.load(model_path, map_location=device)
            
            # Handle models saved with DataParallel
            if list(state_dict.keys())[0].startswith('module.'):
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[name_key] = value
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
                
            # Set model to evaluation mode
            model.eval()
            
            # Store in cache
            self._models[model_path] = {
                "model": model,
                "last_used": 0  # Will be updated on each access
            }
            
            logger.info(f"Successfully loaded and cached model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
            raise
    
    def clear_cache(self) -> None:
        """Clear the entire model cache."""
        self._models.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about the current cache state.
        
        Returns:
            Dict containing cache statistics
        """
        return {
            "cached_models": len(self._models),
            "max_cache_size": self._max_cache_size
        }
    
    def set_max_cache_size(self, size: int) -> None:
        """
        Set the maximum cache size.
        
        Args:
            size: Maximum number of models to keep in cache
        """
        if size < 1:
            raise ValueError("Cache size must be at least 1")
        
        self._max_cache_size = size
        
        # If current cache is larger than new max size, remove oldest entries
        while len(self._models) > self._max_cache_size:
            self._models.pop(next(iter(self._models)))
            
        logger.info(f"Max cache size set to {size}")

"""
Tests for the API extensions.

This module tests the APIExtensions class to ensure it correctly
enhances the API functionality while maintaining compatibility
with the original codebase.
"""

import os
import cv2
import base64
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from custom modules
from custom.api_extensions import APIExtensions


class TestAPIExtensions(unittest.TestCase):
    """Test cases for the APIExtensions class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Draw a face-like shape for testing
        cv2.rectangle(self.test_image, (30, 30), (70, 70), (255, 255, 255), -1)
        cv2.circle(self.test_image, (40, 45), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(self.test_image, (60, 45), 5, (0, 0, 0), -1)  # Right eye
        cv2.line(self.test_image, (45, 60), (55, 60), (0, 0, 0), 2)  # Mouth
        
        # Encode image to bytes
        _, self.image_bytes = cv2.imencode('.jpg', self.test_image)
        self.image_bytes = self.image_bytes.tobytes()
        
        # Create a mock FastAPI app
        self.app = FastAPI()
        
        # Add API extensions
        APIExtensions.add_api_extensions(self.app)
        
        # Create a test client
        self.client = TestClient(self.app)
    
    @patch('custom.api_extensions.AntiSpoofPredict')
    @patch('custom.api_extensions.CropImage')
    @patch('custom.api_extensions.ModelCache')
    def test_process_image(self, mock_model_cache, mock_crop_image, mock_anti_spoof_predict):
        """Test that _process_image processes images correctly."""
        # Mock AntiSpoofPredict.get_bbox
        mock_anti_spoof_instance = mock_anti_spoof_predict.return_value
        mock_anti_spoof_instance.get_bbox.return_value = [30, 30, 40, 40]
        
        # Mock CropImage.crop
        mock_crop_image_instance = mock_crop_image.return_value
        mock_crop_image_instance.crop.return_value = self.test_image
        
        # Mock ModelCache
        mock_cache_instance = mock_model_cache.get_instance.return_value
        mock_model = MagicMock()
        mock_model.forward.return_value = torch.tensor([[0.1, 0.8, 0.1]])
        mock_cache_instance.get_model.return_value = mock_model
        
        # Test _process_image
        with patch('os.listdir', return_value=['2.7_80x80_MiniFASNetV2.pth']):
            with patch('custom.api_extensions.parse_model_name', 
                      return_value=(80, 80, 'MiniFASNetV2', 2.7)):
                
                result = APIExtensions._process_image(
                    self.test_image,
                    confidence_threshold=0.5,
                    include_annotated_image=True
                )
                
                # Check result
                self.assertIsInstance(result, dict)
                self.assertTrue(result['is_real'])
                self.assertGreater(result['confidence'], 0)
                self.assertEqual(result['bbox'], [30, 30, 40, 40])
                self.assertIn('annotated_image_base64', result)
    
    @patch('custom.api_extensions.APIExtensions._process_image')
    async def test_detect_image_enhanced(self, mock_process_image):
        """Test that detect_image_enhanced handles images correctly."""
        # Mock _process_image
        mock_process_image.return_value = {
            'is_real': True,
            'confidence': 0.8,
            'bbox': [30, 30, 40, 40],
            'label': 1,
            'result_text': 'Real Face',
            'annotated_image_base64': 'test_base64'
        }
        
        # Create mock file
        mock_file = MagicMock()
        mock_file.filename = 'test.jpg'
        mock_file.content_type = 'image/jpeg'
        mock_file.read = MagicMock(return_value=self.image_bytes)
        
        # Test detect_image_enhanced
        with patch('custom.api_extensions.config.get', side_effect=lambda key: {
            'allowed_image_types': ['image/jpeg', 'image/png'],
            'max_file_size': 10000000,
            'confidence_threshold': 0.65
        }.get(key)):
            
            result = await APIExtensions.detect_image_enhanced(
                file=mock_file,
                include_annotated_image=True
            )
            
            # Check result
            self.assertIsInstance(result, dict)
            self.assertTrue(result['is_real'])
            self.assertEqual(result['confidence'], 0.8)
            self.assertEqual(result['bbox'], [30, 30, 40, 40])
            self.assertEqual(result['result_text'], 'Real Face')
            self.assertEqual(result['annotated_image_base64'], 'test_base64')
            self.assertIn('processing_time_ms', result)
    
    @patch('custom.api_extensions.APIExtensions._process_image')
    async def test_detect_image_enhanced_invalid_file_type(self, mock_process_image):
        """Test that detect_image_enhanced handles invalid file types correctly."""
        # Create mock file
        mock_file = MagicMock()
        mock_file.filename = 'test.txt'
        mock_file.content_type = 'text/plain'
        
        # Test detect_image_enhanced with invalid file type
        with patch('custom.api_extensions.config.get', side_effect=lambda key: {
            'allowed_image_types': ['image/jpeg', 'image/png'],
            'max_file_size': 10000000
        }.get(key)):
            
            with self.assertRaises(HTTPException) as context:
                await APIExtensions.detect_image_enhanced(
                    file=mock_file,
                    include_annotated_image=True
                )
            
            # Check exception
            self.assertEqual(context.exception.status_code, 400)
            self.assertIn('Unsupported file type', context.exception.detail)
    
    def test_add_api_extensions(self):
        """Test that add_api_extensions adds endpoints correctly."""
        # Check that health endpoint is added
        response = self.client.get("/health/extended")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    import torch
    unittest.main()

"""
API extensions for the Silent-Face-Anti-Spoofing project.

This module provides enhanced API functionality while maintaining
compatibility with the original codebase. It includes improved error
handling, input validation, and performance optimizations.
"""

import os
import cv2
import time
import base64
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse

# Import from custom modules
from custom.logging import get_logger, log_api_request, log_api_response, log_exception
from custom.config import config
from custom.model_cache import ModelCache

# Import from original codebase
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Configure logger
logger = get_logger("api_extensions")

class APIExtensions:
    """
    Extensions for the Silent-Face-Anti-Spoofing API.
    
    This class provides enhanced functionality for the API while
    maintaining compatibility with the original codebase.
    """
    
    @staticmethod
    async def detect_image_enhanced(
        file: UploadFile,
        include_annotated_image: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enhanced image detection with improved error handling and performance.
        
        Args:
            file: Uploaded image file
            include_annotated_image: Whether to include the annotated image in the response
            confidence_threshold: Custom confidence threshold for this request
            
        Returns:
            Detection result
        """
        # Log API request
        request_id = log_api_request(logger, "/detect_image", "POST", {
            "filename": file.filename,
            "content_type": file.content_type,
            "include_annotated_image": include_annotated_image,
            "confidence_threshold": confidence_threshold
        })
        
        start_time = time.time()
        
        try:
            # Validate file type
            if file.content_type not in config.get("allowed_image_types"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.content_type}. " +
                           f"Supported types: {config.get('allowed_image_types')}"
                )
            
            # Read file content
            contents = await file.read()
            if len(contents) > config.get("max_file_size"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {len(contents)} bytes. " +
                           f"Maximum size: {config.get('max_file_size')} bytes"
                )
            
            # Convert to numpy array
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to decode image"
                )
            
            # Use the provided confidence threshold or the default
            if confidence_threshold is None:
                confidence_threshold = config.get("confidence_threshold")
            
            # Process image
            result = APIExtensions._process_image(
                image,
                confidence_threshold,
                include_annotated_image
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time_ms
            
            # Log API response
            log_api_response(logger, "/detect_image", 200, processing_time_ms, {
                "is_real": result["is_real"],
                "confidence": result["confidence"],
                "bbox": result["bbox"]
            })
            
            return result
            
        except HTTPException as e:
            # Re-raise HTTP exceptions
            log_api_response(logger, "/detect_image", e.status_code, 
                            (time.time() - start_time) * 1000, 
                            {"error": e.detail})
            raise
            
        except Exception as e:
            # Log unexpected exceptions
            log_exception(logger, e, {"request_id": request_id})
            
            # Return 500 error
            processing_time_ms = (time.time() - start_time) * 1000
            log_api_response(logger, "/detect_image", 500, processing_time_ms, 
                            {"error": str(e)})
            
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    @staticmethod
    def _process_image(
        image: np.ndarray,
        confidence_threshold: float,
        include_annotated_image: bool
    ) -> Dict[str, Any]:
        """
        Process an image for liveness detection.
        
        Args:
            image: Image as numpy array
            confidence_threshold: Confidence threshold for detection
            include_annotated_image: Whether to include the annotated image
            
        Returns:
            Detection result
        """
        # Get device
        device_id = config.get("device_id")
        if device_id < 0 or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{device_id}")
        
        # Initialize model
        model_test = AntiSpoofPredict(device_id)
        image_cropper = CropImage()
        
        # Get face bounding box
        image_bbox = model_test.get_bbox(image)
        if image_bbox is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image"
            )
        
        # Get model cache
        model_cache = ModelCache.get_instance()
        
        # Initialize prediction array
        prediction = np.zeros((1, 3))
        
        # Process with all models in the model directory
        model_dir = config.get("model_dir")
        for model_name in os.listdir(model_dir):
            # Parse model name
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            
            # Prepare image
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
                
            img = image_cropper.crop(**param)
            
            # Get model from cache
            model_path = os.path.join(model_dir, model_name)
            model = model_cache.get_model(model_path, device)
            
            # Make prediction
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                result = model.forward(img_tensor)
                result = torch.nn.functional.softmax(result).cpu().numpy()
                
            prediction += result
        
        # Determine if real or fake face with confidence threshold
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        
        # Apply confidence threshold (label 1 is real, label 0 is fake)
        is_real = label == 1
        if is_real and value < confidence_threshold:
            # If classified as real but confidence is low, mark as uncertain/fake
            is_real = False
            label = 0
            result_text = f"Uncertain (Low Confidence)"
        elif is_real:
            result_text = "Real Face"
        else:
            result_text = "Fake Face"
        
        # Create result dictionary
        result = {
            "is_real": bool(is_real),
            "confidence": float(value),
            "bbox": [int(x) for x in image_bbox],
            "label": int(label),
            "result_text": result_text
        }
        
        # Create annotated image if requested
        if include_annotated_image:
            annotated_image = image.copy()
            
            # Set color based on result
            if is_real:
                color = (0, 255, 0)  # Green for real
            else:
                color = (0, 0, 255)  # Red for fake
            
            # Draw bounding box and result
            cv2.rectangle(
                annotated_image,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            
            cv2.putText(
                annotated_image,
                f"{result_text}: {value:.2f}",
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
            
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', annotated_image)
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result["annotated_image_base64"] = annotated_image_base64
        
        return result
    
    @staticmethod
    def add_api_extensions(app: FastAPI) -> None:
        """
        Add API extensions to the FastAPI app.
        
        Args:
            app: FastAPI application instance
        """
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            """Log all HTTP requests and responses."""
            start_time = time.time()
            
            # Process the request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Log request and response
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {processing_time_ms:.2f}ms - "
                f"Client: {request.client.host if request.client else 'Unknown'}"
            )
            
            return response
        
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions."""
            logger.warning(
                f"HTTP Exception: {exc.status_code} - {exc.detail} - "
                f"Path: {request.url.path}"
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions."""
            log_exception(logger, exc, {"path": request.url.path})
            
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {str(exc)}"}
            )
        
        # Add health check endpoint
        @app.get("/health/extended")
        async def extended_health_check():
            """Extended health check endpoint with system information."""
            try:
                # Check if model directory exists
                model_dir = config.get("model_dir")
                models_available = os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0
                
                # Check if GPU is available
                gpu_available = torch.cuda.is_available()
                gpu_info = None
                if gpu_available:
                    device_id = config.get("device_id")
                    if device_id >= 0 and device_id < torch.cuda.device_count():
                        gpu_info = {
                            "name": torch.cuda.get_device_name(device_id),
                            "memory_total": torch.cuda.get_device_properties(device_id).total_memory,
                            "memory_allocated": torch.cuda.memory_allocated(device_id),
                            "memory_reserved": torch.cuda.memory_reserved(device_id)
                        }
                
                # Get model cache info
                model_cache = ModelCache.get_instance()
                cache_info = model_cache.get_cache_info()
                
                # Get configuration
                config_info = {
                    key: value for key, value in config.get_all().items()
                    if key not in ["api_ssl_key", "api_ssl_cert"]  # Exclude sensitive info
                }
                
                return {
                    "status": "ok",
                    "version": "1.0.0",
                    "models_available": models_available,
                    "gpu_available": gpu_available,
                    "gpu_info": gpu_info,
                    "model_cache": cache_info,
                    "config": config_info
                }
                
            except Exception as e:
                log_exception(logger, e)
                
                return {
                    "status": "error",
                    "version": "1.0.0",
                    "error": str(e)
                }
        
        logger.info("API extensions added successfully")

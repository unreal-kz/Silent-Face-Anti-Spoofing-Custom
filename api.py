#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import time
import uuid
import json
import numpy as np
import uvicorn
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64

# Import from the existing codebase
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Constants
MODEL_DIR = "./resources/anti_spoof_models"
TEMP_DIR = "./temp_uploads"
CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_WINDOW = 5
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/avi", "video/x-msvideo", "video/quicktime"]

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Pydantic models for API responses
class HealthResponse(BaseModel):
    status: str
    version: str

class ImageDetectionResponse(BaseModel):
    is_real: bool
    confidence: float
    bbox: List[int]
    annotated_image_base64: Optional[str] = None
    processing_time_ms: float

class VideoDetectionResponse(BaseModel):
    overall_result: str
    real_percentage: float
    fake_percentage: float
    uncertain_percentage: float
    real_frames: int
    fake_frames: int
    uncertain_frames: int
    total_processed_frames: int
    total_frames: int
    output_video_path: Optional[str] = None
    processing_time_ms: float
    video_info: Dict

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: Dict):
        await websocket.send_json(data)

# Create FastAPI app
app = FastAPI(
    title="Face Liveness Detection API",
    description="API for detecting fake/real faces in images and videos with real-time WebSocket support",
    version="1.0.0"
)

# Create connection manager instance
manager = ConnectionManager()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving processed videos
app.mount("/static", StaticFiles(directory=TEMP_DIR), name="static")

# Utility functions - adapted from test_video.py
def detect_from_image(image, device_id=-1, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Detect liveness from a single image."""
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    # Get face bounding box
    image_bbox = model_test.get_bbox(image)
    
    # Initialize prediction array
    prediction = np.zeros((1, 3))
    
    # Process with all models in the model directory
    for model_name in os.listdir(MODEL_DIR):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
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
        prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
    
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
    
    # Create annotated image
    annotated_image = image.copy()
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
    
    return {
        "is_real": bool(is_real),
        "confidence": float(value),
        "label": int(label),
        "bbox": [int(x) for x in image_bbox],
        "annotated_image": annotated_image
    }

def detect_from_video(video_path, device_id=-1, confidence_threshold=CONFIDENCE_THRESHOLD, 
                     smoothing_window=SMOOTHING_WINDOW, output_path=None):
    """Detect liveness from a video file - directly adapted from test_video.py."""
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output path is provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # For temporal smoothing
    recent_predictions = []
    
    # For statistics
    frame_count = 0
    real_count = 0
    fake_count = 0
    uncertain_count = 0
    
    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame for speed
            continue
        
        # Get face bounding box
        image_bbox = model_test.get_bbox(frame)
        
        # Initialize prediction array
        prediction = np.zeros((1, 3))
        
        # Process with all models in the model directory
        for model_name in os.listdir(MODEL_DIR):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
        
        # Store the raw prediction for temporal smoothing
        recent_predictions.append(prediction)
        if len(recent_predictions) > smoothing_window:
            recent_predictions.pop(0)  # Remove oldest prediction
        
        # Apply temporal smoothing by averaging recent predictions
        if len(recent_predictions) > 0:
            smoothed_prediction = np.mean(recent_predictions, axis=0)
        else:
            smoothed_prediction = prediction
        
        # Determine if real or fake face with confidence threshold
        label = np.argmax(smoothed_prediction)
        value = smoothed_prediction[0][label]/2
        
        # Apply confidence threshold (label 1 is real, label 0 is fake)
        is_real = label == 1
        if is_real and value < confidence_threshold:
            # If classified as real but confidence is low, mark as uncertain/fake
            is_real = False
            label = 0
            result_text = f"Uncertain (Low Conf)"
            color = (0, 165, 255)  # Orange for uncertain
            uncertain_count += 1
        elif is_real:
            result_text = f"Real Face"
            color = (0, 255, 0)  # Green for real
            real_count += 1
        else:
            result_text = f"Fake Face"
            color = (0, 0, 255)  # Red for fake
            fake_count += 1
        
        # Draw bounding box and result
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            frame,
            f"{result_text}: {value:.2f}",
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)
        
        # Write frame to output video
        if video_writer:
            video_writer.write(frame)
    
    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()
    
    # Calculate statistics
    processed_frames = real_count + fake_count + uncertain_count
    real_percentage = (real_count / processed_frames * 100) if processed_frames > 0 else 0
    fake_percentage = (fake_count / processed_frames * 100) if processed_frames > 0 else 0
    uncertain_percentage = (uncertain_count / processed_frames * 100) if processed_frames > 0 else 0
    
    # Determine overall result (majority vote)
    if real_count > fake_count and real_count > uncertain_count:
        overall_result = "real"
    elif fake_count > real_count and fake_count > uncertain_count:
        overall_result = "fake"
    else:
        overall_result = "uncertain"
    
    return {
        "overall_result": overall_result,
        "real_percentage": float(real_percentage),
        "fake_percentage": float(fake_percentage),
        "uncertain_percentage": float(uncertain_percentage),
        "real_frames": int(real_count),
        "fake_frames": int(fake_count),
        "uncertain_frames": int(uncertain_count),
        "total_processed_frames": int(processed_frames),
        "total_frames": int(total_frames),
        "output_path": output_path
    }

# Helper functions for file handling
async def save_upload_file_temp(upload_file: UploadFile) -> str:
    """Save an uploaded file to a temporary location."""
    file_extension = os.path.splitext(upload_file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    content = await upload_file.read()
    with open(temp_path, "wb") as f:
        f.write(content)
    
    return temp_path

def cleanup_temp_file(file_path: str) -> None:
    """Remove a temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error removing temporary file {file_path}: {e}")

def get_video_info(video_path: str) -> dict:
    """Get information about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": float(fps),
        "width": int(frame_width),
        "height": int(frame_height),
        "total_frames": int(total_frames),
        "duration": float(duration)
    }

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode an image as base64 string."""
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Could not encode image")
    return base64.b64encode(encoded_image).decode('utf-8')

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0"
    }

@app.post("/detect/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    include_annotated_image: bool = True,
    background_tasks: BackgroundTasks = None
):
    """Detect liveness from an uploaded image.
    
    - **file**: Image file to analyze
    - **include_annotated_image**: Whether to include the annotated image in base64 format
    """
    # Validate file type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image file. Allowed types: {ALLOWED_IMAGE_TYPES}"
        )
    
    try:
        # Save uploaded file to temp location
        start_time = time.time()
        temp_file_path = await save_upload_file_temp(file)
        
        # Add cleanup task
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # Read image file
        image = cv2.imread(temp_file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image file")
        
        # Detect liveness
        result = detect_from_image(image)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Prepare response
        response = {
            "is_real": result["is_real"],
            "confidence": result["confidence"],
            "bbox": result["bbox"],
            "processing_time_ms": processing_time_ms
        }
        
        # Include annotated image if requested
        if include_annotated_image:
            response["annotated_image_base64"] = encode_image_to_base64(result["annotated_image"])
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = CONFIDENCE_THRESHOLD,
    smoothing_window: Optional[int] = SMOOTHING_WINDOW,
    save_annotated_video: bool = True,
    background_tasks: BackgroundTasks = None
):
    """Detect liveness from an uploaded video.
    
    - **file**: Video file to analyze
    - **confidence_threshold**: Threshold for real face confidence (0.0-1.0)
    - **smoothing_window**: Number of frames to use for temporal smoothing
    - **save_annotated_video**: Whether to save and return an annotated video
    """
    # Validate file type
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid video file. Allowed types: {ALLOWED_VIDEO_TYPES}"
        )
    
    try:
        # Save uploaded file to temp location
        start_time = time.time()
        temp_file_path = await save_upload_file_temp(file)
        
        # Create output path for annotated video if requested
        output_path = None
        if save_annotated_video:
            output_filename = f"annotated_{os.path.basename(temp_file_path)}"
            output_path = os.path.join(TEMP_DIR, output_filename)
        
        # Get video info
        video_info = get_video_info(temp_file_path)
        
        # Detect liveness
        result = detect_from_video(
            temp_file_path, 
            confidence_threshold=confidence_threshold,
            smoothing_window=smoothing_window,
            output_path=output_path
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Add cleanup task for input file
        if background_tasks:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # Prepare response
        response = {
            "overall_result": result["overall_result"],
            "real_percentage": result["real_percentage"],
            "fake_percentage": result["fake_percentage"],
            "uncertain_percentage": result["uncertain_percentage"],
            "real_frames": result["real_frames"],
            "fake_frames": result["fake_frames"],
            "uncertain_frames": result["uncertain_frames"],
            "total_processed_frames": result["total_processed_frames"],
            "total_frames": result["total_frames"],
            "processing_time_ms": processing_time_ms,
            "video_info": video_info
        }
        
        # Include output video path if saved
        if save_annotated_video and output_path:
            # Convert to relative URL for static file serving
            output_filename = os.path.basename(output_path)
            response["output_video_path"] = f"/static/{output_filename}"
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a processed video file.
    
    - **filename**: Name of the file to download
    """
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time liveness detection.
    
    Clients should send base64-encoded image frames and will receive liveness detection results.
    """
    # Initialize detector
    model_test = AntiSpoofPredict(-1)  # Use CPU for inference
    image_cropper = CropImage()
    
    # For temporal smoothing
    recent_predictions = []
    
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Extract parameters
            confidence_threshold = data.get("confidence_threshold", CONFIDENCE_THRESHOLD)
            smoothing_window = data.get("smoothing_window", SMOOTHING_WINDOW)
            include_annotated_image = data.get("include_annotated_image", False)
            
            # Decode base64 image
            try:
                image_data = base64.b64decode(data["image"])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    await manager.send_json(websocket, {"error": "Invalid image data"})
                    continue
            except Exception as e:
                await manager.send_json(websocket, {"error": f"Error decoding image: {str(e)}"})
                continue
            
            # Start timing
            start_time = time.time()
            
            try:
                # Get face bounding box
                image_bbox = model_test.get_bbox(frame)
                
                # Initialize prediction array
                prediction = np.zeros((1, 3))
                
                # Process with all models in the model directory
                for model_name in os.listdir(MODEL_DIR):
                    h_input, w_input, model_type, scale = parse_model_name(model_name)
                    param = {
                        "org_img": frame,
                        "bbox": image_bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    if scale is None:
                        param["crop"] = False
                    img = image_cropper.crop(**param)
                    prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
                
                # Store the raw prediction for temporal smoothing
                recent_predictions.append(prediction)
                if len(recent_predictions) > smoothing_window:
                    recent_predictions.pop(0)  # Remove oldest prediction
                
                # Apply temporal smoothing by averaging recent predictions
                if len(recent_predictions) > 0:
                    smoothed_prediction = np.mean(recent_predictions, axis=0)
                else:
                    smoothed_prediction = prediction
                
                # Determine if real or fake face with confidence threshold
                label = np.argmax(smoothed_prediction)
                value = smoothed_prediction[0][label]/2
                
                # Apply confidence threshold (label 1 is real, label 0 is fake)
                is_real = label == 1
                if is_real and value < confidence_threshold:
                    # If classified as real but confidence is low, mark as uncertain/fake
                    is_real = False
                    label = 0
                    result_text = f"Uncertain (Low Conf)"
                    color = (0, 165, 255)  # Orange for uncertain
                elif is_real:
                    result_text = f"Real Face"
                    color = (0, 255, 0)  # Green for real
                else:
                    result_text = f"Fake Face"
                    color = (0, 0, 255)  # Red for fake
                
                # Create annotated image if requested
                annotated_image_base64 = None
                if include_annotated_image:
                    annotated_frame = frame.copy()
                    # Draw bounding box and result
                    cv2.rectangle(
                        annotated_frame,
                        (image_bbox[0], image_bbox[1]),
                        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                        color, 2)
                    cv2.putText(
                        annotated_frame,
                        f"{result_text}: {value:.2f}",
                        (image_bbox[0], image_bbox[1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)
                    
                    # Encode annotated image to base64
                    success, encoded_image = cv2.imencode('.jpg', annotated_frame)
                    if success:
                        annotated_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
                
                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Send results back to client
                response = {
                    "is_real": bool(is_real),
                    "confidence": float(value),
                    "result": "real" if is_real else "fake",
                    "bbox": [int(x) for x in image_bbox],
                    "processing_time_ms": float(processing_time_ms)
                }
                
                if annotated_image_base64:
                    response["annotated_image"] = annotated_image_base64
                
                await manager.send_json(websocket, response)
                
            except Exception as e:
                await manager.send_json(websocket, {"error": str(e)})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_json(websocket, {"error": f"Unexpected error: {str(e)}"})
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=9001, reload=True)

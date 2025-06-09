#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import time
import uuid
import json
import numpy as np
import torch
import uvicorn
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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

# Check if CUDA is available and working
USE_GPU = False  # Default to CPU for safety
DEVICE_ID = -1    # Default to CPU

# Try to initialize CUDA only if available
if torch.cuda.is_available():
    try:
        # Test CUDA with a small tensor operation
        test_tensor = torch.zeros(1).cuda()
        test_tensor = test_tensor + 1
        test_tensor.cpu()  # Move back to CPU
        
        # If we get here, CUDA is working
        USE_GPU = True
        DEVICE_ID = 0
        print("\n*** GPU TEST SUCCESSFUL ***")
    except Exception as e:
        print(f"\n*** GPU TEST FAILED: {str(e)} ***")
        print("Falling back to CPU mode")
        USE_GPU = False
        DEVICE_ID = -1

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)

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

# Add new Pydantic model for webcam response
class WebcamDetectionResponse(BaseModel):
    is_real: bool
    confidence: float
    bbox: List[int]
    frame_base64: str
    processing_time_ms: float

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

# Log GPU status
if USE_GPU:
    print(f"\n*** GPU ACCELERATION ENABLED ***")
    print(f"CUDA Device: {torch.cuda.get_device_name(DEVICE_ID)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(DEVICE_ID).total_memory / 1024**3:.2f} GB")
else:
    print("\n*** RUNNING ON CPU ***")
    print("GPU acceleration not available. Install CUDA for better performance.")

# Mount static files directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory=TEMP_DIR), name="uploads")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add caching for model instances
model_cache = {}
image_cropper_cache = None

def get_model(device_id):
    """Get or create model instance from cache."""
    if device_id not in model_cache:
        model_cache[device_id] = AntiSpoofPredict(device_id)
    return model_cache[device_id]

def get_image_cropper():
    """Get or create image cropper instance from cache."""
    global image_cropper_cache
    if image_cropper_cache is None:
        image_cropper_cache = CropImage()
    return image_cropper_cache

def detect_from_image(image, device_id=DEVICE_ID, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Detect liveness from a single image."""
    model_test = get_model(device_id)
    image_cropper = get_image_cropper()
    
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

def detect_from_video(video_path, device_id=DEVICE_ID, confidence_threshold=CONFIDENCE_THRESHOLD, 
                     smoothing_window=SMOOTHING_WINDOW, output_path=None):
    """Detect liveness from a video file - directly adapted from test_video.py."""
    model_test = get_model(device_id)
    image_cropper = get_image_cropper()
    
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
    """Clean up temporary files with proper error handling."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Failed to clean up temporary file {file_path}: {str(e)}")
        # Don't raise the exception as this is a cleanup operation

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

def detect_from_webcam(frame, device_id=DEVICE_ID, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Detect liveness from a webcam frame in real-time."""
    # Get cached instances
    model_test = get_model(device_id)
    image_cropper = get_image_cropper()
    
    # Get face bounding box
    image_bbox = model_test.get_bbox(frame)
    if image_bbox is None:
        return None
    
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
    
    return {
        "is_real": bool(is_real),
        "confidence": float(value),
        "label": int(label),
        "bbox": [int(x) for x in image_bbox]
    }

# API Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Serve the webcam client interface."""
    return FileResponse("static/webcam-client.html")

@app.get("/docs", include_in_schema=False)
async def docs_redirect():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/client", include_in_schema=False)
async def websocket_client():
    """Serve the WebSocket client HTML file."""
    return FileResponse("static/websocket-client.html")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
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
            response["output_video_path"] = f"/uploads/{output_filename}"
        
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

@app.post("/detect/webcam", response_model=WebcamDetectionResponse)
async def detect_webcam(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = CONFIDENCE_THRESHOLD
):
    """Process a single frame from webcam for liveness detection."""
    start_time = time.time()
    
    # Validate file type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        )
    
    # Read and decode the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    # Process the frame
    result = detect_from_webcam(frame, confidence_threshold=confidence_threshold)
    
    if result is None:
        raise HTTPException(status_code=400, detail="No face detected in the frame")
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Encode the annotated image to base64
    _, buffer = cv2.imencode('.jpg', result['annotated_image'])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "is_real": result["is_real"],
        "confidence": result["confidence"],
        "bbox": result["bbox"],
        "frame_base64": frame_base64,
        "processing_time_ms": processing_time
    }

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    try:
        await manager.connect(websocket)
        print(f"WebSocket client connected from {websocket.client.host}:{websocket.client.port}")
        
        while True:
            try:
                # Receive the image data
                data = await websocket.receive_text()
                try:
                    image_data = json.loads(data)
                    if "image" not in image_data:
                        print("No image data found in request")
                        await manager.send_json(websocket, {
                            "error": "No image data found in request"
                        })
                        continue
                    
                    # Decode base64 image
                    try:
                        image_str = image_data["image"]
                        if "," in image_str:
                            # Data URL format (e.g., "data:image/jpeg;base64,/9j/4AAQ...")
                            image_bytes = base64.b64decode(image_str.split(",")[1])
                        else:
                            # Raw base64 format
                            image_bytes = base64.b64decode(image_str)
                            
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError("Could not decode image data")
                            
                    except Exception as e:
                        print(f"Image decoding error: {str(e)}")
                        await manager.send_json(websocket, {
                            "error": f"Failed to decode image: {str(e)}"
                        })
                        continue

                    # Process the image
                    try:
                        start_time = time.time()
                        result = detect_from_image(image)
                        processing_time_ms = (time.time() - start_time) * 1000
                        
                        # Convert numpy array to base64 for JSON serialization
                        if "annotated_image" in result:
                            result["annotated_image_base64"] = encode_image_to_base64(result["annotated_image"])
                            del result["annotated_image"]  # Remove the numpy array
                        
                        # Add processing time
                        result["processing_time_ms"] = processing_time_ms
                        
                        await manager.send_json(websocket, result)
                    except Exception as e:
                        print(f"Image processing error: {str(e)}")
                        await manager.send_json(websocket, {
                            "error": f"Failed to process image: {str(e)}"
                        })
                        continue

                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {str(e)}")
                    await manager.send_json(websocket, {
                        "error": "Invalid JSON data received"
                    })
                    continue
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    await manager.send_json(websocket, {
                        "error": f"Unexpected error: {str(e)}"
                    })
                    continue

            except WebSocketDisconnect:
                print(f"WebSocket client disconnected from {websocket.client.host}:{websocket.client.port}")
                manager.disconnect(websocket)
                break
            except Exception as e:
                print(f"WebSocket error: {str(e)}")
                await manager.send_json(websocket, {
                    "error": f"WebSocket error: {str(e)}"
                })
                break

    except Exception as e:
        print(f"Connection error: {str(e)}")
        try:
            await manager.send_json(websocket, {
                "error": f"Connection error: {str(e)}"
            })
        except:
            pass
        manager.disconnect(websocket)

# Add WebSocket endpoint for real-time webcam streaming
@app.websocket("/ws/webcam")
async def websocket_webcam(websocket: WebSocket):
    """WebSocket endpoint for real-time webcam streaming and detection."""
    await manager.connect(websocket)
    try:
        while True:
            # Receive base64 encoded frame
            data = await websocket.receive_text()
            try:
                # Decode base64 image
                img_data = base64.b64decode(data.split(',')[1] if ',' in data else data)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Could not decode image"})
                    continue
                
                # Process frame
                start_time = time.time()
                result = detect_from_webcam(frame)
                processing_time = (time.time() - start_time) * 1000
                
                if result is None:
                    await websocket.send_json({"error": "No face detected"})
                    continue
                
                # Send results
                await websocket.send_json({
                    "is_real": result["is_real"],
                    "confidence": result["confidence"],
                    "bbox": result["bbox"],
                    "processing_time_ms": processing_time
                })
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Check if SSL certificates exist for HTTPS
    ssl_keyfile = os.environ.get('SSL_KEYFILE', None)
    ssl_certfile = os.environ.get('SSL_CERTFILE', None)
    
    # Use SSL if certificates are provided
    if ssl_keyfile and ssl_certfile and os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        print(f"\n*** STARTING SERVER WITH HTTPS SUPPORT ***")
        uvicorn.run(
            "api:app", 
            host="0.0.0.0", 
            port=9001, 
            reload=True,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile
        )
    else:
        print(f"\n*** STARTING SERVER WITH HTTP ONLY ***")
        print(f"To enable HTTPS, set SSL_KEYFILE and SSL_CERTFILE environment variables")
        uvicorn.run("api:app", host="0.0.0.0", port=9001, reload=True)

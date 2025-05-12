# -*- coding: utf-8 -*-
# @File : test_video.py

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def is_running_in_wsl():
    """Check if running in WSL"""
    import platform
    return 'Linux' in platform.system() and 'microsoft' in platform.release().lower()

def convert_windows_path_if_wsl(path):
    """Convert Windows path to WSL path if running in WSL"""
    import re
    
    # Check if running in WSL
    if is_running_in_wsl():
        # Check if it's a Windows path (starts with C:, D:, etc.)
        if re.match(r'^[a-zA-Z]:', path):
            try:
                # Convert Windows path to WSL path
                drive = path[0].lower()
                path_without_drive = path[2:].replace('\\', '/')
                wsl_path = f"/mnt/{drive}{path_without_drive}"
                return wsl_path
            except Exception as e:
                print(f"Warning: Failed to convert Windows path: {e}")
    return path

def test_video(video_path, model_dir, device_id, output_path=None, display=True):
    # Force CPU usage if CUDA is causing issues
    if device_id >= 0:
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device_id = -1
        except Exception as e:
            print(f"Error checking CUDA availability: {e}. Falling back to CPU.")
            device_id = -1
    
    print(f"Using device_id: {device_id} ({'-1 means CPU' if device_id < 0 else 'GPU'})")
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    # Convert paths if running in WSL
    video_path = convert_windows_path_if_wsl(video_path)
    model_dir = convert_windows_path_if_wsl(model_dir)
    if output_path:
        output_path = convert_windows_path_if_wsl(output_path)
    
    print(f"Attempting to open video: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if output path is provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    processing_times = []
    
    print("Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame to speed up
            continue
        
        start_time = time.time()
        
        # Get face bounding box
        image_bbox = model_test.get_bbox(frame)
        
        # Initialize prediction array
        prediction = np.zeros((1, 3))
        
        # Process with all models in the model directory
        for model_name in os.listdir(model_dir):
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
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        
        # Determine if real or fake face
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        
        if label == 1:
            result_text = f"Real Face: {value:.2f}"
            color = (0, 255, 0)  # Green for real
        else:
            result_text = f"Fake Face: {value:.2f}"
            color = (0, 0, 255)  # Red for fake
        
        # Draw bounding box and result
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            frame,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*frame.shape[0]/1024, color)
        
        # Calculate processing time
        process_time = time.time() - start_time
        processing_times.append(process_time)
        
        # Display the frame
        if display:
            cv2.imshow('Anti-Spoofing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write frame to output video
        if video_writer:
            video_writer.write(frame)
    
    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Processed {frame_count} frames")
        print(f"Average processing time per frame: {avg_time:.4f} seconds")
        print(f"Average FPS: {1/avg_time:.2f}")
    
    print("Video processing complete!")


if __name__ == "__main__":
    desc = "test video for face anti-spoofing"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=-1,  # Changed default to -1 (CPU) to avoid CUDA issues
        help="which gpu id, [0/1/2/3] or -1 for CPU")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="video file path to test")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output video file path")
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="disable display of video processing")
    
    args = parser.parse_args()
    
    # Force CPU usage to avoid CUDA errors
    device_id = -1
    print("Forcing CPU usage to avoid CUDA compatibility issues")
    
    # Handle display settings based on environment
    display = not args.no_display
    if is_running_in_wsl() and display:
        print("Running in WSL environment - display will be disabled to avoid Qt/XCB errors")
        display = False
    
    # Set default output path if none provided and display is disabled
    output_path = args.output_path
    if not display and output_path is None:
        import os
        # Extract filename from video path
        video_filename = os.path.basename(args.video_path)
        base_name = os.path.splitext(video_filename)[0]
        output_path = f"{base_name}_result.avi"
        print(f"Display disabled - setting default output path to: {output_path}")
    
    test_video(args.video_path, args.model_dir, device_id, 
               output_path, display)

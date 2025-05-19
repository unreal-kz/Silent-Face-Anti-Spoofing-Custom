"""
Utility functions for the Silent-Face-Anti-Spoofing project.

This module provides helper functions and utilities that can be used
across the project to improve code reuse and maintainability.
"""

import os
import cv2
import time
import uuid
import shutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# Import from custom modules
from custom.logging import get_logger

# Configure logger
logger = get_logger("utils")

def ensure_directory(directory: str) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        return False

def generate_unique_filename(extension: str = "") -> str:
    """
    Generate a unique filename using UUID.
    
    Args:
        extension: File extension (with or without dot)
        
    Returns:
        Unique filename
    """
    # Ensure extension starts with a dot if provided
    if extension and not extension.startswith("."):
        extension = f".{extension}"
    
    return f"{uuid.uuid4()}{extension}"

def save_temp_file(file_content: bytes, extension: str = "") -> Tuple[str, str]:
    """
    Save content to a temporary file with a unique name.
    
    Args:
        file_content: File content as bytes
        extension: File extension
        
    Returns:
        Tuple of (file path, filename)
    """
    from custom.config import config
    
    # Get temp directory from config
    temp_dir = config.get("temp_dir")
    
    # Ensure temp directory exists
    ensure_directory(temp_dir)
    
    # Generate unique filename
    filename = generate_unique_filename(extension)
    file_path = os.path.join(temp_dir, filename)
    
    # Write file
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        logger.info(f"Saved temporary file: {file_path}")
        return file_path, filename
        
    except Exception as e:
        logger.error(f"Failed to save temporary file: {str(e)}")
        raise

def cleanup_temp_file(file_path: str) -> bool:
    """
    Remove a temporary file.
    
    Args:
        file_path: Path to the file to remove
        
    Returns:
        True if file was removed, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed temporary file: {file_path}")
            return True
        else:
            logger.warning(f"Temporary file not found: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to remove temporary file {file_path}: {str(e)}")
        return False

def cleanup_old_temp_files(max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than the specified age.
    
    Args:
        max_age_hours: Maximum age of files in hours
        
    Returns:
        Number of files removed
    """
    from custom.config import config
    
    # Get temp directory from config
    temp_dir = config.get("temp_dir")
    
    if not os.path.exists(temp_dir):
        logger.warning(f"Temp directory not found: {temp_dir}")
        return 0
    
    # Calculate cutoff time
    cutoff_time = time.time() - (max_age_hours * 3600)
    
    # Count removed files
    removed_count = 0
    
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check file age
            file_modified_time = os.path.getmtime(file_path)
            if file_modified_time < cutoff_time:
                # Remove file
                os.remove(file_path)
                removed_count += 1
                
        logger.info(f"Cleaned up {removed_count} old temporary files")
        return removed_count
        
    except Exception as e:
        logger.error(f"Error cleaning up old temporary files: {str(e)}")
        return removed_count

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Release video
        cap.release()
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "file_size": file_size,
            "file_size_mb": file_size / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Error getting video info for {video_path}: {str(e)}")
        raise

def resize_image_aspect_ratio(
    image: np.ndarray,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    max_dimension: Optional[int] = None
) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: Image as numpy array
        target_width: Target width (optional)
        target_height: Target height (optional)
        max_dimension: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    if max_dimension is not None:
        # Scale based on maximum dimension
        if width > height:
            target_width = max_dimension
            target_height = int(height * (max_dimension / width))
        else:
            target_height = max_dimension
            target_width = int(width * (max_dimension / height))
    elif target_width is not None and target_height is None:
        # Scale based on target width
        target_height = int(height * (target_width / width))
    elif target_height is not None and target_width is None:
        # Scale based on target height
        target_width = int(width * (target_height / height))
    elif target_width is None and target_height is None:
        # No resizing needed
        return image
    
    # Resize image
    resized_image = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )
    
    return resized_image

def batch_process_images(
    images: List[np.ndarray],
    batch_size: int,
    process_func: callable,
    *args, **kwargs
) -> List[Any]:
    """
    Process a list of images in batches.
    
    Args:
        images: List of images as numpy arrays
        batch_size: Batch size
        process_func: Function to process each batch
        *args, **kwargs: Additional arguments for process_func
        
    Returns:
        List of results from process_func
    """
    results = []
    
    # Process in batches
    for i in range(0, len(images), batch_size):
        # Get batch
        batch = images[i:i+batch_size]
        
        # Process batch
        batch_results = process_func(batch, *args, **kwargs)
        
        # Add results
        results.extend(batch_results)
    
    return results

def create_montage(
    images: List[np.ndarray],
    grid_size: Optional[Tuple[int, int]] = None,
    padding: int = 5,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a montage of images.
    
    Args:
        images: List of images as numpy arrays
        grid_size: Grid size as (rows, cols)
        padding: Padding between images
        background_color: Background color
        
    Returns:
        Montage image
    """
    if not images:
        raise ValueError("No images provided")
    
    # Calculate grid size if not provided
    if grid_size is None:
        grid_cols = int(np.ceil(np.sqrt(len(images))))
        grid_rows = int(np.ceil(len(images) / grid_cols))
        grid_size = (grid_rows, grid_cols)
    else:
        grid_rows, grid_cols = grid_size
    
    # Ensure all images have the same shape
    first_image = images[0]
    for i in range(1, len(images)):
        if images[i].shape != first_image.shape:
            images[i] = cv2.resize(
                images[i],
                (first_image.shape[1], first_image.shape[0]),
                interpolation=cv2.INTER_AREA
            )
    
    # Calculate montage dimensions
    img_height, img_width = first_image.shape[:2]
    montage_width = grid_cols * img_width + (grid_cols - 1) * padding
    montage_height = grid_rows * img_height + (grid_rows - 1) * padding
    
    # Create montage image
    montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)
    montage[:] = background_color
    
    # Add images to montage
    img_idx = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            if img_idx >= len(images):
                break
                
            # Calculate position
            y = row * (img_height + padding)
            x = col * (img_width + padding)
            
            # Add image
            montage[y:y+img_height, x:x+img_width] = images[img_idx]
            img_idx += 1
    
    return montage

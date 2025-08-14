"""
Moondream Vision MCP Server

An MCP server that provides computer vision capabilities using Moondream VLM.
Enables AI assistants to analyze images, extract text, detect objects, and generate captions
from local image files.

Author: Luke Allen
Version: 1.0.0
"""

from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage, ImageDraw
import moondream as md
import json
import os
import logging
from io import BytesIO
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("moondream-vision")

# Initialize Moondream model
try:
    model = md.vl(endpoint="http://127.0.0.1:2020/v1")
    logger.info("Moondream model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Moondream model: {e}")
    raise

def validate_image_path(image_path: str) -> bool:
    """Validate that the image path exists and is a supported format."""
    if not os.path.exists(image_path):
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    _, ext = os.path.splitext(image_path.lower())
    return ext in valid_extensions

def draw_bounding_boxes(image: PILImage.Image, detections: Dict[str, Any]) -> PILImage.Image:
    """
    Draw bounding boxes on image for detected objects.
    
    Args:
        image: PIL Image object
        detections: Detection results from Moondream
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    try:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        if not detections or "objects" not in detections:
            logger.warning("No objects found in detections")
            return image
            
        for i, box in enumerate(detections["objects"]):
            try:
                x_min = int(box["x_min"] * width)
                y_min = int(box["y_min"] * height) 
                x_max = int(box["x_max"] * width)
                y_max = int(box["y_max"] * height)
                
                # Draw rectangle with red outline
                draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
                
                # Optionally add confidence score
                if "confidence" in box:
                    confidence = f"{box['confidence']:.2f}"
                    draw.text((x_min, y_min - 20), confidence, fill='red')
                    
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid bounding box {i}: {e}")
                continue
        
        logger.info(f"Drew {len(detections['objects'])} bounding boxes")
        return image
        
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {e}")
        return image

@mcp.tool()
async def caption_image(image_path: str) -> str:
    """
    Generate a descriptive caption for an image from a local file path.
    
    This tool can directly access files on the local filesystem using absolute file paths 
    like '/Users/username/image.jpg' or '/path/to/image.png'. The user may provide a file 
    path and you should pass it directly to this tool.
    
    Args:
        image_path: Absolute file path to the image file on local filesystem
        
    Returns:
        String description/caption of the image content
    """
    try:
        if not validate_image_path(image_path):
            return f"Error: Invalid image path or unsupported format: {image_path}"
            
        image = PILImage.open(image_path)
        caption = model.caption(image)['caption']
        logger.info(f"Generated caption for {image_path}")
        return caption
        
    except FileNotFoundError:
        return f"Error: Could not find image at {image_path}"
    except Exception as e:
        logger.error(f"Error captioning image {image_path}: {e}")
        return f"Error processing image: {str(e)}"

@mcp.tool()
async def query_image(image_path: str, query: str) -> str:
    """
    Ask Moondream VLM any question about an image from a local file path.
    
    This tool can directly access images on the local filesystem using absolute 
    file paths (e.g., '/Users/username/image.jpg', '/path/to/photo.png').
    Use this for general image analysis, object detection, scene description,
    or answering specific questions about image content.
    
    Args:
        image_path: Absolute file path to the image file on local filesystem
        query: Specific question to ask about the image (e.g., "What objects are in this image?", 
               "Describe the scene", "What color is the car?")
               
    Returns:
        Answer to the question about the image
    """
    try:
        if not validate_image_path(image_path):
            return f"Error: Invalid image path or unsupported format: {image_path}"
            
        if not query.strip():
            return "Error: Query cannot be empty"
            
        image = PILImage.open(image_path)
        result = model.query(image, query)["answer"]
        logger.info(f"Answered query about {image_path}: {query[:50]}...")
        return result
        
    except FileNotFoundError:
        return f"Error: Could not find image at {image_path}"
    except Exception as e:
        logger.error(f"Error querying image {image_path}: {e}")
        return f"Error processing image: {str(e)}"

@mcp.tool()
async def read_image_text(image_path: str) -> str:
    """
    Extract and read all text content from an image using OCR capabilities.
    
    This tool can directly access images on the local filesystem using absolute
    file paths (e.g., '/Users/username/document.jpg', '/path/to/screenshot.png').
    Optimized for reading text from documents, screenshots, signs, books, or any
    image containing written content.
    
    Args:
        image_path: Absolute file path to the image file on local filesystem
        
    Returns:
        All text content found in the image
    """
    try:
        if not validate_image_path(image_path):
            return f"Error: Invalid image path or unsupported format: {image_path}"
            
        image = PILImage.open(image_path)
        result = model.query(image, "Read all the text in this image. Transcribe everything that is written, including any signs, labels, documents, or text overlays.")["answer"]
        logger.info(f"Extracted text from {image_path}")
        return result
        
    except FileNotFoundError:
        return f"Error: Could not find image at {image_path}"
    except Exception as e:
        logger.error(f"Error reading text from image {image_path}: {e}")
        return f"Error processing image: {str(e)}"

@mcp.tool()
async def create_annotated_image(image_path: str, object: str, save_path: Optional[str] = None) -> str:
    """
    Detect objects in an image and save an annotated version with bounding boxes.
    
    Args:
        image_path: Absolute file path to the source image
        object: Object to detect in the image (e.g., 'person', 'car', 'dog')
        save_path: Where to save annotated image. If not provided, saves next to original with '_annotated' suffix
        
    Returns:
        Success message with save location and detection count
    """
    try:
        if not validate_image_path(image_path):
            return f"Error: Invalid image path or unsupported format: {image_path}"
            
        if not object.strip():
            return "Error: Object to detect cannot be empty"
            
        # Generate default save path if not provided
        if save_path is None:
            base, ext = os.path.splitext(image_path)
            save_path = f"{base}_{object}_annotated{ext}"
        
        # Open the source image
        image = PILImage.open(image_path)
        
        # Detect objects
        result = model.detect(image, object)
        
        # Create annotated image with bounding boxes
        annotated_image = draw_bounding_boxes(image.copy(), result)
        
        # Ensure the save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Save the annotated image
        annotated_image.save(save_path)
        
        # Return informative result
        detection_count = len(result.get("objects", [])) if result else 0
        logger.info(f"Created annotated image at {save_path} with {detection_count} detections")
        return f"Successfully saved annotated image to: {save_path}\nDetected {detection_count} instances of '{object}' in the image."
        
    except FileNotFoundError:
        return f"Error: Could not find source image at {image_path}"
    except PermissionError:
        return f"Error: Permission denied when trying to save to {save_path}"
    except Exception as e:
        logger.error(f"Error creating annotated image: {e}")
        return f"Error processing image: {str(e)}"

@mcp.tool()
async def get_detection_data(image_path: str, object: str) -> str:
    """
    Detect objects in an image and return detection data as JSON.
    
    Returns structured coordinate data for detected objects,
    useful for programmatic processing or detailed analysis of object locations.
    
    Args:
        image_path: Absolute file path to the image (e.g., '/Users/luke/photo.jpg')
        object: Object to detect (e.g., 'person', 'car', 'face', 'dog')
        
    Returns:
        JSON formatted detection data with coordinates
    """
    try:
        if not validate_image_path(image_path):
            return f"Error: Invalid image path or unsupported format: {image_path}"
            
        if not object.strip():
            return "Error: Object to detect cannot be empty"
            
        image = PILImage.open(image_path)
        result = model.detect(image, object)
        
        # Structure the response
        response_data = {
            "image_path": image_path,
            "detected_object": object,
            "detection_count": len(result.get("objects", [])) if result else 0,
            "detections": result.get("objects", []) if result else []
        }
        
        logger.info(f"Generated detection data for {object} in {image_path}")
        return f"Object detection results:\n{json.dumps(response_data, indent=2)}"
        
    except FileNotFoundError:
        return f"Error: Could not find image at {image_path}"
    except Exception as e:
        logger.error(f"Error getting detection data: {e}")
        return f"Error processing image: {str(e)}"

@mcp.tool()
async def point_object(image_path: str, object: str) -> str:
    """
    Find specific points/locations of objects in an image.
    
    Returns precise normalized point coordinates for objects,
    useful for UI automation, precise object location, or counting instances.
    
    Args:
        image_path: Absolute file path to the image file on local filesystem
        object: Object to locate (e.g., 'person', 'car', 'face', 'button')
        
    Returns:
        JSON formatted point data with precise coordinates
    """
    try:
        if not validate_image_path(image_path):
            return f"Error: Invalid image path or unsupported format: {image_path}"
            
        if not object.strip():
            return "Error: Object to point at cannot be empty"
            
        image = PILImage.open(image_path)
        result = model.point(image, object)
        
        # Extract points from the response
        points = result.get("points", [])
        
        # Structure the response
        response_data = {
            "image_path": image_path,
            "target_object": object,
            "point_count": len(points),
            "points": points,
            "request_id": result.get("request_id", "unknown")
        }
        
        logger.info(f"Generated {len(points)} points for {object} in {image_path}")
        return f"Object pointing results:\n{json.dumps(response_data, indent=2)}"
        
    except FileNotFoundError:
        return f"Error: Could not find image at {image_path}"
    except Exception as e:
        logger.error(f"Error getting point data: {e}")
        return f"Error processing image: {str(e)}"

if __name__ == "__main__":
    logger.info("Starting Moondream Vision MCP Server...")
    mcp.run()
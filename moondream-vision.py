"""
Moondream Vision MCP Server

An MCP server that provides computer vision capabilities using Moondream VLM.
Enables AI assistants to analyze images, extract text, detect objects, and generate captions
from local image files.

Author: Luke Allen
Version: 1.0.0
"""

from mcp.server.fastmcp import FastMCP
from PIL import Image as PILImage, ImageDraw, ImageOps
import moondream as md
import json
import os
import logging
from typing import Optional, Dict, Any
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("moondream-vision")

# Get allowed access root of image files (defaults to home directory)
ALLOW_ROOT=Path(os.getenv("MD_IMAGE_ROOT", Path.home())).resolve()

# Get Moondream endpoint
endpoint = os.getenv("MOONDREAM_ENDPOINT", "http://127.0.0.1:2020/v1")

# Limit Moondream concurrent calls
SEM = asyncio.Semaphore(int(os.getenv("MD_MAX_CONCURRENCY", "4")))

PILImage.MAX_IMAGE_PIXELS = int(os.getenv("MD_MAX_PIXELS", "178956970"))

# Currently supported Moondream filetypes
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Initialize Moondream model
try:
    model = md.vl(endpoint=endpoint)
    logger.info("Moondream model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Moondream model: {e}")
    raise

def safe_open(image_path: str) -> PILImage.Image:
    with PILImage.open(image_path) as im:
        im.verify()
    
    im = PILImage.open(image_path)
    return ImageOps.exif_transpose(im)

def validate_image_path(image_path: str, must_exist: bool = True) -> bool:
    """Validate that the image path exists and is a supported format."""
    path = Path(image_path)
    if not path.is_absolute():
        return False
    
    p = path.expanduser().resolve()
    
    if must_exist:
        path_check = p.exists()
    else:
        path_check = p.parent.exists()

    if not (str(p).startswith(str(ALLOW_ROOT)) and path_check):
        return False
    
    valid_extensions = SUPPORTED_EXTENSIONS 
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
                x_min = max(0, min(int(box["x_min"] * width), width-1))
                x_max = max(0, min(int(box["x_max"] * width), width-1))
                y_min = max(0, min(int(box["y_min"] * height), height-1))
                y_max = max(0, min(int(box["y_max"] * height), height-1))
                
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
async def caption_image(image_path: str) -> dict:
    """
    Generate a descriptive caption for an image from a local file path.
    
    This tool can directly access files on the local filesystem using absolute file paths 
    like '/Users/username/image.jpg' or '/path/to/image.png'. The user may provide a file 
    path and you should pass it directly to this tool.
    
    Args:
        image_path: Absolute file path to the image file on local filesystem
        
    Returns:
        JSON message with a result value of either success or error and a value with 
        either the caption or the error message.
    """
    try:
        if not validate_image_path(image_path):
            return {
                "success": False,
                "data": None,
                "error": f"Invalid image path or unsupported format: {image_path}"
            }
            
        image = await asyncio.to_thread(safe_open,image_path)
        async with SEM:
            caption = await asyncio.to_thread(lambda: model.caption(image)['caption'])
        logger.info(f"Generated caption for {image_path}")
        image.close()
        return {
            "success": True,
            "data": caption,
            "error": None
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "data": None, 
            "error": f"Failed to find the file: {image_path}"
        }
    except Exception as e:
        logger.error(f"Error captioning image {image_path}: {e}")
        return {
            "success": False,
            "data": None, 
            "error": f"Could not caption the image: {str(e)}"
        }

@mcp.tool()
async def query_image(image_path: str, query: str) -> dict:
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
            return {
                "success": False,
                "data": None,
                "error": f"Invalid image path or unsupported format: {image_path}"
            }
            
        if not query.strip():
            return {
                "success": False,
                "data": None,
                "error": "Query cannot be empty"
            }
            
        image = await asyncio.to_thread(safe_open,image_path)
        async with SEM:
            result = await asyncio.to_thread(lambda: model.query(image, query)["answer"])
        logger.info(f"Answered query about {image_path}: {query[:50]}...")
        image.close()
        return {
            "success": True,
            "data": result,
            "error": None
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "error": f"Could not find image at {image_path}"
        }
    except Exception as e:
        logger.error(f"Error querying image {image_path}: {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Error processing image: {str(e)}"
        }

@mcp.tool()
async def read_image_text(image_path: str) -> dict:
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
            return {
                "success": False,
                "data": None,
                "error": f"Invalid image path or unsupported format: {image_path}"
            }
            
        image = await asyncio.to_thread(safe_open,image_path)
        async with SEM:
            result = await asyncio.to_thread(lambda: model.query(
                image, 
                "Read all the text in this image. Transcribe everything that is written, " \
                "including any signs, labels, documents, or text overlays.")["answer"]
                )
        logger.info(f"Extracted text from {image_path}")
        image.close()
        return {
            "success": True,
            "data": result,
            "error": None
        }

    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "error": f"Could not find image at {image_path}"
        }
    except Exception as e:
        logger.error(f"Error querying image {image_path}: {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Error processing image: {str(e)}"
        }

@mcp.tool()
async def create_annotated_image(image_path: str, target_object: str, save_path: Optional[str] = None) -> dict:
    """
    Detect objects in an image and save an annotated version with bounding boxes.
    
    Args:
        image_path: Absolute file path to the source image
        target_object: Object to detect in the image (e.g., 'person', 'car', 'dog')
        save_path: Where to save annotated image. If not provided, saves next to original with '_annotated' suffix
        
    Returns:
        Success message with save location and detection count
    """
    try:
        if not validate_image_path(image_path):
            return {
                "success": False,
                "data": None,
                "error": f"Invalid image path or unsupported format: {image_path}"
            }
            
        if not target_object.strip():
            return {
                "success": False,
                "data": None,
                "error": "Object to detect cannot be empty"
            }
            
        # Generate default save path if not provided
        if save_path is None:
            base, ext = os.path.splitext(image_path)
            save_path = f"{base}_{target_object}_annotated{ext}"
        
        if not validate_image_path(save_path, must_exist=False):
            return {
                "success": False,
                "data": None,
                "error": f"Invalid save path or unsupported save format: {save_path}"
            }
        
        # Open the source image
        image = await asyncio.to_thread(safe_open,image_path)
        
        # Detect objects
        async with SEM:
            result = await asyncio.to_thread(lambda: model.detect(image, target_object))
        
        # Create annotated image with bounding boxes
        annotated_image = draw_bounding_boxes(image.copy(), result)
        
        # Ensure the save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # If image is jpg, make sure it doesn't have alpha
        if save_path.lower().endswith((".jpg", ".jpeg")) and annotated_image.mode in ("RGBA", "LA"):
            annotated_image = annotated_image.convert("RGB")
        
        # Save the annotated image
        await asyncio.to_thread(annotated_image.save, save_path)
        
        # Return informative result
        detection_count = len(result.get("objects", [])) if result else 0
        logger.info(f"Created annotated image at {save_path} with {detection_count} detections")
        image.close()
        return {
            "success": True,
            "data": f"Successfully saved annotated image to: {save_path}\nDetected {detection_count} instances of '{target_object}' in the image.",
            "error": None
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "error": f"Could not find image at {image_path}"
        }
    except PermissionError:
        return {
            "success": False,
            "data": None,
            "error": f"Permission denied when trying to save to {save_path}"
        }
    except Exception as e:
        logger.error(f"Error querying image {image_path}: {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Error processing image: {str(e)}"
        }

@mcp.tool()
async def get_detection_data(image_path: str, target_object: str) -> dict:
    """
    Detect objects in an image and return detection data as JSON.
    
    Returns structured coordinate data for detected objects,
    useful for programmatic processing or detailed analysis of object locations.
    
    Args:
        image_path: Absolute file path to the image (e.g., '/Users/luke/photo.jpg')
        target_object: Object to detect (e.g., 'person', 'car', 'face', 'dog')
        
    Returns:
        JSON formatted detection data with coordinates
    """
    try:
        if not validate_image_path(image_path):
            return {
                "success": False,
                "data": None,
                "error": f"Invalid image path or unsupported format: {image_path}"
            }
            
        if not target_object.strip():
            return {
                "success": False,
                "data": None,
                "error": "Object to detect cannot be empty"
            }
            
        image = await asyncio.to_thread(safe_open,image_path)
        async with SEM:
            result = await asyncio.to_thread(lambda: model.detect(image, target_object))
        
        # Structure the response
        response_data = {
            "image_path": image_path,
            "detected_object": target_object,
            "detection_count": len(result.get("objects", [])) if result else 0,
            "detections": result.get("objects", []) if result else []
        }
        
        logger.info(f"Generated detection data for {target_object} in {image_path}")
        image.close()
        return {
            "success": True,
            "data": response_data,
            "error": None
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "error": f"Could not find image at {image_path}"
        }
    except Exception as e:
        logger.error(f"Error querying image {image_path}: {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Error processing image: {str(e)}"
        }

@mcp.tool()
async def point_object(image_path: str, target_object: str) -> dict:
    """
    Find specific points/locations of objects in an image.
    
    Returns precise normalized point coordinates for objects,
    useful for UI automation, precise object location, or counting instances.
    
    Args:
        image_path: Absolute file path to the image file on local filesystem
        target_object: Object to locate (e.g., 'person', 'car', 'face', 'button')
        
    Returns:
        JSON formatted point data with precise coordinates
    """
    try:
        if not validate_image_path(image_path):
            return {
                "success": False,
                "data": None,
                "error": f"Invalid image path or unsupported format: {image_path}"
            }
            
        if not target_object.strip():
            return {
                "success": False,
                "data": None,
                "error": "Object to point at cannot be empty"
            } 

        image = await asyncio.to_thread(safe_open,image_path)
        async with SEM:
            result = await asyncio.to_thread(lambda: model.point(image, target_object))
        
        # Extract points from the response
        points = result.get("points", [])
        
        # Structure the response
        response_data = {
            "image_path": image_path,
            "target_object": target_object,
            "point_count": len(points),
            "points": points,
            "request_id": result.get("request_id", "unknown")
        }
        
        logger.info(f"Generated {len(points)} points for {target_object} in {image_path}")
        image.close()
        return {
            "success": True,
            "data": response_data,
            "error": None
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "error": f"Could not find image at {image_path}"
        }
    except Exception as e:
        logger.error(f"Error querying image {image_path}: {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Error processing image: {str(e)}"
        }

if __name__ == "__main__":
    logger.info("Starting Moondream Vision MCP Server...")
    mcp.run()

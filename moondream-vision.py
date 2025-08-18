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
    
    This tool uses Moondream VLM to analyze images and generate natural language
    descriptions. Supports common image formats and provides detailed captions
    describing the scene, objects, and context within the image.
    
    Args:
        image_path (str): Absolute file path to the image file on local filesystem.
                         Must be within the configured ALLOW_ROOT directory.
                         Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp
        
    Returns:
        dict: Structured response containing:
            - success (bool): True if caption generated successfully
            - data (str|None): Generated image caption or None on error  
            - error (str|None): Error message or None on success
            
    Raises:
        FileNotFoundError: Image file not found or inaccessible
        PermissionError: Insufficient permissions to read file
        ValueError: Unsupported image format or corrupted file
        
    Examples:
        >>> result = await caption_image("/Users/luke/vacation.jpg")
        >>> if result["success"]:
        ...     print(result["data"])  # "A beautiful sunset over the ocean with people walking on the beach"
        
        >>> # Error case
        >>> result = await caption_image("/invalid/path.jpg")
        >>> print(result["error"])  # "Invalid image path or unsupported format: /invalid/path.jpg"
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
    Ask specific questions about an image using natural language queries.
    
    This tool enables interactive analysis of images by asking targeted questions.
    Use this for detailed inspection, object counting, color identification,
    spatial relationships, or any specific visual analysis task.
    
    Args:
        image_path (str): Absolute file path to the image file on local filesystem.
                         Must be within the configured ALLOW_ROOT directory.
                         Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp
        query (str): Natural language question about the image content.
                    Cannot be empty or whitespace-only.
        
    Returns:
        dict: Structured response containing:
            - success (bool): True if query answered successfully
            - data (str|None): Answer to the question or None on error  
            - error (str|None): Error message or None on success
            
    Raises:
        FileNotFoundError: Image file not found or inaccessible
        PermissionError: Insufficient permissions to read file
        ValueError: Empty query or unsupported image format
        
    Examples:
        >>> result = await query_image("/Users/luke/street.jpg", "How many cars are visible?")
        >>> if result["success"]:
        ...     print(result["data"])  # "I can see 3 cars in the image - two sedans and one SUV"
        
        >>> result = await query_image("/Users/luke/room.jpg", "What color are the walls?")
        >>> print(result["data"])  # "The walls appear to be painted in a light blue color"
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
    Extract and transcribe all text content from an image using OCR capabilities.
    
    This tool is optimized for reading text from documents, screenshots, signs,
    books, handwritten notes, or any image containing written content. It can
    handle multiple text orientations and various fonts.
    
    Args:
        image_path (str): Absolute file path to the image file on local filesystem.
                         Must be within the configured ALLOW_ROOT directory.
                         Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp
        
    Returns:
        dict: Structured response containing:
            - success (bool): True if text extraction completed successfully
            - data (str|None): All extracted text content or None on error  
            - error (str|None): Error message or None on success
            
    Raises:
        FileNotFoundError: Image file not found or inaccessible
        PermissionError: Insufficient permissions to read file
        ValueError: Unsupported image format or corrupted file
        
    Examples:
        >>> result = await read_image_text("/Users/luke/document.png")
        >>> if result["success"]:
        ...     print(result["data"])  # "Invoice #12345\nDate: 2024-01-15\nTotal: $299.99"
        
        >>> result = await read_image_text("/Users/luke/sign.jpg")
        >>> print(result["data"])  # "STOP\nDo Not Enter\nAuthorized Personnel Only"
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
    
    This tool performs object detection and creates a new image with red bounding
    boxes drawn around detected instances of the specified object. Useful for
    visual verification of detection results and creating annotated datasets.
    
    Args:
        image_path (str): Absolute file path to the source image file.
                         Must be within the configured ALLOW_ROOT directory.
                         Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp
        target_object (str): Object type to detect and annotate (e.g., 'person', 'car', 'dog').
                           Cannot be empty or whitespace-only.
        save_path (str, optional): Absolute path where annotated image will be saved.
                                 If None, saves next to original with '_annotated' suffix.
                                 Must be within the configured ALLOW_ROOT directory.
        
    Returns:
        dict: Structured response containing:
            - success (bool): True if annotation completed and saved successfully
            - data (str|None): Success message with save location and detection count, or None on error  
            - error (str|None): Error message or None on success
            
    Raises:
        FileNotFoundError: Source image file not found or inaccessible
        PermissionError: Insufficient permissions to read source or write to save location
        ValueError: Empty target_object or unsupported image format
        
    Examples:
        >>> result = await create_annotated_image("/Users/luke/street.jpg", "car")
        >>> if result["success"]:
        ...     print(result["data"])  
        # "Successfully saved annotated image to: /Users/luke/street_car_annotated.jpg
        #  Detected 3 instances of 'car' in the image."
        
        >>> # Custom save path
        >>> result = await create_annotated_image(
        ...     "/Users/luke/photo.jpg", 
        ...     "person", 
        ...     "/Users/luke/annotated/people_detected.jpg"
        ... )
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
    Detect objects in an image and return structured coordinate data as JSON.
    
    This tool provides programmatic access to object detection results, returning
    precise bounding box coordinates and metadata. Useful for automated analysis,
    data processing pipelines, or integration with other computer vision tools.
    
    Args:
        image_path (str): Absolute file path to the image file.
                         Must be within the configured ALLOW_ROOT directory.
                         Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp
        target_object (str): Object type to detect (e.g., 'person', 'car', 'face', 'dog').
                           Cannot be empty or whitespace-only.
        
    Returns:
        dict: Structured response containing:
            - success (bool): True if detection completed successfully
            - data (dict|None): Detection results with structure:
                - image_path (str): Path to analyzed image
                - detected_object (str): Object type that was searched for
                - detection_count (int): Number of objects found
                - detections (list): List of detection objects with coordinates
            - error (str|None): Error message or None on success
            
    Raises:
        FileNotFoundError: Image file not found or inaccessible
        PermissionError: Insufficient permissions to read file
        ValueError: Empty target_object or unsupported image format
        
    Examples:
        >>> result = await get_detection_data("/Users/luke/crowd.jpg", "person")
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Found {data['detection_count']} people")
        ...     for detection in data["detections"]:
        ...         print(f"Person at: {detection['x_min']}, {detection['y_min']}")
        
        >>> # Result structure example:
        # {
        #   "success": True,
        #   "data": {
        #     "image_path": "/Users/luke/crowd.jpg",
        #     "detected_object": "person", 
        #     "detection_count": 2,
        #     "detections": [
        #       {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.8, "confidence": 0.95},
        #       {"x_min": 0.6, "y_min": 0.1, "x_max": 0.8, "y_max": 0.9, "confidence": 0.87}
        #     ]
        #   },
        #   "error": None
        # }
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
    Find precise point coordinates for objects in an image.
    
    This tool returns normalized point coordinates (center points) for detected objects,
    useful for UI automation, click targeting, object counting, or spatial analysis.
    Points are returned as normalized coordinates (0.0-1.0) relative to image dimensions.
    
    Args:
        image_path (str): Absolute file path to the image file.
                         Must be within the configured ALLOW_ROOT directory.
                         Supported formats: .jpg, .jpeg, .png, .gif, .bmp, .webp
        target_object (str): Object type to locate (e.g., 'person', 'button', 'face', 'car').
                           Cannot be empty or whitespace-only.
        
    Returns:
        dict: Structured response containing:
            - success (bool): True if pointing completed successfully
            - data (dict|None): Pointing results with structure:
                - image_path (str): Path to analyzed image
                - target_object (str): Object type that was searched for
                - point_count (int): Number of points found
                - points (list): List of normalized coordinate points [x, y]
                - request_id (str): Unique identifier for this request
            - error (str|None): Error message or None on success
            
    Raises:
        FileNotFoundError: Image file not found or inaccessible
        PermissionError: Insufficient permissions to read file
        ValueError: Empty target_object or unsupported image format
        
    Examples:
        >>> result = await point_object("/Users/luke/interface.png", "button")
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Found {data['point_count']} buttons")
        ...     for point in data["points"]:
        ...         x, y = point
        ...         print(f"Button center at: ({x:.3f}, {y:.3f})")
        
        >>> # Convert to pixel coordinates
        >>> image_width, image_height = 1920, 1080
        >>> for point in data["points"]:
        ...     pixel_x = int(point[0] * image_width)
        ...     pixel_y = int(point[1] * image_height)
        ...     print(f"Pixel coordinates: ({pixel_x}, {pixel_y})")
        
        >>> # Result structure example:
        # {
        #   "success": True,
        #   "data": {
        #     "image_path": "/Users/luke/interface.png",
        #     "target_object": "button",
        #     "point_count": 3,
        #     "points": [[0.25, 0.75], [0.5, 0.25], [0.8, 0.6]],
        #     "request_id": "abc123"
        #   },
        #   "error": None
        # }
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

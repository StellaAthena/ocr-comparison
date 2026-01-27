"""Utility functions for OCR comparison system."""

from pathlib import Path
from typing import List, Tuple, Union
import json

from PIL import Image
import numpy as np

from .models import BoundingBox, OCRWord


def load_ground_truth(path: Union[str, Path]) -> dict:
    """Load ground truth from a JSON file.

    Expected format:
    {
        "text": "full text content",
        "boxes": [
            {"x": 10, "y": 20, "width": 100, "height": 30, "text": "word"},
            ...
        ]
    }

    Args:
        path: Path to JSON file

    Returns:
        Dictionary with 'text' and optional 'boxes'
    """
    with open(path) as f:
        data = json.load(f)

    result = {'text': data.get('text', '')}

    if 'boxes' in data:
        result['boxes'] = [
            BoundingBox(
                x=b['x'],
                y=b['y'],
                width=b['width'],
                height=b['height']
            )
            for b in data['boxes']
        ]

    return result


def save_ground_truth(
    path: Union[str, Path],
    text: str,
    boxes: List[BoundingBox] = None
) -> None:
    """Save ground truth to a JSON file.

    Args:
        path: Path to save file
        text: Ground truth text
        boxes: Optional list of bounding boxes
    """
    data = {'text': text}

    if boxes:
        data['boxes'] = [
            {
                'x': b.x,
                'y': b.y,
                'width': b.width,
                'height': b.height
            }
            for b in boxes
        ]

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def preprocess_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    grayscale: bool = False,
    resize: Tuple[int, int] = None,
    denoise: bool = False,
    threshold: bool = False
) -> np.ndarray:
    """Preprocess image for better OCR results.

    Args:
        image: Input image (path, PIL Image, or numpy array)
        grayscale: Convert to grayscale
        resize: Target size as (width, height)
        denoise: Apply denoising (requires opencv)
        threshold: Apply adaptive thresholding

    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image

    # Resize if specified
    if resize:
        img = img.resize(resize, Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Convert to grayscale
    if grayscale and len(img_array.shape) == 3:
        # Use PIL for conversion
        img = Image.fromarray(img_array).convert('L')
        img_array = np.array(img)

    # Advanced preprocessing requires OpenCV
    if denoise or threshold:
        try:
            import cv2

            if denoise:
                if len(img_array.shape) == 2:
                    img_array = cv2.fastNlMeansDenoising(img_array)
                else:
                    img_array = cv2.fastNlMeansDenoisingColored(img_array)

            if threshold:
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                img_array = cv2.adaptiveThreshold(
                    img_array, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

        except ImportError:
            import warnings
            warnings.warn(
                "OpenCV not available for advanced preprocessing. "
                "Install with: pip install opencv-python"
            )

    return img_array


def crop_to_text_region(
    image: Union[str, Path, Image.Image],
    words: List[OCRWord],
    padding: int = 10
) -> Image.Image:
    """Crop image to the region containing detected text.

    Args:
        image: Input image
        words: List of detected words
        padding: Padding around detected region

    Returns:
        Cropped PIL Image
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image

    if not words:
        return img

    # Find bounding region of all text
    x_min = min(w.bbox.x for w in words)
    y_min = min(w.bbox.y for w in words)
    x_max = max(w.bbox.x2 for w in words)
    y_max = max(w.bbox.y2 for w in words)

    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img.width, x_max + padding)
    y_max = min(img.height, y_max + padding)

    return img.crop((x_min, y_min, x_max, y_max))


def merge_nearby_boxes(
    words: List[OCRWord],
    distance_threshold: int = 10,
    merge_text: bool = True
) -> List[OCRWord]:
    """Merge horizontally adjacent word boxes into lines.

    Args:
        words: List of OCR words
        distance_threshold: Max horizontal gap to merge
        merge_text: Whether to concatenate text

    Returns:
        List of merged OCRWord objects
    """
    if not words:
        return []

    # Sort by y position (line), then x position
    sorted_words = sorted(words, key=lambda w: (w.bbox.y, w.bbox.x))

    merged = []
    current_line = [sorted_words[0]]

    for word in sorted_words[1:]:
        last = current_line[-1]

        # Check if on same line (similar y)
        y_overlap = (
            word.bbox.y < last.bbox.y2 and
            word.bbox.y2 > last.bbox.y
        )

        # Check horizontal distance
        h_gap = word.bbox.x - last.bbox.x2

        if y_overlap and h_gap <= distance_threshold:
            current_line.append(word)
        else:
            # Merge current line
            merged.append(_merge_word_list(current_line, merge_text))
            current_line = [word]

    # Don't forget last line
    if current_line:
        merged.append(_merge_word_list(current_line, merge_text))

    return merged


def _merge_word_list(words: List[OCRWord], merge_text: bool) -> OCRWord:
    """Merge a list of words into a single word.

    Args:
        words: List of words to merge
        merge_text: Whether to concatenate text

    Returns:
        Single merged OCRWord
    """
    if len(words) == 1:
        return words[0]

    x_min = min(w.bbox.x for w in words)
    y_min = min(w.bbox.y for w in words)
    x_max = max(w.bbox.x2 for w in words)
    y_max = max(w.bbox.y2 for w in words)

    merged_bbox = BoundingBox(
        x=x_min,
        y=y_min,
        width=x_max - x_min,
        height=y_max - y_min
    )

    if merge_text:
        merged_text = ' '.join(w.text for w in words)
    else:
        merged_text = words[0].text

    avg_confidence = sum(w.confidence for w in words) / len(words)

    return OCRWord(
        text=merged_text,
        bbox=merged_bbox,
        confidence=avg_confidence
    )


def find_image_files(
    directory: Union[str, Path],
    extensions: List[str] = None,
    recursive: bool = True
) -> List[Path]:
    """Find all image files in a directory.

    Args:
        directory: Directory to search
        extensions: File extensions to include (default: common image formats)
        recursive: Whether to search subdirectories

    Returns:
        List of image file paths
    """
    directory = Path(directory)
    extensions = extensions or ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp']

    # Normalize extensions
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                  for ext in extensions]

    pattern = '**/*' if recursive else '*'
    files = []

    for ext in extensions:
        files.extend(directory.glob(f'{pattern}{ext}'))
        files.extend(directory.glob(f'{pattern}{ext.upper()}'))

    return sorted(set(files))


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1.23s", "45.6ms")
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}us"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def calculate_scale_factor(
    original_size: Tuple[int, int],
    target_size: Tuple[int, int]
) -> float:
    """Calculate scale factor to fit image within target size while preserving aspect ratio.

    Args:
        original_size: Original (width, height)
        target_size: Target (width, height)

    Returns:
        Scale factor
    """
    width_ratio = target_size[0] / original_size[0]
    height_ratio = target_size[1] / original_size[1]
    return min(width_ratio, height_ratio)


def resize_with_aspect_ratio(
    image: Union[str, Path, Image.Image],
    max_width: int = None,
    max_height: int = None
) -> Image.Image:
    """Resize image while preserving aspect ratio.

    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height

    Returns:
        Resized PIL Image
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image.copy()

    width, height = img.size

    if max_width and width > max_width:
        ratio = max_width / width
        width = max_width
        height = int(height * ratio)

    if max_height and height > max_height:
        ratio = max_height / height
        height = max_height
        width = int(width * ratio)

    if (width, height) != img.size:
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    return img

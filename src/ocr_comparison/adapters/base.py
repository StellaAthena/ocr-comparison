"""Abstract base class for OCR adapters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import time

from PIL import Image
import numpy as np

from ..models import OCRResult


class BaseOCRAdapter(ABC):
    """Abstract base class for OCR engine adapters."""

    def __init__(self, **kwargs):
        """Initialize the adapter with engine-specific options."""
        self.options = kwargs
        self._engine = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the OCR engine."""
        pass

    @abstractmethod
    def _initialize_engine(self) -> None:
        """Initialize the underlying OCR engine."""
        pass

    @abstractmethod
    def _process(self, image: np.ndarray) -> OCRResult:
        """Process an image and return OCR results.

        Args:
            image: Image as numpy array (RGB format)

        Returns:
            OCRResult containing detected words with bounding boxes
        """
        pass

    def process(self, image: Union[str, Path, Image.Image, np.ndarray]) -> OCRResult:
        """Process an image and return OCR results.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            OCRResult containing detected words with bounding boxes
        """
        # Ensure engine is initialized
        if self._engine is None:
            self._initialize_engine()

        # Load and convert image to numpy array
        img_array, image_size = self._load_image(image)

        # Process and time the operation
        start_time = time.time()
        result = self._process(img_array)
        processing_time = time.time() - start_time

        # Update result with timing and size info
        result.processing_time = processing_time
        result.image_size = image_size

        return result

    def _load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> tuple:
        """Load image and convert to numpy array.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            Tuple of (numpy array in RGB format, (width, height))
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            # Assume it's already in correct format
            if len(image.shape) == 2:
                # Grayscale - convert to RGB
                pil_image = Image.fromarray(image).convert('RGB')
            elif image.shape[2] == 4:
                # RGBA - convert to RGB
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                return image, (image.shape[1], image.shape[0])
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        return np.array(pil_image), pil_image.size

    def is_available(self) -> bool:
        """Check if the OCR engine is available/installed."""
        try:
            self._initialize_engine()
            return True
        except Exception:
            return False

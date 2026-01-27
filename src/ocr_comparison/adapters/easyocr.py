"""EasyOCR adapter."""

from typing import List, Tuple
import numpy as np

from .base import BaseOCRAdapter
from ..models import BoundingBox, OCRWord, OCRResult


class EasyOCRAdapter(BaseOCRAdapter):
    """Adapter for EasyOCR engine."""

    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = True,
        **kwargs
    ):
        """Initialize EasyOCR adapter.

        Args:
            languages: List of language codes (default: ['en'])
            gpu: Whether to use GPU if available
            **kwargs: Additional options passed to base class
        """
        super().__init__(**kwargs)
        self.languages = languages or ['en']
        self.gpu = gpu

    @property
    def name(self) -> str:
        return "easyocr"

    def _initialize_engine(self) -> None:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            self._engine = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
        except ImportError:
            raise ImportError(
                "easyocr is not installed. "
                "Install with: pip install easyocr"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR: {e}")

    def _process(self, image: np.ndarray) -> OCRResult:
        """Process image with EasyOCR.

        Args:
            image: Image as numpy array (RGB format)

        Returns:
            OCRResult with detected words and bounding boxes
        """
        # EasyOCR readtext returns list of (bbox, text, confidence)
        # bbox is 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        results = self._engine.readtext(image)

        words = self._extract_words(results)
        image_size = (image.shape[1], image.shape[0])

        return OCRResult(
            words=words,
            engine_name=self.name,
            processing_time=0.0,  # Will be set by base class
            image_size=image_size
        )

    def _extract_words(self, results: List[Tuple]) -> List[OCRWord]:
        """Extract words from EasyOCR output.

        Args:
            results: List of (bbox, text, confidence) tuples from readtext()

        Returns:
            List of OCRWord objects
        """
        words = []

        for bbox_points, text, confidence in results:
            text = text.strip()
            if not text:
                continue

            # Convert 4-point polygon to axis-aligned bounding box
            bbox = self._polygon_to_bbox(bbox_points)

            words.append(OCRWord(
                text=text,
                bbox=bbox,
                confidence=float(confidence)
            ))

        return words

    def _polygon_to_bbox(self, points: List[List[float]]) -> BoundingBox:
        """Convert 4-point polygon to axis-aligned bounding box.

        EasyOCR returns 4 corner points in order:
        top-left, top-right, bottom-right, bottom-left

        Args:
            points: List of 4 [x, y] coordinate pairs

        Returns:
            BoundingBox with axis-aligned rectangle
        """
        # Extract all x and y coordinates
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Get bounding rectangle
        x_min = int(min(xs))
        y_min = int(min(ys))
        x_max = int(max(xs))
        y_max = int(max(ys))

        # Calculate angle from the top edge (for potential future use)
        # Top-left to top-right vector
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        angle = np.degrees(np.arctan2(dy, dx))

        return BoundingBox(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
            angle=angle
        )

    def get_full_text(self, image: np.ndarray) -> str:
        """Get just the extracted text without bounding boxes.

        Args:
            image: Image as numpy array

        Returns:
            Extracted text as string
        """
        if self._engine is None:
            self._initialize_engine()

        results = self._engine.readtext(image)
        return ' '.join(text for _, text, _ in results if text.strip())

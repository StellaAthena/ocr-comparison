"""Tesseract OCR adapter."""

from typing import List
import numpy as np

from .base import BaseOCRAdapter
from ..models import BoundingBox, OCRWord, OCRResult


class TesseractAdapter(BaseOCRAdapter):
    """Adapter for Tesseract OCR engine."""

    def __init__(self, lang: str = 'eng', config: str = '', **kwargs):
        """Initialize Tesseract adapter.

        Args:
            lang: Language code for Tesseract (default: 'eng')
            config: Additional Tesseract configuration string
            **kwargs: Additional options passed to base class
        """
        super().__init__(**kwargs)
        self.lang = lang
        self.config = config

    @property
    def name(self) -> str:
        return "tesseract"

    def _initialize_engine(self) -> None:
        """Import and verify pytesseract is available."""
        try:
            import pytesseract
            # Verify tesseract is actually installed
            pytesseract.get_tesseract_version()
            self._engine = pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is not installed. "
                "Install with: pip install pytesseract"
            )
        except Exception as e:
            raise RuntimeError(
                f"Tesseract OCR is not properly installed: {e}. "
                "Make sure tesseract-ocr is installed on your system."
            )

    def _process(self, image: np.ndarray) -> OCRResult:
        """Process image with Tesseract.

        Args:
            image: Image as numpy array (RGB format)

        Returns:
            OCRResult with detected words and bounding boxes
        """
        from pytesseract import Output

        # Get word-level data
        data = self._engine.image_to_data(
            image,
            lang=self.lang,
            config=self.config,
            output_type=Output.DICT
        )

        words = self._extract_words(data)
        image_size = (image.shape[1], image.shape[0])

        return OCRResult(
            words=words,
            engine_name=self.name,
            processing_time=0.0,  # Will be set by base class
            image_size=image_size
        )

    def _extract_words(self, data: dict) -> List[OCRWord]:
        """Extract words from Tesseract output data.

        Args:
            data: Dictionary output from pytesseract.image_to_data()

        Returns:
            List of OCRWord objects
        """
        words = []
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            text = data['text'][i].strip()

            # Skip empty text
            if not text:
                continue

            # Get confidence (Tesseract returns -1 for certain elements)
            conf = data['conf'][i]
            if conf == -1:
                continue

            # Convert confidence from 0-100 to 0-1
            confidence = conf / 100.0

            # Create bounding box
            bbox = BoundingBox(
                x=data['left'][i],
                y=data['top'][i],
                width=data['width'][i],
                height=data['height'][i]
            )

            words.append(OCRWord(
                text=text,
                bbox=bbox,
                confidence=confidence
            ))

        return words

    def get_full_text(self, image: np.ndarray) -> str:
        """Get just the extracted text without bounding boxes.

        Args:
            image: Image as numpy array

        Returns:
            Extracted text as string
        """
        if self._engine is None:
            self._initialize_engine()

        return self._engine.image_to_string(
            image,
            lang=self.lang,
            config=self.config
        ).strip()

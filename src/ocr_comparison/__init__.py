"""OCR Comparison System - Compare Tesseract and EasyOCR results."""

from .models import BoundingBox, OCRWord, OCRResult, AccuracyMetrics, ComparisonResult
from .comparator import OCRComparator

__version__ = "0.1.0"

__all__ = [
    "BoundingBox",
    "OCRWord",
    "OCRResult",
    "AccuracyMetrics",
    "ComparisonResult",
    "OCRComparator",
]

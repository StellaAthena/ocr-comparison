"""OCR engine adapters."""

from .base import BaseOCRAdapter
from .tesseract import TesseractAdapter
from .easyocr import EasyOCRAdapter

__all__ = ["BaseOCRAdapter", "TesseractAdapter", "EasyOCRAdapter"]

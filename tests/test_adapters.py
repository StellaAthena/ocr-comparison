"""Tests for OCR adapters."""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_comparison.adapters.base import BaseOCRAdapter
from ocr_comparison.adapters.tesseract import TesseractAdapter
from ocr_comparison.adapters.easyocr import EasyOCRAdapter
from ocr_comparison.models import OCRResult


def create_test_image(text: str = "Hello World", size: tuple = (200, 50)) -> Image.Image:
    """Create a simple test image with text."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    # Use default font
    draw.text((10, 10), text, fill='black')
    return img


class TestBaseOCRAdapter:
    """Tests for BaseOCRAdapter."""

    def test_abstract_methods(self):
        """Test that BaseOCRAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseOCRAdapter()

    def test_load_image_from_path(self, tmp_path):
        """Test loading image from file path."""
        # Create a simple adapter subclass for testing
        class DummyAdapter(BaseOCRAdapter):
            @property
            def name(self):
                return "dummy"

            def _initialize_engine(self):
                self._engine = True

            def _process(self, image):
                return OCRResult(
                    words=[],
                    engine_name=self.name,
                    processing_time=0.0,
                    image_size=(image.shape[1], image.shape[0])
                )

        # Create test image
        img_path = tmp_path / "test.png"
        img = create_test_image()
        img.save(img_path)

        adapter = DummyAdapter()
        img_array, size = adapter._load_image(str(img_path))

        assert isinstance(img_array, np.ndarray)
        assert size == (200, 50)
        assert img_array.shape == (50, 200, 3)

    def test_load_image_from_pil(self):
        """Test loading from PIL Image."""
        class DummyAdapter(BaseOCRAdapter):
            @property
            def name(self):
                return "dummy"

            def _initialize_engine(self):
                self._engine = True

            def _process(self, image):
                return OCRResult(
                    words=[],
                    engine_name=self.name,
                    processing_time=0.0,
                    image_size=(image.shape[1], image.shape[0])
                )

        adapter = DummyAdapter()
        pil_img = create_test_image()
        img_array, size = adapter._load_image(pil_img)

        assert isinstance(img_array, np.ndarray)
        assert size == (200, 50)

    def test_load_image_from_array(self):
        """Test loading from numpy array."""
        class DummyAdapter(BaseOCRAdapter):
            @property
            def name(self):
                return "dummy"

            def _initialize_engine(self):
                self._engine = True

            def _process(self, image):
                return OCRResult(
                    words=[],
                    engine_name=self.name,
                    processing_time=0.0,
                    image_size=(image.shape[1], image.shape[0])
                )

        adapter = DummyAdapter()
        np_img = np.zeros((50, 200, 3), dtype=np.uint8)
        img_array, size = adapter._load_image(np_img)

        assert isinstance(img_array, np.ndarray)
        assert size == (200, 50)


class TestTesseractAdapter:
    """Tests for TesseractAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Tesseract adapter."""
        return TesseractAdapter()

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "tesseract"

    @pytest.mark.skipif(
        not TesseractAdapter().is_available(),
        reason="Tesseract not installed"
    )
    def test_process_simple_image(self, adapter, tmp_path):
        """Test processing a simple image with Tesseract."""
        # Create test image
        img_path = tmp_path / "test.png"
        img = create_test_image("Hello World")
        img.save(img_path)

        result = adapter.process(str(img_path))

        assert isinstance(result, OCRResult)
        assert result.engine_name == "tesseract"
        assert result.processing_time > 0
        # Note: OCR results depend on tesseract installation and may vary

    @pytest.mark.skipif(
        not TesseractAdapter().is_available(),
        reason="Tesseract not installed"
    )
    def test_get_full_text(self, adapter, tmp_path):
        """Test get_full_text method."""
        img_path = tmp_path / "test.png"
        img = create_test_image("Simple Test")
        img.save(img_path)

        from PIL import Image as PILImage
        img_array = np.array(PILImage.open(img_path))
        text = adapter.get_full_text(img_array)

        assert isinstance(text, str)


class TestEasyOCRAdapter:
    """Tests for EasyOCRAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create EasyOCR adapter."""
        return EasyOCRAdapter(gpu=False)

    def test_name(self, adapter):
        """Test adapter name."""
        assert adapter.name == "easyocr"

    def test_polygon_to_bbox(self, adapter):
        """Test polygon to bounding box conversion."""
        # Rectangle points: top-left, top-right, bottom-right, bottom-left
        points = [[10, 20], [110, 20], [110, 70], [10, 70]]
        bbox = adapter._polygon_to_bbox(points)

        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50

    def test_polygon_to_bbox_rotated(self, adapter):
        """Test polygon to bbox with rotated text."""
        # Slightly rotated rectangle
        points = [[10, 25], [110, 15], [115, 65], [15, 75]]
        bbox = adapter._polygon_to_bbox(points)

        # Should capture the axis-aligned bounding box
        assert bbox.x == 10
        assert bbox.y == 15
        assert bbox.width == 105
        assert bbox.height == 60

    @pytest.mark.skipif(
        not EasyOCRAdapter(gpu=False).is_available(),
        reason="EasyOCR not installed"
    )
    def test_process_simple_image(self, adapter, tmp_path):
        """Test processing a simple image with EasyOCR."""
        img_path = tmp_path / "test.png"
        img = create_test_image("Hello World")
        img.save(img_path)

        result = adapter.process(str(img_path))

        assert isinstance(result, OCRResult)
        assert result.engine_name == "easyocr"
        assert result.processing_time > 0

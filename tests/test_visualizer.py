"""Tests for OCR visualizer."""

import pytest
from pathlib import Path
from PIL import Image

from ocr_comparison.models import BoundingBox, OCRWord, OCRResult
from ocr_comparison.visualizer import OCRVisualizer, visualize_overlay


def create_test_image(size: tuple = (400, 200)) -> Image.Image:
    """Create a blank test image."""
    return Image.new('RGB', size, color='white')


def create_test_result(engine_name: str = "test") -> OCRResult:
    """Create a test OCR result."""
    words = [
        OCRWord(
            text="Hello",
            bbox=BoundingBox(x=10, y=10, width=50, height=20),
            confidence=0.95
        ),
        OCRWord(
            text="World",
            bbox=BoundingBox(x=70, y=10, width=50, height=20),
            confidence=0.90
        ),
    ]
    return OCRResult(
        words=words,
        engine_name=engine_name,
        processing_time=0.1,
        image_size=(400, 200)
    )


class TestOCRVisualizer:
    """Tests for OCRVisualizer class."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return OCRVisualizer()

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create and save test image."""
        img_path = tmp_path / "test.png"
        img = create_test_image()
        img.save(img_path)
        return img_path

    def test_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.font_size == 12
        assert visualizer.box_thickness == 2
        assert visualizer.show_confidence is False
        assert visualizer.min_confidence == 0.0

    def test_custom_initialization(self):
        """Test visualizer with custom options."""
        vis = OCRVisualizer(
            font_size=16,
            box_thickness=3,
            show_confidence=True,
            min_confidence=0.5
        )
        assert vis.font_size == 16
        assert vis.box_thickness == 3
        assert vis.show_confidence is True
        assert vis.min_confidence == 0.5

    def test_overlay_single(self, visualizer, test_image):
        """Test overlaying single result."""
        result = create_test_result()
        output = visualizer.overlay_single(test_image, result)

        assert isinstance(output, Image.Image)
        assert output.mode == 'RGB'
        assert output.size == (400, 200)

    def test_overlay_single_from_pil(self, visualizer):
        """Test overlay with PIL Image input."""
        img = create_test_image()
        result = create_test_result()
        output = visualizer.overlay_single(img, result)

        assert isinstance(output, Image.Image)
        assert output.size == img.size

    def test_overlay_multiple(self, visualizer, test_image):
        """Test overlaying multiple results."""
        results = {
            'tesseract': create_test_result('tesseract'),
            'easyocr': create_test_result('easyocr'),
        }
        output = visualizer.overlay_multiple(test_image, results)

        assert isinstance(output, Image.Image)
        assert output.mode == 'RGB'

    def test_side_by_side(self, visualizer, test_image):
        """Test side-by-side view."""
        results = {
            'tesseract': create_test_result('tesseract'),
            'easyocr': create_test_result('easyocr'),
        }
        output = visualizer.side_by_side(test_image, results, labels=True)

        assert isinstance(output, Image.Image)
        # Width should be doubled for 2 engines
        assert output.width == 800
        # Height should include label bar (30px)
        assert output.height == 230

    def test_side_by_side_no_labels(self, visualizer, test_image):
        """Test side-by-side without labels."""
        results = {
            'engine1': create_test_result('engine1'),
        }
        output = visualizer.side_by_side(test_image, results, labels=False)

        # Height should not include label bar
        assert output.height == 200

    def test_diff_view(self, visualizer, test_image):
        """Test diff view."""
        # Create results with different detections
        result1 = OCRResult(
            words=[
                OCRWord(
                    text="Hello",
                    bbox=BoundingBox(x=10, y=10, width=50, height=20),
                    confidence=0.9
                ),
            ],
            engine_name="engine1",
            processing_time=0.1,
            image_size=(400, 200)
        )
        result2 = OCRResult(
            words=[
                OCRWord(
                    text="World",
                    bbox=BoundingBox(x=100, y=10, width=50, height=20),
                    confidence=0.9
                ),
            ],
            engine_name="engine2",
            processing_time=0.1,
            image_size=(400, 200)
        )

        results = {'engine1': result1, 'engine2': result2}
        output = visualizer.diff_view(test_image, results)

        assert isinstance(output, Image.Image)

    def test_min_confidence_filtering(self, visualizer, test_image):
        """Test that min_confidence filters words."""
        vis = OCRVisualizer(min_confidence=0.92)
        result = create_test_result()

        # Only "Hello" has confidence >= 0.92
        output = vis.overlay_single(test_image, result)
        assert isinstance(output, Image.Image)

    def test_create_legend(self, visualizer):
        """Test legend creation."""
        legend = visualizer.create_legend(['tesseract', 'easyocr'])

        assert isinstance(legend, Image.Image)
        assert legend.width == 200
        # 2 items * 30px + 20px padding
        assert legend.height == 80

    def test_empty_results(self, visualizer, test_image):
        """Test handling empty results."""
        output = visualizer.overlay_multiple(test_image, {})
        # Should still work, just return image without overlays
        assert isinstance(output, Image.Image)


class TestVisualizeOverlay:
    """Tests for visualize_overlay convenience function."""

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create test image file."""
        img_path = tmp_path / "test.png"
        img = create_test_image()
        img.save(img_path)
        return img_path

    def test_overlay_mode(self, test_image):
        """Test overlay mode."""
        results = {'test': create_test_result()}
        output = visualize_overlay(str(test_image), results, mode='overlay')
        assert isinstance(output, Image.Image)

    def test_side_by_side_mode(self, test_image):
        """Test side_by_side mode."""
        results = {'test1': create_test_result('test1'), 'test2': create_test_result('test2')}
        output = visualize_overlay(str(test_image), results, mode='side_by_side')
        assert isinstance(output, Image.Image)

    def test_diff_mode(self, test_image):
        """Test diff mode."""
        results = {'test1': create_test_result('test1'), 'test2': create_test_result('test2')}
        output = visualize_overlay(str(test_image), results, mode='diff')
        assert isinstance(output, Image.Image)

    def test_invalid_mode(self, test_image):
        """Test that invalid mode raises error."""
        results = {'test': create_test_result()}
        with pytest.raises(ValueError):
            visualize_overlay(str(test_image), results, mode='invalid')

    def test_save_output(self, test_image, tmp_path):
        """Test saving output to file."""
        results = {'test': create_test_result()}
        output_path = tmp_path / "output.png"

        output = visualize_overlay(
            str(test_image),
            results,
            mode='overlay',
            output_path=str(output_path)
        )

        assert output_path.exists()
        saved = Image.open(output_path)
        assert saved.size == output.size

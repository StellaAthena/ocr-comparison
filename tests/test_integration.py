"""Integration tests for OCR Comparison System."""

import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ocr_comparison import OCRComparator, BoundingBox, OCRWord, OCRResult
from ocr_comparison.adapters import TesseractAdapter, EasyOCRAdapter


def create_test_document(size=(600, 400), text_lines=None):
    """Create a test document image with text."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)

    if text_lines is None:
        text_lines = [
            "This is a test document",
            "with multiple lines of text",
            "for OCR comparison testing."
        ]

    y = 30
    for line in text_lines:
        draw.text((30, y), line, fill='black')
        y += 40

    return img


class TestOCRComparator:
    """Tests for OCRComparator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        comparator = OCRComparator()
        assert 'tesseract' in comparator.engines
        assert 'easyocr' in comparator.engines

    def test_initialization_custom_engines(self):
        """Test initialization with custom engines."""
        comparator = OCRComparator(engines=['tesseract'])
        assert comparator.engines == ['tesseract']
        assert 'easyocr' not in comparator._adapters

    def test_initialization_invalid_engine(self):
        """Test initialization with invalid engine."""
        with pytest.raises(ValueError):
            OCRComparator(engines=['nonexistent_engine'])

    def test_list_available_engines(self):
        """Test listing available engines."""
        engines = OCRComparator.list_available_engines()
        assert 'tesseract' in engines
        assert 'easyocr' in engines

    def test_check_availability(self):
        """Test checking engine availability."""
        comparator = OCRComparator()
        availability = comparator.check_availability()

        assert isinstance(availability, dict)
        assert 'tesseract' in availability
        assert 'easyocr' in availability
        # Values should be boolean
        assert all(isinstance(v, bool) for v in availability.values())

    @pytest.fixture
    def test_image(self, tmp_path):
        """Create test image file."""
        img_path = tmp_path / "test_document.png"
        img = create_test_document()
        img.save(img_path)
        return img_path

    @pytest.mark.skipif(
        not TesseractAdapter().is_available(),
        reason="Tesseract not installed"
    )
    def test_process_image_tesseract(self, test_image):
        """Test processing image with Tesseract only."""
        comparator = OCRComparator(engines=['tesseract'])
        results = comparator.process_image(test_image)

        assert 'tesseract' in results
        assert isinstance(results['tesseract'], OCRResult)
        assert results['tesseract'].processing_time > 0

    @pytest.mark.skipif(
        not EasyOCRAdapter(gpu=False).is_available(),
        reason="EasyOCR not installed"
    )
    def test_process_image_easyocr(self, test_image):
        """Test processing image with EasyOCR only."""
        comparator = OCRComparator(
            engines=['easyocr'],
            adapter_options={'easyocr': {'gpu': False}}
        )
        results = comparator.process_image(test_image)

        assert 'easyocr' in results
        assert isinstance(results['easyocr'], OCRResult)

    def test_process_image_not_found(self):
        """Test processing non-existent image."""
        comparator = OCRComparator(engines=['tesseract'])
        with pytest.raises(FileNotFoundError):
            comparator.process_image('/nonexistent/path/image.png')

    @pytest.mark.skipif(
        not TesseractAdapter().is_available(),
        reason="Tesseract not installed"
    )
    def test_visualize_overlay(self, test_image, tmp_path):
        """Test visualization generation."""
        comparator = OCRComparator(engines=['tesseract'])
        results = comparator.process_image(test_image)

        # Test overlay mode
        output = comparator.visualize_overlay(test_image, results, mode='overlay')
        assert isinstance(output, Image.Image)

        # Test saving
        output_path = tmp_path / "output.png"
        comparator.visualize_overlay(
            test_image, results, mode='overlay', output_path=str(output_path)
        )
        assert output_path.exists()

    @pytest.mark.skipif(
        not TesseractAdapter().is_available(),
        reason="Tesseract not installed"
    )
    def test_visualize_auto_process(self, test_image):
        """Test visualization with automatic processing."""
        comparator = OCRComparator(engines=['tesseract'])

        # Don't pass results - should auto-process
        output = comparator.visualize_overlay(test_image, mode='overlay')
        assert isinstance(output, Image.Image)

    @pytest.mark.skipif(
        not TesseractAdapter().is_available(),
        reason="Tesseract not installed"
    )
    def test_compare_accuracy_with_ground_truth(self, test_image):
        """Test accuracy comparison with ground truth."""
        comparator = OCRComparator(engines=['tesseract'])
        results = comparator.process_image(test_image)

        ground_truth = "This is a test document"
        metrics = comparator.compare_accuracy(results, ground_truth)

        assert 'tesseract' in metrics
        assert hasattr(metrics['tesseract'], 'cer')
        assert hasattr(metrics['tesseract'], 'wer')

    def test_compare_accuracy_no_ground_truth(self):
        """Test that compare_accuracy returns empty dict without ground truth."""
        comparator = OCRComparator(engines=['tesseract'])
        # Create dummy result
        result = OCRResult(
            words=[],
            engine_name='tesseract',
            processing_time=0.1,
            image_size=(100, 100)
        )
        results = {'tesseract': result}

        metrics = comparator.compare_accuracy(results)
        assert metrics == {}

    @pytest.mark.skipif(
        not TesseractAdapter().is_available(),
        reason="Tesseract not installed"
    )
    def test_generate_report(self, test_image, tmp_path):
        """Test full report generation."""
        comparator = OCRComparator(engines=['tesseract'])
        output_dir = tmp_path / "report"

        comparison = comparator.generate_report(
            test_image,
            output_dir=str(output_dir),
            ground_truth="This is a test document"
        )

        # Check comparison result
        assert comparison.image_path == str(test_image)
        assert 'tesseract' in comparison.results
        assert 'tesseract' in comparison.metrics

        # Check output files
        assert output_dir.exists()
        assert (output_dir / "test_document_comparison.png").exists()
        assert (output_dir / "test_document_side_by_side.png").exists()
        assert (output_dir / "test_document_report.txt").exists()
        assert (output_dir / "test_document_data.json").exists()


class TestCustomAdapter:
    """Test registering custom adapters."""

    def test_register_custom_adapter(self):
        """Test registering a custom OCR adapter."""
        from ocr_comparison.adapters.base import BaseOCRAdapter

        class CustomAdapter(BaseOCRAdapter):
            @property
            def name(self):
                return "custom"

            def _initialize_engine(self):
                self._engine = True

            def _process(self, image):
                return OCRResult(
                    words=[],
                    engine_name=self.name,
                    processing_time=0.0,
                    image_size=(image.shape[1], image.shape[0])
                )

        OCRComparator.register_adapter('custom', CustomAdapter)

        # Now we can use it
        comparator = OCRComparator(engines=['custom'])
        assert 'custom' in comparator._adapters

    def test_register_invalid_adapter(self):
        """Test that non-adapter classes are rejected."""
        class NotAnAdapter:
            pass

        with pytest.raises(TypeError):
            OCRComparator.register_adapter('invalid', NotAnAdapter)


class TestEndToEnd:
    """End-to-end workflow tests."""

    @pytest.fixture
    def simple_image(self, tmp_path):
        """Create simple image with known text."""
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 35), "HELLO WORLD", fill='black')
        img_path = tmp_path / "simple.png"
        img.save(img_path)
        return img_path, "HELLO WORLD"

    @pytest.mark.skipif(
        not TesseractAdapter().is_available(),
        reason="Tesseract not installed"
    )
    def test_full_workflow_single_engine(self, simple_image):
        """Test complete workflow with single engine."""
        img_path, ground_truth = simple_image

        # Initialize
        comparator = OCRComparator(engines=['tesseract'])

        # Process
        results = comparator.process_image(img_path)
        assert results is not None

        # Visualize
        vis = comparator.visualize_overlay(img_path, results)
        assert vis is not None

        # Evaluate
        metrics = comparator.compare_accuracy(results, ground_truth)
        assert 'tesseract' in metrics

        # Generate report text
        report = comparator.evaluator.generate_report(results, ground_truth)
        assert "OCR Comparison Report" in report

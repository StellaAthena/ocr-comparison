"""Tests for OCR data models."""

import pytest
from ocr_comparison.models import BoundingBox, OCRWord, OCRResult


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_basic_creation(self):
        """Test basic bounding box creation."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.angle == 0.0

    def test_derived_properties(self):
        """Test x2, y2, center, and area properties."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.x2 == 110
        assert bbox.y2 == 70
        assert bbox.center == (60, 45)
        assert bbox.area == 5000

    def test_to_tuple(self):
        """Test conversion to tuple."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.to_tuple() == (10, 20, 100, 50)

    def test_to_corners(self):
        """Test conversion to corner points."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        assert bbox.to_corners() == ((10, 20), (110, 70))

    def test_iou_perfect_overlap(self):
        """Test IoU with identical boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=0, y=0, width=100, height=100)
        assert bbox1.iou(bbox2) == 1.0

    def test_iou_no_overlap(self):
        """Test IoU with non-overlapping boxes."""
        bbox1 = BoundingBox(x=0, y=0, width=50, height=50)
        bbox2 = BoundingBox(x=100, y=100, width=50, height=50)
        assert bbox1.iou(bbox2) == 0.0

    def test_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=50, y=50, width=100, height=100)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500/17500 ≈ 0.143
        iou = bbox1.iou(bbox2)
        assert 0.14 < iou < 0.15

    def test_iou_symmetric(self):
        """Test that IoU is symmetric."""
        bbox1 = BoundingBox(x=0, y=0, width=100, height=100)
        bbox2 = BoundingBox(x=25, y=25, width=100, height=100)
        assert bbox1.iou(bbox2) == bbox2.iou(bbox1)


class TestOCRWord:
    """Tests for OCRWord class."""

    def test_basic_creation(self):
        """Test basic word creation."""
        bbox = BoundingBox(x=10, y=20, width=50, height=15)
        word = OCRWord(text="hello", bbox=bbox, confidence=0.95)
        assert word.text == "hello"
        assert word.confidence == 0.95
        assert word.bbox == bbox

    def test_confidence_validation_too_high(self):
        """Test that confidence > 1.0 raises error."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        with pytest.raises(ValueError):
            OCRWord(text="test", bbox=bbox, confidence=1.5)

    def test_confidence_validation_too_low(self):
        """Test that confidence < 0.0 raises error."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        with pytest.raises(ValueError):
            OCRWord(text="test", bbox=bbox, confidence=-0.1)

    def test_boundary_confidence_values(self):
        """Test that boundary confidence values (0.0, 1.0) are valid."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        word_zero = OCRWord(text="test", bbox=bbox, confidence=0.0)
        word_one = OCRWord(text="test", bbox=bbox, confidence=1.0)
        assert word_zero.confidence == 0.0
        assert word_one.confidence == 1.0


class TestOCRResult:
    """Tests for OCRResult class."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = OCRResult(
            words=[],
            engine_name="test",
            processing_time=0.5,
            image_size=(800, 600)
        )
        assert result.engine_name == "test"
        assert result.processing_time == 0.5
        assert result.image_size == (800, 600)

    def test_full_text(self):
        """Test full_text property."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        words = [
            OCRWord(text="Hello", bbox=bbox, confidence=0.9),
            OCRWord(text="world", bbox=bbox, confidence=0.8),
        ]
        result = OCRResult(
            words=words,
            engine_name="test",
            processing_time=0.1,
            image_size=(100, 100)
        )
        assert result.full_text == "Hello world"

    def test_word_count(self):
        """Test word_count property."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        words = [OCRWord(text=f"word{i}", bbox=bbox, confidence=0.9) for i in range(5)]
        result = OCRResult(
            words=words,
            engine_name="test",
            processing_time=0.1,
            image_size=(100, 100)
        )
        assert result.word_count == 5

    def test_average_confidence(self):
        """Test average_confidence property."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        words = [
            OCRWord(text="a", bbox=bbox, confidence=0.8),
            OCRWord(text="b", bbox=bbox, confidence=0.9),
            OCRWord(text="c", bbox=bbox, confidence=1.0),
        ]
        result = OCRResult(
            words=words,
            engine_name="test",
            processing_time=0.1,
            image_size=(100, 100)
        )
        assert result.average_confidence == pytest.approx(0.9, rel=0.01)

    def test_average_confidence_empty(self):
        """Test average_confidence with no words."""
        result = OCRResult(
            words=[],
            engine_name="test",
            processing_time=0.1,
            image_size=(100, 100)
        )
        assert result.average_confidence == 0.0

    def test_filter_by_confidence(self):
        """Test filter_by_confidence method."""
        bbox = BoundingBox(x=0, y=0, width=10, height=10)
        words = [
            OCRWord(text="low", bbox=bbox, confidence=0.3),
            OCRWord(text="medium", bbox=bbox, confidence=0.6),
            OCRWord(text="high", bbox=bbox, confidence=0.9),
        ]
        result = OCRResult(
            words=words,
            engine_name="test",
            processing_time=0.1,
            image_size=(100, 100)
        )

        filtered = result.filter_by_confidence(0.5)
        assert filtered.word_count == 2
        assert "low" not in filtered.full_text
        assert "medium" in filtered.full_text
        assert "high" in filtered.full_text
        # Original should be unchanged
        assert result.word_count == 3

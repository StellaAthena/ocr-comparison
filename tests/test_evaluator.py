"""Tests for OCR evaluator."""

import pytest

from ocr_comparison.models import BoundingBox, OCRWord, OCRResult
from ocr_comparison.evaluator import (
    levenshtein_distance,
    calculate_cer,
    calculate_wer,
    calculate_detection_metrics,
    normalize_text,
    OCREvaluator
)


class TestLevenshteinDistance:
    """Tests for Levenshtein distance calculation."""

    def test_identical_strings(self):
        """Test distance between identical strings."""
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        """Test with empty strings."""
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_insertion(self):
        """Test single character insertion."""
        assert levenshtein_distance("abc", "abcd") == 1

    def test_single_deletion(self):
        """Test single character deletion."""
        assert levenshtein_distance("abcd", "abc") == 1

    def test_single_substitution(self):
        """Test single character substitution."""
        assert levenshtein_distance("abc", "adc") == 1

    def test_known_distances(self):
        """Test known edit distances."""
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("saturday", "sunday") == 3


class TestCalculateCER:
    """Tests for Character Error Rate calculation."""

    def test_perfect_match(self):
        """Test CER with identical strings."""
        cer, correct, total = calculate_cer("hello world", "hello world")
        assert cer == 0.0
        assert correct == 11
        assert total == 11

    def test_complete_mismatch(self):
        """Test CER with completely different strings."""
        cer, correct, total = calculate_cer("abc", "xyz")
        assert cer == 1.0  # All characters different
        assert total == 3

    def test_empty_ground_truth(self):
        """Test CER with empty ground truth."""
        cer, correct, total = calculate_cer("some text", "")
        assert cer == 1.0
        assert total == 0

    def test_partial_match(self):
        """Test CER with partial match."""
        cer, correct, total = calculate_cer("hello", "hallo")
        # 1 substitution, 5 chars = 0.2 CER
        assert cer == pytest.approx(0.2, rel=0.01)


class TestCalculateWER:
    """Tests for Word Error Rate calculation."""

    def test_perfect_match(self):
        """Test WER with identical text."""
        wer, correct, total = calculate_wer("hello world", "hello world")
        assert wer == 0.0
        assert correct == 2
        assert total == 2

    def test_single_word_error(self):
        """Test WER with one word different."""
        wer, correct, total = calculate_wer("hello world", "hello earth")
        assert wer == pytest.approx(0.5, rel=0.01)

    def test_case_insensitive(self):
        """Test that WER is case insensitive."""
        wer, _, _ = calculate_wer("Hello World", "hello world")
        assert wer == 0.0

    def test_extra_words(self):
        """Test WER with extra words in prediction."""
        wer, _, _ = calculate_wer("hello world today", "hello world")
        assert wer == 0.5  # 1 insertion / 2 words


class TestCalculateDetectionMetrics:
    """Tests for detection precision/recall/F1."""

    def test_perfect_detection(self):
        """Test with perfectly matching boxes."""
        pred = [BoundingBox(x=0, y=0, width=100, height=50)]
        gt = [BoundingBox(x=0, y=0, width=100, height=50)]

        precision, recall, f1 = calculate_detection_metrics(pred, gt)

        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_no_predictions(self):
        """Test with no predictions."""
        pred = []
        gt = [BoundingBox(x=0, y=0, width=100, height=50)]

        precision, recall, f1 = calculate_detection_metrics(pred, gt)

        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_no_ground_truth(self):
        """Test with no ground truth."""
        pred = [BoundingBox(x=0, y=0, width=100, height=50)]
        gt = []

        precision, recall, f1 = calculate_detection_metrics(pred, gt)

        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_partial_overlap(self):
        """Test with partial overlap below threshold."""
        pred = [BoundingBox(x=0, y=0, width=100, height=100)]
        gt = [BoundingBox(x=80, y=0, width=100, height=100)]
        # Overlap is small, won't meet 0.5 IoU threshold

        precision, recall, f1 = calculate_detection_metrics(pred, gt, iou_threshold=0.5)

        assert precision == 0.0
        assert recall == 0.0

    def test_multiple_boxes(self):
        """Test with multiple boxes."""
        pred = [
            BoundingBox(x=0, y=0, width=50, height=50),
            BoundingBox(x=100, y=0, width=50, height=50),
            BoundingBox(x=200, y=0, width=50, height=50),  # Extra detection
        ]
        gt = [
            BoundingBox(x=0, y=0, width=50, height=50),
            BoundingBox(x=100, y=0, width=50, height=50),
        ]

        precision, recall, f1 = calculate_detection_metrics(pred, gt)

        # 2 true positives, 3 predictions, 2 ground truth
        assert precision == pytest.approx(2/3, rel=0.01)
        assert recall == 1.0


class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self):
        """Test lowercase conversion."""
        assert normalize_text("Hello World") == "hello world"

    def test_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_text("hello   world") == "hello world"
        assert normalize_text("  hello  world  ") == "hello world"

    def test_mixed(self):
        """Test combined normalization."""
        assert normalize_text("  Hello   WORLD  ") == "hello world"


class TestOCREvaluator:
    """Tests for OCREvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return OCREvaluator()

    @pytest.fixture
    def sample_result(self):
        """Create sample OCR result."""
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
            engine_name="test",
            processing_time=0.1,
            image_size=(200, 50)
        )

    def test_evaluate_perfect(self, evaluator, sample_result):
        """Test evaluation with perfect match."""
        metrics = evaluator.evaluate(sample_result, "Hello World")

        assert metrics.cer == 0.0
        assert metrics.wer == 0.0

    def test_evaluate_with_normalization(self, evaluator, sample_result):
        """Test that normalization is applied."""
        metrics = evaluator.evaluate(sample_result, "hello world")

        assert metrics.cer == 0.0  # Should match after normalization

    def test_evaluate_without_normalization(self, sample_result):
        """Test without normalization."""
        evaluator = OCREvaluator(normalize=False)
        metrics = evaluator.evaluate(sample_result, "hello world")

        # "Hello World" vs "hello world" - case differences
        assert metrics.cer > 0

    def test_compare_engines(self, evaluator):
        """Test comparing multiple engines."""
        result1 = OCRResult(
            words=[
                OCRWord(
                    text="Hello",
                    bbox=BoundingBox(x=0, y=0, width=50, height=20),
                    confidence=0.9
                )
            ],
            engine_name="engine1",
            processing_time=0.1,
            image_size=(100, 50)
        )
        result2 = OCRResult(
            words=[
                OCRWord(
                    text="Hallo",
                    bbox=BoundingBox(x=0, y=0, width=50, height=20),
                    confidence=0.9
                )
            ],
            engine_name="engine2",
            processing_time=0.1,
            image_size=(100, 50)
        )

        results = {'engine1': result1, 'engine2': result2}
        metrics = evaluator.compare_engines(results, "Hello")

        assert 'engine1' in metrics
        assert 'engine2' in metrics
        assert metrics['engine1'].cer == 0.0
        assert metrics['engine2'].cer > 0

    def test_pairwise_agreement(self, evaluator):
        """Test pairwise agreement calculation."""
        result1 = OCRResult(
            words=[OCRWord(text="Hello World", bbox=BoundingBox(0, 0, 100, 20), confidence=0.9)],
            engine_name="engine1",
            processing_time=0.1,
            image_size=(100, 50)
        )
        result2 = OCRResult(
            words=[OCRWord(text="Hello World", bbox=BoundingBox(0, 0, 100, 20), confidence=0.9)],
            engine_name="engine2",
            processing_time=0.1,
            image_size=(100, 50)
        )

        results = {'engine1': result1, 'engine2': result2}
        agreements = evaluator.pairwise_agreement(results)

        assert ('engine1', 'engine2') in agreements
        assert agreements[('engine1', 'engine2')] == 1.0

    def test_generate_report(self, evaluator, sample_result):
        """Test report generation."""
        results = {'test_engine': sample_result}
        report = evaluator.generate_report(results, "Hello World")

        assert "OCR Comparison Report" in report
        assert "test_engine" in report
        assert "CER" in report
        assert "WER" in report

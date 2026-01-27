"""Data models for the OCR Comparison System."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BoundingBox:
    """Represents a bounding box for detected text."""
    x: int      # top-left x
    y: int      # top-left y
    width: int
    height: int
    angle: float = 0.0  # rotation angle in degrees (optional)

    @property
    def x2(self) -> int:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height

    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    def to_corners(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return as ((x1, y1), (x2, y2)) corner tuple."""
        return ((self.x, self.y), (self.x2, self.y2))


@dataclass
class OCRWord:
    """Represents a single word detected by OCR."""
    text: str
    bbox: BoundingBox
    confidence: float  # 0.0 to 1.0

    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class OCRResult:
    """Result from an OCR engine for a single image."""
    words: List[OCRWord]
    engine_name: str
    processing_time: float  # seconds
    image_size: Tuple[int, int]  # (width, height)

    @property
    def full_text(self) -> str:
        """Concatenate all words into full text."""
        return ' '.join(word.text for word in self.words)

    @property
    def word_count(self) -> int:
        """Number of words detected."""
        return len(self.words)

    @property
    def average_confidence(self) -> float:
        """Average confidence across all words."""
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    def filter_by_confidence(self, min_confidence: float) -> 'OCRResult':
        """Return new result with only words above confidence threshold."""
        filtered = [w for w in self.words if w.confidence >= min_confidence]
        return OCRResult(
            words=filtered,
            engine_name=self.engine_name,
            processing_time=self.processing_time,
            image_size=self.image_size
        )


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for OCR evaluation."""
    cer: float  # Character Error Rate
    wer: float  # Word Error Rate
    precision: float  # Detection precision (IoU-based)
    recall: float  # Detection recall (IoU-based)
    f1: float  # F1 score
    total_characters: int
    total_words: int
    correct_characters: int
    correct_words: int

    @classmethod
    def calculate(
        cls,
        predicted: str,
        ground_truth: str,
        predicted_words: List[OCRWord] = None,
        ground_truth_boxes: List[BoundingBox] = None,
        iou_threshold: float = 0.5
    ) -> 'AccuracyMetrics':
        """Calculate accuracy metrics from predicted vs ground truth text."""
        from .evaluator import calculate_cer, calculate_wer, calculate_detection_metrics

        cer, correct_chars, total_chars = calculate_cer(predicted, ground_truth)
        wer, correct_words_count, total_words_count = calculate_wer(predicted, ground_truth)

        # Calculate detection metrics if bounding boxes provided
        if predicted_words and ground_truth_boxes:
            precision, recall, f1 = calculate_detection_metrics(
                [w.bbox for w in predicted_words],
                ground_truth_boxes,
                iou_threshold
            )
        else:
            precision = recall = f1 = 0.0

        return cls(
            cer=cer,
            wer=wer,
            precision=precision,
            recall=recall,
            f1=f1,
            total_characters=total_chars,
            total_words=total_words_count,
            correct_characters=correct_chars,
            correct_words=correct_words_count
        )


@dataclass
class ComparisonResult:
    """Result of comparing multiple OCR engines on a single image."""
    image_path: str
    results: dict  # Dict[str, OCRResult]
    metrics: dict = field(default_factory=dict)  # Dict[str, AccuracyMetrics]
    ground_truth: Optional[str] = None

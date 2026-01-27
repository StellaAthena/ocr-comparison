"""Accuracy evaluation for OCR results."""

from typing import Dict, List, Tuple, Optional

from .models import OCRResult, OCRWord, BoundingBox, AccuracyMetrics


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of edits to transform s1 into s2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate cost of insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_cer(predicted: str, ground_truth: str) -> Tuple[float, int, int]:
    """Calculate Character Error Rate.

    CER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total chars in reference

    Args:
        predicted: Predicted text from OCR
        ground_truth: Ground truth text

    Returns:
        Tuple of (CER, correct_characters, total_characters)
    """
    if not ground_truth:
        return 0.0 if not predicted else 1.0, 0, 0

    distance = levenshtein_distance(predicted, ground_truth)
    total_chars = len(ground_truth)
    correct_chars = max(0, total_chars - distance)
    cer = distance / total_chars

    return cer, correct_chars, total_chars


def calculate_wer(predicted: str, ground_truth: str) -> Tuple[float, int, int]:
    """Calculate Word Error Rate.

    WER = (S + D + I) / N at word level

    Args:
        predicted: Predicted text from OCR
        ground_truth: Ground truth text

    Returns:
        Tuple of (WER, correct_words, total_words)
    """
    pred_words = predicted.split()
    gt_words = ground_truth.split()

    if not gt_words:
        return 0.0 if not pred_words else 1.0, 0, 0

    distance = levenshtein_distance_words(pred_words, gt_words)
    total_words = len(gt_words)
    correct_words = max(0, total_words - distance)
    wer = distance / total_words

    return wer, correct_words, total_words


def levenshtein_distance_words(s1: List[str], s2: List[str]) -> int:
    """Calculate Levenshtein distance at word level.

    Args:
        s1: First list of words
        s2: Second list of words

    Returns:
        Minimum number of word-level edits
    """
    if len(s1) < len(s2):
        return levenshtein_distance_words(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, w1 in enumerate(s1):
        current_row = [i + 1]
        for j, w2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1.lower() != w2.lower())
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_detection_metrics(
    predicted_boxes: List[BoundingBox],
    ground_truth_boxes: List[BoundingBox],
    iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1 for text detection.

    Args:
        predicted_boxes: Bounding boxes from OCR
        ground_truth_boxes: Ground truth bounding boxes
        iou_threshold: IoU threshold for matching

    Returns:
        Tuple of (precision, recall, f1)
    """
    if not predicted_boxes and not ground_truth_boxes:
        return 1.0, 1.0, 1.0

    if not predicted_boxes:
        return 0.0, 0.0, 0.0

    if not ground_truth_boxes:
        return 0.0, 0.0, 0.0

    # Match predicted boxes to ground truth
    matched_gt = set()
    true_positives = 0

    for pred_box in predicted_boxes:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue

            iou = pred_box.iou(gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            true_positives += 1

    precision = true_positives / len(predicted_boxes)
    recall = true_positives / len(ground_truth_boxes)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Args:
        text: Input text

    Returns:
        Normalized text (lowercase, single spaces, stripped)
    """
    # Convert to lowercase
    text = text.lower()
    # Replace multiple whitespace with single space
    text = ' '.join(text.split())
    # Strip leading/trailing whitespace
    return text.strip()


class OCREvaluator:
    """Evaluate OCR accuracy against ground truth."""

    def __init__(
        self,
        normalize: bool = True,
        iou_threshold: float = 0.5
    ):
        """Initialize evaluator.

        Args:
            normalize: Whether to normalize text before comparison
            iou_threshold: IoU threshold for box matching
        """
        self.normalize = normalize
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        result: OCRResult,
        ground_truth_text: str,
        ground_truth_boxes: List[BoundingBox] = None
    ) -> AccuracyMetrics:
        """Evaluate OCR result against ground truth.

        Args:
            result: OCR result to evaluate
            ground_truth_text: Expected text
            ground_truth_boxes: Optional ground truth bounding boxes

        Returns:
            AccuracyMetrics with CER, WER, and detection metrics
        """
        predicted = result.full_text

        if self.normalize:
            predicted = normalize_text(predicted)
            ground_truth_text = normalize_text(ground_truth_text)

        # Calculate text metrics
        cer, correct_chars, total_chars = calculate_cer(predicted, ground_truth_text)
        wer, correct_words, total_words = calculate_wer(predicted, ground_truth_text)

        # Calculate detection metrics if boxes provided
        if ground_truth_boxes:
            precision, recall, f1 = calculate_detection_metrics(
                [w.bbox for w in result.words],
                ground_truth_boxes,
                self.iou_threshold
            )
        else:
            precision = recall = f1 = 0.0

        return AccuracyMetrics(
            cer=cer,
            wer=wer,
            precision=precision,
            recall=recall,
            f1=f1,
            total_characters=total_chars,
            total_words=total_words,
            correct_characters=correct_chars,
            correct_words=correct_words
        )

    def compare_engines(
        self,
        results: Dict[str, OCRResult],
        ground_truth_text: str,
        ground_truth_boxes: List[BoundingBox] = None
    ) -> Dict[str, AccuracyMetrics]:
        """Compare multiple OCR engines against ground truth.

        Args:
            results: Dictionary mapping engine names to OCR results
            ground_truth_text: Expected text
            ground_truth_boxes: Optional ground truth bounding boxes

        Returns:
            Dictionary mapping engine names to AccuracyMetrics
        """
        return {
            name: self.evaluate(result, ground_truth_text, ground_truth_boxes)
            for name, result in results.items()
        }

    def pairwise_agreement(
        self,
        results: Dict[str, OCRResult]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise text agreement between engines.

        Args:
            results: Dictionary mapping engine names to OCR results

        Returns:
            Dictionary mapping engine pairs to agreement scores (0-1)
        """
        agreements = {}
        engine_names = list(results.keys())

        for i, name1 in enumerate(engine_names):
            for name2 in engine_names[i + 1:]:
                text1 = results[name1].full_text
                text2 = results[name2].full_text

                if self.normalize:
                    text1 = normalize_text(text1)
                    text2 = normalize_text(text2)

                # Calculate similarity (1 - normalized edit distance)
                if not text1 and not text2:
                    similarity = 1.0
                elif not text1 or not text2:
                    similarity = 0.0
                else:
                    distance = levenshtein_distance(text1, text2)
                    max_len = max(len(text1), len(text2))
                    similarity = 1 - (distance / max_len)

                agreements[(name1, name2)] = similarity

        return agreements

    def generate_report(
        self,
        results: Dict[str, OCRResult],
        ground_truth_text: str = None,
        ground_truth_boxes: List[BoundingBox] = None
    ) -> str:
        """Generate a text report comparing OCR engines.

        Args:
            results: Dictionary mapping engine names to OCR results
            ground_truth_text: Optional expected text
            ground_truth_boxes: Optional ground truth bounding boxes

        Returns:
            Formatted report string
        """
        lines = ["=" * 60, "OCR Comparison Report", "=" * 60, ""]

        # Basic statistics
        lines.append("Detection Statistics:")
        lines.append("-" * 40)
        for name, result in results.items():
            lines.append(f"  {name}:")
            lines.append(f"    Words detected: {result.word_count}")
            lines.append(f"    Avg confidence: {result.average_confidence:.1%}")
            lines.append(f"    Processing time: {result.processing_time:.3f}s")
        lines.append("")

        # Accuracy metrics if ground truth provided
        if ground_truth_text:
            lines.append("Accuracy Metrics (vs Ground Truth):")
            lines.append("-" * 40)
            metrics = self.compare_engines(results, ground_truth_text, ground_truth_boxes)
            for name, m in metrics.items():
                lines.append(f"  {name}:")
                lines.append(f"    CER: {m.cer:.2%}")
                lines.append(f"    WER: {m.wer:.2%}")
                if ground_truth_boxes:
                    lines.append(f"    Precision: {m.precision:.2%}")
                    lines.append(f"    Recall: {m.recall:.2%}")
                    lines.append(f"    F1: {m.f1:.2%}")
            lines.append("")

        # Pairwise agreement
        if len(results) > 1:
            lines.append("Pairwise Agreement:")
            lines.append("-" * 40)
            agreements = self.pairwise_agreement(results)
            for (name1, name2), score in agreements.items():
                lines.append(f"  {name1} vs {name2}: {score:.1%}")
            lines.append("")

        # Extracted text comparison
        lines.append("Extracted Text:")
        lines.append("-" * 40)
        for name, result in results.items():
            lines.append(f"  [{name}]")
            text = result.full_text
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"  {text}")
            lines.append("")

        if ground_truth_text:
            lines.append("  [Ground Truth]")
            gt = ground_truth_text
            if len(gt) > 200:
                gt = gt[:200] + "..."
            lines.append(f"  {gt}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

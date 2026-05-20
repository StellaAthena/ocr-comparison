# Python API Reference

## OCRComparator

The main class for comparing OCR engines.

```python
from ocr_comparison import OCRComparator

comparator = OCRComparator(
    engines=['tesseract', 'easyocr'],          # which engines to use
    adapter_options={'easyocr': {'gpu': True}}  # engine-specific options
)
```

### Methods

#### `process_image(image_path) -> Dict[str, OCRResult]`

Run all configured engines on an image.

```python
results = comparator.process_image("image.png")
for engine_name, result in results.items():
    print(f"{engine_name}: {result.word_count} words, {result.average_confidence:.1%}")
    print(result.full_text)
```

#### `visualize_overlay(image_path, results, mode, output_path)`

Create a visualization. Modes: `overlay`, `side_by_side`, `diff`, `textmap`, `margin`.

```python
comparator.visualize_overlay("image.png", results, mode="side_by_side", output_path="out.png")
```

#### `compare_accuracy(results, ground_truth) -> Dict[str, AccuracyMetrics]`

Evaluate OCR results against ground truth text.

```python
metrics = comparator.compare_accuracy(results, ground_truth="expected text")
for engine, m in metrics.items():
    print(f"{engine}: CER={m.cer:.2%}, WER={m.wer:.2%}, F1={m.f1:.2%}")
```

#### `generate_report(image_path, output_dir, ground_truth)`

Generate a full comparison report (visualization images, text report, JSON data).

```python
comparator.generate_report("image.png", output_dir="./report", ground_truth="expected text")
```

#### `check_availability() -> Dict[str, bool]`

Check which engines are installed and working.

## Data Models

### BoundingBox

Represents a detected text region.

```python
@dataclass
class BoundingBox:
    x: int          # top-left x coordinate
    y: int          # top-left y coordinate
    width: int
    height: int
    angle: float    # rotation angle in degrees (optional)

    # Properties
    x2: int         # right edge
    y2: int         # bottom edge
    center: Tuple[int, int]
    area: int

    # Methods
    def iou(self, other: BoundingBox) -> float  # Intersection over Union
```

### OCRWord

A single detected word with its location and confidence.

```python
@dataclass
class OCRWord:
    text: str
    bbox: BoundingBox
    confidence: float  # 0.0 to 1.0
```

### OCRResult

Complete result from one OCR engine.

```python
@dataclass
class OCRResult:
    words: List[OCRWord]
    engine_name: str
    processing_time: float  # seconds
    image_size: Tuple[int, int]  # (width, height)

    # Properties
    full_text: str
    word_count: int
    average_confidence: float

    # Methods
    def filter_by_confidence(self, min_confidence: float) -> OCRResult
```

### AccuracyMetrics

Evaluation metrics when ground truth is available.

```python
@dataclass
class AccuracyMetrics:
    cer: float       # Character Error Rate (0.0 = perfect)
    wer: float       # Word Error Rate (0.0 = perfect)
    precision: float # Detection precision (IoU-based)
    recall: float    # Detection recall (IoU-based)
    f1: float        # F1 score
```

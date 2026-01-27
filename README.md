# OCR Comparison System

A Python library for comparing OCR (Optical Character Recognition) engines with visualization and accuracy metrics. Currently supports Tesseract and EasyOCR, with an extensible adapter architecture for adding new engines.

## Features

- **Multi-engine comparison**: Run multiple OCR engines on the same image and compare results
- **Visualization**: Overlay bounding boxes, side-by-side comparisons, and diff views
- **Accuracy metrics**: CER (Character Error Rate), WER (Word Error Rate), precision, recall, F1
- **Batch processing**: Process entire directories of images
- **CLI and Python API**: Use from command line or integrate into your code
- **Extensible**: Add new OCR engines via the adapter interface

## Installation

### From source

```bash
cd ocr_comparison
pip install -e .
```

### Dependencies

**Python packages** (installed automatically):
- Pillow
- numpy
- pytesseract
- easyocr

**System dependencies**:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### Optional dependencies

```bash
# For development/testing
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

```bash
# Check available OCR engines
ocr-compare engines

# Compare OCR engines on a single image
ocr-compare compare image.png --show-text

# Save visualization
ocr-compare compare image.png --output result.png

# Generate full report with ground truth evaluation
ocr-compare report image.png --output-dir ./report --ground-truth "expected text"

# Batch process a directory
ocr-compare batch ./images/ --output-dir ./results
```

### Python API

```python
from ocr_comparison import OCRComparator

# Initialize comparator
comparator = OCRComparator()

# Process a single image
results = comparator.process_image("image.png")

# Access results by engine
for engine_name, result in results.items():
    print(f"{engine_name}: {result.word_count} words, {result.average_confidence:.1%} confidence")
    print(f"Text: {result.full_text}")

# Generate comparison visualization
comparator.visualize("image.png", output_path="comparison.png")

# Evaluate against ground truth
metrics = comparator.evaluate("image.png", ground_truth="expected text")
for engine_name, m in metrics.items():
    print(f"{engine_name}: CER={m.cer:.2%}, WER={m.wer:.2%}")
```

### Working with Results

```python
from ocr_comparison import OCRComparator, OCRResult

comparator = OCRComparator()
results = comparator.process_image("image.png")

# Get result for specific engine
tesseract_result: OCRResult = results["tesseract"]

# Access detected words
for word in tesseract_result.words:
    print(f"'{word.text}' at ({word.bbox.x}, {word.bbox.y}) conf={word.confidence:.2f}")

# Filter low-confidence words
filtered = tesseract_result.filter_by_confidence(min_confidence=0.8)

# Get full extracted text
print(tesseract_result.full_text)
```

## CLI Reference

### `ocr-compare engines`

Check which OCR engines are available.

### `ocr-compare compare <image>`

Compare OCR engines on a single image.

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Save visualization to file |
| `--mode`, `-m` | Visualization mode: `overlay`, `side_by_side`, `diff` |
| `--show-text` | Print extracted text to console |
| `--engines`, `-e` | Comma-separated list of engines to use |

### `ocr-compare report <image>`

Generate a full comparison report.

| Option | Description |
|--------|-------------|
| `--output-dir`, `-o` | Directory for report files |
| `--ground-truth`, `-g` | Ground truth text for accuracy evaluation |

### `ocr-compare batch <directory>`

Process all images in a directory.

| Option | Description |
|--------|-------------|
| `--output-dir`, `-o` | Directory for output files |
| `--recursive`, `-r` | Process subdirectories |
| `--pattern`, `-p` | Glob pattern for image files (default: `*.png,*.jpg,*.jpeg`) |

### `ocr-compare split <image>`

Generate separate visualization images for each OCR engine. Each engine's results are shown on its own copy of the image, making it easy to compare without overlapping boxes.

| Option | Description |
|--------|-------------|
| `--output-dir`, `-o` | Directory for output files |
| `--show-confidence` | Show confidence scores on boxes |
| `--min-confidence` | Minimum confidence threshold (0.0-1.0) |

```bash
# Generate split images
ocr-compare split document.png --output-dir ./split_results

# Output: split_results/document_tesseract.png
#         split_results/document_easyocr.png
#         split_results/document_manifest.txt
```

### `ocr-compare view <input>`

Launch an interactive viewer to flip between OCR results. The viewer keeps images aligned so you can see exactly what each engine detected by pressing arrow keys.

**Controls:**
- `←` `→` or `↑` `↓`: Switch between images
- `1-9`: Jump to specific image
- `Q` or `Esc`: Quit

```bash
# View split images from a directory
ocr-compare view ./split_results/

# Process an image and immediately view the results
ocr-compare view document.png

# View from a manifest file
ocr-compare view split_results/document_manifest.txt
```

**Note:** The viewer requires a graphical display (uses tkinter).

## Visualization Modes

The library provides several ways to visualize and compare OCR results:

### Split + Flip Viewer (Recommended)

The clearest way to compare results. Each engine gets its own image, and you flip between them with arrow keys:

```bash
ocr-compare split image.png -o ./results
ocr-compare view ./results
```

This keeps the underlying image perfectly aligned, so differences in detection are immediately visible.

### Side-by-Side

All engines displayed horizontally (or in a grid for many engines):

```bash
ocr-compare compare image.png --mode side_by_side -o comparison.png
```

Good for static comparison or documentation, but images may be small.

### Overlay

All engine results overlaid on a single image with different colors:

```bash
ocr-compare compare image.png --mode overlay -o comparison.png
```

- Blue = Tesseract
- Green = EasyOCR
- Additional engines get orange, pink, cyan

Can be cluttered if engines detect many overlapping regions.

### Diff View

Highlights disagreements between engines. Matched detections shown in gray, unique detections in engine colors:

```bash
ocr-compare compare image.png --mode diff -o comparison.png
```

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

## Extending the Library

### Adding a New OCR Engine

The library uses an adapter pattern for OCR engines. To add a new engine:

1. **Create a new adapter file** in `src/ocr_comparison/adapters/`
2. **Inherit from `BaseOCRAdapter`**
3. **Implement required methods**
4. **Register the adapter**

#### Step 1: Create the Adapter

Create `src/ocr_comparison/adapters/my_ocr.py`:

```python
"""Adapter for MyOCR engine."""

from typing import List
import numpy as np

from .base import BaseOCRAdapter
from ..models import BoundingBox, OCRWord, OCRResult


class MyOCRAdapter(BaseOCRAdapter):
    """Adapter for MyOCR engine."""

    def __init__(self, language: str = "en", **kwargs):
        """Initialize MyOCR adapter.

        Args:
            language: Language code for OCR
            **kwargs: Additional options passed to base class
        """
        super().__init__(**kwargs)
        self.language = language

    @property
    def name(self) -> str:
        """Return the name of this OCR engine."""
        return "myocr"

    def _initialize_engine(self) -> None:
        """Initialize the underlying OCR engine.

        This is called lazily on first use. Raise ImportError if
        the required library is not installed.
        """
        try:
            import myocr_library
            self._engine = myocr_library.OCR(lang=self.language)
        except ImportError:
            raise ImportError(
                "myocr_library is not installed. "
                "Install with: pip install myocr-library"
            )

    def _process(self, image: np.ndarray) -> OCRResult:
        """Process an image and return OCR results.

        Args:
            image: Image as numpy array in RGB format

        Returns:
            OCRResult containing detected words with bounding boxes
        """
        # Call your OCR engine
        raw_results = self._engine.detect(image)

        # Convert to OCRWord objects
        words = []
        for item in raw_results:
            bbox = BoundingBox(
                x=item["x"],
                y=item["y"],
                width=item["width"],
                height=item["height"]
            )
            words.append(OCRWord(
                text=item["text"],
                bbox=bbox,
                confidence=item["confidence"]
            ))

        return OCRResult(
            words=words,
            engine_name=self.name,
            processing_time=0.0,  # Set by base class
            image_size=(image.shape[1], image.shape[0])
        )
```

#### Step 2: Register the Adapter

Edit `src/ocr_comparison/adapters/__init__.py`:

```python
"""OCR engine adapters."""

from .base import BaseOCRAdapter
from .tesseract import TesseractAdapter
from .easyocr import EasyOCRAdapter
from .my_ocr import MyOCRAdapter  # Add this line

__all__ = [
    "BaseOCRAdapter",
    "TesseractAdapter",
    "EasyOCRAdapter",
    "MyOCRAdapter",  # Add this line
]
```

#### Step 3: Add to Comparator

Edit `src/ocr_comparison/comparator.py` to include your adapter in the default engines:

```python
from .adapters import TesseractAdapter, EasyOCRAdapter, MyOCRAdapter

class OCRComparator:
    DEFAULT_ADAPTERS = {
        "tesseract": TesseractAdapter,
        "easyocr": EasyOCRAdapter,
        "myocr": MyOCRAdapter,  # Add this line
    }
```

### Adapter Interface Reference

Your adapter must implement these methods:

| Method | Description |
|--------|-------------|
| `name` (property) | Return the engine name as a string |
| `_initialize_engine()` | Set up `self._engine`. Called lazily on first use. |
| `_process(image: np.ndarray) -> OCRResult` | Run OCR and return results |

The base class provides these methods for free:

| Method | Description |
|--------|-------------|
| `process(image)` | Public API - handles image loading, timing, calls `_process()` |
| `_load_image(image)` | Convert path/PIL/numpy to RGB numpy array |
| `is_available()` | Check if engine is installed (calls `_initialize_engine()`) |

### Tips for Writing Adapters

1. **Lazy initialization**: The engine is initialized on first use, not at import time
2. **Handle missing dependencies**: Raise `ImportError` with installation instructions
3. **Normalize confidence**: Ensure confidence values are in range 0.0-1.0
4. **Convert coordinates**: BoundingBox expects integer pixel coordinates
5. **Handle rotated text**: Store rotation angle in `BoundingBox.angle` if available

## Project Structure

```
ocr_comparison/
├── pyproject.toml           # Package configuration
├── README.md
├── requirements.txt
├── src/
│   └── ocr_comparison/      # Main package
│       ├── __init__.py      # Public API exports
│       ├── __main__.py      # CLI entry point
│       ├── models.py        # Data classes
│       ├── comparator.py    # Main OCRComparator class
│       ├── visualizer.py    # Visualization functions
│       ├── evaluator.py     # Accuracy metrics
│       ├── utils.py         # Helper functions
│       └── adapters/        # OCR engine adapters
│           ├── __init__.py
│           ├── base.py      # BaseOCRAdapter ABC
│           ├── tesseract.py
│           └── easyocr.py
├── examples/                # Example scripts
│   ├── quick_test.py
│   ├── basic_usage.py
│   ├── batch_comparison.py
│   └── test_with_images.py
├── tests/                   # Unit tests
│   ├── test_models.py
│   ├── test_adapters.py
│   ├── test_visualizer.py
│   ├── test_evaluator.py
│   └── test_integration.py
└── images/                  # Test images
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=ocr_comparison
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

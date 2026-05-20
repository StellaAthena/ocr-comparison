# Adding a New OCR Engine

The library uses an adapter pattern for OCR engines. To add a new engine, you need to:

1. Create an adapter class
2. Register it
3. Add it to the comparator

## Step 1: Create the Adapter

Create `src/ocr_comparison/adapters/my_ocr.py`:

```python
"""Adapter for MyOCR engine."""

from typing import List
import numpy as np

from .base import BaseOCRAdapter
from ..models import BoundingBox, OCRWord, OCRResult


class MyOCRAdapter(BaseOCRAdapter):

    def __init__(self, language: str = "en", **kwargs):
        super().__init__(**kwargs)
        self.language = language

    @property
    def name(self) -> str:
        return "myocr"

    def _initialize_engine(self) -> None:
        """Called lazily on first use. Raise ImportError if not installed."""
        try:
            import myocr_library
            self._engine = myocr_library.OCR(lang=self.language)
        except ImportError:
            raise ImportError("myocr_library is not installed. Install with: pip install myocr-library")

    def _process(self, image: np.ndarray) -> OCRResult:
        """Run OCR on an RGB numpy array and return results."""
        raw_results = self._engine.detect(image)

        words = []
        for item in raw_results:
            bbox = BoundingBox(x=item["x"], y=item["y"], width=item["width"], height=item["height"])
            words.append(OCRWord(text=item["text"], bbox=bbox, confidence=item["confidence"]))

        return OCRResult(
            words=words,
            engine_name=self.name,
            processing_time=0.0,  # Set by base class
            image_size=(image.shape[1], image.shape[0])
        )
```

## Step 2: Register the Adapter

Edit `src/ocr_comparison/adapters/__init__.py`:

```python
from .my_ocr import MyOCRAdapter  # Add this line

__all__ = [
    ...,
    "MyOCRAdapter",
]
```

## Step 3: Add to Comparator

Edit `src/ocr_comparison/comparator.py`:

```python
from .adapters import MyOCRAdapter

ADAPTER_REGISTRY = {
    ...,
    "myocr": MyOCRAdapter,
}
```

## Adapter Interface

Your adapter must implement:

| Method | Description |
|--------|-------------|
| `name` (property) | Engine name as a string |
| `_initialize_engine()` | Set up `self._engine`. Called lazily on first use. |
| `_process(image: np.ndarray) -> OCRResult` | Run OCR and return results |

The base class handles image loading, timing, and the public `process()` API.

## Tips

- **Lazy initialization**: Don't import heavy libraries at module level. Do it in `_initialize_engine()`.
- **Normalize confidence**: Values must be in range 0.0-1.0.
- **Integer coordinates**: BoundingBox expects integer pixel coordinates.
- **Rotation**: Store rotation angle in `BoundingBox.angle` if the engine detects rotated text.

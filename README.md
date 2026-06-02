# OCR Comparison System

Compare OCR engines visually. Run multiple engines on the same image, then flip between results in an interactive viewer to see exactly what each engine detected. Currently supports Tesseract and EasyOCR, with an extensible adapter architecture for adding new engines.

<table>
<tr>
<td><img src="images/demo_pythia_header.gif" alt="Academic paper demo"></td>
<td><img src="images/demo_math.gif" alt="Handwritten math demo"></td>
<td><img src="images/demo_newspaper.gif" alt="Newspaper demo"></td>
</tr>
<tr>
<td align="center"><em>Academic paper</em></td>
<td align="center"><em>Handwritten math</em></td>
<td align="center"><em>Newspaper</em></td>
</tr>
</table>

## Quick Start

### Windows (PowerShell)

```powershell
# Install Python package
cd ocr-comparison
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Install Tesseract OCR separately (Windows installer)
# Then verify:
tesseract --version
ocr-compare engines

# Launch the interactive viewer on any image
ocr-compare view .\your_image.png
```

### macOS / Linux

```bash
cd ocr-comparison

# Install Tesseract OCR system dependency
# macOS:
brew install tesseract
# Ubuntu/Debian:
sudo apt install tesseract-ocr

pip install -e .
ocr-compare view your_image.png
```

This runs all available OCR engines and opens a viewer where you can:
- **Arrow keys**: flip between engines
- **I**: toggle between the original image and a "text map" showing what the engine extracted
- **Q**: quit

That's it. One command to see how your OCR engines compare on a document.

## Other Ways to Use It

### Print extracted text

```bash
ocr-compare compare image.png --show-text
```

### Save a static visualization

```bash
# Side-by-side comparison
ocr-compare compare image.png --mode side_by_side -o comparison.png

# Overlay (each engine gets a distinct color)
ocr-compare compare image.png --mode overlay -o comparison.png

# Diff view (highlights disagreements between engines)
ocr-compare compare image.png --mode diff -o comparison.png

# Margin view (extracted text in a right-side margin with leader lines)
ocr-compare compare image.png --mode margin -o margin.png
```

### Batch process a directory

```bash
ocr-compare batch ./images/ --output-dir ./results
```

### Generate a full report

```bash
ocr-compare report image.png --output-dir ./report --ground-truth "expected text"
```

### Check available engines

```bash
ocr-compare engines
```

## Installation Details

**Python packages** (installed automatically): Pillow, numpy, pytesseract, easyocr

**System dependency** -- Tesseract requires a separate install:
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr
```

On Windows, install Tesseract via a Windows installer, then ensure:
- `tesseract.exe` exists (commonly `C:\Program Files\Tesseract-OCR\tesseract.exe`)
- `C:\Program Files\Tesseract-OCR\` is on your `PATH`
- A new terminal is opened after PATH changes

Verify installation:
```powershell
tesseract --version
ocr-compare engines
```

### Windows Troubleshooting

- If `.\.venv\Scripts\Activate.ps1` is blocked, allow local scripts:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
- If `tesseract` is not recognized, add `C:\Program Files\Tesseract-OCR\` to PATH and open a new terminal.
- If `ocr-compare` is not recognized, use module form:
  ```powershell
  python -m ocr_comparison engines
  ```

Use `ocr-compare engines` to see which engines are available on your system. You can select a subset with `--engines`, e.g. `--engines tesseract`.

For development/testing: `pip install -e ".[dev]"`

## Python API

```python
from ocr_comparison import OCRComparator

comparator = OCRComparator()
results = comparator.process_image("image.png")

for engine_name, result in results.items():
    print(f"{engine_name}: {result.word_count} words, {result.average_confidence:.1%} confidence")
    print(result.full_text)
```

See `examples/` for more usage patterns including batch processing and ground truth evaluation.

## Documentation

- [CLI Reference](docs/cli.md) -- all commands and options
- [Python API](docs/api.md) -- data models and programmatic usage
- [Adding OCR Engines](docs/adding-engines.md) -- how to write a new adapter

## License

Apache 2.0

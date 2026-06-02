# CLI Reference

All commands are available via `ocr-compare` after installing with `pip install -e .`

## Install & Verify (Windows PowerShell)

```powershell
cd .\ocr-comparison
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Verify Tesseract is available
tesseract --version

# Verify OCR engines recognized by this project
ocr-compare engines
```

If `ocr-compare` is not recognized in your shell, use:

```powershell
python -m ocr_comparison engines
```

## Windows Troubleshooting

- **Activation blocked (`Activate.ps1`)**
  - Run:
    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    ```
  - Then open a new PowerShell session and activate again.
- **`tesseract` not recognized**
  - Confirm `tesseract.exe` exists (commonly `C:\Program Files\Tesseract-OCR\tesseract.exe`).
  - Add `C:\Program Files\Tesseract-OCR\` to your PATH and open a new terminal.
  - Re-test with `tesseract --version`.
- **`ocr-compare` not recognized**
  - Ensure your venv is active, then run `python -m ocr_comparison ...` as a fallback.

## `ocr-compare view <input>`

Launch the interactive viewer. This is the recommended way to compare engines.

```bash
ocr-compare view image.png              # process and view immediately
ocr-compare view ./split_results/       # view pre-computed results
ocr-compare view manifest.txt           # view from a manifest file
```

**Controls:**
- Arrow keys: switch between engines
- `I`: toggle between original image and text map
- `1-9`: jump to specific engine
- `Q` / `Esc`: quit

Requires a graphical display (uses tkinter).

## `ocr-compare compare <image>`

Run OCR engines and print results. Optionally save a visualization.

```bash
ocr-compare compare image.png --show-text
ocr-compare compare image.png -o result.png
ocr-compare compare image.png --mode side_by_side -o comparison.png
```

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Save visualization to file |
| `--mode`, `-m` | `overlay`, `side_by_side`, `diff`, `textmap`, `margin` |
| `--show-text`, `-t` | Print extracted text to console |
| `--full-text` | Don't truncate long text output |
| `--engines`, `-e` | Comma-separated engine list (default: `tesseract,easyocr`) |
| `--ground-truth`, `-g` | Ground truth text for accuracy metrics |
| `--gpu` | Use GPU for EasyOCR |

## `ocr-compare split <image>`

Generate separate visualization images for each engine (one image per engine).

```bash
ocr-compare split document.png -o ./split_results
```

| Option | Description |
|--------|-------------|
| `--output-dir`, `-o` | Output directory |
| `--show-confidence` | Show confidence scores on boxes |
| `--min-confidence` | Minimum confidence threshold (0.0-1.0) |
| `--font-size`, `-f` | Font size for labels (default: 18) |
| `--box-thickness`, `-b` | Box line thickness (default: 3) |

## `ocr-compare report <image>`

Generate a full report: overlay, side-by-side, text report, and JSON data.

```bash
ocr-compare report image.png -o ./report
ocr-compare report image.png -o ./report --ground-truth "expected text"
```

| Option | Description |
|--------|-------------|
| `--output-dir`, `-o` | Output directory |
| `--ground-truth`, `-g` | Ground truth text for accuracy evaluation |

## `ocr-compare batch <directory>`

Process all images in a directory.

```bash
ocr-compare batch ./images/ -o ./results
ocr-compare batch ./images/ -o ./results --recursive
```

| Option | Description |
|--------|-------------|
| `--output-dir`, `-o` | Output directory |
| `--recursive`, `-r` | Search subdirectories |
| `--mode`, `-m` | Visualization mode for saved images |

## `ocr-compare engines`

Check which OCR engines are installed and available.

```bash
ocr-compare engines
```

## Common Options

These options are available on most commands:

| Option | Description |
|--------|-------------|
| `--engines`, `-e` | Comma-separated engine list |
| `--gpu` | Use GPU for EasyOCR |
| `--lang`, `-l` | Tesseract language (default: `eng`) |

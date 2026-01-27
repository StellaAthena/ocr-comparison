#!/usr/bin/env python3
"""Test OCR comparison system with sample images.

This script tests the OCR comparison system using the images in the
../images/ directory.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ocr_comparison import OCRComparator


# Test images with descriptions and partial ground truth where known
TEST_IMAGES = {
    "Screenshot 2026-01-27 at 1.31.40 PM.png": {
        "description": "Pythia paper - title page with abstract",
        "expected_snippets": ["Pythia", "Large Language Models", "Abstract", "EleutherAI"],
    },
    "Screenshot 2026-01-27 at 1.31.56 PM.png": {
        "description": "Pythia paper - table comparing model suites",
        "expected_snippets": ["GPT-2", "GPT-3", "BLOOM", "Pythia", "Public Models"],
    },
    "Screenshot 2026-01-27 at 1.32.07 PM.png": {
        "description": "Pythia paper - graphs and acknowledgments",
        "expected_snippets": ["Accuracy", "Trivia QA", "Acknowledgments", "Stability AI"],
    },
    "Screenshot 2026-01-27 at 1.32.47 PM.png": {
        "description": "Twitter/X interface - dark theme",
        "expected_snippets": ["Home", "Explore", "Notifications", "Premium"],
    },
    "Screenshot 2026-01-27 at 1.34.19 PM.png": {
        "description": "Handwritten math equations",
        "expected_snippets": [],  # Handwriting is hard to predict
    },
    "Screenshot 2026-01-27 at 11.24.01 AM.png": {
        "description": "HuggingFace dataset page - dark theme",
        "expected_snippets": ["Datasets", "NVIDIA", "EleutherAI"],
    },
}


def check_snippets(text: str, snippets: list) -> dict:
    """Check which expected snippets were found in text."""
    text_lower = text.lower()
    results = {}
    for snippet in snippets:
        results[snippet] = snippet.lower() in text_lower
    return results


def main():
    images_dir = Path(__file__).parent.parent / "images"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("OCR Comparison System - Image Test Suite")
    print("=" * 60)

    # Check available engines
    print("\nInitializing OCR engines...")
    comparator = OCRComparator(
        engines=['tesseract', 'easyocr'],
        adapter_options={'easyocr': {'gpu': False}}
    )

    availability = comparator.check_availability()
    print(f"Engine availability:")
    for engine, available in availability.items():
        status = "OK" if available else "NOT INSTALLED"
        print(f"  {engine}: {status}")

    available_engines = [name for name, avail in availability.items() if avail]
    if not available_engines:
        print("\nERROR: No OCR engines available!")
        print("Install with: pip install pytesseract easyocr")
        print("Also install tesseract: brew install tesseract (macOS)")
        return 1

    # Reinitialize with only available engines
    comparator = OCRComparator(
        engines=available_engines,
        adapter_options={'easyocr': {'gpu': False}} if 'easyocr' in available_engines else {}
    )

    # Process each test image
    all_results = {}

    for filename, info in TEST_IMAGES.items():
        image_path = images_dir / filename
        if not image_path.exists():
            print(f"\nSkipping {filename} - file not found")
            continue

        print(f"\n{'─' * 60}")
        print(f"Processing: {filename}")
        print(f"Description: {info['description']}")
        print("─" * 60)

        try:
            # Process image
            results = comparator.process_image(image_path)
            all_results[filename] = results

            for engine_name, result in results.items():
                print(f"\n[{engine_name.upper()}]")
                print(f"  Words detected: {result.word_count}")
                print(f"  Avg confidence: {result.average_confidence:.1%}")
                print(f"  Processing time: {result.processing_time:.2f}s")

                # Show snippet of detected text
                text = result.full_text
                preview = text[:150] + "..." if len(text) > 150 else text
                print(f"  Text preview: {preview}")

                # Check expected snippets
                if info['expected_snippets']:
                    found = check_snippets(text, info['expected_snippets'])
                    found_count = sum(found.values())
                    total = len(found)
                    print(f"  Expected snippets found: {found_count}/{total}")
                    for snippet, was_found in found.items():
                        mark = "✓" if was_found else "✗"
                        print(f"    {mark} '{snippet}'")

            # Generate visualizations
            print(f"\n  Generating visualizations...")

            # Overlay view
            overlay_path = output_dir / f"{image_path.stem}_overlay.png"
            comparator.visualize_overlay(
                image_path, results, mode='overlay', output_path=str(overlay_path)
            )
            print(f"    Saved: {overlay_path.name}")

            # Side-by-side (if multiple engines)
            if len(results) > 1:
                side_path = output_dir / f"{image_path.stem}_sidebyside.png"
                comparator.visualize_overlay(
                    image_path, results, mode='side_by_side', output_path=str(side_path)
                )
                print(f"    Saved: {side_path.name}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Images processed: {len(all_results)}")
    print(f"Output directory: {output_dir}")

    for engine in available_engines:
        times = [r[engine].processing_time for r in all_results.values() if engine in r]
        if times:
            avg_time = sum(times) / len(times)
            print(f"{engine} average time: {avg_time:.2f}s")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

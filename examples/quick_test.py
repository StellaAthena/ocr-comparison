#!/usr/bin/env python3
"""Quick test to verify OCR comparison system works.

Run this first to check your installation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    print("Testing OCR Comparison System installation...\n")

    # Test 1: Import check
    print("1. Checking imports...")
    try:
        from ocr_comparison import OCRComparator, BoundingBox, OCRWord, OCRResult
        print("   ✓ Core imports OK")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        print("   Install dependencies: pip install Pillow numpy")
        return 1

    # Test 2: Engine availability
    print("\n2. Checking OCR engines...")
    comparator = OCRComparator(
        engines=['tesseract', 'easyocr'],
        adapter_options={'easyocr': {'gpu': False}}
    )
    availability = comparator.check_availability()

    any_available = False
    for engine, available in availability.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"   {engine}: {status}")
        if available:
            any_available = True

    if not any_available:
        print("\n   No OCR engines available!")
        print("   Install Tesseract: brew install tesseract (macOS)")
        print("   Install EasyOCR: pip install easyocr")
        return 1

    # Test 3: Process a test image
    print("\n3. Testing image processing...")
    images_dir = Path(__file__).parent.parent / "images"
    test_images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

    if not test_images:
        print("   ✗ No test images found in ../images/")
        print("   Add some images to test with")
        return 1

    test_image = test_images[0]
    print(f"   Using: {test_image.name}")

    available_engines = [e for e, a in availability.items() if a]
    comparator = OCRComparator(
        engines=available_engines,
        adapter_options={'easyocr': {'gpu': False}} if 'easyocr' in available_engines else {}
    )

    try:
        results = comparator.process_image(test_image)
        for engine, result in results.items():
            print(f"   ✓ {engine}: detected {result.word_count} words in {result.processing_time:.2f}s")
    except Exception as e:
        print(f"   ✗ Processing failed: {e}")
        return 1

    # Test 4: Visualization
    print("\n4. Testing visualization...")
    try:
        output_path = Path(__file__).parent / "output" / "quick_test_output.png"
        output_path.parent.mkdir(exist_ok=True)
        comparator.visualize_overlay(
            test_image, results, mode='overlay', output_path=str(output_path)
        )
        print(f"   ✓ Visualization saved to: {output_path}")
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
        return 1

    print("\n" + "=" * 40)
    print("All tests passed! System is working.")
    print("=" * 40)
    print(f"\nRun the full test suite with:")
    print(f"  python {Path(__file__).parent}/test_with_images.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())

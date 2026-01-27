#!/usr/bin/env python3
"""Basic usage example for OCR Comparison System.

This script demonstrates the core functionality of the OCR comparison system:
1. Processing an image with multiple OCR engines
2. Creating overlay visualizations
3. Comparing accuracy against ground truth
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_comparison import OCRComparator


def main():
    # Initialize comparator with both engines
    # Set gpu=False for EasyOCR if you don't have CUDA
    print("Initializing OCR comparator...")
    comparator = OCRComparator(
        engines=['tesseract', 'easyocr'],
        adapter_options={
            'tesseract': {'lang': 'eng'},
            'easyocr': {'gpu': False}
        }
    )

    # Check which engines are available
    availability = comparator.check_availability()
    print(f"Engine availability: {availability}")

    # Use only available engines
    available_engines = [name for name, available in availability.items() if available]
    if not available_engines:
        print("No OCR engines available. Please install tesseract or easyocr.")
        return

    print(f"Using engines: {available_engines}")

    # Example with sample image (you would replace this with your own image)
    # For this demo, we'll create a simple test image
    from PIL import Image, ImageDraw

    # Create a test image with text
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((20, 35), "Hello World - OCR Test", fill='black')

    # Save test image
    test_image_path = Path(__file__).parent / "sample_images" / "test.png"
    test_image_path.parent.mkdir(exist_ok=True)
    img.save(test_image_path)
    print(f"Created test image: {test_image_path}")

    # Process the image
    print("\nProcessing image...")
    results = comparator.process_image(test_image_path, engines=available_engines)

    # Print results for each engine
    for engine_name, result in results.items():
        print(f"\n{engine_name.upper()} Results:")
        print(f"  Detected text: {result.full_text}")
        print(f"  Words found: {result.word_count}")
        print(f"  Average confidence: {result.average_confidence:.1%}")
        print(f"  Processing time: {result.processing_time:.3f}s")

    # Create overlay visualization
    print("\nCreating visualizations...")

    # Overlay mode - all engines on one image
    overlay_path = test_image_path.parent / "overlay_comparison.png"
    comparator.visualize_overlay(
        test_image_path,
        results,
        mode='overlay',
        output_path=str(overlay_path)
    )
    print(f"  Overlay saved to: {overlay_path}")

    # Side-by-side mode
    if len(results) > 1:
        sidebyside_path = test_image_path.parent / "sidebyside_comparison.png"
        comparator.visualize_overlay(
            test_image_path,
            results,
            mode='side_by_side',
            output_path=str(sidebyside_path)
        )
        print(f"  Side-by-side saved to: {sidebyside_path}")

        # Diff mode
        diff_path = test_image_path.parent / "diff_comparison.png"
        comparator.visualize_overlay(
            test_image_path,
            results,
            mode='diff',
            output_path=str(diff_path)
        )
        print(f"  Diff view saved to: {diff_path}")

    # Compare accuracy against ground truth
    ground_truth = "Hello World - OCR Test"
    print(f"\nComparing against ground truth: '{ground_truth}'")

    metrics = comparator.compare_accuracy(results, ground_truth)
    for engine_name, m in metrics.items():
        print(f"\n{engine_name.upper()} Accuracy:")
        print(f"  Character Error Rate (CER): {m.cer:.2%}")
        print(f"  Word Error Rate (WER): {m.wer:.2%}")

    # Generate full report
    print("\nGenerating full report...")
    report = comparator.evaluator.generate_report(results, ground_truth)
    print("\n" + report)

    print("\nDone! Check the sample_images directory for output files.")


if __name__ == "__main__":
    main()

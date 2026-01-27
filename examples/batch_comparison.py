#!/usr/bin/env python3
"""Batch processing example for OCR Comparison System.

This script demonstrates how to:
1. Process multiple images in batch
2. Load ground truth from files
3. Generate aggregate statistics
4. Create summary reports
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_comparison import OCRComparator
from ocr_comparison.utils import find_image_files, load_ground_truth, format_time


def create_sample_dataset(output_dir: Path):
    """Create a sample dataset with images and ground truth."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(exist_ok=True)
    ground_truths = {}

    # Sample texts
    samples = [
        "The quick brown fox",
        "jumps over the lazy dog",
        "Pack my box with five",
        "dozen liquor jugs",
    ]

    for i, text in enumerate(samples):
        # Create image
        img = Image.new('RGB', (300, 60), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 20), text, fill='black')

        filename = f"sample_{i+1}.png"
        img.save(output_dir / filename)
        ground_truths[filename] = text

    # Save ground truth file
    with open(output_dir / "ground_truth.json", 'w') as f:
        json.dump(ground_truths, f, indent=2)

    return ground_truths


def main():
    # Setup paths
    samples_dir = Path(__file__).parent / "sample_images"
    output_dir = Path(__file__).parent / "batch_output"

    # Create sample dataset if it doesn't have many images
    existing_images = list(find_image_files(samples_dir)) if samples_dir.exists() else []
    if len(existing_images) < 4:
        print("Creating sample dataset...")
        ground_truths = create_sample_dataset(samples_dir)
    else:
        # Try to load existing ground truth
        gt_file = samples_dir / "ground_truth.json"
        if gt_file.exists():
            with open(gt_file) as f:
                ground_truths = json.load(f)
        else:
            ground_truths = {}

    # Initialize comparator
    print("Initializing OCR comparator...")
    comparator = OCRComparator(
        engines=['tesseract', 'easyocr'],
        adapter_options={'easyocr': {'gpu': False}}
    )

    # Check availability and filter engines
    availability = comparator.check_availability()
    available_engines = [name for name, avail in availability.items() if avail]

    if not available_engines:
        print("No OCR engines available!")
        return

    print(f"Using engines: {available_engines}")

    # Reinitialize with only available engines
    comparator = OCRComparator(
        engines=available_engines,
        adapter_options={'easyocr': {'gpu': False}} if 'easyocr' in available_engines else {}
    )

    # Find all images
    image_files = find_image_files(samples_dir)
    print(f"\nFound {len(image_files)} images to process")

    if not image_files:
        print("No images found!")
        return

    # Process all images
    print("\nProcessing images...")
    output_dir.mkdir(exist_ok=True)

    all_results = []
    all_metrics = {engine: {'cer': [], 'wer': [], 'time': []} for engine in available_engines}

    for i, img_path in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] Processing {img_path.name}...")

        # Get ground truth if available
        gt = ground_truths.get(img_path.name)

        # Generate report for this image
        comparison = comparator.generate_report(
            img_path,
            output_dir=str(output_dir / img_path.stem),
            ground_truth=gt
        )

        all_results.append(comparison)

        # Collect metrics
        for engine_name, result in comparison.results.items():
            all_metrics[engine_name]['time'].append(result.processing_time)

            if gt and engine_name in comparison.metrics:
                m = comparison.metrics[engine_name]
                all_metrics[engine_name]['cer'].append(m.cer)
                all_metrics[engine_name]['wer'].append(m.wer)

    # Generate summary statistics
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"\nImages processed: {len(image_files)}")
    print(f"Images with ground truth: {sum(1 for r in all_results if r.ground_truth)}")

    for engine_name in available_engines:
        metrics = all_metrics[engine_name]
        print(f"\n{engine_name.upper()}:")

        # Timing stats
        if metrics['time']:
            avg_time = sum(metrics['time']) / len(metrics['time'])
            total_time = sum(metrics['time'])
            print(f"  Average processing time: {format_time(avg_time)}")
            print(f"  Total processing time: {format_time(total_time)}")

        # Accuracy stats (if ground truth available)
        if metrics['cer']:
            avg_cer = sum(metrics['cer']) / len(metrics['cer'])
            avg_wer = sum(metrics['wer']) / len(metrics['wer'])
            print(f"  Average CER: {avg_cer:.2%}")
            print(f"  Average WER: {avg_wer:.2%}")

    # Write summary to file
    summary_path = output_dir / "batch_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("OCR Batch Processing Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Images processed: {len(image_files)}\n\n")

        for engine_name in available_engines:
            metrics = all_metrics[engine_name]
            f.write(f"{engine_name.upper()}:\n")
            if metrics['time']:
                avg_time = sum(metrics['time']) / len(metrics['time'])
                f.write(f"  Average time: {format_time(avg_time)}\n")
            if metrics['cer']:
                avg_cer = sum(metrics['cer']) / len(metrics['cer'])
                avg_wer = sum(metrics['wer']) / len(metrics['wer'])
                f.write(f"  Average CER: {avg_cer:.2%}\n")
                f.write(f"  Average WER: {avg_wer:.2%}\n")
            f.write("\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"Individual reports saved to: {output_dir}/")


if __name__ == "__main__":
    main()

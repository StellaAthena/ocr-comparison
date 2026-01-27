#!/usr/bin/env python3
"""Command-line interface for OCR Comparison System.

Usage:
    python -m ocr_comparison compare image.png
    python -m ocr_comparison compare image.png --output result.png
    python -m ocr_comparison compare image.png --mode side_by_side
    python -m ocr_comparison report image.png --output-dir ./reports
    python -m ocr_comparison batch ./images/ --output-dir ./results
"""

import argparse
import sys
from pathlib import Path

from . import OCRComparator
from .utils import find_image_files, format_time


def cmd_compare(args):
    """Compare OCR engines on a single image."""
    comparator = _get_comparator(args)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        return 1

    print(f"Processing: {image_path.name}")
    results = comparator.process_image(image_path)

    # Print results
    for engine, result in results.items():
        print(f"\n[{engine.upper()}]")
        print(f"  Words: {result.word_count}")
        print(f"  Confidence: {result.average_confidence:.1%}")
        print(f"  Time: {format_time(result.processing_time)}")

        if args.show_text:
            text = result.full_text
            if len(text) > 500 and not args.full_text:
                text = text[:500] + "..."
            print(f"  Text: {text}")

    # Compare with ground truth if provided
    if args.ground_truth:
        print(f"\n[ACCURACY vs Ground Truth]")
        metrics = comparator.compare_accuracy(results, args.ground_truth)
        for engine, m in metrics.items():
            print(f"  {engine}: CER={m.cer:.2%}, WER={m.wer:.2%}")

    # Generate visualization if output specified
    if args.output:
        comparator.visualize_overlay(
            image_path,
            results,
            mode=args.mode,
            output_path=args.output
        )
        print(f"\nVisualization saved: {args.output}")

    return 0


def cmd_visualize(args):
    """Generate visualization for an image."""
    comparator = _get_comparator(args)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        return 1

    output_path = args.output or f"{image_path.stem}_{args.mode}.png"

    print(f"Processing: {image_path.name}")
    results = comparator.process_image(image_path)

    comparator.visualize_overlay(
        image_path,
        results,
        mode=args.mode,
        output_path=output_path,
        show_confidence=args.show_confidence,
        min_confidence=args.min_confidence
    )

    print(f"Saved: {output_path}")
    return 0


def cmd_report(args):
    """Generate full report for an image."""
    comparator = _get_comparator(args)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or f"{image_path.stem}_report"

    print(f"Processing: {image_path.name}")
    comparison = comparator.generate_report(
        image_path,
        output_dir=output_dir,
        ground_truth=args.ground_truth
    )

    print(f"\nReport generated in: {output_dir}/")
    print(f"  - {image_path.stem}_comparison.png")
    print(f"  - {image_path.stem}_side_by_side.png")
    print(f"  - {image_path.stem}_report.txt")
    print(f"  - {image_path.stem}_data.json")

    return 0


def cmd_batch(args):
    """Process multiple images."""
    comparator = _get_comparator(args)

    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = find_image_files(input_path, recursive=args.recursive)
    else:
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        return 1

    if not image_files:
        print("No images found.", file=sys.stderr)
        return 1

    print(f"Found {len(image_files)} images")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {image_path.name}")

        try:
            results = comparator.process_image(image_path)

            for engine, result in results.items():
                print(f"  {engine}: {result.word_count} words, {format_time(result.processing_time)}")

            if output_dir:
                out_path = output_dir / f"{image_path.stem}_comparison.png"
                comparator.visualize_overlay(
                    image_path, results, mode=args.mode, output_path=str(out_path)
                )
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)

    print(f"\nProcessed {len(image_files)} images")
    if output_dir:
        print(f"Output saved to: {output_dir}/")

    return 0


def cmd_engines(args):
    """List available OCR engines."""
    print("Checking OCR engine availability...\n")

    comparator = OCRComparator(
        engines=['tesseract', 'easyocr'],
        adapter_options={'easyocr': {'gpu': False}}
    )

    availability = comparator.check_availability()

    for engine, available in availability.items():
        status = "Available" if available else "Not installed"
        symbol = "+" if available else "-"
        print(f"  {symbol} {engine}: {status}")

    print("\nInstallation:")
    print("  Tesseract: brew install tesseract (macOS) / apt install tesseract-ocr (Linux)")
    print("  EasyOCR: pip install easyocr")

    return 0


def cmd_split(args):
    """Generate separate images for each OCR engine."""
    from .visualizer import save_individual_images

    comparator = _get_comparator(args)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or f"{image_path.stem}_split"

    print(f"Processing: {image_path.name}")
    results = comparator.process_image(image_path)

    for engine, result in results.items():
        print(f"  {engine}: {result.word_count} words, {format_time(result.processing_time)}")

    saved_paths = save_individual_images(
        str(image_path),
        results,
        output_dir,
        font_size=args.font_size,
        box_thickness=args.box_thickness,
        show_confidence=args.show_confidence,
        min_confidence=args.min_confidence
    )

    print(f"\nSaved {len(saved_paths)} images to: {output_dir}/")
    for path in saved_paths:
        print(f"  - {Path(path).name}")

    print(f"\nTo view with flip comparison:")
    print(f"  ocr-compare view {output_dir}/")

    return 0


def cmd_view(args):
    """Launch interactive flip viewer to compare OCR results."""
    from .visualizer import create_flip_viewer, save_individual_images

    input_path = Path(args.input)

    # Determine image paths to view
    if input_path.is_dir():
        # Look for manifest or image files
        manifest = input_path / f"{input_path.name}_manifest.txt"
        if not manifest.exists():
            # Try to find any manifest
            manifests = list(input_path.glob("*_manifest.txt"))
            if manifests:
                manifest = manifests[0]

        if manifest.exists():
            with open(manifest) as f:
                image_paths = [line.strip() for line in f
                              if line.strip() and not line.startswith('#')]
        else:
            # Just find PNG files (exclude _clean versions, viewer will find them)
            all_pngs = sorted(input_path.glob("*.png"))
            image_paths = [str(p) for p in all_pngs if '_clean' not in p.stem]
    elif input_path.is_file():
        if input_path.suffix == '.txt':
            # Manifest file
            with open(input_path) as f:
                image_paths = [line.strip() for line in f
                              if line.strip() and not line.startswith('#')]
        else:
            # Single image - process it first
            print(f"Processing image: {input_path.name}")
            comparator = _get_comparator(args)
            results = comparator.process_image(input_path)

            output_dir = Path(f"{input_path.stem}_split")
            image_paths = save_individual_images(
                str(input_path),
                results,
                str(output_dir)
            )
            print(f"Created split images in: {output_dir}/")
    else:
        print(f"Error: Path not found: {input_path}", file=sys.stderr)
        return 1

    if not image_paths:
        print("No images found to view.", file=sys.stderr)
        return 1

    # Count actual images (manifest entries may be pairs)
    num_engines = len(image_paths)

    print(f"\nLaunching viewer with {num_engines} engine(s)...")
    print("Controls: ← → to switch, I to toggle boxes, Q to quit")

    try:
        create_flip_viewer(image_paths, title=f"OCR Comparison - {input_path.name}")
    except Exception as e:
        print(f"Error launching viewer: {e}", file=sys.stderr)
        print("Note: The viewer requires a graphical display (tkinter).")
        return 1

    return 0


def _get_comparator(args):
    """Create comparator from args."""
    engines = args.engines.split(',') if hasattr(args, 'engines') and args.engines else ['tesseract', 'easyocr']

    # Filter to only requested engines
    adapter_options = {}
    if 'easyocr' in engines:
        adapter_options['easyocr'] = {'gpu': getattr(args, 'gpu', False)}
    if 'tesseract' in engines and hasattr(args, 'lang') and args.lang:
        adapter_options['tesseract'] = {'lang': args.lang}

    return OCRComparator(engines=engines, adapter_options=adapter_options)


def main():
    parser = argparse.ArgumentParser(
        prog='python -m ocr_comparison',
        description='Compare OCR engines (Tesseract vs EasyOCR)'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--engines', '-e', default='tesseract,easyocr',
                        help='Comma-separated list of engines (default: tesseract,easyocr)')
    common.add_argument('--gpu', action='store_true',
                        help='Use GPU for EasyOCR')
    common.add_argument('--lang', '-l', default=None,
                        help='Language for Tesseract (default: eng)')

    # compare command
    p_compare = subparsers.add_parser('compare', parents=[common],
                                       help='Compare OCR engines on an image')
    p_compare.add_argument('image', help='Image file to process')
    p_compare.add_argument('--output', '-o', help='Save visualization to file')
    p_compare.add_argument('--mode', '-m', default='overlay',
                           choices=['overlay', 'side_by_side', 'diff'],
                           help='Visualization mode (default: overlay)')
    p_compare.add_argument('--ground-truth', '-g', help='Ground truth text for accuracy')
    p_compare.add_argument('--show-text', '-t', action='store_true',
                           help='Show extracted text')
    p_compare.add_argument('--full-text', action='store_true',
                           help='Show full text (no truncation)')
    p_compare.set_defaults(func=cmd_compare)

    # visualize command
    p_vis = subparsers.add_parser('visualize', parents=[common],
                                   help='Generate visualization')
    p_vis.add_argument('image', help='Image file to process')
    p_vis.add_argument('--output', '-o', help='Output file path')
    p_vis.add_argument('--mode', '-m', default='overlay',
                       choices=['overlay', 'side_by_side', 'diff'],
                       help='Visualization mode')
    p_vis.add_argument('--show-confidence', action='store_true',
                       help='Show confidence scores')
    p_vis.add_argument('--min-confidence', type=float, default=0.0,
                       help='Minimum confidence threshold (0.0-1.0)')
    p_vis.set_defaults(func=cmd_visualize)

    # report command
    p_report = subparsers.add_parser('report', parents=[common],
                                      help='Generate full report')
    p_report.add_argument('image', help='Image file to process')
    p_report.add_argument('--output-dir', '-o', help='Output directory')
    p_report.add_argument('--ground-truth', '-g', help='Ground truth text')
    p_report.set_defaults(func=cmd_report)

    # batch command
    p_batch = subparsers.add_parser('batch', parents=[common],
                                     help='Process multiple images')
    p_batch.add_argument('input', help='Image file or directory')
    p_batch.add_argument('--output-dir', '-o', help='Output directory')
    p_batch.add_argument('--mode', '-m', default='overlay',
                         choices=['overlay', 'side_by_side', 'diff'])
    p_batch.add_argument('--recursive', '-r', action='store_true',
                         help='Search directories recursively')
    p_batch.set_defaults(func=cmd_batch)

    # engines command
    p_engines = subparsers.add_parser('engines', help='List available engines')
    p_engines.set_defaults(func=cmd_engines)

    # split command - generate individual images per engine
    p_split = subparsers.add_parser('split', parents=[common],
                                     help='Generate separate image per OCR engine')
    p_split.add_argument('image', help='Image file to process')
    p_split.add_argument('--output-dir', '-o', help='Output directory')
    p_split.add_argument('--font-size', '-f', type=int, default=18,
                         help='Font size for text labels (default: 18)')
    p_split.add_argument('--box-thickness', '-b', type=int, default=3,
                         help='Thickness of bounding boxes (default: 3)')
    p_split.add_argument('--show-confidence', action='store_true',
                         help='Show confidence scores on boxes')
    p_split.add_argument('--min-confidence', type=float, default=0.0,
                         help='Minimum confidence threshold (0.0-1.0)')
    p_split.set_defaults(func=cmd_split)

    # view command - interactive flip viewer
    p_view = subparsers.add_parser('view', parents=[common],
                                    help='Interactive flip viewer for comparing results')
    p_view.add_argument('input', help='Directory with split images, manifest file, or image to process')
    p_view.set_defaults(func=cmd_view)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

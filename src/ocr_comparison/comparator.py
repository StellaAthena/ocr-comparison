"""Main OCR Comparator class that orchestrates the comparison system."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from PIL import Image

from .models import OCRResult, AccuracyMetrics, BoundingBox, ComparisonResult
from .adapters import TesseractAdapter, EasyOCRAdapter, BaseOCRAdapter
from .visualizer import OCRVisualizer, visualize_overlay
from .evaluator import OCREvaluator


# Registry of available adapters
ADAPTER_REGISTRY = {
    'tesseract': TesseractAdapter,
    'easyocr': EasyOCRAdapter,
}


class OCRComparator:
    """Main class for comparing OCR engines."""

    def __init__(
        self,
        engines: List[str] = None,
        adapter_options: Dict[str, dict] = None
    ):
        """Initialize OCR comparator with specified engines.

        Args:
            engines: List of engine names to use (default: ['tesseract', 'easyocr'])
            adapter_options: Dictionary of engine-specific options
                e.g., {'tesseract': {'lang': 'eng+fra'}, 'easyocr': {'gpu': False}}
        """
        self.engines = engines or ['tesseract', 'easyocr']
        self.adapter_options = adapter_options or {}

        self._adapters: Dict[str, BaseOCRAdapter] = {}
        self._visualizer: Optional[OCRVisualizer] = None
        self._evaluator: Optional[OCREvaluator] = None

        self._initialize_adapters()

    def _initialize_adapters(self) -> None:
        """Initialize OCR adapters for configured engines."""
        for engine_name in self.engines:
            if engine_name not in ADAPTER_REGISTRY:
                raise ValueError(
                    f"Unknown engine: {engine_name}. "
                    f"Available engines: {list(ADAPTER_REGISTRY.keys())}"
                )

            adapter_class = ADAPTER_REGISTRY[engine_name]
            options = self.adapter_options.get(engine_name, {})
            self._adapters[engine_name] = adapter_class(**options)

    @property
    def visualizer(self) -> OCRVisualizer:
        """Get or create visualizer instance."""
        if self._visualizer is None:
            self._visualizer = OCRVisualizer()
        return self._visualizer

    @property
    def evaluator(self) -> OCREvaluator:
        """Get or create evaluator instance."""
        if self._evaluator is None:
            self._evaluator = OCREvaluator()
        return self._evaluator

    def process_image(
        self,
        image_path: Union[str, Path],
        engines: List[str] = None
    ) -> Dict[str, OCRResult]:
        """Run all configured engines on an image.

        Args:
            image_path: Path to the image file
            engines: Optional subset of engines to use

        Returns:
            Dictionary mapping engine names to OCR results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        engines_to_use = engines or list(self._adapters.keys())
        results = {}

        for engine_name in engines_to_use:
            if engine_name not in self._adapters:
                raise ValueError(f"Engine not configured: {engine_name}")

            adapter = self._adapters[engine_name]
            results[engine_name] = adapter.process(str(image_path))

        return results

    def visualize_overlay(
        self,
        image_path: Union[str, Path],
        results: Dict[str, OCRResult] = None,
        mode: str = 'overlay',
        output_path: str = None,
        **visualizer_kwargs
    ) -> Image.Image:
        """Create visualization with OCR results overlaid.

        Args:
            image_path: Path to the original image
            results: OCR results (will process image if not provided)
            mode: Visualization mode ('overlay', 'side_by_side', 'diff')
            output_path: Optional path to save the visualization
            **visualizer_kwargs: Additional arguments for OCRVisualizer

        Returns:
            PIL Image with visualization
        """
        image_path = Path(image_path)

        # Process image if results not provided
        if results is None:
            results = self.process_image(image_path)

        # Update visualizer settings if provided
        if visualizer_kwargs:
            self._visualizer = OCRVisualizer(**visualizer_kwargs)

        # Generate visualization based on mode
        if mode == 'overlay':
            result_image = self.visualizer.overlay_multiple(image_path, results)
        elif mode == 'side_by_side':
            result_image = self.visualizer.side_by_side(image_path, results)
        elif mode == 'diff':
            result_image = self.visualizer.diff_view(image_path, results)
        elif mode == 'textmap':
            # Create side-by-side text maps for multiple engines
            text_maps = []
            for engine_name, result in results.items():
                text_maps.append(
                    self.visualizer.text_map(image_path, result, engine_name)
                )
            if len(text_maps) == 1:
                result_image = text_maps[0]
            else:
                total_width = sum(img.width for img in text_maps)
                max_height = max(img.height for img in text_maps)
                result_image = Image.new('RGB', (total_width, max_height), (245, 245, 245))
                x_offset = 0
                for img in text_maps:
                    result_image.paste(img, (x_offset, 0))
                    x_offset += img.width
        elif mode == 'margin':
            result_image = self.visualizer.margin_view(
                image_path, list(results.values())[0],
                list(results.keys())[0]
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'overlay', 'side_by_side', 'diff', 'textmap', or 'margin'"
            )

        # Save if output path provided
        if output_path:
            result_image.save(output_path)

        return result_image

    def compare_accuracy(
        self,
        results: Dict[str, OCRResult],
        ground_truth: str = None,
        ground_truth_boxes: List[BoundingBox] = None
    ) -> Dict[str, AccuracyMetrics]:
        """Calculate accuracy metrics for OCR results.

        Args:
            results: Dictionary mapping engine names to OCR results
            ground_truth: Optional ground truth text
            ground_truth_boxes: Optional ground truth bounding boxes

        Returns:
            Dictionary mapping engine names to AccuracyMetrics
            (empty dict if no ground truth provided)
        """
        if ground_truth is None:
            return {}

        return self.evaluator.compare_engines(
            results, ground_truth, ground_truth_boxes
        )

    def generate_report(
        self,
        image_path: Union[str, Path],
        output_dir: str = None,
        ground_truth: str = None,
        ground_truth_boxes: List[BoundingBox] = None,
        visualization_mode: str = 'overlay'
    ) -> ComparisonResult:
        """Run full comparison and generate report.

        Args:
            image_path: Path to the image
            output_dir: Directory to save outputs (optional)
            ground_truth: Ground truth text (optional)
            ground_truth_boxes: Ground truth boxes (optional)
            visualization_mode: Mode for visualization

        Returns:
            ComparisonResult with all data
        """
        image_path = Path(image_path)

        # Process image
        results = self.process_image(image_path)

        # Calculate metrics if ground truth provided
        metrics = {}
        if ground_truth:
            metrics = self.compare_accuracy(results, ground_truth, ground_truth_boxes)

        # Create comparison result
        comparison = ComparisonResult(
            image_path=str(image_path),
            results=results,
            metrics=metrics,
            ground_truth=ground_truth
        )

        # Save outputs if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save visualization
            vis_path = output_dir / f"{image_path.stem}_comparison.png"
            self.visualize_overlay(
                image_path, results, mode=visualization_mode, output_path=str(vis_path)
            )

            # Save side-by-side view
            side_path = output_dir / f"{image_path.stem}_side_by_side.png"
            self.visualize_overlay(
                image_path, results, mode='side_by_side', output_path=str(side_path)
            )

            # Save text report
            report_path = output_dir / f"{image_path.stem}_report.txt"
            report_text = self.evaluator.generate_report(
                results, ground_truth, ground_truth_boxes
            )
            report_path.write_text(report_text)

            # Save JSON data
            json_path = output_dir / f"{image_path.stem}_data.json"
            self._save_json(comparison, json_path)

        return comparison

    def _save_json(self, comparison: ComparisonResult, path: Path) -> None:
        """Save comparison result as JSON.

        Args:
            comparison: ComparisonResult to save
            path: Path to save JSON file
        """
        data = {
            'image_path': comparison.image_path,
            'ground_truth': comparison.ground_truth,
            'results': {},
            'metrics': {}
        }

        for name, result in comparison.results.items():
            data['results'][name] = {
                'full_text': result.full_text,
                'word_count': result.word_count,
                'average_confidence': result.average_confidence,
                'processing_time': result.processing_time,
                'words': [
                    {
                        'text': w.text,
                        'confidence': w.confidence,
                        'bbox': {
                            'x': w.bbox.x,
                            'y': w.bbox.y,
                            'width': w.bbox.width,
                            'height': w.bbox.height
                        }
                    }
                    for w in result.words
                ]
            }

        for name, metrics in comparison.metrics.items():
            data['metrics'][name] = {
                'cer': metrics.cer,
                'wer': metrics.wer,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1': metrics.f1
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def batch_process(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: str = None,
        ground_truths: Dict[str, str] = None
    ) -> List[ComparisonResult]:
        """Process multiple images.

        Args:
            image_paths: List of image paths
            output_dir: Directory to save outputs
            ground_truths: Dictionary mapping image filenames to ground truth text

        Returns:
            List of ComparisonResult objects
        """
        ground_truths = ground_truths or {}
        results = []

        for image_path in image_paths:
            image_path = Path(image_path)
            gt = ground_truths.get(image_path.name)

            comparison = self.generate_report(
                image_path,
                output_dir=output_dir,
                ground_truth=gt
            )
            results.append(comparison)

        return results

    def check_availability(self) -> Dict[str, bool]:
        """Check which OCR engines are available.

        Returns:
            Dictionary mapping engine names to availability status
        """
        return {
            name: adapter.is_available()
            for name, adapter in self._adapters.items()
        }

    @staticmethod
    def list_available_engines() -> List[str]:
        """List all registered OCR engines.

        Returns:
            List of engine names
        """
        return list(ADAPTER_REGISTRY.keys())

    @staticmethod
    def register_adapter(name: str, adapter_class: type) -> None:
        """Register a new OCR adapter.

        Args:
            name: Name for the engine
            adapter_class: Adapter class (must inherit from BaseOCRAdapter)
        """
        if not issubclass(adapter_class, BaseOCRAdapter):
            raise TypeError("Adapter must inherit from BaseOCRAdapter")
        ADAPTER_REGISTRY[name] = adapter_class

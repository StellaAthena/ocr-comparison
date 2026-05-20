"""Visualization tools for OCR comparison results."""

import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .models import OCRResult, OCRWord, BoundingBox


# Default color scheme for engines
ENGINE_COLORS = {
    'tesseract': {
        'box': (0, 100, 255),      # Blue
        'text': (0, 50, 200),       # Dark blue
        'fill': (0, 100, 255, 50),  # Semi-transparent blue
    },
    'easyocr': {
        'box': (0, 200, 100),       # Green
        'text': (0, 150, 50),       # Dark green
        'fill': (0, 200, 100, 50),  # Semi-transparent green
    },
    'overlap': {
        'box': (180, 0, 180),       # Purple
        'text': (150, 0, 150),      # Dark purple
        'fill': (180, 0, 180, 70),  # Semi-transparent purple
    }
}

# Additional colors for more engines
EXTRA_COLORS = [
    {'box': (255, 150, 0), 'text': (200, 100, 0), 'fill': (255, 150, 0, 50)},   # Orange
    {'box': (255, 0, 100), 'text': (200, 0, 80), 'fill': (255, 0, 100, 50)},    # Pink
    {'box': (100, 200, 255), 'text': (50, 150, 200), 'fill': (100, 200, 255, 50)},  # Cyan
]


class OCRVisualizer:
    """Visualize OCR results on images."""

    def __init__(
        self,
        font_size: int = 18,
        box_thickness: int = 3,
        show_confidence: bool = False,
        min_confidence: float = 0.0
    ):
        """Initialize visualizer.

        Args:
            font_size: Font size for text labels
            box_thickness: Thickness of bounding box lines
            show_confidence: Whether to show confidence scores
            min_confidence: Minimum confidence to display (0.0-1.0)
        """
        self.font_size = font_size
        self.box_thickness = box_thickness
        self.show_confidence = show_confidence
        self.min_confidence = min_confidence
        self._font = None

    def _get_font(self) -> ImageFont.FreeTypeFont:
        """Get or create font for text rendering."""
        if self._font is None:
            try:
                # Try to use a system font
                self._font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc",
                    self.font_size
                )
            except (IOError, OSError):
                try:
                    # Fallback to DejaVu (common on Linux)
                    self._font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        self.font_size
                    )
                except (IOError, OSError):
                    # Use default font
                    self._font = ImageFont.load_default()
        return self._font

    def _get_engine_colors(self, engine_name: str) -> Dict[str, Tuple]:
        """Get colors for an engine, assigning new colors if needed."""
        if engine_name in ENGINE_COLORS:
            return ENGINE_COLORS[engine_name]

        # Assign from extra colors
        idx = hash(engine_name) % len(EXTRA_COLORS)
        return EXTRA_COLORS[idx]

    def overlay_single(
        self,
        image: Union[str, Path, Image.Image],
        result: OCRResult,
        engine_name: Optional[str] = None,
        draw_boxes: bool = True
    ) -> Image.Image:
        """Overlay single OCR result on image.

        Args:
            image: Image path or PIL Image
            result: OCR result to overlay
            engine_name: Override engine name for colors
            draw_boxes: Whether to draw bounding boxes (if False, only draws labels)

        Returns:
            PIL Image with overlay
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGBA')
        else:
            img = image.convert('RGBA')

        # Create overlay layer
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        name = engine_name or result.engine_name
        colors = self._get_engine_colors(name)
        font = self._get_font()

        for word in result.words:
            if word.confidence < self.min_confidence:
                continue

            self._draw_word(draw, word, colors, font, draw_boxes=draw_boxes)

        # Composite overlay onto image
        return Image.alpha_composite(img, overlay).convert('RGB')

    def overlay_multiple(
        self,
        image: Union[str, Path, Image.Image],
        results: Dict[str, OCRResult]
    ) -> Image.Image:
        """Overlay multiple OCR results on single image.

        Args:
            image: Image path or PIL Image
            results: Dictionary mapping engine name to OCR result

        Returns:
            PIL Image with all results overlaid
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGBA')
        else:
            img = image.convert('RGBA')

        # Create overlay layer
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._get_font()

        for engine_name, result in results.items():
            colors = self._get_engine_colors(engine_name)

            for word in result.words:
                if word.confidence < self.min_confidence:
                    continue

                self._draw_word(draw, word, colors, font)

        # Composite overlay onto image
        return Image.alpha_composite(img, overlay).convert('RGB')

    def side_by_side(
        self,
        image: Union[str, Path, Image.Image],
        results: Dict[str, OCRResult],
        labels: bool = True
    ) -> Image.Image:
        """Create side-by-side comparison of OCR results.

        Args:
            image: Original image
            results: Dictionary mapping engine name to OCR result
            labels: Whether to add engine name labels

        Returns:
            Combined image with side-by-side comparisons
        """
        # Create individual overlays
        overlays = []
        for engine_name, result in results.items():
            overlay = self.overlay_single(image, result, engine_name)
            overlays.append((engine_name, overlay))

        if not overlays:
            if isinstance(image, (str, Path)):
                return Image.open(image).convert('RGB')
            return image.convert('RGB')

        # Calculate combined dimensions
        sample = overlays[0][1]
        single_width = sample.width
        single_height = sample.height
        label_height = 30 if labels else 0

        total_width = single_width * len(overlays)
        total_height = single_height + label_height

        # Create combined image
        combined = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(combined)
        font = self._get_font()

        for i, (engine_name, overlay) in enumerate(overlays):
            x_offset = i * single_width

            # Paste overlay
            combined.paste(overlay, (x_offset, label_height))

            # Add label
            if labels:
                colors = self._get_engine_colors(engine_name)
                label = engine_name.upper()
                draw.rectangle(
                    [(x_offset, 0), (x_offset + single_width, label_height)],
                    fill=colors['box']
                )
                draw.text(
                    (x_offset + 10, 5),
                    label,
                    fill=(255, 255, 255),
                    font=font
                )

        return combined

    def diff_view(
        self,
        image: Union[str, Path, Image.Image],
        results: Dict[str, OCRResult],
        iou_threshold: float = 0.5
    ) -> Image.Image:
        """Create diff view highlighting disagreements between engines.

        Args:
            image: Original image
            results: Dictionary mapping engine name to OCR result
            iou_threshold: IoU threshold for matching boxes

        Returns:
            Image highlighting differences
        """
        if len(results) < 2:
            return self.overlay_multiple(image, results)

        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGBA')
        else:
            img = image.convert('RGBA')

        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = self._get_font()

        engine_names = list(results.keys())
        all_words = [(name, w) for name, r in results.items() for w in r.words
                     if w.confidence >= self.min_confidence]

        # Find matching and unique detections
        matched = set()
        unique = []

        for i, (name1, word1) in enumerate(all_words):
            is_matched = False
            for j, (name2, word2) in enumerate(all_words):
                if i >= j or name1 == name2:
                    continue

                iou = word1.bbox.iou(word2.bbox)
                if iou >= iou_threshold:
                    # Check if text matches
                    if word1.text.lower() == word2.text.lower():
                        matched.add(i)
                        matched.add(j)
                    is_matched = True

            if not is_matched:
                unique.append((name1, word1))

        # Draw unique detections (disagreements) with engine colors
        for engine_name, word in unique:
            colors = self._get_engine_colors(engine_name)
            self._draw_word(draw, word, colors, font, highlight=True)

        # Draw matched detections in muted gray
        gray_colors = {
            'box': (150, 150, 150),
            'text': (100, 100, 100),
            'fill': (150, 150, 150, 30)
        }
        for i, (name, word) in enumerate(all_words):
            if i in matched:
                self._draw_word(draw, word, gray_colors, font)

        return Image.alpha_composite(img, overlay).convert('RGB')

    def _draw_word(
        self,
        draw: ImageDraw.ImageDraw,
        word: OCRWord,
        colors: Dict[str, Tuple],
        font: ImageFont.FreeTypeFont,
        highlight: bool = False,
        draw_boxes: bool = True
    ) -> None:
        """Draw a single word with bounding box and label.

        Args:
            draw: ImageDraw object
            word: OCR word to draw
            colors: Color scheme dictionary
            font: Font for text
            highlight: Whether to highlight (thicker box)
            draw_boxes: Whether to draw bounding boxes (if False, only draws labels)
        """
        bbox = word.bbox
        thickness = self.box_thickness * 2 if highlight else self.box_thickness

        if draw_boxes:
            # Draw semi-transparent fill
            if 'fill' in colors and len(colors['fill']) == 4:
                draw.rectangle(
                    [(bbox.x, bbox.y), (bbox.x2, bbox.y2)],
                    fill=colors['fill']
                )

            # Draw box outline
            for i in range(thickness):
                draw.rectangle(
                    [(bbox.x - i, bbox.y - i), (bbox.x2 + i, bbox.y2 + i)],
                    outline=colors['box']
                )

        # Draw text label above box
        label = word.text
        if self.show_confidence:
            label = f"{word.text} ({word.confidence:.0%})"

        # Position text above box
        text_y = max(0, bbox.y - self.font_size - 2)
        draw.text(
            (bbox.x, text_y),
            label,
            fill=colors['text'],
            font=font
        )

    def _get_font_for_size(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a font at a specific size."""
        try:
            return ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", size
            )
        except (IOError, OSError):
            try:
                return ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
                )
            except (IOError, OSError):
                return ImageFont.load_default()

    def text_map(
        self,
        image: Union[str, Path, Image.Image],
        result: OCRResult,
        engine_name: Optional[str] = None,
        background_color: Tuple[int, int, int] = (245, 245, 245),
    ) -> Image.Image:
        """Create a text map view: extracted text rendered on a blank background.

        Each detected word is drawn inside its bounding box region, with text
        auto-sized to fit. Areas with no detections remain blank, creating a
        skeleton of what the OCR engine saw.

        Args:
            image: Image path or PIL Image (used only for dimensions)
            result: OCR result to render
            engine_name: Override engine name for colors
            background_color: RGB background color

        Returns:
            PIL Image with text map
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image
        width, height = img.size

        text_map_img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(text_map_img)

        name = engine_name or result.engine_name
        colors = self._get_engine_colors(name)

        for word in result.words:
            if word.confidence < self.min_confidence:
                continue

            bbox = word.bbox
            if bbox.width < 2 or bbox.height < 2:
                continue

            # Draw thin gray box outline
            draw.rectangle(
                [(bbox.x, bbox.y), (bbox.x2, bbox.y2)],
                outline=(200, 200, 200),
                width=1,
            )

            font_size = max(6, int(bbox.height * 1.0))
            font = self._get_font_for_size(font_size)

            # Shrink font if text overflows the bounding box width
            text_bbox = draw.textbbox((0, 0), word.text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            if text_width > bbox.width and bbox.width > 0:
                font_size = max(6, int(font_size * bbox.width / text_width))
                font = self._get_font_for_size(font_size)

            draw.text(
                (bbox.x, bbox.y),
                word.text,
                fill=colors['text'],
                font=font,
            )

        return text_map_img

    def margin_view(
        self,
        image: Union[str, Path, Image.Image],
        result: OCRResult,
        engine_name: Optional[str] = None,
        margin_width: int = 350,
        max_margin_width: int = 800,
        margin_font_size: int = 13,
        line_grouping_threshold: int = 10
    ) -> Image.Image:
        """Create a margin-annotated visualization of OCR results.

        Draws bounding box outlines on the original image and places
        extracted text in a right margin, connected by thin leader lines.
        Words on the same horizontal line are grouped into a single
        margin annotation. The margin auto-expands to fit text, and
        wraps lines that exceed max_margin_width.
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGBA')
        else:
            img = image.convert('RGBA')

        name = engine_name or result.engine_name
        colors = self._get_engine_colors(name)

        margin_font = self._get_font_for_size(margin_font_size)

        words = [w for w in result.words if w.confidence >= self.min_confidence]
        lines = self._group_words_into_lines(words, line_grouping_threshold)
        lines.sort(key=lambda line: min(w.bbox.y for w in line))

        margin_padding = 20
        margin_x_offset = 10
        line_height = margin_font_size + 6
        margin_labels = []

        for line_words in lines:
            line_words.sort(key=lambda w: w.bbox.x)
            line_text = " ".join(w.text for w in line_words)
            if self.show_confidence:
                avg_conf = sum(w.confidence for w in line_words) / len(line_words)
                line_text += f" ({avg_conf:.0%})"

            min_x = min(w.bbox.x for w in line_words)
            min_y = min(w.bbox.y for w in line_words)
            max_x2 = max(w.bbox.x2 for w in line_words)
            max_y2 = max(w.bbox.y2 for w in line_words)
            center_y = (min_y + max_y2) // 2

            margin_labels.append({
                'desired_y': center_y - margin_font_size // 2,
                'text': line_text,
                'leader_start_x': max_x2,
                'leader_start_y': center_y,
                'words': line_words,
            })

        # Auto-size margin to fit longest text
        tmp_img = Image.new('RGBA', (1, 1))
        tmp_draw = ImageDraw.Draw(tmp_img)

        max_text_width = 0
        for label in margin_labels:
            bbox = tmp_draw.textbbox((0, 0), label['text'], font=margin_font)
            max_text_width = max(max_text_width, bbox[2] - bbox[0])

        required_width = max_text_width + margin_x_offset + margin_padding
        actual_margin_width = max(margin_width, required_width)

        if actual_margin_width > max_margin_width:
            actual_margin_width = max_margin_width
            usable_width = actual_margin_width - margin_x_offset - margin_padding
            self._wrap_margin_labels(margin_labels, margin_font, tmp_draw, usable_width)

        # Create canvas
        new_width = img.width + actual_margin_width
        canvas = Image.new('RGBA', (new_width, img.height), (245, 245, 245, 255))
        canvas.paste(img, (0, 0))

        draw = ImageDraw.Draw(canvas)
        draw.line(
            [(img.width, 0), (img.width, img.height)],
            fill=(200, 200, 200, 255), width=1
        )

        margin_x = img.width + margin_x_offset

        # Resolve overlaps
        for label in margin_labels:
            label_lines = label['text'].count('\n') + 1
            label['height'] = label_lines * line_height
        self._resolve_margin_overlaps_variable(margin_labels, img.height)

        # Draw
        for label in margin_labels:
            actual_y = label['actual_y']

            for word in label['words']:
                bbox = word.bbox
                draw.rectangle(
                    [(bbox.x, bbox.y), (bbox.x2, bbox.y2)],
                    outline=(200, 200, 200), width=1
                )

            leader_color = colors['box'] + (100,) if len(colors['box']) == 3 else colors['box']
            leader_start_x = label['leader_start_x']
            leader_start_y = label['leader_start_y']
            margin_text_y = actual_y + margin_font_size // 2

            mid_x = img.width + 3
            draw.line(
                [(leader_start_x + 2, leader_start_y), (mid_x, leader_start_y)],
                fill=leader_color, width=1
            )
            if abs(leader_start_y - margin_text_y) > 1:
                draw.line(
                    [(mid_x, leader_start_y), (mid_x, margin_text_y)],
                    fill=leader_color, width=1
                )
            draw.line(
                [(mid_x, margin_text_y), (margin_x - 2, margin_text_y)],
                fill=leader_color, width=1
            )

            text_fill = colors['text'] + (255,) if len(colors['text']) == 3 else colors['text']
            draw.text((margin_x, actual_y), label['text'], fill=text_fill, font=margin_font)

        return canvas.convert('RGB')

    def _group_words_into_lines(
        self, words: List[OCRWord], threshold: int = 10
    ) -> List[List[OCRWord]]:
        """Group words into lines based on vertical proximity."""
        if not words:
            return []
        sorted_words = sorted(words, key=lambda w: w.bbox.center[1])
        lines = []
        current_line = [sorted_words[0]]
        for word in sorted_words[1:]:
            prev_center_y = sum(w.bbox.center[1] for w in current_line) / len(current_line)
            if abs(word.bbox.center[1] - prev_center_y) <= threshold:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)
        return lines

    def _wrap_margin_labels(self, labels, font, draw, max_width):
        """Wrap margin label text to fit within max_width pixels."""
        for label in labels:
            text = label['text']
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= max_width:
                continue
            if len(text) > 0:
                avg_char_w = text_w / len(text)
                chars_per_line = max(1, int(max_width / avg_char_w))
            else:
                continue
            wrapped = textwrap.fill(text, width=chars_per_line)
            for wrapped_line in wrapped.split('\n'):
                lbbox = draw.textbbox((0, 0), wrapped_line, font=font)
                if lbbox[2] - lbbox[0] > max_width and chars_per_line > 5:
                    chars_per_line = max(5, chars_per_line - 2)
                    wrapped = textwrap.fill(text, width=chars_per_line)
                    break
            label['text'] = wrapped

    def _resolve_margin_overlaps_variable(self, labels, image_height):
        """Resolve vertical overlaps for labels with variable heights."""
        if not labels:
            return
        for label in labels:
            label['actual_y'] = max(0, label['desired_y'])
        for i in range(1, len(labels)):
            prev_bottom = labels[i - 1]['actual_y'] + labels[i - 1]['height']
            if labels[i]['actual_y'] < prev_bottom:
                labels[i]['actual_y'] = prev_bottom
        if labels:
            last = labels[-1]
            if last['actual_y'] + last['height'] > image_height:
                last['actual_y'] = max(0, image_height - last['height'])
                for i in range(len(labels) - 2, -1, -1):
                    next_top = labels[i + 1]['actual_y']
                    if labels[i]['actual_y'] + labels[i]['height'] > next_top:
                        labels[i]['actual_y'] = max(0, next_top - labels[i]['height'])

    def create_legend(
        self,
        engine_names: List[str],
        width: int = 200,
        height: int = None
    ) -> Image.Image:
        """Create a legend image for the color scheme.

        Args:
            engine_names: List of engine names to include
            width: Width of legend image
            height: Height (auto-calculated if None)

        Returns:
            Legend image
        """
        item_height = 30
        height = height or (len(engine_names) * item_height + 20)

        legend = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(legend)
        font = self._get_font()

        for i, name in enumerate(engine_names):
            colors = self._get_engine_colors(name)
            y = 10 + i * item_height

            # Draw color swatch
            draw.rectangle(
                [(10, y), (30, y + 20)],
                fill=colors['box'],
                outline=(0, 0, 0)
            )

            # Draw label
            draw.text((40, y + 3), name, fill=(0, 0, 0), font=font)

        return legend


def visualize_overlay(
    image_path: str,
    results: Dict[str, OCRResult],
    mode: str = 'overlay',
    output_path: str = None,
    **kwargs
) -> Image.Image:
    """Convenience function to create OCR visualization.

    Args:
        image_path: Path to input image
        results: Dictionary mapping engine names to OCR results
        mode: Visualization mode ('overlay', 'side_by_side', 'diff')
        output_path: Optional path to save result
        **kwargs: Additional arguments for OCRVisualizer

    Returns:
        Resulting PIL Image
    """
    visualizer = OCRVisualizer(**kwargs)

    if mode == 'overlay':
        result = visualizer.overlay_multiple(image_path, results)
    elif mode == 'side_by_side':
        result = visualizer.side_by_side(image_path, results)
    elif mode == 'diff':
        result = visualizer.diff_view(image_path, results)
    elif mode == 'textmap':
        # Create side-by-side text maps for all engines
        maps = []
        for engine_name, engine_result in results.items():
            maps.append(visualizer.text_map(image_path, engine_result, engine_name))
        if len(maps) == 1:
            result = maps[0]
        else:
            total_width = sum(m.width for m in maps)
            max_height = max(m.height for m in maps)
            result = Image.new('RGB', (total_width, max_height), (255, 255, 255))
            x_offset = 0
            for m in maps:
                result.paste(m, (x_offset, 0))
                x_offset += m.width
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'overlay', 'side_by_side', 'diff', or 'textmap'")

    if output_path:
        result.save(output_path)

    return result


def save_individual_images(
    image_path: str,
    results: Dict[str, OCRResult],
    output_dir: str,
    base_name: str = None,
    mode: str = 'textmap',
    **kwargs
) -> List[str]:
    """Save separate visualization images for each OCR engine.

    Creates two images per engine:
    - One with bounding box outlines on the original image
    - One alternate view (text map or margin) toggled with 'I' key

    Args:
        image_path: Path to input image
        results: Dictionary mapping engine names to OCR results
        output_dir: Directory to save output images
        base_name: Base filename (default: derived from input)
        mode: Alternate view mode ('textmap' or 'margin')
        **kwargs: Additional arguments for OCRVisualizer

    Returns:
        List of paths to saved images (boxes view)
    """
    from pathlib import Path

    visualizer = OCRVisualizer(**kwargs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_name is None:
        base_name = Path(image_path).stem

    saved_paths = []
    alt_paths = []
    alt_suffix = '_margin' if mode == 'margin' else '_textmap'

    for engine_name, result in results.items():
        colors = visualizer._get_engine_colors(engine_name)

        # Original image with bounding box outlines only
        img = Image.open(image_path).convert('RGBA')
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        for word in result.words:
            if word.confidence < visualizer.min_confidence:
                continue
            bbox = word.bbox
            draw_overlay.rectangle(
                [(bbox.x, bbox.y), (bbox.x2, bbox.y2)],
                outline=(200, 200, 200),
                width=1
            )

        img_with_boxes = Image.alpha_composite(img, overlay).convert('RGB')
        labeled_boxes = _add_label_bar(img_with_boxes, engine_name, colors)
        output_path = output_dir / f"{base_name}_{engine_name}.png"
        labeled_boxes.save(output_path)
        saved_paths.append(str(output_path))

        # Alternate view
        if mode == 'margin':
            img_alt = visualizer.margin_view(image_path, result, engine_name)
        else:
            img_alt = visualizer.text_map(image_path, result, engine_name)
        labeled_alt = _add_label_bar(img_alt, engine_name, colors)
        alt_path = output_dir / f"{base_name}_{engine_name}{alt_suffix}.png"
        labeled_alt.save(alt_path)
        alt_paths.append(str(alt_path))

    # Save manifest file for the viewer
    manifest_path = output_dir / f"{base_name}_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"# Format: boxes_path,alt_path\n")
        for boxes_path, alt_path in zip(saved_paths, alt_paths):
            f.write(f"{boxes_path},{alt_path}\n")

    return saved_paths


def _add_label_bar(
    image: Image.Image,
    label: str,
    colors: Dict[str, Tuple],
    bar_height: int = 40
) -> Image.Image:
    """Add a colored label bar at the top of an image.

    Args:
        image: PIL Image to label
        label: Text label
        colors: Color scheme with 'box' color
        bar_height: Height of label bar

    Returns:
        New image with label bar
    """
    new_height = image.height + bar_height
    labeled = Image.new('RGB', (image.width, new_height), (255, 255, 255))

    # Draw label bar
    draw = ImageDraw.Draw(labeled)
    draw.rectangle([(0, 0), (image.width, bar_height)], fill=colors['box'])

    # Draw label text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except (IOError, OSError):
            font = ImageFont.load_default()

    # Center the text
    bbox = draw.textbbox((0, 0), label.upper(), font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (image.width - text_width) // 2
    draw.text((text_x, 8), label.upper(), fill=(255, 255, 255), font=font)

    # Paste original image below
    labeled.paste(image, (0, bar_height))

    return labeled


def create_flip_viewer(image_paths: List[str], title: str = "OCR Comparison Viewer"):
    """Create and launch an interactive viewer for comparing OCR results.

    Use arrow keys (Left/Right) to flip between engines.
    Use 'I' to toggle between original (with bounding boxes) and text map view.
    The images remain aligned so differences are easy to spot.

    Args:
        image_paths: List of paths to images to compare (or "boxes,textmap" pairs)
        title: Window title
    """
    import tkinter as tk
    from tkinter import ttk

    class FlipViewer:
        def __init__(self, paths: List[str], title: str):
            # Parse paths - check if we have pairs (boxes,textmap) or single paths
            self.image_pairs = []  # List of (boxes_path, textmap_path) tuples
            self.has_textmap_versions = False

            for path in paths:
                if ',' in path:
                    # Pair format: boxes_path,textmap_path
                    parts = [p.strip() for p in path.split(',')]
                    boxes_path = parts[0]
                    textmap_path = parts[-1]
                    self.image_pairs.append((boxes_path, textmap_path))
                    self.has_textmap_versions = True
                else:
                    # Single path - check if alt version exists (textmap or margin)
                    p = Path(path)
                    alt_path = None
                    for suffix in ('_textmap', '_margin'):
                        candidate = p.parent / f"{p.stem}{suffix}{p.suffix}"
                        if candidate.exists():
                            alt_path = str(candidate)
                            break
                    if alt_path:
                        self.image_pairs.append((path, alt_path))
                        self.has_textmap_versions = True
                    else:
                        self.image_pairs.append((path, None))

            self.current_index = 0
            self.show_original = True  # True = original with boxes, False = text map

            # Load all images
            self.boxes_images = []
            self.textmap_images = []
            self.boxes_photos = []
            self.textmap_photos = []

            for boxes_path, textmap_path in self.image_pairs:
                self.boxes_images.append(Image.open(boxes_path))
                if textmap_path and Path(textmap_path).exists():
                    self.textmap_images.append(Image.open(textmap_path))
                else:
                    self.textmap_images.append(None)

            # Create window
            self.root = tk.Tk()
            self.root.title(title)

            # Get screen dimensions for sizing
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Calculate display size (fit to screen with margin)
            img_width, img_height = self.boxes_images[0].size
            max_width = int(screen_width * 0.9)
            max_height = int(screen_height * 0.85)

            scale = min(max_width / img_width, max_height / img_height, 1.0)
            self.display_width = int(img_width * scale)
            self.display_height = int(img_height * scale)
            self.scale = scale

            # Pre-render all images at display size
            from PIL import ImageTk

            for i, img in enumerate(self.boxes_images):
                if scale < 1.0:
                    display_img = img.resize(
                        (self.display_width, self.display_height),
                        Image.Resampling.LANCZOS
                    )
                else:
                    display_img = img
                self.boxes_photos.append(ImageTk.PhotoImage(display_img))

                # Also render textmap version if available
                textmap_img = self.textmap_images[i]
                if textmap_img:
                    if scale < 1.0:
                        display_textmap = textmap_img.resize(
                            (self.display_width, self.display_height),
                            Image.Resampling.LANCZOS
                        )
                    else:
                        display_textmap = textmap_img
                    self.textmap_photos.append(ImageTk.PhotoImage(display_textmap))
                else:
                    self.textmap_photos.append(None)

            # Create main frame
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True)

            # Image display
            self.canvas = tk.Canvas(
                self.main_frame,
                width=self.display_width,
                height=self.display_height,
                bg='gray20'
            )
            self.canvas.pack(pady=10)

            # Navigation info
            self.info_frame = ttk.Frame(self.main_frame)
            self.info_frame.pack(fill=tk.X, padx=20, pady=10)

            self.label_var = tk.StringVar()
            self.update_label()

            info_label = ttk.Label(
                self.info_frame,
                textvariable=self.label_var,
                font=('Helvetica', 14)
            )
            info_label.pack()

            help_parts = ["<- -> to switch engines", "I to toggle original/text map"]
            if len(self.image_pairs) > 2:
                help_parts.append("1-9 to jump")
            help_parts.append("Q to quit")
            help_text = "  |  ".join(help_parts)

            help_label = ttk.Label(
                self.info_frame,
                text=help_text,
                font=('Helvetica', 10),
                foreground='gray'
            )
            help_label.pack(pady=5)

            # Bind keys
            self.root.bind('<Left>', self.prev_image)
            self.root.bind('<Right>', self.next_image)
            self.root.bind('<Up>', self.prev_image)
            self.root.bind('<Down>', self.next_image)
            self.root.bind('i', self.toggle_view)
            self.root.bind('I', self.toggle_view)
            self.root.bind('q', lambda e: self.root.quit())
            self.root.bind('Q', lambda e: self.root.quit())
            self.root.bind('<Escape>', lambda e: self.root.quit())

            # Number keys 1-9 for quick jump
            for i in range(1, 10):
                self.root.bind(str(i), lambda e, idx=i-1: self.jump_to(idx))

            # Display first image
            self.show_current()

        def update_label(self):
            boxes_path = self.image_pairs[self.current_index][0]
            path = Path(boxes_path)
            # Extract engine name from filename (assumes format: base_engine.png)
            parts = path.stem.rsplit('_', 1)
            engine = parts[-1] if len(parts) > 1 else path.stem

            view_label = "ORIGINAL" if self.show_original else "TEXT MAP"
            self.label_var.set(
                f"{engine.upper()}  ({self.current_index + 1} / {len(self.image_pairs)})  [{view_label}]"
            )

        def show_current(self):
            self.canvas.delete("all")

            if self.show_original or self.textmap_photos[self.current_index] is None:
                photo = self.boxes_photos[self.current_index]
            else:
                photo = self.textmap_photos[self.current_index]

            self.canvas.create_image(
                self.display_width // 2,
                self.display_height // 2,
                image=photo,
                anchor=tk.CENTER
            )
            self.update_label()

        def toggle_view(self, event=None):
            if self.has_textmap_versions:
                self.show_original = not self.show_original
                self.show_current()

        def next_image(self, event=None):
            self.current_index = (self.current_index + 1) % len(self.image_pairs)
            self.show_current()

        def prev_image(self, event=None):
            self.current_index = (self.current_index - 1) % len(self.image_pairs)
            self.show_current()

        def jump_to(self, index: int):
            if 0 <= index < len(self.image_pairs):
                self.current_index = index
                self.show_current()

        def run(self):
            self.root.mainloop()

    viewer = FlipViewer(image_paths, title)
    viewer.run()

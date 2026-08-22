"""Defect Visualization Engine.

Generates annotated inspection images with bounding boxes, severity badges,
and defect classification overlays using PIL and OpenCV.
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Severity color schemes (RGB)
SEVERITY_COLORS = {
    "minor": (46, 204, 113),      # Emerald Green
    "moderate": (243, 156, 18),   # Amber Orange
    "severe": (231, 76, 60),      # Crimson Red
    "none": (52, 152, 219),       # Sky Blue
}


def get_font(size: int = 14) -> ImageFont.ImageFont:
    """Load default or truetype font safely."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()


def visualize_detections(
    image_input: Union[str, Path, Image.Image],
    detections: List[Dict[str, Any]],
    overall_severity: str = "none",
    save_path: Union[str, Path, None] = None,
) -> Image.Image:
    """Draw bounding boxes, labels, and an industrial inspection summary badge on the image.

    Args:
        image_input: Filepath or PIL Image.
        detections: List of detection dictionaries.
        overall_severity: Overall severity string.
        save_path: Optional file path to save the annotated image.

    Returns:
        Annotated PIL Image.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    draw = ImageDraw.Draw(img)
    font_main = get_font(13)
    font_header = get_font(15)

    img_w, img_h = img.size

    # Draw each detection box
    for det in detections:
        bbox = det.get("bbox", {})
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))

        cls_name = det.get("class_name", "defect")
        conf = det.get("confidence", 0.0)
        sev = det.get("severity", "minor").lower()
        area_pct = det.get("area_percentage", 0.0)

        color = SEVERITY_COLORS.get(sev, (255, 255, 0))

        # Draw thick bounding box
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

        # Label text
        label_text = f"{cls_name.upper()} | {conf:.0%} | {sev.title()} ({area_pct:.1f}%)"
        
        # Calculate text bounding box
        text_bbox = draw.textbbox((x1, y1), label_text, font=font_main)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        label_y = max(0.0, y1 - th - 6)
        label_bg = [x1, label_y, x1 + tw + 8, label_y + th + 6]

        # Draw filled label background
        draw.rectangle(label_bg, fill=color)
        draw.text((x1 + 4, label_y + 2), label_text, fill=(255, 255, 255), font=font_main)

    # Draw Top Header Inspection Badge
    defect_count = len(detections)
    header_color = SEVERITY_COLORS.get(overall_severity.lower(), (100, 100, 100))
    header_text = f"QC STATUS: {overall_severity.upper()} | DEFECTS: {defect_count}"

    head_bbox = draw.textbbox((10, 10), header_text, font=font_header)
    hw = head_bbox[2] - head_bbox[0]
    hh = head_bbox[3] - head_bbox[1]

    draw.rectangle([10, 10, 10 + hw + 16, 10 + hh + 12], fill=(20, 24, 33))
    draw.rectangle([10, 10, 10 + hw + 16, 10 + hh + 12], outline=header_color, width=2)
    draw.text((18, 16), header_text, fill=header_color, font=font_header)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path, "JPEG", quality=95)

    return img


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 JPEG string."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

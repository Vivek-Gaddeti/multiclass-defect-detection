"""Tests for dataset conversion and configuration utilities."""

from pathlib import Path
import pytest
from src.data.prepare_dataset import convert_to_yolo_format, load_config
from src.inference.visualize import visualize_detections


def test_config_loading():
    """Verify master configuration loads expected sections."""
    config = load_config("configs/config.yaml")
    assert "project" in config
    assert "dataset" in config
    assert "model" in config
    assert "severity" in config
    assert len(config["dataset"]["classes"]) == 6


def test_yolo_coord_conversion():
    """Verify conversion from pixel bbox to normalized YOLO bbox."""
    # 200x200 image, box from (50, 50) to (150, 150)
    # x_center = 100/200 = 0.5, y_center = 100/200 = 0.5, w = 100/200 = 0.5, h = 100/200 = 0.5
    xc, yc, w, h = convert_to_yolo_format(50, 50, 150, 150, 200, 200)
    assert pytest.approx(xc, 0.001) == 0.5
    assert pytest.approx(yc, 0.001) == 0.5
    assert pytest.approx(w, 0.001) == 0.5
    assert pytest.approx(h, 0.001) == 0.5


def test_visualizer_annotation():
    """Verify image annotation engine renders properly without error."""
    from PIL import Image

    test_img = Image.new("RGB", (300, 300), color=(100, 100, 100))
    sample_detections = [
        {
            "class_id": 0,
            "class_name": "scratches",
            "confidence": 0.88,
            "bbox": {"x1": 20, "y1": 20, "x2": 100, "y2": 80},
            "area_pixels": 4800,
            "area_percentage": 5.33,
            "severity": "severe",
        }
    ]

    annotated = visualize_detections(
        image_input=test_img,
        detections=sample_detections,
        overall_severity="severe",
    )
    assert annotated.size == (300, 300)

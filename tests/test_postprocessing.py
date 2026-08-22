"""Unit tests for defect post-processing, IoU suppression, and severity calculation."""

import pytest
from src.postprocessing.defect_processor import (
    DefectProcessor,
    BoundingBox,
    Detection,
    SeverityLevel,
    ProcessedResult,
)


@pytest.fixture
def processor():
    """Default processor fixture."""
    return DefectProcessor(
        min_confidence=0.25,
        duplicate_iou_threshold=0.45,
        minor_max_percent=1.0,
        moderate_max_percent=5.0,
        min_box_area_pixels=16.0,
    )


def test_bounding_box_properties():
    """Verify BoundingBox geometry, width, height, and area calculations."""
    box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
    assert box.width == 100.0
    assert box.height == 100.0
    assert box.area == 10000.0
    d = box.to_dict()
    assert d["x1"] == 10.0
    assert d["x2"] == 110.0


def test_iou_calculation(processor):
    """Verify IoU calculation for identical, disjoint, and intersecting boxes."""
    box_a = BoundingBox(0, 0, 100, 100)
    box_b = BoundingBox(0, 0, 100, 100)
    # Identical
    assert pytest.approx(processor.calculate_iou(box_a, box_b), 0.01) == 1.0

    # Disjoint
    box_c = BoundingBox(200, 200, 300, 300)
    assert processor.calculate_iou(box_a, box_c) == 0.0

    # Partial overlap (50x100 overlap)
    box_d = BoundingBox(50, 0, 150, 100)
    # intersection: 50 * 100 = 5000, union: 10000 + 10000 - 5000 = 15000 => 5000/15000 = 1/3
    assert pytest.approx(processor.calculate_iou(box_a, box_d), 0.01) == 1.0 / 3.0


def test_clamp_and_validate_bbox(processor):
    """Verify clamping to image boundaries and rejecting inverted coordinates."""
    # Box extending past boundaries
    out_box = BoundingBox(-10, -5, 650, 490)
    clamped = processor.validate_and_clamp_bbox(out_box, image_width=640, image_height=480)
    assert clamped is not None
    assert clamped.x1 == 0.0
    assert clamped.y1 == 0.0
    assert clamped.x2 == 640.0
    assert clamped.y2 == 480.0

    # Inverted box (invalid)
    inverted = BoundingBox(100, 100, 50, 50)
    assert processor.validate_and_clamp_bbox(inverted, 640, 480) is None

    # Too small area (< min_box_area_pixels)
    tiny = BoundingBox(10, 10, 12, 12)  # area = 4
    assert processor.validate_and_clamp_bbox(tiny, 640, 480) is None


def test_severity_classification(processor):
    """Verify minor, moderate, and severe threshold classification."""
    # Image area = 1000 x 1000 = 1,000,000 px
    img_w, img_h = 1000, 1000

    # Minor: area = 5,000 px = 0.5% (< 1%)
    box_minor = BoundingBox(0, 0, 50, 100)
    _, pct_min, sev_min = processor.compute_defect_area_and_severity(box_minor, img_w, img_h)
    assert pct_min == 0.5
    assert sev_min == SeverityLevel.MINOR

    # Moderate: area = 30,000 px = 3.0% (1% <= x < 5%)
    box_mod = BoundingBox(0, 0, 150, 200)
    _, pct_mod, sev_mod = processor.compute_defect_area_and_severity(box_mod, img_w, img_h)
    assert pct_mod == 3.0
    assert sev_mod == SeverityLevel.MODERATE

    # Severe: area = 80,000 px = 8.0% (>= 5%)
    box_sev = BoundingBox(0, 0, 200, 400)
    _, pct_sev, sev_sev = processor.compute_defect_area_and_severity(box_sev, img_w, img_h)
    assert pct_sev == 8.0
    assert sev_sev == SeverityLevel.SEVERE


def test_duplicate_suppression(processor):
    """Verify Non-Maximum Suppression removes overlapping boxes of same class."""
    det1 = Detection(
        class_id=0,
        class_name="crazing",
        confidence=0.90,
        bbox=BoundingBox(0, 0, 100, 100),
    )
    # Highly overlapping box with lower confidence
    det2 = Detection(
        class_id=0,
        class_name="crazing",
        confidence=0.75,
        bbox=BoundingBox(5, 5, 105, 105),
    )
    # Non-overlapping box
    det3 = Detection(
        class_id=0,
        class_name="crazing",
        confidence=0.80,
        bbox=BoundingBox(200, 200, 300, 300),
    )

    suppressed = processor.suppress_duplicates([det1, det2, det3], class_aware=True)
    assert len(suppressed) == 2
    assert suppressed[0].confidence == 0.90
    assert suppressed[1].confidence == 0.80


def test_empty_detections_handling(processor):
    """Verify processing empty detections returns zero count and 'none' severity."""
    result: ProcessedResult = processor.process([], image_width=640, image_height=480)
    assert result.defect_count == 0
    assert result.overall_severity == SeverityLevel.NONE
    assert result.total_affected_area_percent == 0.0
    assert len(result.detections) == 0

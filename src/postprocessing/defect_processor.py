"""Defect post-processing, IoU duplicate suppression, area calculation, and severity estimation.

Follows industrial quality inspection heuristics for defect assessment.
Note: Bounding-box area is an approximation of the affected surface area.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    """Defect severity level classification."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class BoundingBox:
    """Bounding box coordinates in pixel space (x1, y1, x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
        }


@dataclass
class Detection:
    """Raw or processed detection entity."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    area_pixels: float = 0.0
    area_percentage: float = 0.0
    severity: SeverityLevel = SeverityLevel.MINOR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict(),
            "area_pixels": round(float(self.area_pixels), 2),
            "area_percentage": round(float(self.area_percentage), 4),
            "severity": self.severity.value,
        }


@dataclass
class ProcessedResult:
    """Final aggregated post-processing output."""
    defect_count: int
    overall_severity: SeverityLevel
    total_affected_area_percent: float
    detections: List[Detection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_count": self.defect_count,
            "overall_severity": self.overall_severity.value,
            "total_affected_area_percent": round(float(self.total_affected_area_percent), 4),
            "detections": [d.to_dict() for d in self.detections],
        }


class DefectProcessor:
    """Post-processor handling geometry validation, duplicate suppression, and severity calculation."""

    def __init__(
        self,
        min_confidence: float = 0.25,
        duplicate_iou_threshold: float = 0.45,
        minor_max_percent: float = 1.0,
        moderate_max_percent: float = 5.0,
        min_box_area_pixels: float = 16.0,
    ):
        """Initialize post-processor with configurable thresholds.

        Args:
            min_confidence: Minimum confidence threshold.
            duplicate_iou_threshold: IoU threshold for overlapping duplicate removal.
            minor_max_percent: Max area percentage for 'minor' severity.
            moderate_max_percent: Max area percentage for 'moderate' severity (>= is severe).
            min_box_area_pixels: Minimum box area to filter out noise artifacts.
        """
        self.min_confidence = min_confidence
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.minor_max_percent = minor_max_percent
        self.moderate_max_percent = moderate_max_percent
        self.min_box_area_pixels = min_box_area_pixels

    @staticmethod
    def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x_left = max(box_a.x1, box_b.x1)
        y_top = max(box_a.y1, box_b.y1)
        x_right = min(box_a.x2, box_b.x2)
        y_bottom = min(box_a.y2, box_b.y2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area_a = box_a.area
        area_b = box_b.area
        union_area = area_a + area_b - intersection_area

        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    def validate_and_clamp_bbox(
        self,
        bbox: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> Optional[BoundingBox]:
        """Validate and clamp bounding box coordinates to image boundaries.

        Returns:
            Clamped BoundingBox if valid, None if degenerate or completely out of bounds.
        """
        if image_width <= 0 or image_height <= 0:
            return None

        # Clamp coordinates
        x1 = max(0.0, min(float(image_width), float(bbox.x1)))
        y1 = max(0.0, min(float(image_height), float(bbox.y1)))
        x2 = max(0.0, min(float(image_width), float(bbox.x2)))
        y2 = max(0.0, min(float(image_height), float(bbox.y2)))

        # Ensure valid non-degenerate box
        if x2 <= x1 or y2 <= y1:
            return None

        clamped_box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        if clamped_box.area < self.min_box_area_pixels:
            return None

        return clamped_box

    def suppress_duplicates(
        self,
        detections: List[Detection],
        class_aware: bool = True,
    ) -> List[Detection]:
        """Perform Non-Maximum Suppression (NMS) duplicate filtering.

        Args:
            detections: List of candidate detections.
            class_aware: If True, suppression only applies between detections of the same class.

        Returns:
            Filtered list of detections sorted by descending confidence.
        """
        if not detections:
            return []

        # Sort detections by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep: List[Detection] = []

        for candidate in sorted_dets:
            should_keep = True
            for existing in keep:
                if class_aware and candidate.class_id != existing.class_id:
                    continue
                iou = self.calculate_iou(candidate.bbox, existing.bbox)
                if iou >= self.duplicate_iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(candidate)

        return keep

    def compute_defect_area_and_severity(
        self,
        bbox: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, SeverityLevel]:
        """Compute bounding box area, area percentage, and heuristic severity.

        Args:
            bbox: Validated bounding box.
            image_width: Total image width in pixels.
            image_height: Total image height in pixels.

        Returns:
            Tuple of (area_pixels, area_percentage, SeverityLevel).
        """
        total_image_area = float(image_width * image_height)
        if total_image_area <= 0.0:
            return 0.0, 0.0, SeverityLevel.NONE

        area_pixels = bbox.area
        area_percentage = (area_pixels / total_image_area) * 100.0

        if area_percentage < self.minor_max_percent:
            severity = SeverityLevel.MINOR
        elif area_percentage < self.moderate_max_percent:
            severity = SeverityLevel.MODERATE
        else:
            severity = SeverityLevel.SEVERE

        return area_pixels, area_percentage, severity

    def process(
        self,
        raw_detections: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> ProcessedResult:
        """Process raw model detections into a validated, deduplicated, and ranked result.

        Args:
            raw_detections: List of dicts containing class_id, class_name, confidence, bbox.
            image_width: Input image width.
            image_height: Input image height.

        Returns:
            ProcessedResult containing overall severity and sanitized detections.
        """
        valid_detections: List[Detection] = []

        for det_dict in raw_detections:
            conf = float(det_dict.get("confidence", 0.0))
            if conf < self.min_confidence:
                continue

            raw_bbox_dict = det_dict.get("bbox", {})
            raw_bbox = BoundingBox(
                x1=float(raw_bbox_dict.get("x1", 0.0)),
                y1=float(raw_bbox_dict.get("y1", 0.0)),
                x2=float(raw_bbox_dict.get("x2", 0.0)),
                y2=float(raw_bbox_dict.get("y2", 0.0)),
            )

            clamped_bbox = self.validate_and_clamp_bbox(raw_bbox, image_width, image_height)
            if clamped_bbox is None:
                continue

            area_pixels, area_pct, severity = self.compute_defect_area_and_severity(
                clamped_bbox, image_width, image_height
            )

            detection = Detection(
                class_id=int(det_dict.get("class_id", 0)),
                class_name=str(det_dict.get("class_name", "defect")),
                confidence=conf,
                bbox=clamped_bbox,
                area_pixels=area_pixels,
                area_percentage=area_pct,
                severity=severity,
            )
            valid_detections.append(detection)

        # Suppress duplicate overlapping boxes
        filtered_detections = self.suppress_duplicates(valid_detections)

        # Compute aggregate metrics
        if not filtered_detections:
            return ProcessedResult(
                defect_count=0,
                overall_severity=SeverityLevel.NONE,
                total_affected_area_percent=0.0,
                detections=[],
            )

        # Overall severity is the maximum severity among detections
        severity_rank = {
            SeverityLevel.NONE: 0,
            SeverityLevel.MINOR: 1,
            SeverityLevel.MODERATE: 2,
            SeverityLevel.SEVERE: 3,
        }
        max_severity = max(
            filtered_detections, key=lambda d: severity_rank.get(d.severity, 0)
        ).severity

        total_area_pct = sum(d.area_percentage for d in filtered_detections)

        return ProcessedResult(
            defect_count=len(filtered_detections),
            overall_severity=max_severity,
            total_affected_area_percent=total_area_pct,
            detections=filtered_detections,
        )

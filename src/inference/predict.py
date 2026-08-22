"""Inference engine for industrial defect detection.

Loads YOLO models, manages preprocessing, runs forward inference,
and applies defect post-processing and severity estimation.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import yaml
from PIL import Image
import numpy as np
from ultralytics import YOLO

from src.postprocessing.defect_processor import DefectProcessor, ProcessedResult

logger = logging.getLogger(__name__)


class DefectPredictor:
    """End-to-end defect prediction service."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: str = "configs/config.yaml",
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ):
        """Initialize predictor with model weights and configuration."""
        self.config = self._load_config(config_path)
        
        # Determine model path
        default_model = self.config.get("paths", {}).get("models_dir", "artifacts/models") + "/best.pt"
        self.model_path = model_path or default_model
        
        # Fallback to base YOLO if custom trained model doesn't exist yet
        if not Path(self.model_path).exists():
            fallback_model = self.config.get("model", {}).get("name", "yolo11n.pt")
            logger.warning(
                "Model '%s' not found. Falling back to base model '%s'",
                self.model_path,
                fallback_model,
            )
            self.model_path = fallback_model

        logger.info("Loading YOLO model from: %s", self.model_path)
        self.model = YOLO(self.model_path)

        # Post-processing configuration
        min_conf = conf_threshold or self.config.get("model", {}).get("confidence_threshold", 0.25)
        dup_iou = iou_threshold or self.config.get("postprocessing", {}).get("duplicate_iou_threshold", 0.45)
        minor_max = self.config.get("severity", {}).get("minor_max_percent", 1.0)
        moderate_max = self.config.get("severity", {}).get("moderate_max_percent", 5.0)

        self.processor = DefectProcessor(
            min_confidence=min_conf,
            duplicate_iou_threshold=dup_iou,
            minor_max_percent=minor_max,
            moderate_max_percent=moderate_max,
        )

        # Extract class mapping
        self.class_names = self.model.names if hasattr(self.model, "names") else {}

    def _load_config(self, config_path: str) -> dict:
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        imgsz: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full prediction and post-processing pipeline on an image.

        Args:
            image_input: Filepath, PIL Image, or numpy array.
            imgsz: Inference input resolution (defaults to config image_size).

        Returns:
            Dictionary containing metadata, defect count, severity, latency, and detections.
        """
        start_time = time.perf_counter()
        target_imgsz = imgsz or self.config.get("model", {}).get("image_size", 256)

        # Resolve image and dimensions
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
            filename = Path(image_input).name
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            filename = "uploaded_image.jpg"
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
            filename = "numpy_image.jpg"
        else:
            raise ValueError("Unsupported image input type.")

        img_w, img_h = pil_img.size

        # Run YOLO inference
        results = self.model.predict(
            source=pil_img,
            imgsz=target_imgsz,
            conf=self.processor.min_confidence,
            verbose=False,
        )

        raw_detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(boxes.conf[i].cpu().item())
                    cls_id = int(boxes.cls[i].cpu().item())
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                    raw_detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": {
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                        },
                    })

        # Apply post-processing (IoU deduplication, area calculation, severity assignment)
        processed: ProcessedResult = self.processor.process(
            raw_detections=raw_detections,
            image_width=img_w,
            image_height=img_h,
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        output = {
            "filename": filename,
            "image_width": img_w,
            "image_height": img_h,
            "inference_time_ms": round(inference_time_ms, 2),
            "defect_count": processed.defect_count,
            "overall_severity": processed.overall_severity.value,
            "total_affected_area_percent": round(processed.total_affected_area_percent, 4),
            "detections": [d.to_dict() for d in processed.detections],
        }

        return output

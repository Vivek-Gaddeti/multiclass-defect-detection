"""Dataset Preparation Pipeline for Multiclass Industrial Defect Detection.

Parses raw Pascal VOC XML or text annotations, converts them to normalized YOLO format,
validates integrity, performs stratified train/val/test splits, creates data/dataset.yaml,
and exports a comprehensive preparation summary report.
"""

import os
import json
import random
import shutil
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[str, float, float, float, float]]]:
    """Parse Pascal VOC XML annotation file.

    Returns:
        Tuple of (width, height, list of (class_name, xmin, ymin, xmax, ymax))
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)
    else:
        width, height = 0, 0

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax))

    return width, height, boxes


def convert_to_yolo_format(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert bounding box (xmin, ymin, xmax, ymax) to normalized YOLO (x_center, y_center, width, height)."""
    # Clamp to boundaries
    xmin = max(0.0, min(float(img_w), xmin))
    ymin = max(0.0, min(float(img_h), ymin))
    xmax = max(0.0, min(float(img_w), xmax))
    ymax = max(0.0, min(float(img_h), ymax))

    dw = 1.0 / img_w
    dh = 1.0 / img_h

    x_center = (xmin + xmax) / 2.0 * dw
    y_center = (ymin + ymax) / 2.0 * dh
    width = (xmax - xmin) * dw
    height = (ymax - ymin) * dh

    return max(0.0, min(1.0, x_center)), max(0.0, min(1.0, y_center)), max(0.0, min(1.0, width)), max(0.0, min(1.0, height))


def prepare_dataset(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Execute the full dataset preparation, conversion, and splitting pipeline."""
    config = load_config(config_path)
    seed = config["project"].get("seed", 42)
    random.seed(seed)

    raw_dir = Path(config["dataset"]["raw_dir"])
    processed_dir = Path(config["dataset"]["processed_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    classes: List[str] = config["dataset"]["classes"]
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}

    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing processed directory if needed
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    for split in ["train", "val", "test"]:
        (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Locate raw images and annotations
    raw_images = list(raw_dir.glob("**/*.jpg")) + list(raw_dir.glob("**/*.bmp")) + list(raw_dir.glob("**/*.png"))
    # Filter out any files in processed or non-raw
    raw_images = [p for p in raw_images if "processed" not in str(p)]

    logger.info("Found %d raw image files.", len(raw_images))

    valid_samples: List[Dict[str, Any]] = []
    invalid_files: List[str] = []
    class_counts: Dict[str, int] = {c: 0 for c in classes}

    for img_path in raw_images:
        stem = img_path.stem
        # Look for corresponding xml file in raw_dir
        xml_candidates = list(raw_dir.glob(f"**/{stem}.xml"))
        if not xml_candidates:
            invalid_files.append(f"Missing XML for {img_path.name}")
            continue

        xml_path = xml_candidates[0]

        # Verify image readability
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            invalid_files.append(f"Corrupted image {img_path.name}: {e}")
            continue

        # Parse XML
        try:
            xml_w, xml_h, boxes = parse_voc_xml(xml_path)
            if img_w <= 0 or img_h <= 0:
                img_w, img_h = (xml_w, xml_h) if xml_w > 0 and xml_h > 0 else (200, 200)
        except Exception as e:
            invalid_files.append(f"XML parse error {xml_path.name}: {e}")
            continue

        yolo_lines = []
        sample_classes = []
        for cls_name, xmin, ymin, xmax, ymax in boxes:
            if cls_name not in class_to_id:
                # normalize name if possible (e.g. rolled-in scale -> rolled-in_scale)
                normalized_name = cls_name.replace(" ", "_").replace("-", "_")
                matching = [c for c in classes if c.replace("-", "_") == normalized_name]
                if matching:
                    cls_name = matching[0]
                else:
                    continue

            cls_id = class_to_id[cls_name]
            xc, yc, w, h = convert_to_yolo_format(xmin, ymin, xmax, ymax, img_w, img_h)
            if w > 0 and h > 0:
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                class_counts[cls_name] += 1
                sample_classes.append(cls_name)

        if yolo_lines:
            valid_samples.append({
                "img_path": img_path,
                "stem": stem,
                "yolo_lines": yolo_lines,
                "primary_class": sample_classes[0] if sample_classes else "unknown",
            })

    # Group by primary class to perform stratified split
    class_groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in valid_samples:
        grp = sample["primary_class"]
        class_groups.setdefault(grp, []).append(sample)

    train_pct = config["dataset"]["splits"].get("train", 0.70)
    val_pct = config["dataset"]["splits"].get("val", 0.15)

    split_counts = {"train": 0, "val": 0, "test": 0}

    for grp_name, items in class_groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]

        for split_name, split_list in [("train", train_items), ("val", val_items), ("test", test_items)]:
            for item in split_list:
                dest_img = processed_dir / "images" / split_name / f"{item['stem']}.jpg"
                dest_lbl = processed_dir / "labels" / split_name / f"{item['stem']}.txt"

                # Copy and save as standard RGB JPEG
                with Image.open(item["img_path"]) as img:
                    img.convert("RGB").save(dest_img, "JPEG", quality=95)

                with open(dest_lbl, "w", encoding="utf-8") as f:
                    f.write("\n".join(item["yolo_lines"]) + "\n")

                split_counts[split_name] += 1

    # Create dataset.yaml
    dataset_yaml_data = {
        "path": str(processed_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
    }

    dataset_yaml_path = Path(config["dataset"]["yaml_path"])
    with open(dataset_yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(dataset_yaml_data, yf, default_flow_style=False, sort_keys=False)

    summary_report = {
        "dataset_name": config["dataset"]["name"],
        "total_images_processed": len(valid_samples),
        "split_counts": split_counts,
        "class_annotation_counts": class_counts,
        "classes": classes,
        "invalid_or_skipped_files": len(invalid_files),
        "invalid_file_log": invalid_files[:20],
        "random_seed": seed,
        "dataset_yaml": str(dataset_yaml_path),
    }

    report_path = metrics_dir / "dataset_summary.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(summary_report, rf, indent=2)

    logger.info("Dataset preparation complete! Report saved to %s", report_path)
    logger.info("Train: %d, Val: %d, Test: %d", split_counts["train"], split_counts["val"], split_counts["test"])
    return summary_report


if __name__ == "__main__":
    prepare_dataset()

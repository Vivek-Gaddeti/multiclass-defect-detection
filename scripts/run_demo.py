"""CLI Demonstration Script for Industrial Defect Detection.

Runs an end-to-end inspection on a sample test surface, computes defect metrics,
saves annotated prediction imagery, and outputs a formatted terminal report.
"""

import sys
import json
import logging
from pathlib import Path
from PIL import Image

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.inference.predict import DefectPredictor
from src.inference.visualize import visualize_detections
from src.data.download_dataset import generate_benchmark_industrial_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo")


def run_demo(target_image: str = None):
    print("=" * 60)
    print(" INDUSTRIAL SURFACE DEFECT DETECTION & ANALYSIS DEMO ")
    print("=" * 60)

    # 1. Initialize predictor
    print("\n[1/4] Initializing DefectPredictor...")
    predictor = DefectPredictor()

    # 2. Select image
    print("\n[2/4] Locating sample test images...")
    if target_image:
        sample_images = [Path(target_image)]
    else:
        test_img_dir = Path("data/processed/images/test")
        sample_images = list(test_img_dir.glob("*.jpg")) if test_img_dir.exists() else []

    if not sample_images:
        demo_dir = Path("artifacts/sample_inputs")
        demo_dir.mkdir(parents=True, exist_ok=True)
        generate_benchmark_industrial_dataset(demo_dir, num_samples_per_class=1)
        sample_images = list((demo_dir / "IMAGES").glob("*.jpg"))

    # Test first 3 images across classes
    predictions_dir = Path("artifacts/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for sample_img_path in sample_images[:3]:
        print(f"\n[3/4] Running inspection on '{sample_img_path.name}'...")
        result = predictor.predict(sample_img_path)

        out_img_path = predictions_dir / f"annotated_{sample_img_path.name}"
        visualize_detections(
            image_input=sample_img_path,
            detections=result["detections"],
            overall_severity=result["overall_severity"],
            save_path=out_img_path,
        )
        print(f"-> Saved annotated inspection result to: {out_img_path}")

        print("\n" + "-" * 60)
        print(f" Image Filename      : {result['filename']}")
        print(f" Dimensions          : {result['image_width']} x {result['image_height']} px")
        print(f" Inference Latency   : {result['inference_time_ms']} ms")
        print(f" Total Defects Found : {result['defect_count']}")
        print(f" Overall Severity    : {result['overall_severity'].upper()}")
        print(f" Total Area Impact   : {result['total_affected_area_percent']:.2f}%")
        print("-" * 60)

        if result["detections"]:
            print(f"{'#':<3} {'Class':<18} {'Confidence':<12} {'Severity':<10} {'Area %':<8} {'BBox [x1,y1,x2,y2]'}")
            print("-" * 60)
            for idx, d in enumerate(result["detections"], 1):
                bb = d["bbox"]
                box_str = f"[{bb['x1']:.0f}, {bb['y1']:.0f}, {bb['x2']:.0f}, {bb['y2']:.0f}]"
                print(
                    f"{idx:<3} {d['class_name']:<18} {d['confidence']*100:>5.1f}%      {d['severity']:<10} {d['area_percentage']:>5.2f}%   {box_str}"
                )
        else:
            print(" No surface defects detected above confidence threshold.")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(img_arg)

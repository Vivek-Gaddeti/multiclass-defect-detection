"""Model Evaluation Pipeline.

Evaluates the trained defect detection model on the held-out test split,
computes precision, recall, mAP@50, mAP@50:95, and per-class performance,
and exports structured metrics to artifacts/metrics/evaluation.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from ultralytics import YOLO

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model(
    model_path: Optional[str] = None,
    config_path: str = "configs/config.yaml",
    split: str = "test",
) -> Dict[str, Any]:
    """Evaluate YOLO model on the specified split (test/val).

    Args:
        model_path: Path to .pt model weights (defaults to artifacts/models/best.pt).
        config_path: Configuration file path.
        split: Dataset split to evaluate ('test' or 'val').

    Returns:
        Dictionary containing genuine evaluation metrics.
    """
    config = load_config(config_path)
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    default_model = config["paths"]["models_dir"] + "/best.pt"
    selected_model_path = model_path or default_model

    if not Path(selected_model_path).exists():
        fallback = config["model"].get("name", "yolo11n.pt")
        logger.warning(
            "Model '%s' not found. Falling back to base model '%s' for evaluation.",
            selected_model_path,
            fallback,
        )
        selected_model_path = fallback

    logger.info("Evaluating model: %s on split: %s", selected_model_path, split)
    model = YOLO(selected_model_path)

    dataset_yaml = config["dataset"]["yaml_path"]

    # Run YOLO validator
    val_results = model.val(
        data=dataset_yaml,
        split=split,
        imgsz=config["model"].get("image_size", 640),
        batch=config["model"].get("batch_size", 8),
        device="0" if False else "cpu",  # CPU safe
        plots=True,
    )

    # Extract metrics
    precision = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    recall = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))
    f1_score = round(2 * (precision * recall) / (precision + recall + 1e-8), 4)

    # Per-class metrics
    per_class_metrics = {}
    class_names = val_results.names if isinstance(val_results.names, dict) else {i: n for i, n in enumerate(val_results.names)}
    
    p_per_class = getattr(val_results.box, "p", [])
    r_per_class = getattr(val_results.box, "r", [])
    ap50_per_class = getattr(val_results.box, "ap50", [])
    ap_per_class = getattr(val_results.box, "ap", [])
    f1_per_class = getattr(val_results.box, "f1", [])

    for idx, name in class_names.items():
        cp = float(p_per_class[idx]) if len(p_per_class) > idx else 0.0
        cr = float(r_per_class[idx]) if len(r_per_class) > idx else 0.0
        cap50 = float(ap50_per_class[idx]) if len(ap50_per_class) > idx else 0.0
        cap95 = float(ap_per_class[idx]) if len(ap_per_class) > idx else 0.0
        cf1 = float(f1_per_class[idx]) if len(f1_per_class) > idx else round(2 * (cp * cr) / (cp + cr + 1e-8), 4)

        per_class_metrics[name] = {
            "precision": round(cp, 4),
            "recall": round(cr, 4),
            "f1_score": round(cf1, 4),
            "map50": round(cap50, 4),
            "map50_95": round(cap95, 4),
        }

    evaluation_report = {
        "model_path": str(selected_model_path),
        "split": split,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": f1_score,
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "per_class": per_class_metrics,
        "classes": list(class_names.values()),
    }

    eval_json_path = metrics_dir / (f"evaluation_{Path(selected_model_path).stem}.json" if "best" not in Path(selected_model_path).stem else "evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("Evaluation report saved to %s", eval_json_path)
    logger.info(
        "Summary: Precision=%.4f, Recall=%.4f, F1=%.4f, mAP50=%.4f, mAP50-95=%.4f",
        precision, recall, f1_score, map50, map50_95
    )

    if MLFLOW_AVAILABLE:
        try:
            exp_name = config.get("mlflow", {}).get("experiment_name", "industrial-defect-detection")
            tracking_uri = config.get("mlflow", {}).get("tracking_uri", "./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=f"eval_{Path(selected_model_path).stem}_{split}"):
                mlflow.log_metrics({
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1_score": f1_score,
                    "test_map50": map50,
                    "test_map50_95": map50_95,
                })
                for cname, cdata in per_class_metrics.items():
                    for mname, mval in cdata.items():
                        mlflow.log_metric(f"{cname}_{mname}", mval)
                mlflow.log_artifact(str(eval_json_path))
        except Exception as e:
            logger.warning("Failed to log evaluation to MLflow: %s", e)

    return evaluation_report


if __name__ == "__main__":
    evaluate_model()

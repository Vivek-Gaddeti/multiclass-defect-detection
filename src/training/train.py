"""Training Pipeline with MLflow Experiment Tracking.

Fine-tunes a lightweight YOLO11 model on industrial surface defect datasets,
tracks hyperparameters and real metrics via MLflow, and exports artifacts.
"""

import time
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
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


def train_model(
    config_path: str = "configs/config.yaml",
    epochs_override: Optional[int] = None,
    batch_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute YOLO training and track results with MLflow.

    Args:
        config_path: Path to configuration file.
        epochs_override: Optional override for number of epochs.
        batch_override: Optional override for batch size.

    Returns:
        Summary dictionary containing training metrics and artifact locations.
    """
    config = load_config(config_path)

    # Directories
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Dataset & Training hyperparameters
    dataset_yaml = config["dataset"]["yaml_path"]
    model_name = config["model"].get("name", "yolo11n.pt")
    epochs = epochs_override or config["model"].get("epochs", 15)
    batch_size = batch_override or config["model"].get("batch_size", 8)
    imgsz = config["model"].get("image_size", 640)
    seed = config["project"].get("seed", 42)
    device = "0" if torch.cuda.is_available() else "cpu"

    logger.info("Initializing YOLO model '%s' on device '%s'...", model_name, device)
    model = YOLO(model_name)

    # MLflow Setup
    exp_name = config.get("mlflow", {}).get("experiment_name", "industrial-defect-detection")
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "./mlruns")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)
        active_run = mlflow.start_run()
        run_id = active_run.info.run_id
        logger.info("MLflow active run started (Run ID: %s)", run_id)
        
        mlflow.log_params({
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": imgsz,
            "seed": seed,
            "device": device,
            "dataset": config["dataset"]["name"],
        })
    else:
        run_id = "local_no_mlflow"
        logger.warning("MLflow not available; skipping remote tracking.")

    start_time = time.time()

    # Train YOLO model
    logger.info("Starting training for %d epochs...", epochs)
    train_results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        seed=seed,
        device=device,
        project="runs/detect",
        name="defect_model",
        exist_ok=True,
        verbose=True,
        workers=2,
    )

    training_duration_sec = round(time.time() - start_time, 2)
    logger.info("Training completed in %.2f seconds.", training_duration_sec)

    # Locate and copy best and last model weights
    save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir") else Path("runs/detect/defect_model")
    best_weights_src = save_dir / "weights" / "best.pt"
    last_weights_src = save_dir / "weights" / "last.pt"

    dest_best = models_dir / "best.pt"
    dest_last = models_dir / "last.pt"

    if best_weights_src.exists():
        shutil.copy2(best_weights_src, dest_best)
        logger.info("Saved best model to %s", dest_best)
    elif last_weights_src.exists():
        shutil.copy2(last_weights_src, dest_best)
        logger.info("Saved last model as best to %s", dest_best)

    if last_weights_src.exists():
        shutil.copy2(last_weights_src, dest_last)

    # Extract real training metrics
    metrics_summary = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "device": device,
        "training_duration_seconds": training_duration_sec,
        "best_model_path": str(dest_best),
        "run_id": run_id,
    }

    # Extract metrics from training results if available
    if hasattr(train_results, "results_dict") and train_results.results_dict:
        rd = train_results.results_dict
        metrics_summary["precision"] = float(rd.get("metrics/precision(B)", 0.0))
        metrics_summary["recall"] = float(rd.get("metrics/recall(B)", 0.0))
        metrics_summary["map50"] = float(rd.get("metrics/mAP50(B)", 0.0))
        metrics_summary["map50_95"] = float(rd.get("metrics/mAP50-95(B)", 0.0))
        metrics_summary["fitness"] = float(rd.get("fitness", 0.0))

    if MLFLOW_AVAILABLE:
        for k in ["precision", "recall", "map50", "map50_95", "training_duration_seconds"]:
            if k in metrics_summary:
                mlflow.log_metric(k, metrics_summary[k])
        if dest_best.exists():
            mlflow.log_artifact(str(dest_best), artifact_path="model_weights")
        mlflow.end_run()

    # Save summary report
    summary_path = metrics_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(metrics_summary, sf, indent=2)

    logger.info("Training summary exported to %s", summary_path)
    return metrics_summary


if __name__ == "__main__":
    train_model()

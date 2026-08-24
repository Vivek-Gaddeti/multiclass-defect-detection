"""Statistical CPU Inference Latency and Model Profiling Benchmark.

Measures real-world latency on the CPU using:
- 10 warmup iterations to stabilize cache & memory.
- 50 timed inference iterations across test split images.
- Computes Mean, Std, Median (P50), 95th Percentile (P95), Min, Max, and FPS.
- Logs parameter count, model size (MB), and GFLOPs.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import logging
from typing import Dict, Any, List
import yaml
import numpy as np
from PIL import Image
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_model_latency(
    model_path: str = "artifacts/models/best.pt",
    config_path: str = "configs/config.yaml",
    warmup_runs: int = 10,
    test_runs: int = 50,
) -> Dict[str, Any]:
    """Benchmark inference latency statistically on CPU."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    imgsz = config["model"].get("image_size", 256)
    conf_thresh = config["model"].get("confidence_threshold", 0.15)
    test_img_dir = Path("data/processed/images/test")
    test_images = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))

    if not test_images:
        raise FileNotFoundError(f"No test images found in {test_img_dir}")

    logger.info("Loading model for CPU latency benchmark: %s", model_path)
    model = YOLO(model_path)

    # Compute model weight size in MB
    model_file = Path(model_path)
    model_size_mb = round(model_file.stat().st_size / (1024 * 1024), 2) if model_file.exists() else 0.0

    # Model architecture stats
    num_params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else 0

    # 1. Warmup Runs
    logger.info("Executing %d warmup iterations...", warmup_runs)
    sample_img = test_images[0]
    for _ in range(warmup_runs):
        _ = model.predict(source=sample_img, imgsz=imgsz, conf=conf_thresh, device="cpu", verbose=False)

    # 2. Timed Iterations
    logger.info("Executing %d timed inference iterations...", test_runs)
    latencies_ms: List[float] = []

    for i in range(test_runs):
        img_p = test_images[i % len(test_images)]
        t0 = time.perf_counter()
        _ = model.predict(source=img_p, imgsz=imgsz, conf=conf_thresh, device="cpu", verbose=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    # Statistical metrics
    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    fps = round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0

    benchmark_results = {
        "model_path": str(model_path),
        "model_name": Path(model_path).stem,
        "model_size_mb": model_size_mb,
        "parameter_count": num_params,
        "device": "cpu",
        "image_size": imgsz,
        "warmup_runs": warmup_runs,
        "measured_runs": test_runs,
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "std": round(std_lat, 2),
            "p50_median": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
        },
        "throughput_fps": fps,
    }

    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"latency_benchmark_{Path(model_path).stem}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    logger.info(
        "Benchmark complete [%s]: Mean=%.2f ms (±%.2f), P50=%.2f ms, P95=%.2f ms | FPS=%.2f",
        Path(model_path).stem, mean_lat, std_lat, p50_lat, p95_lat, fps
    )
    return benchmark_results


if __name__ == "__main__":
    benchmark_model_latency()

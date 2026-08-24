"""End-to-End Experimentation, Benchmarking & Model Comparison Pipeline.

Orchestrates:
1. Training YOLO11n (nano baseline) and YOLO11s (small comparison model).
2. Evaluating both on the identical held-out test split.
3. Conducting confusion matrix & diagnostic error analysis.
4. Running statistical CPU latency profiling.
5. Aggregating results into artifacts/metrics/model_comparison.json.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
from typing import Dict, Any

from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model
from src.evaluation.error_analysis import run_error_analysis
from scripts.benchmark_latency import benchmark_model_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("experiments")


def run_full_comparison_pipeline(epochs: int = 15, batch_size: int = 16) -> Dict[str, Any]:
    """Train and evaluate YOLO11n vs YOLO11s on the exact same dataset."""
    logger.info("============================================================")
    logger.info("  STARTING DISCIPLINED MODEL EXPERIMENTATION & COMPARISON   ")
    logger.info("============================================================")

    models_to_train = [
        {"name": "yolo11n.pt", "label": "YOLO11n (Nano)"},
        {"name": "yolo11s.pt", "label": "YOLO11s (Small)"},
    ]

    comparison_results = {}

    for m in models_to_train:
        m_name = m["name"]
        m_label = m["label"]
        stem = Path(m_name).stem
        logger.info("\n>>> [1/4] Training %s for %d epochs...", m_label, epochs)
        train_summary = train_model(
            config_path="configs/config.yaml",
            model_name_override=m_name,
            epochs_override=epochs,
            batch_override=batch_size,
            run_name=f"train_{stem}",
        )

        weights_path = f"artifacts/models/{stem}_best.pt"
        if not Path(weights_path).exists():
            weights_path = "artifacts/models/best.pt"

        logger.info("\n>>> [2/4] Evaluating %s on held-out test set...", m_label)
        eval_metrics = evaluate_model(
            model_path=weights_path,
            config_path="configs/config.yaml",
            split="test",
        )

        logger.info("\n>>> [3/4] Running Error Analysis on %s...", m_label)
        error_diag = run_error_analysis(
            model_path=weights_path,
            config_path="configs/config.yaml",
        )

        logger.info("\n>>> [4/4] Profiling CPU Latency on %s...", m_label)
        latency_diag = benchmark_model_latency(
            model_path=weights_path,
            config_path="configs/config.yaml",
            warmup_runs=10,
            test_runs=50,
        )

        comparison_results[stem] = {
            "model_variant": m_label,
            "weights_path": weights_path,
            "parameters": latency_diag["parameter_count"],
            "model_size_mb": latency_diag["model_size_mb"],
            "test_precision": eval_metrics["precision"],
            "test_recall": eval_metrics["recall"],
            "test_f1_score": eval_metrics["f1_score"],
            "test_map50": eval_metrics["map50"],
            "test_map50_95": eval_metrics["map50_95"],
            "latency_mean_ms": latency_diag["latency_ms"]["mean"],
            "latency_p50_ms": latency_diag["latency_ms"]["p50_median"],
            "latency_p95_ms": latency_diag["latency_ms"]["p95"],
            "throughput_fps": latency_diag["throughput_fps"],
            "per_class": eval_metrics["per_class"],
        }

    # Save comprehensive comparison artifact
    comp_file = Path("artifacts/metrics/model_comparison.json")
    comp_file.parent.mkdir(parents=True, exist_ok=True)
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    logger.info("\n============================================================")
    logger.info("  EXPERIMENTATION COMPLETE — MODEL COMPARISON SUMMARY       ")
    logger.info("============================================================")
    for model_key, r in comparison_results.items():
        logger.info(
            "%s -> Precision: %.3f | Recall: %.3f | F1: %.3f | mAP50: %.3f | mAP50-95: %.3f | Latency: %.2f ms (%.1f FPS)",
            r["model_variant"], r["test_precision"], r["test_recall"], r["test_f1_score"],
            r["test_map50"], r["test_map50_95"], r["latency_mean_ms"], r["throughput_fps"]
        )

    return comparison_results


if __name__ == "__main__":
    run_full_comparison_pipeline()

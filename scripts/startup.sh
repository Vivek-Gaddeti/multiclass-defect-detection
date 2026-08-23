#!/usr/bin/env bash
# startup.sh — Production startup script for cloud deployments (Render / Railway / Docker)
#
# Strategy:
#   1. Start the API server IMMEDIATELY so Render detects the port right away.
#   2. If model weights are missing, run training in the BACKGROUND.
#   3. The /health endpoint returns {"model_loaded": false} while training.
#   4. Once training finishes, the next request auto-loads the new weights.

echo "=========================================="
echo "  Industrial Defect Detection API Startup"
echo "=========================================="

MODEL_PATH="artifacts/models/best.pt"
PORT="${PORT:-10000}"

# ── Background training function ──────────────────────────────────────────────
run_training() {
    echo "[TRAIN] No model found. Starting background training..."

    python -m src.data.download_dataset  && echo "[TRAIN] Dataset generated."
    python -m src.data.prepare_dataset   && echo "[TRAIN] Dataset prepared."
    python -m src.training.train          && echo "[TRAIN] ✅ Training complete! Model saved to $MODEL_PATH"
    echo "[TRAIN] Reload the page or wait — the next request will auto-load the model."
}

# ── Start API server immediately (port opens in <5 seconds) ───────────────────
if [ ! -f "$MODEL_PATH" ]; then
    echo "[INFO] Model not found — launching background training..."
    run_training &   # Run training in background, don't block server startup
else
    echo "[INFO] Model found at $MODEL_PATH — skipping training."
fi

echo "[INFO] Starting FastAPI server on 0.0.0.0:${PORT} ..."
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"

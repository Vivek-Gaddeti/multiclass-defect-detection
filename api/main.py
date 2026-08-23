"""FastAPI REST API for Industrial Defect Detection & Analysis.

Provides high-throughput endpoints for defect prediction, model metadata,
health inspection, and serving the interactive web dashboard.
"""

import io
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from src.inference.predict import DefectPredictor
from src.inference.visualize import visualize_detections, image_to_base64

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("defect-api")

# Global predictor instance (cached during application lifespan)
predictor: Optional[DefectPredictor] = None

# Max upload file size in bytes (10 MB default)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# Pydantic Schemas for Swagger Documentation
class BoundingBoxSchema(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")


class DetectionItemSchema(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBoxSchema
    area_pixels: float
    area_percentage: float
    severity: str


class PredictionResponseSchema(BaseModel):
    filename: str
    image_width: int
    image_height: int
    inference_time_ms: float
    defect_count: int
    overall_severity: str
    total_affected_area_percent: float
    detections: List[DetectionItemSchema]
    annotated_image_base64: Optional[str] = None


class HealthResponseSchema(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponseSchema(BaseModel):
    model_path: str
    supported_classes: List[str]
    confidence_threshold: float
    iou_threshold: float
    severity_rules: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: Load model once at startup and clean up on shutdown."""
    global predictor
    logger.info("Starting up Defect Detection API service...")

    model_path = Path("artifacts/models/best.pt")
    if not model_path.exists():
        logger.warning(
            "Model weights not found at %s. "
            "API is running in DEGRADED mode — background training may be in progress. "
            "The model will load automatically on the next /predict request once training completes.",
            model_path,
        )
        predictor = None
    else:
        try:
            predictor = DefectPredictor()
            logger.info("DefectPredictor initialized successfully with model: %s", predictor.model_path)
        except Exception as e:
            logger.error("Failed to initialize DefectPredictor on startup: %s", e)
            predictor = None

    yield

    logger.info("Shutting down Defect Detection API service...")
    predictor = None



# Initialize FastAPI App
app = FastAPI(
    title="Industrial Defect Detection & Analysis API",
    description="Production-grade CV API for automated surface defect detection, area estimation, and severity rating.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Frontend static assets
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve web user interface dashboard."""
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Industrial Defect Detection API is online. Visit /docs for Swagger UI."})


@app.get("/health", response_model=HealthResponseSchema, tags=["System"])
async def health_check():
    """Health check endpoint to verify system status and model readiness."""
    is_ready = predictor is not None and predictor.model is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "version": "1.0.0",
    }


@app.get("/model-info", response_model=ModelInfoResponseSchema, tags=["Model"])
async def get_model_info():
    """Return active model parameters, supported defect classes, and thresholds."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor is not initialized.",
        )

    classes = list(predictor.class_names.values()) if isinstance(predictor.class_names, dict) else list(predictor.class_names)
    return {
        "model_path": str(predictor.model_path),
        "supported_classes": classes,
        "confidence_threshold": predictor.processor.min_confidence,
        "iou_threshold": predictor.processor.duplicate_iou_threshold,
        "severity_rules": {
            "minor_max_percent": predictor.processor.minor_max_percent,
            "moderate_max_percent": predictor.processor.moderate_max_percent,
            "severe_min_percent": predictor.processor.moderate_max_percent,
        },
    }


@app.post("/predict", response_model=PredictionResponseSchema, tags=["Inference"])
async def predict_defect(
    file: UploadFile = File(..., description="Surface image file (JPG, PNG, WEBP)"),
    conf_threshold: Optional[float] = Query(None, ge=0.05, le=1.0, description="Override confidence threshold"),
    iou_threshold: Optional[float] = Query(None, ge=0.1, le=0.9, description="Override IoU duplicate threshold"),
    include_image: bool = Query(True, description="Include base64 annotated image in response"),
):
    """Accept an industrial surface image, detect defects, calculate surface area impact and severity."""
    global predictor

    # Auto-reload model if background training just finished
    if predictor is None:
        model_path = Path("artifacts/models/best.pt")
        if model_path.exists():
            logger.info("Model weights detected — loading predictor now...")
            try:
                predictor = DefectPredictor()
                logger.info("Predictor loaded successfully after background training.")
            except Exception as e:
                logger.error("Failed to load predictor: %s", e)

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not ready yet. Background training is in progress. "
                "Please wait 5–10 minutes and try again. "
                "Check /health to see when model_loaded becomes true."
            ),
        )


    # 1. Validate file extension
    filename = file.filename or "unknown.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Read and validate content size
    try:
        contents = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file content.",
        )

    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024 * 1024):.1f} MB.",
        )

    # 3. Decode image
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.warning("Corrupt image received '%s': %s", filename, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be parsed as a valid image.",
        )

    # 4. Perform Inference & Post-processing
    try:
        # Override processor thresholds if provided for this request
        orig_conf = predictor.processor.min_confidence
        orig_iou = predictor.processor.duplicate_iou_threshold

        if conf_threshold is not None:
            predictor.processor.min_confidence = conf_threshold
        if iou_threshold is not None:
            predictor.processor.duplicate_iou_threshold = iou_threshold

        pred_result = predictor.predict(pil_img)
        pred_result["filename"] = filename

        # Restore thresholds
        predictor.processor.min_confidence = orig_conf
        predictor.processor.duplicate_iou_threshold = orig_iou

        # 5. Generate Annotated Image if requested
        if include_image:
            annotated_pil = visualize_detections(
                image_input=pil_img,
                detections=pred_result["detections"],
                overall_severity=pred_result["overall_severity"],
            )
            pred_result["annotated_image_base64"] = image_to_base64(annotated_pil)
        else:
            pred_result["annotated_image_base64"] = None

        return pred_result

    except Exception as e:
        logger.error("Inference pipeline failed for '%s': %s", filename, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference processing failed on the server.",
        )

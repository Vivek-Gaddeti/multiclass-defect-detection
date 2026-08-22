"""API integration and endpoint tests using FastAPI TestClient."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create test client within app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def create_dummy_image_bytes(width=200, height=200, color=(128, 128, 128)) -> bytes:
    """Helper to create dummy in-memory JPEG bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint(client):
    """Verify /health endpoint returns HTTP 200 and schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_model_info_endpoint(client):
    """Verify /model-info returns configuration and supported classes."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_path" in data
    assert "supported_classes" in data
    assert "confidence_threshold" in data
    assert "severity_rules" in data


def test_predict_invalid_extension(client):
    """Verify /predict rejects unsupported file extensions with HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]


def test_predict_empty_file(client):
    """Verify /predict rejects 0-byte uploads with HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_valid_image(client):
    """Verify /predict successfully processes a valid image and returns schema."""
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("sample_steel.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample_steel.jpg"
    assert "defect_count" in data
    assert "overall_severity" in data
    assert "detections" in data
    assert "inference_time_ms" in data
    assert "annotated_image_base64" in data

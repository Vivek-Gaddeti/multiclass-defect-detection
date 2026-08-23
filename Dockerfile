# Multi-stage / lightweight production Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    HOST=0.0.0.0

# Install system dependencies required by OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs/ ./configs/
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/
COPY data/ ./data/

# Make startup script executable
RUN chmod +x scripts/startup.sh

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Start using the startup script (trains model if missing, then serves API)
CMD ["bash", "scripts/startup.sh"]

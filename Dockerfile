# Unified Dockerfile for Audio Fingerprinting Application
# Supports both API and background worker containers from single image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ src/
COPY stats_updater_runner.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create necessary directories with proper permissions
RUN mkdir -p temp_samples music samples database

# Expose port 8000 for the API service
EXPOSE 8000

# Default command runs the API server (can be overridden for worker)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
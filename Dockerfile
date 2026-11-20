# =====================================================================
# Production Dockerfile: Document Q&A Assistant
# - Base image: python:3.11-slim
# - Security: Non-root execution with least privilege
# - Performance: Dependency caching & build-time model pre-download
# =====================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered terminal logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOME=/home/appuser

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Create dedicated non-root user and user home directory
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m -s /bin/bash appuser

# Layer caching: Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download local FastEmbed embedding and reranker models at image build time
# This avoids runtime latency spikes and guarantees container works offline
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

# Copy application source code and data assets
COPY app/ ./app/
COPY data/ ./data/

# Adjust file permissions for non-root execution
RUN chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Healthcheck probing GET /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose HTTP port
EXPOSE 8000

# Start Uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

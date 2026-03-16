FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY serving/ ./serving/
COPY monitoring/ ./monitoring/

# Environment variables (override at runtime)
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_MODEL_NAME=credit-risk-pd-lightgbm
ENV PORT=8000

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

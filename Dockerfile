FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*
    
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES="" \
    FORCE_CPU=1

# Copy requirements first to leverage Docker cache
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the application code
COPY api.py .
COPY src/ ./src/
COPY resources/ ./resources/

# Create temp directory for file uploads
RUN mkdir -p temp_uploads

# Expose the port the app runs on
EXPOSE 9001

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9001"]

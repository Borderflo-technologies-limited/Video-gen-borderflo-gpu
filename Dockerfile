# Video Generation Service Dockerfile
# Python 3.7 is required by some Wav2Lip dependency chains; use 3.7 base
FROM python:3.7-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch (CUDA 11.3) first with generous timeouts, then the rest
ENV PIP_DEFAULT_TIMEOUT=1800 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir --upgrade pip==23.2.1 && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113 \
      torch==1.10.2+cu113 torchvision==0.11.3+cu113 && \
    pip install --no-cache-dir \
      --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org \
      -r requirements.txt

# Production stage
FROM python:3.7-slim

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Clone Wav2Lip repository
RUN git clone https://github.com/Rudrabha/Wav2Lip.git

# Note: We intentionally avoid installing Wav2Lip/requirements.txt to prevent conflicts

# Create temp directory
RUN mkdir -p temp

# Create environment file template
COPY env.example .env

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget -qO- http://localhost:8001/health || exit 1

# Run the Flask application
CMD ["python", "app/main.py"]
# Video Generation Service Dockerfile
# Python 3.8 (bullseye) to keep Debian repos current while maintaining Wav2Lip compatibility
FROM python:3.8-slim-bullseye as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Clone Wav2Lip repository (shallow) in builder
RUN git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git

# Install Wav2Lip requirements (this was missing!)
RUN pip install --no-cache-dir -r Wav2Lip/requirements.txt

# Download Wav2Lip model during build
RUN mkdir -p /app/models && \
    curl -L -o /app/models/wav2lip_gan.pth \
    "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"

# Production stage
FROM python:3.8-slim-bullseye

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Copy Wav2Lip from builder instead of cloning at runtime
COPY --from=builder /app/Wav2Lip ./Wav2Lip

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
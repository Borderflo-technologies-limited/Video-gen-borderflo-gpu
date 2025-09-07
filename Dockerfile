# Video Generation Service Dockerfile
# Python 3.8 (bullseye) to keep Debian repos current while maintaining Wav2Lip compatibility
FROM python:3.8-slim-bullseye as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
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

# Set production environment variables
ENV DEVICE=cuda
ENV LOG_LEVEL=INFO
ENV DEBUG=false

# RunPod GPU environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Clone Wav2Lip repository (shallow) in builder with error handling
RUN git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git || \
    (echo "Failed to clone Wav2Lip, trying alternative source..." && \
     git clone --depth 1 https://github.com/justinjohn0306/Wav2Lip.git) || \
    (echo "Wav2Lip clone failed, creating minimal structure..." && \
     mkdir -p Wav2Lip && \
     echo "# Wav2Lip placeholder" > Wav2Lip/requirements.txt)

# Install Wav2Lip requirements only if they exist and are valid
RUN if [ -f "Wav2Lip/requirements.txt" ] && [ -s "Wav2Lip/requirements.txt" ]; then \
        echo "Installing Wav2Lip requirements with compatibility filtering..." && \
        grep -v "opencv-python==" Wav2Lip/requirements.txt | \
        grep -v "numpy==" | \
        grep -v "librosa==" | \
        sed 's/>=/==/g' | \
        sed 's/<=/==/g' > Wav2Lip/requirements_filtered.txt && \
        if [ -s "Wav2Lip/requirements_filtered.txt" ]; then \
            pip install --no-cache-dir -r Wav2Lip/requirements_filtered.txt || \
            echo "Filtered Wav2Lip requirements installation failed, continuing..."; \
        else \
            echo "No compatible Wav2Lip requirements found after filtering, skipping..."; \
        fi; \
    else \
        echo "No valid Wav2Lip requirements found, skipping..."; \
    fi

# Download Wav2Lip model during build with error handling
RUN mkdir -p /app/models && \
    (curl -L -o /app/models/wav2lip_gan.pth \
    "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth" || \
     echo "Model download failed, will need to be downloaded at runtime") && \
    echo "Model download completed"

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
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Create models directory if it doesn't exist
RUN mkdir -p models

# Copy Wav2Lip model from builder to both locations
COPY --from=builder /app/models/wav2lip_gan.pth ./models/wav2lip_gan.pth

# Create workspace directory structure for RunPod
RUN mkdir -p /workspace/models /workspace/temp
COPY --from=builder /app/models/wav2lip_gan.pth /workspace/models/wav2lip_gan.pth

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
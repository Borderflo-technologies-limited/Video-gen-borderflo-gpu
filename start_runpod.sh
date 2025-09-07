#!/bin/bash
# RunPod startup script with GPU detection

echo "🚀 Starting Video Generation Service on RunPod"
echo "=============================================="

# Set GPU environment variables explicitly
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda

# Debug GPU availability
echo "🔍 Running GPU diagnostics..."
python debug_gpu.py

echo ""
echo "🚀 Starting Flask application..."
echo "=============================================="

# Start the Flask application
python app/main.py

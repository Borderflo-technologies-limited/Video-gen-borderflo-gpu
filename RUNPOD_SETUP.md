# RunPod Deployment Guide

## 🚀 Quick Setup for RTX A4000

### 1. RunPod Template Configuration

When creating your RunPod instance, use these settings:

**Container Image:**
```
your-registry/visa-ai-video-generation:latest
```

**Container Disk:** 10 GB (minimum)

**Exposed HTTP Ports:** 8001

**Docker Command:**
```bash
docker run -d \
  --name video-gen-prod \
  -p 8001:8001 \
  --gpus all \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e DEVICE=cuda \
  -e LOG_LEVEL=INFO \
  -e DEBUG=false \
  your-registry/visa-ai-video-generation:latest
```

### 2. Alternative: RunPod Pod Configuration

If using RunPod's web interface:

1. **Select GPU:** RTX A4000
2. **Container Image:** `your-registry/visa-ai-video-generation:latest`
3. **Expose HTTP:** Port 8001
4. **Environment Variables:**
   ```
   DEVICE=cuda
   LOG_LEVEL=INFO
   DEBUG=false
   NVIDIA_VISIBLE_DEVICES=all
   NVIDIA_DRIVER_CAPABILITIES=compute,utility
   ```

### 3. Troubleshooting GPU Access

If you still see "CUDA not available":

#### Check GPU Access in Container:
```bash
# SSH into your RunPod container
nvidia-smi

# Should show your RTX A4000
# If not, GPU access is not properly configured
```

#### Check PyTorch CUDA:
```bash
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"
```

#### Force CPU Mode (Temporary):
```bash
# Add this environment variable to force CPU mode
DEVICE=cpu
```

### 4. Expected Performance on RTX A4000

- **10-second audio:** 5-10 seconds processing
- **Memory usage:** ~3-4GB VRAM
- **Batch size:** 128 (optimal)
- **Quality:** Full Wav2Lip accuracy

### 5. Health Check Verification

Once running, check: `http://your-runpod-url:8001/health`

**Success (GPU mode):**
```json
{
  "device": "cuda",
  "wav2lip_available": true,
  "mode": "PRODUCTION",
  "gpu_available": true
}
```

**Fallback (CPU mode):**
```json
{
  "device": "cpu", 
  "wav2lip_available": false,
  "mode": "MOCK",
  "gpu_available": false
}
```

### 6. Common RunPod Issues

1. **"CUDA driver initialization failed"**
   - Ensure `--gpus all` flag is used
   - Check NVIDIA_VISIBLE_DEVICES is set
   - Verify RunPod instance has GPU access

2. **"Model file not found"**
   - Container built successfully but model not copied
   - Check build logs for download failures

3. **"Service in mock mode"**
   - Wav2Lip dependencies failed to load
   - Usually due to NumPy version conflicts (fixed)

### 7. Monitoring and Logs

```bash
# View container logs
docker logs video-gen-prod

# Monitor GPU usage
nvidia-smi -l 1

# Check service health
curl http://localhost:8001/health
```

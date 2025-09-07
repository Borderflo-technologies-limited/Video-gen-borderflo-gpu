# RunPod GPU Fix Guide

## 🚨 Common GPU Issues on RunPod

### Issue: "CUDA driver initialization failed"
This happens when the Docker container can't access the GPU properly.

## ✅ Solution Steps

### 1. **RunPod Template Configuration**

**Container Image:** `your-registry/visa-ai-video-generation:latest`

**Docker Command:**
```bash
docker run -d --name video-gen-prod \
  --gpus all \
  --runtime=nvidia \
  -p 8001:8001 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e DEVICE=cuda \
  -v /workspace:/workspace \
  your-registry/visa-ai-video-generation:latest
```

**Environment Variables:**
```
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_VISIBLE_DEVICES=0
DEVICE=cuda
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Volume Mount Path:** `workspace`

**Expose HTTP Ports:** `8001`

### 2. **Alternative: Use RunPod's GPU Template**

If the above doesn't work, try RunPod's pre-configured GPU template:

1. Go to RunPod Templates
2. Search for "PyTorch" or "CUDA" templates
3. Use their base configuration
4. Replace the Docker image with yours

### 3. **Debug GPU Access**

The container now includes a GPU debug script. Check the logs for:

```
🚀 RunPod GPU Debug Script
==================================================
🔍 Environment Variables:
  NVIDIA_VISIBLE_DEVICES: all
  NVIDIA_DRIVER_CAPABILITIES: compute,utility
  CUDA_VISIBLE_DEVICES: 0
  DEVICE: cuda

🔍 nvidia-smi Check:
✅ nvidia-smi available:
[GPU info here]

🔍 PyTorch CUDA Check:
✅ PyTorch version: 1.10.2+cu113
✅ CUDA available: True
✅ CUDA device count: 1
✅ Device name: NVIDIA RTX A4000
```

### 4. **If GPU Still Not Working**

**Option A: Force CPU Mode**
```bash
docker run -d --name video-gen-prod \
  -p 8001:8001 \
  -e DEVICE=cpu \
  -v /workspace:/workspace \
  your-registry/visa-ai-video-generation:latest
```

**Option B: Check RunPod Pod Configuration**
- Ensure your pod has GPU enabled
- Check if the pod is using the correct GPU runtime
- Try restarting the pod

### 5. **Expected Success Logs**

When working correctly, you should see:
```
✅ CUDA available: 1 GPU(s) - NVIDIA RTX A4000
✅ Model file found: 438.7 MB
✅ Wav2Lip dependencies loaded successfully. Running in PRODUCTION mode.
✅ Video Generation Service started successfully
```

### 6. **Test GPU Access**

Once running, test with:
```bash
curl http://localhost:8001/health
```

Should return:
```json
{
  "status": "healthy",
  "device": "cuda",
  "wav2lip_available": true,
  "mode": "PRODUCTION",
  "gpu_available": true
}
```

## 🔧 Troubleshooting

### Problem: Container starts but GPU not detected
**Solution:** Check RunPod pod configuration, ensure GPU is enabled

### Problem: "nvidia-smi not found"
**Solution:** RunPod pod doesn't have GPU runtime enabled

### Problem: PyTorch CUDA not available
**Solution:** Container built without CUDA support, rebuild with proper base image

### Problem: Model file not found
**Solution:** Check volume mount path is `/workspace` and model is copied correctly

## 📞 Support

If issues persist:
1. Check RunPod pod logs
2. Run the debug script: `python debug_gpu.py`
3. Verify pod has GPU enabled
4. Try the CPU fallback mode

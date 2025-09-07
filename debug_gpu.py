#!/usr/bin/env python3
"""
GPU Detection and Debug Script for RunPod
"""

import os
import sys
import subprocess

def check_nvidia_smi():
    """Check if nvidia-smi is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi available:")
            print(result.stdout)
            return True
        else:
            print(f"❌ nvidia-smi failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ nvidia-smi error: {e}")
        return False

def check_environment_variables():
    """Check GPU-related environment variables"""
    print("\n🔍 Environment Variables:")
    gpu_vars = [
        'NVIDIA_VISIBLE_DEVICES',
        'NVIDIA_DRIVER_CAPABILITIES', 
        'CUDA_VISIBLE_DEVICES',
        'DEVICE',
        'PYTORCH_CUDA_ALLOC_CONF'
    ]
    
    for var in gpu_vars:
        value = os.getenv(var, 'NOT SET')
        print(f"  {var}: {value}")

def check_pytorch_cuda():
    """Check PyTorch CUDA availability"""
    print("\n🔍 PyTorch CUDA Check:")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.current_device()}")
            print(f"✅ Device name: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA version: {torch.version.cuda}")
        else:
            print("❌ CUDA not available in PyTorch")
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
    except Exception as e:
        print(f"❌ PyTorch CUDA check failed: {e}")

def check_cuda_libraries():
    """Check CUDA libraries"""
    print("\n🔍 CUDA Libraries:")
    try:
        import torch
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"✅ PyTorch CUDA version: {torch.version.cuda}")
        else:
            print("❌ PyTorch not compiled with CUDA")
    except Exception as e:
        print(f"❌ CUDA library check failed: {e}")

def main():
    print("🚀 RunPod GPU Debug Script")
    print("=" * 50)
    
    check_environment_variables()
    check_nvidia_smi()
    check_pytorch_cuda()
    check_cuda_libraries()
    
    print("\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    main()

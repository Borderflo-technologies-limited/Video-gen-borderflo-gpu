#!/usr/bin/env python3
"""
Test script to check Wav2Lip imports and availability
"""

import os
import sys
from pathlib import Path

print("🔍 Testing Wav2Lip Availability")
print("=" * 50)

# Check if Wav2Lip directory exists
wav2lip_path = Path("Wav2Lip")
print(f"Wav2Lip directory exists: {wav2lip_path.exists()}")

if wav2lip_path.exists():
    print(f"Wav2Lip directory contents:")
    for item in wav2lip_path.iterdir():
        print(f"  - {item.name}")
    
    # Check models subdirectory
    models_path = wav2lip_path / "models"
    if models_path.exists():
        print(f"\nWav2Lip/models contents:")
        for item in models_path.iterdir():
            print(f"  - {item.name}")
    else:
        print(f"\nWav2Lip/models directory missing!")

# Check if model file exists
model_path = Path("models/wav2lip_gan.pth")
print(f"\nModel file exists: {model_path.exists()}")
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model file size: {size_mb:.1f} MB")

# Test imports
print(f"\n🧪 Testing Python Imports")
print("-" * 30)

# Add Wav2Lip to path
sys.path.append('Wav2Lip')
print(f"Added Wav2Lip to Python path")

# Test individual imports
imports_to_test = [
    ("librosa", "import librosa"),
    ("torch", "import torch"),
    ("cv2", "import cv2"),
    ("numpy", "import numpy as np"),
    ("models", "from models import Wav2Lip"),
    ("face_detection", "import face_detection"),
    ("audio", "import audio"),
]

for name, import_stmt in imports_to_test:
    try:
        exec(import_stmt)
        print(f"✅ {name}: SUCCESS")
    except ImportError as e:
        print(f"❌ {name}: FAILED - {e}")
    except Exception as e:
        print(f"⚠️  {name}: ERROR - {e}")

# Test if we can create Wav2Lip model
print(f"\n🏗️  Testing Wav2Lip Model Creation")
print("-" * 35)

try:
    sys.path.append('Wav2Lip')
    from models import Wav2Lip
    model = Wav2Lip()
    print("✅ Wav2Lip model creation: SUCCESS")
    print(f"   Model type: {type(model)}")
except Exception as e:
    print(f"❌ Wav2Lip model creation: FAILED - {e}")

print(f"\n📊 Summary")
print("=" * 20)
print("If all imports succeed, WAV2LIP_AVAILABLE should be True")
print("If any imports fail, the service runs in mock mode")

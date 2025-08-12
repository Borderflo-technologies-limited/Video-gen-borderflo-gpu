#!/usr/bin/env python3
"""
Setup script for Video Generation Service
Downloads required models and sets up dependencies
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """Download a file from URL to destination"""
    logger.info(f"Downloading {url} to {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded {destination}")

def setup_wav2lip():
    """Set up Wav2Lip repository"""
    wav2lip_dir = Path("Wav2Lip")
    
    if wav2lip_dir.exists():
        logger.info("Wav2Lip directory already exists")
        return
    
    logger.info("Cloning Wav2Lip repository")
    os.system("git clone https://github.com/Rudrabha/Wav2Lip.git")
    
    # Download face detection model
    face_detection_dir = wav2lip_dir / "face_detection" / "detection" / "sfd"
    face_detection_dir.mkdir(parents=True, exist_ok=True)
    
    face_model_url = "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/s3fd.pth"
    face_model_path = face_detection_dir / "s3fd.pth"
    
    if not face_model_path.exists():
        download_file(face_model_url, face_model_path)

def setup_models():
    """Set up model directory and download models"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    wav2lip_model_path = models_dir / "wav2lip_gan.pth"
    
    if not wav2lip_model_path.exists():
        logger.info("Wav2Lip model not found. Please download it manually:")
        logger.info("1. Go to https://github.com/Rudrabha/Wav2Lip")
        logger.info("2. Download wav2lip_gan.pth from releases")
        logger.info(f"3. Place it at {wav2lip_model_path}")
        logger.warning("Service will run in mock mode without the model")

def setup_directories():
    """Create necessary directories"""
    directories = ["temp", "models", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main setup function"""
    logger.info("Setting up Video Generation Service")
    
    # Create directories
    setup_directories()
    
    # Set up Wav2Lip
    setup_wav2lip()
    
    # Set up models
    setup_models()
    
    logger.info("Setup completed!")
    logger.info("To start the service, run: python app/main.py")

if __name__ == "__main__":
    main()
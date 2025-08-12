#!/usr/bin/env python3
"""
Configuration for Video Generation Service
"""

import os
from pathlib import Path

class Settings:
    """Video Generation Service Settings"""
    
    # Service Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Model Configuration
    MODEL_PATH: str = "models/wav2lip_gan.pth"
    DEVICE: str = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'
    
    # Processing Configuration
    IMG_SIZE: int = 96  # 96 for fast, 288 for high quality
    WAV2LIP_BATCH_SIZE: int = 128
    FPS: float = 25.0
    
    # File Storage
    TEMP_DIR: str = "temp"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_AUDIO_TYPES: list = ["audio/wav", "audio/mp3", "audio/mpeg"]
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/jpg", "image/png"]
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/avi", "video/mov"]
    
    # Cleanup Configuration
    AUTO_CLEANUP_HOURS: int = 24
    CLEANUP_INTERVAL_MINUTES: int = 60
    
    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(Path(self.MODEL_PATH).parent, exist_ok=True)

# Global settings instance
settings = Settings()
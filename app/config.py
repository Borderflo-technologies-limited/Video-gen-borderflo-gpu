#!/usr/bin/env python3
"""
Configuration for Video Generation Service
"""

import os
from pathlib import Path
from dotenv import load_dotenv

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Video Generation Service Settings"""
    
    # Service Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/wav2lip_gan.pth")
    
    # Device selection with better error handling
    def _get_device():
        device_env = os.getenv("DEVICE")
        if device_env:
            return device_env.lower()
        
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    return 'cpu'
            except Exception:
                return 'cpu'
        return 'cpu'
    
    DEVICE: str = _get_device()
    
    # Default Video Configuration
    DEFAULT_VIDEO_PATH: str = os.getenv("DEFAULT_VIDEO_PATH", "models/default_face.mp4")
    
    # Processing Configuration
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "96"))  # 96 for fast, 288 for high quality
    WAV2LIP_BATCH_SIZE: int = int(os.getenv("WAV2LIP_BATCH_SIZE", "128"))
    FPS: float = float(os.getenv("FPS", "25.0"))
    
    # File Storage
    TEMP_DIR: str = os.path.abspath(os.getenv("TEMP_DIR", "temp"))
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB
    ALLOWED_AUDIO_TYPES: list = ["audio/wav", "audio/mp3", "audio/mpeg"]
    ALLOWED_IMAGE_TYPES: list = ["image/jpeg", "image/jpg", "image/png"]
    ALLOWED_VIDEO_TYPES: list = ["video/mp4", "video/avi", "video/mov"]
    
    # Cleanup Configuration
    AUTO_CLEANUP_HOURS: int = int(os.getenv("AUTO_CLEANUP_HOURS", "24"))
    CLEANUP_INTERVAL_MINUTES: int = int(os.getenv("CLEANUP_INTERVAL_MINUTES", "60"))
    
    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(Path(self.MODEL_PATH).parent, exist_ok=True)

# Global settings instance
settings = Settings()
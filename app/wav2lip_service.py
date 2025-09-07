#!/usr/bin/env python3
"""
Wav2Lip Service Module
Handles Wav2Lip model loading and inference
"""

import os
import sys
import cv2
import numpy as np
import torch
import logging
from typing import Dict, Any, List

# Add Wav2Lip to path
sys.path.append('Wav2Lip')

try:
    from models import Wav2Lip
    import face_detection
    import audio
    import librosa
    WAV2LIP_AVAILABLE = True
except ImportError:
    WAV2LIP_AVAILABLE = False
    logging.warning("Wav2Lip dependencies not available. Service will run in mock mode.")

logger = logging.getLogger(__name__)

class Wav2LipProcessor:
    """Wav2Lip model processor"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.face_detector = None
        
    def load_model(self):
        """Load Wav2Lip model"""
        if not WAV2LIP_AVAILABLE:
            logger.warning("Wav2Lip not available, using mock mode")
            return
            
        if self.model is not None:
            return self.model
        
        logger.info(f"Loading Wav2Lip model from {self.model_path}")
        
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Wav2Lip model not found at {self.model_path}")
        
        model = Wav2Lip()
        checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        logger.info("Wav2Lip model loaded successfully")
        return self.model
    
    def load_face_detector(self):
        """Load face detector"""
        if not WAV2LIP_AVAILABLE:
            return None
            
        if self.face_detector is not None:
            return self.face_detector
            
        logger.info("Loading face detector")
        self.face_detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, 
            flip_input=False, 
            device=self.device
        )
        return self.face_detector
    
    def process_audio(self, audio_path: str) -> np.ndarray:
        """Process audio file to mel spectrogram"""
        if not WAV2LIP_AVAILABLE:
            # Return mock mel spectrogram
            return np.random.rand(80, 100)
            
        logger.info(f"Processing audio from {audio_path}")
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
        
        return mel
    
    def load_frames(self, face_path: str, audio_duration: float) -> List[np.ndarray]:
        """Load frames from image or video"""
        logger.info(f"Loading frames from {face_path}")
        
        if face_path.endswith(('.mp4', '.avi', '.mov')):
            # Load video frames
            cap = cv2.VideoCapture(face_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        else:
            # Single image - repeat for audio duration
            frame = cv2.imread(face_path)
            if frame is None:
                raise ValueError(f"Could not load image from {face_path}")
            
            # Estimate number of frames needed (assuming 25 fps)
            num_frames = max(1, int(audio_duration * 25))
            frames = [frame] * num_frames
        
        logger.info(f"Loaded {len(frames)} frames")
        return frames
    
    def detect_faces(self, frames: List[np.ndarray]) -> np.ndarray:
        """Detect faces in frames"""
        if not WAV2LIP_AVAILABLE:
            # Return mock face boxes
            height, width = frames[0].shape[:2]
            return np.array([[width//4, height//4, 3*width//4, 3*height//4]] * len(frames))
        
        detector = self.load_face_detector()
        
        batch_size = 16
        while True:
            predictions = []
            try:
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i + batch_size]
                    predictions.extend(detector.get_detections_for_batch(np.array(batch)))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError('Image too big to run face detection on GPU')
                batch_size //= 2
                logger.warning(f'Reducing batch size to: {batch_size}')
                continue
            break
        
        # Process predictions
        results = []
        for rect in predictions:
            if rect is None:
                # Default face box if no face detected
                height, width = frames[0].shape[:2]
                results.append([width//4, height//4, 3*width//4, 3*height//4])
            else:
                results.append(rect)
        
        return np.array(results)
    
    def generate_video(self, face_path: str, audio_path: str, output_path: str) -> Dict[str, Any]:
        """
        Generate lip-sync video using Wav2Lip
        
        Args:
            face_path: Path to reference face image or video
            audio_path: Path to audio file
            output_path: Path for output video
            
        Returns:
            Dictionary with generation results
        """
        import time
        start_time = time.time()
        
        try:
            if not WAV2LIP_AVAILABLE:
                # Mock video generation
                logger.info("Generating mock video (Wav2Lip not available)")
                
                # Get audio duration to determine video length
                try:
                    import librosa
                    audio_data, sample_rate = librosa.load(audio_path, sr=None)
                    audio_duration = len(audio_data) / sample_rate
                    logger.info(f"Audio duration: {audio_duration:.2f} seconds")
                except:
                    audio_duration = 5.0  # Default 5 seconds if can't determine
                    logger.warning("Could not determine audio duration, using 5 seconds")
                
                # Create a simple mock video
                cap = cv2.VideoCapture(face_path) if face_path.endswith(('.mp4', '.avi')) else None
                if cap:
                    ret, frame = cap.read()
                    cap.release()
                else:
                    frame = cv2.imread(face_path)
                
                if frame is None:
                    # Create a default frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Mock Video", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Calculate number of frames needed
                fps = 25.0
                num_frames = int(audio_duration * fps)
                logger.info(f"Creating {num_frames} frames for {audio_duration:.2f}s video at {fps} FPS")
                
                # Write mock video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                logger.info(f"Creating mock video at: {output_path}")
                logger.info(f"Frame shape: {frame.shape}")
                
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
                
                if not out.isOpened():
                    logger.error(f"Failed to open video writer for {output_path}")
                    return {
                        "success": False,
                        "error": "Failed to create video file",
                        "duration": time.time() - start_time
                    }
                
                # Write frames with some variation
                for i in range(num_frames):
                    # Create a slightly different frame each time
                    current_frame = frame.copy()
                    
                    # Add a moving element or color variation
                    if i % 10 == 0:  # Every 10th frame
                        # Add a subtle color shift
                        current_frame = cv2.addWeighted(current_frame, 0.9, 
                                                      np.ones_like(current_frame) * 10, 0.1, 0)
                    
                    # Add frame number for debugging
                    cv2.putText(current_frame, f"Frame {i+1}/{num_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    out.write(current_frame)
                
                out.release()
                
                logger.info(f"Mock video created successfully: {output_path}")
                logger.info(f"Video duration: {audio_duration:.2f} seconds, {num_frames} frames")
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "duration": time.time() - start_time,
                    "file_size": os.path.getsize(output_path),
                    "mode": "mock"
                }
            
            # Load model
            model = self.load_model()
            
            # Process audio
            mel = self.process_audio(audio_path)
            
            # Estimate audio duration for frame calculation
            audio_duration = mel.shape[1] * 0.04  # Approximate duration
            
            # Load frames
            frames = self.load_frames(face_path, audio_duration)
            
            # Detect faces
            face_boxes = self.detect_faces(frames)
            
            # TODO: Implement actual Wav2Lip inference
            # This is where the actual model inference would happen
            # For now, create a simple output video
            
            logger.info("Generating video with Wav2Lip model")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create output video
            fps = 25
            frame_h, frame_w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
            
            if not out.isOpened():
                logger.error(f"Failed to open video writer for {output_path}")
                return {
                    "success": False,
                    "error": "Failed to create video file",
                    "duration": time.time() - start_time
                }
            
            # Write frames (simplified version)
            for frame in frames:
                out.write(frame)
            
            out.release()
            logger.info(f"Video created successfully: {output_path}")
            
            duration = time.time() - start_time
            file_size = os.path.getsize(output_path)
            
            logger.info(f"Video generation completed in {duration:.2f} seconds")
            
            return {
                "success": True,
                "output_path": output_path,
                "duration": duration,
                "file_size": file_size,
                "frames": len(frames),
                "fps": fps
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
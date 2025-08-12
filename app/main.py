#!/usr/bin/env python3
"""
Video Generation Service using Wav2Lip
Flask service for generating lip-sync videos (Python 3.7 compatible)
"""

import os
import uuid
import logging
import threading
import time
import shutil
from pathlib import Path
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS

from wav2lip_service import Wav2LipProcessor
from config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = settings.MAX_FILE_SIZE

# Add CORS support
CORS(app)

# Global processor instance
processor = None

def initialize_processor():
    """Initialize the Wav2Lip processor"""
    global processor
    try:
        logger.info("Initializing Wav2Lip processor...")
        processor = Wav2LipProcessor(settings.MODEL_PATH, settings.DEVICE)
        processor.load_model()
        logger.info("Video Generation Service started successfully")
        logger.info(f"Device: {settings.DEVICE}")
        logger.info(f"Model path: {settings.MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        # Continue anyway, processor will work in mock mode

# Initialize processor on startup
initialize_processor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "video-generation",
        "device": settings.DEVICE,
        "model_path": settings.MODEL_PATH,
        "model_loaded": processor is not None and processor.model is not None
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "Video Generation Service",
        "status": "running",
        "service": "video-generation", 
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate-video",
            "download": "/download/<filename>",
            "cleanup": "/cleanup/<task_id>"
        }
    })

@app.route('/generate-video', methods=['POST'])
def generate_video():
    """
    Generate lip-sync video from audio and face image/video
    
    Expects multipart/form-data with:
    - audio_file: Audio file (WAV, MP3, etc.)
    - face_file: Face image (JPG, PNG) or video (MP4, AVI, etc.)
    - session_id: Optional session identifier
    - question_id: Optional question identifier
    
    Returns:
        JSON response with task ID and status
    """
    if not processor:
        return jsonify({"error": "Video processor not initialized"}), 503
    
    # Check if files are present
    if 'audio_file' not in request.files:
        return jsonify({"error": "audio_file is required"}), 400
        
    if 'face_file' not in request.files:
        return jsonify({"error": "face_file is required"}), 400
    
    audio_file = request.files['audio_file']
    face_file = request.files['face_file']
    
    # Get optional parameters
    session_id = request.form.get('session_id')
    question_id = request.form.get('question_id')
    
    # Validate file selection
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
        
    if face_file.filename == '':
        return jsonify({"error": "No face file selected"}), 400
    
    # Validate file types
    if audio_file.content_type not in settings.ALLOWED_AUDIO_TYPES:
        return jsonify({
            "error": f"Audio file type {audio_file.content_type} not supported"
        }), 400
    
    face_types = settings.ALLOWED_IMAGE_TYPES + settings.ALLOWED_VIDEO_TYPES
    if face_file.content_type not in face_types:
        return jsonify({
            "error": f"Face file type {face_file.content_type} not supported"
        }), 400
    
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create temp file paths
        temp_audio_path = os.path.join(settings.TEMP_DIR, f"{task_id}_audio.wav")
        face_extension = Path(face_file.filename).suffix
        temp_face_path = os.path.join(settings.TEMP_DIR, f"{task_id}_face{face_extension}")
        output_path = os.path.join(settings.TEMP_DIR, f"{task_id}_output.mp4")
        
        # Ensure temp directory exists
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # Save audio file
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.stream, buffer)
        
        # Save face file
        with open(temp_face_path, "wb") as buffer:
            shutil.copyfileobj(face_file.stream, buffer)
        
        # Generate video
        result = processor.generate_video(temp_face_path, temp_audio_path, output_path)
        
        if result["success"]:
            # Schedule cleanup in background thread
            cleanup_thread = threading.Thread(
                target=cleanup_task_files,
                args=(task_id, 3600)  # 1 hour delay
            )
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            return jsonify({
                "task_id": task_id,
                "status": "completed",
                "message": "Video generated successfully",
                "video_url": f"/download/{task_id}_output.mp4",
                "duration": result.get("duration"),
                "session_id": session_id,
                "question_id": question_id
            })
        else:
            return jsonify({
                "task_id": task_id,
                "status": "failed",
                "error": result.get("error", "Unknown error occurred")
            }), 500
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return jsonify({"error": f"Video generation failed: {str(e)}"}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download generated video file"""
    try:
        file_path = os.path.join(settings.TEMP_DIR, filename)
        
        if not os.path.exists(file_path):
            abort(404, description="File not found")
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return jsonify({"error": "Download failed"}), 500

@app.route('/cleanup/<task_id>', methods=['DELETE'])
def cleanup_task(task_id):
    """Clean up temporary files for a task"""
    try:
        files_to_remove = [
            f"{task_id}_audio.wav",
            f"{task_id}_face.jpg",
            f"{task_id}_face.png", 
            f"{task_id}_face.mp4",
            f"{task_id}_face.avi",
            f"{task_id}_output.mp4"
        ]
        
        removed_files = []
        for filename in files_to_remove:
            file_path = os.path.join(settings.TEMP_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_files.append(filename)
        
        return jsonify({
            "message": f"Cleaned up files for task {task_id}",
            "removed_files": removed_files
        })
            
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return jsonify({"error": "Cleanup failed"}), 500

def cleanup_task_files(task_id, delay=3600):
    """Background function to clean up temporary files"""
    try:
        # Wait for the specified delay
        time.sleep(delay)
        
        files_to_remove = [
            f"{task_id}_audio.wav",
            f"{task_id}_face.jpg",
            f"{task_id}_face.png",
            f"{task_id}_face.mp4", 
            f"{task_id}_face.avi",
            f"{task_id}_output.mp4"
        ]
        
        removed_count = 0
        for filename in files_to_remove:
            file_path = os.path.join(settings.TEMP_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Background cleanup removed {removed_count} files for task {task_id}")
            
    except Exception as e:
        logger.error(f"Background cleanup failed: {e}")

if __name__ == "__main__":
    app.run(
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG
    )
# Video Generation Service

A Flask-based service for generating lip-sync videos using Wav2Lip technology. This service takes an audio file and a face image/video as input and generates a synchronized video output.

## 🚀 Quick Start

### Setup Default Video (Optional)
For audio-only video generation, you can provide a default video file:

```bash
# Place your default video file in the models directory
cp your_default_video.mp4 models/default_face.mp4

# Or set a custom path via environment variable
export DEFAULT_VIDEO_PATH="/path/to/your/default_video.mp4"
```

If no default video is provided, the service will automatically create a simple face image as fallback.

### Build and Run Container

```bash
# Build the Docker image
docker build -t visa-ai-video-generation:latest .

# Run the container
docker run -d --name video-gen-test \
  -p 8001:8001 \
  -e HOST=0.0.0.0 \
  -e PORT=8001 \
  -e DEVICE=cpu \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/temp:/app/temp \
  visa-ai-video-generation:latest
```

### Test Endpoints

```bash
# 1. Health Check
curl http://localhost:8001/health

# 2. Service Info
curl http://localhost:8001/

# 3. Generate Video (requires files)
curl -X POST http://localhost:8001/generate-video \
  -F "audio_file=@/path/to/audio.wav" \
  -F "face_file=@/path/to/face.jpg" \
  -F "session_id=test123" \
  -F "question_id=q1"

# 4. Generate Video (audio only - uses default face)
curl -X POST http://localhost:8001/generate-video-audio-only \
  -F "audio_file=@/path/to/audio.wav" \
  -F "session_id=test123" \
  -F "question_id=q1"
```

## 📋 API Endpoints

### 1. Health Check
- **URL**: `GET /health`
- **Description**: Check service health and model status
- **Response**:
```json
{
  "status": "healthy",
  "service": "video-generation",
  "device": "cpu",
  "model_path": "models/wav2lip_gan.pth",
  "model_loaded": true
}
```

### 2. Service Information
- **URL**: `GET /`
- **Description**: Get service details and available endpoints
- **Response**:
```json
{
  "message": "Video Generation Service",
  "status": "running",
  "service": "video-generation",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "generate": "/generate-video",
    "generate_audio_only": "/generate-video-audio-only",
    "download": "/download/<filename>",
    "cleanup": "/cleanup/<task_id>"
  }
}
```

### 3. Generate Video
- **URL**: `POST /generate-video`
- **Description**: Generate lip-sync video from audio and face input
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio_file` (required): Audio file (WAV, MP3, MPEG)
  - `face_file` (required): Face image (JPG, PNG) or video (MP4, AVI, MOV)
  - `session_id` (optional): Session identifier
  - `question_id` (optional): Question identifier

**Response (Success)**:
```json
{
  "task_id": "uuid-12345",
  "status": "success",
  "message": "Video generated successfully",
  "video_url": "/download/uuid-12345_output.mp4",
  "duration": 15.5,
  "session_id": "test123",
  "question_id": "q1"
}
```

**Response (Error)**:
```json
{
  "error": "audio_file is required"
}
```

### 4. Generate Video (Audio Only)
- **URL**: `POST /generate-video-audio-only`
- **Description**: Generate lip-sync video from audio only using a default video file or fallback face image
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio_file` (required): Audio file (WAV, MP3, MPEG)
  - `session_id` (optional): Session identifier
  - `question_id` (optional): Question identifier

**Response (Success)**:
```json
{
  "task_id": "uuid-12345",
  "status": "completed",
  "message": "Video generated successfully using default face/video",
  "video_url": "/download/uuid-12345_output.mp4",
  "duration": 15.5,
  "session_id": "test123",
  "question_id": "q1",
  "face_type": "default_video"
}
```

**Response (Error)**:
```json
{
  "error": "audio_file is required"
}
```

### 5. Download Video
- **URL**: `GET /download/<filename>`
- **Description**: Download generated video file
- **Parameters**: `filename` - The filename returned from generate-video
- **Response**: Video file (MP4)

### 6. Cleanup Files
- **URL**: `DELETE /cleanup/<task_id>`
- **Description**: Clean up temporary files for a specific task
- **Parameters**: `task_id` - The task ID from generate-video
- **Response**:
```json
{
  "message": "Cleaned up files for task uuid-12345",
  "removed_files": ["uuid-12345_audio.wav", "uuid-12345_output.mp4"]
}
```

## 🧪 Testing Scenarios

### Basic Health Check
```bash
# Test if service is running
curl -s http://localhost:8001/health | jq '.status'
# Expected: "healthy"
```

### Service Information
```bash
# Get service details
curl -s http://localhost:8001/ | jq '.endpoints'
# Expected: List of available endpoints
```

### Video Generation Test
```bash
# Create test files (if you don't have them)
echo "test audio" > test_audio.wav
convert -size 100x100 xc:white test_face.jpg

# Generate video
curl -X POST http://localhost:8001/generate-video \
  -F "audio_file=@test_audio.wav" \
  -F "face_file=@test_face.jpg" \
  -F "session_id=test123"

# Generate video (audio only)
curl -X POST http://localhost:8001/generate-video-audio-only \
  -F "audio_file=@test_audio.wav" \
  -F "session_id=test123"
```

### Download Test
```bash
# Download generated video (replace with actual task_id)
curl -O http://localhost:8001/download/task_id_output.mp4
```

### Cleanup Test
```bash
# Clean up files (replace with actual task_id)
curl -X DELETE http://localhost:8001/cleanup/task_id
```

## ⚙️ Configuration

### Environment Variables
- `HOST`: Service host (default: 0.0.0.0)
- `PORT`: Service port (default: 8001)
- `DEBUG`: Debug mode (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)
- `DEVICE`: Processing device (default: cuda, fallback: cpu)
- `MODEL_PATH`: Path to Wav2Lip model (default: models/wav2lip_gan.pth)
- `DEFAULT_VIDEO_PATH`: Path to default video for audio-only generation (default: models/default_face.mp4)

### Model Requirements
- **Wav2Lip Model**: `wav2lip_gan.pth` in `models/` directory
- **Default Video**: `default_face.mp4` in `models/` directory (for audio-only generation)
- **Supported Audio**: WAV, MP3, MPEG
- **Supported Images**: JPG, PNG
- **Supported Videos**: MP4, AVI, MOV

## 🔧 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```bash
   # Check if model file exists
   ls -la models/wav2lip_gan.pth
   
   # Ensure model directory is mounted
   docker run -v $(pwd)/models:/app/models ...
   ```

2. **Service Not Starting**
   ```bash
   # Check container logs
   docker logs video-gen-test
   
   # Check if port is available
   netstat -tulpn | grep 8001
   ```

3. **File Upload Issues**
   ```bash
   # Check file size limits
   ls -lh your_file.wav
   
   # Ensure correct content type
   file your_file.wav
   ```

### Debug Mode
```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG -e DEBUG=true ...
```

## 📊 Performance

- **Processing Time**: 10-30 seconds depending on video length
- **Memory Usage**: 2-4GB RAM
- **GPU**: CUDA support for faster processing
- **File Size Limit**: 100MB per file

## 🔗 Dependencies

- **Flask**: Web framework
- **Wav2Lip**: Core video generation
- **OpenCV**: Image/video processing
- **FFmpeg**: Audio/video conversion
- **NumPy**: Numerical operations

## 📝 Notes

- Temporary files are automatically cleaned up after 24 hours
- Service supports both CPU and GPU processing
- Input files are validated for type and size
- Generated videos are stored temporarily and can be downloaded
- Background cleanup runs automatically to manage disk space

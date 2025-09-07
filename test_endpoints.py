#!/usr/bin/env python3
"""
Test script for Video Generation Service endpoints
Run this after starting the container to verify all endpoints work correctly
"""

import requests
import time
import os

# Service configuration
BASE_URL = "http://localhost:8001"
TEST_FILES_DIR = "test_files"

def create_test_files():
    """Create simple test files for testing"""
    os.makedirs(TEST_FILES_DIR, exist_ok=True)

    # Create a simple test audio file (WAV format)
    audio_path = os.path.join(TEST_FILES_DIR, "test_audio.wav")
    if not os.path.exists(audio_path):
        # Create a minimal WAV file (44.1kHz, 16-bit, mono, 1 second)
        with open(audio_path, 'wb') as f:
            # WAV header for 1 second of silence
            f.write(b'RIFF')
            f.write((36).to_bytes(4, 'little'))  # File size
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))  # Chunk size
            f.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
            f.write((1).to_bytes(2, 'little'))   # Channels (mono)
            f.write((44100).to_bytes(4, 'little'))  # Sample rate
            f.write((88200).to_bytes(4, 'little'))  # Byte rate
            f.write((2).to_bytes(2, 'little'))   # Block align
            f.write((16).to_bytes(2, 'little'))  # Bits per sample
            f.write(b'data')
            f.write((0).to_bytes(4, 'little'))   # Data chunk size
            f.write(b'\x00' * 44100)  # 1 second of silence
    
    # Create a simple test image file (PNG format)
    image_path = os.path.join(TEST_FILES_DIR, "test_face.png")
    if not os.path.exists(image_path):
        # Create a minimal PNG file (1x1 white pixel)
        png_data = (
            b'\x89PNG\r\n\x1a\n'  # PNG signature
            b'\x00\x00\x00\r'     # IHDR chunk length
            b'IHDR'               # IHDR chunk type
            b'\x00\x00\x00\x01'   # Width: 1
            b'\x00\x00\x00\x01'   # Height: 1
            b'\x08'               # Bit depth: 8
            b'\x02'               # Color type: RGB
            b'\x00'               # Compression: deflate
            b'\x00'               # Filter: none
            b'\x00'               # Interlace: none
            b'\x1f\x15\xc4\x89'   # CRC
            b'\x00\x00\x00\x0c'   # IDAT chunk length
            b'IDAT'               # IDAT chunk type
            b'\x08\x1d\x01\x01\x00\x00\x00\xff\xff'  # Compressed data
            b'\x00\x00\x00\x00'   # CRC
            b'\x00\x00\x00\x00'   # IEND chunk length
            b'IEND'               # IEND chunk type
            b'\xae\x42\x60\x82'   # CRC
        )
        with open(image_path, 'wb') as f:
            f.write(png_data)
    
    print(f"✅ Test files created in {TEST_FILES_DIR}/")
    return audio_path, image_path

def test_health_check():
    """Test the health check endpoint"""
    print("\n🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   Device: {data['device']}")
            print(f"   Model Loaded: {data['model_loaded']}")
            return True
        else:
            print(f"❌ Health Check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check error: {e}")
        return False

def test_service_info():
    """Test the service information endpoint"""
    print("\n🔍 Testing Service Info...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service Info: {data['message']}")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Available endpoints: {list(data['endpoints'].keys())}")
            return True
        else:
            print(f"❌ Service Info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Service Info error: {e}")
        return False

def test_video_generation(audio_path, image_path):
    """Test the video generation endpoint"""
    print("\n🔍 Testing Video Generation...")
    try:
        with open(audio_path, 'rb') as audio_file, open(image_path, 'rb') as image_file:
            files = {
                'audio_file': ('test_audio.wav', audio_file, 'audio/wav'),
                'face_file': ('test_face.png', image_file, 'image/png')
            }
            data = {
                'session_id': 'test_session_123',
                'question_id': 'test_question_1'
            }
            
            response = requests.post(f"{BASE_URL}/generate-video", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Video Generation: {result['status']}")
                print(f"   Task ID: {result['task_id']}")
                print(f"   Video URL: {result['video_url']}")
                return result['task_id']
            else:
                print(f"❌ Video Generation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
    except Exception as e:
        print(f"❌ Video Generation error: {e}")
        return None

def test_audio_only_generation(audio_path):
    """Test the audio-only video generation endpoint"""
    print("\n🔍 Testing Audio-Only Video Generation...")
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {
                'audio_file': ('test_audio.wav', audio_file, 'audio/wav')
            }
            data = {
                'session_id': 'test_session_audio_only',
                'question_id': 'test_question_audio_only'
            }
            
            response = requests.post(f"{BASE_URL}/generate-video-audio-only", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Audio-Only Video Generation: {result['status']}")
                print(f"   Task ID: {result['task_id']}")
                print(f"   Video URL: {result['video_url']}")
                print(f"   Face Type: {result.get('face_type', 'unknown')}")
                return result['task_id']
            else:
                print(f"❌ Audio-Only Video Generation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
    except Exception as e:
        print(f"❌ Audio-Only Video Generation error: {e}")
        return None

def test_download(task_id):
    """Test the download endpoint"""
    print(f"\n🔍 Testing Download for task {task_id}...")
    try:
        filename = f"{task_id}_output.mp4"
        response = requests.get(f"{BASE_URL}/download/{filename}")
        
        if response.status_code == 200:
            # Save the downloaded file
            download_path = os.path.join(TEST_FILES_DIR, filename)
            with open(download_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Download successful: {filename}")
            print(f"   File size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Download failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Download error: {e}")
        return False

def test_cleanup(task_id):
    """Test the cleanup endpoint"""
    print(f"\n🔍 Testing Cleanup for task {task_id}...")
    try:
        response = requests.delete(f"{BASE_URL}/cleanup/{task_id}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cleanup successful")
            print(f"   Removed files: {result['removed_files']}")
            return True
        else:
            print(f"❌ Cleanup failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Video Generation Service Tests")
    print("=" * 50)
    
    # Check if service is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except:
        print("❌ Service is not running!")
        print("Please start the container first:")
        print("docker run -d --name video-gen-test -p 8001:8001 visa-ai-video-generation:latest")
        return
    
    # Create test files
    audio_path, image_path = create_test_files()
    
    # Run tests
    tests_passed = 0
    total_tests = 6
    
    if test_health_check():
        tests_passed += 1
    
    if test_service_info():
        tests_passed += 1
    
    task_id = test_video_generation(audio_path, image_path)
    if task_id:
        tests_passed += 1
        
        # Wait a bit for processing
        print("⏳ Waiting for video processing...")
        time.sleep(5)
        
        if test_download(task_id):
            tests_passed += 1
        
        if test_cleanup(task_id):
            tests_passed += 1
    
    # Test audio-only generation
    audio_only_task_id = test_audio_only_generation(audio_path)
    if audio_only_task_id:
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Service is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the service logs for details.")
    
    # Cleanup test files
    if os.path.exists(TEST_FILES_DIR):
        import shutil
        shutil.rmtree(TEST_FILES_DIR)
        print(f"🧹 Cleaned up test files in {TEST_FILES_DIR}/")

if __name__ == "__main__":
    main()

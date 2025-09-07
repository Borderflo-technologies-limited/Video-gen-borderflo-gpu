#!/usr/bin/env python3
"""
Test script to check audio duration detection methods
"""

import os
import sys

def test_audio_duration(audio_path):
    """Test different methods to get audio duration"""
    print(f"Testing audio file: {audio_path}")
    print(f"File size: {os.path.getsize(audio_path)} bytes")
    print("-" * 50)
    
    # Method 1: Librosa
    try:
        import librosa
        print("Testing librosa...")
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        duration = len(audio_data) / sample_rate
        print(f"✅ Librosa duration: {duration:.2f} seconds")
        print(f"   Sample rate: {sample_rate}")
        print(f"   Audio data shape: {audio_data.shape}")
        return duration
    except Exception as e:
        print(f"❌ Librosa failed: {e}")
    
    # Method 2: Wave module
    try:
        import wave
        print("\nTesting wave module...")
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)
            print(f"✅ Wave duration: {duration:.2f} seconds")
            print(f"   Sample rate: {sample_rate}")
            print(f"   Frames: {frames}")
            return duration
    except Exception as e:
        print(f"❌ Wave module failed: {e}")
    
    # Method 3: Pydub
    try:
        from pydub import AudioSegment
        print("\nTesting pydub...")
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
        print(f"✅ Pydub duration: {duration:.2f} seconds")
        print(f"   Sample rate: {audio.frame_rate}")
        print(f"   Channels: {audio.channels}")
        return duration
    except Exception as e:
        print(f"❌ Pydub failed: {e}")
    
    # Method 4: File size estimation
    try:
        print("\nTesting file size estimation...")
        file_size = os.path.getsize(audio_path)
        estimated_duration = file_size / 16000  # 16KB per second
        duration = max(1.0, min(estimated_duration, 300))
        print(f"✅ Estimated duration: {duration:.2f} seconds")
        print(f"   File size: {file_size} bytes")
        print(f"   Estimation: {file_size / 16000:.2f} seconds")
        return duration
    except Exception as e:
        print(f"❌ File size estimation failed: {e}")
    
    print("\n❌ All methods failed!")
    return 5.0

if __name__ == "__main__":
    audio_path = "models/base-audio.mp3"
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    duration = test_audio_duration(audio_path)
    print(f"\nFinal duration: {duration:.2f} seconds")

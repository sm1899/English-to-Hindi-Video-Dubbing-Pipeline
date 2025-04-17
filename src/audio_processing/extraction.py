"""Audio extraction module using FFmpeg."""

import os
import subprocess
import time
from pathlib import Path

from src.utils.logger import logger

def extract_audio(
    video_path, 
    output_path=None, 
    sample_rate=16000, 
    channels=1, 
    format="wav"
):
    """Extract audio from a video file using FFmpeg.
    
    Args:
        video_path: Path to the input video file
        output_path: Path for the output audio file. If None, a path is generated.
        sample_rate: Audio sample rate (default: 16000 Hz)
        channels: Number of audio channels (default: 1 for mono)
        format: Output audio format (default: wav)
        
    Returns:
        str: Path to the extracted audio file
    """
    video_path = Path(video_path)
    
    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("processed_data/audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}.{format}"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting audio from {video_path} to {output_path}")
    start_time = time.time()
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:a", "0",                # Best quality
        "-ac", str(channels),       # Set number of channels
        "-ar", str(sample_rate),    # Set sample rate
        "-vn",                      # No video
        str(output_path),
        "-y"                        # Overwrite if exists
    ]
    
    try:
        # Run ffmpeg command
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        _, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error extracting audio: {stderr.decode()}")
            raise Exception(f"FFmpeg error: {stderr.decode()}")
        
        duration = time.time() - start_time
        logger.info(f"Audio extraction completed in {duration:.2f} seconds")
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error during audio extraction: {str(e)}")
        raise 
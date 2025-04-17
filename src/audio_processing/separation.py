#!/usr/bin/env python3
"""
Audio separation module using Demucs.

This module provides functions to separate speech from background noise/music
using Facebook's Demucs model.
"""

import os
import subprocess
import time
import logging
import torch
from pathlib import Path
import tempfile
import shutil

try:
    from src.utils.logger import logger
except ImportError:
    # Configure logging if not imported
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("audio_separation")

def separate_audio(
    audio_path,
    output_dir=None,
    model_name="htdemucs",
    use_gpu=True,
    output_format="wav"
):
    """
    Separate speech from background noise/music using Demucs.
    
    Args:
        audio_path: Path to the input audio file
        output_dir: Directory to save separated stems. If None, a temp dir is used.
        model_name: Demucs model to use ('htdemucs' recommended for voice)
        use_gpu: Whether to use GPU for separation
        output_format: Output audio format
        
    Returns:
        dict: Dictionary with paths to separated audio stems (vocals, other, etc.)
    """
    audio_path = Path(audio_path)
    
    # Create temporary directory if no output directory specified
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Separating audio using Demucs: {audio_path}")
    start_time = time.time()
    
    # Check if demucs is installed
    try:
        import demucs
    except ImportError:
        logger.error("Demucs not installed. Install with: pip install demucs")
        raise
    
    # Prepare command
    cmd = ["demucs", "--out", str(output_dir)]
    
    # Add model name
    cmd.extend(["-n", model_name])
    
    # Add GPU/CPU selection
    if use_gpu and torch.cuda.is_available():
        cmd.extend(["-d", "cuda"])
    else:
        cmd.extend(["-d", "cpu"])
        if use_gpu:
            logger.warning("GPU requested but not available, using CPU instead")
    
    # Add the audio file (must be at the end)
    cmd.append(str(audio_path))
    
    try:
        # Run demucs command
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error separating audio: {stderr}")
            raise Exception(f"Demucs error: {stderr}")
        
        # Get paths to separated stems
        model_output_dir = output_dir / model_name / audio_path.stem
        
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            stem_path = model_output_dir / f"{stem}.{output_format}"
            if stem_path.exists():
                stems[stem] = str(stem_path)
        
        duration = time.time() - start_time
        logger.info(f"Audio separation completed in {duration:.2f} seconds")
        
        return {
            "speech": stems.get("vocals"),
            "background": stems.get("other"),
            "all_stems": stems,
            "temp_dir": temp_dir
        }
    
    except Exception as e:
        logger.error(f"Error during audio separation: {str(e)}")
        # Clean up temporary directory if created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise

def separate_audio_from_video(
    video_path,
    output_dir=None,
    model_name="htdemucs",
    use_gpu=True,
    output_format="wav"
):
    """
    Extract and separate audio from a video file using Demucs.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save separated stems
        model_name: Demucs model to use
        use_gpu: Whether to use GPU for separation
        output_format: Output audio format
        
    Returns:
        dict: Dictionary with paths to separated audio stems
    """
    video_path = Path(video_path)
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract audio from video
        temp_audio = Path(temp_dir) / f"{video_path.stem}.wav"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",                     # No video
            "-acodec", "pcm_s16le",    # PCM 16-bit encoding
            "-ar", "44100",            # 44.1kHz sample rate
            "-ac", "2",                # Stereo
            str(temp_audio)
        ]
        
        try:
            # Run ffmpeg command
            logger.info(f"Extracting audio from video: {video_path}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Separate the audio
            return separate_audio(
                temp_audio,
                output_dir,
                model_name,
                use_gpu,
                output_format
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio from video: {e.stderr.decode() if e.stderr else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error during audio extraction and separation: {str(e)}")
            raise

def cleanup_separation_temp(separation_result):
    """
    Clean up temporary directory created during separation.
    
    Args:
        separation_result: Result dictionary from separate_audio function
    """
    temp_dir = separation_result.get("temp_dir")
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Separate speech from background in audio file")
    parser.add_argument("input_path", help="Path to input audio or video file")
    parser.add_argument("--output-dir", help="Directory to save separated audio")
    parser.add_argument("--model", default="htdemucs", help="Demucs model to use")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        result = separate_audio_from_video(
            input_path,
            args.output_dir,
            args.model,
            not args.no_gpu
        )
    else:
        result = separate_audio(
            input_path,
            args.output_dir,
            args.model,
            not args.no_gpu
        )
    
    print(f"Separated audio files:")
    for stem_name, stem_path in result.get("all_stems", {}).items():
        print(f"  - {stem_name}: {stem_path}") 
#!/usr/bin/env python3
"""
Simple script to combine TTS segments into a single audio file.
This is a direct approach that doesn't rely on the pipeline's combine function.
"""

import os
import sys
import glob
import argparse
import subprocess
import tempfile
from pathlib import Path

def combine_segments(input_dir, output_path, background_path=None, background_volume=0.05):
    """
    Combine all segment*.wav files from input_dir into a single audio file.
    
    Args:
        input_dir: Directory containing segment WAV files
        output_path: Path to save combined output
        background_path: Optional path to background audio
        background_volume: Volume level for background (0.0-1.0)
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Find all segment files
    segment_pattern = os.path.join(input_dir, "segment_*.wav")
    segment_files = sorted(glob.glob(segment_pattern))
    
    if not segment_files:
        print(f"No segment files found in {input_dir}")
        return None
    
    print(f"Found {len(segment_files)} segment files")
    for i, file in enumerate(segment_files[:5]):
        print(f"  {i}: {file}")
    
    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create list file for ffmpeg concat
        concat_list_path = os.path.join(temp_dir, "segments.txt")
        with open(concat_list_path, 'w') as f:
            for segment_path in segment_files:
                f.write(f"file '{os.path.abspath(segment_path)}'\n")
        
        # Path for speech-only output
        speech_output = os.path.join(temp_dir, "speech_combined.wav")
        
        # Concatenate all segments
        try:
            print("Concatenating segments...")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
                speech_output
            ], check=True)
            
            print(f"Segments combined to {speech_output}")
            
            # Get speech duration
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", speech_output
            ], capture_output=True, text=True, check=True)
            
            speech_duration = float(result.stdout.strip())
            print(f"Combined speech duration: {speech_duration:.2f}s")
            
            # If background audio is provided, mix with it
            if background_path and os.path.exists(background_path):
                print(f"Mixing with background audio: {background_path}")
                
                # Adjust background length
                bg_adjusted_path = os.path.join(temp_dir, "bg_adjusted.wav")
                subprocess.run([
                    "ffmpeg", "-y", "-i", background_path,
                    "-t", str(speech_duration),
                    "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    bg_adjusted_path
                ], check=True)
                
                # Mix speech with background
                filter_complex = f"[0:a]volume=1.5[speech];[1:a]volume={background_volume}[bg];[speech][bg]amix=inputs=2:duration=longest:weights=5 1"
                
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", speech_output,
                    "-i", bg_adjusted_path,
                    "-filter_complex", filter_complex,
                    "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    output_path
                ], check=True)
                
                print(f"Final audio (with background) saved to: {output_path}")
            else:
                # Just copy speech output
                print("No background audio provided, using speech only")
                subprocess.run([
                    "ffmpeg", "-y", "-i", speech_output,
                    "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    output_path
                ], check=True)
                
                print(f"Final audio saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error combining segments: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Combine TTS segments into a single audio file")
    parser.add_argument("--input_dir", required=True, help="Directory containing segment WAV files")
    parser.add_argument("--output_path", required=True, help="Path to save combined audio")
    parser.add_argument("--background", help="Optional background audio file")
    parser.add_argument("--bg_volume", type=float, default=0.05, help="Background volume (0.0-1.0)")
    
    args = parser.parse_args()
    
    combine_segments(
        input_dir=args.input_dir,
        output_path=args.output_path,
        background_path=args.background,
        background_volume=args.bg_volume
    )

if __name__ == "__main__":
    main() 
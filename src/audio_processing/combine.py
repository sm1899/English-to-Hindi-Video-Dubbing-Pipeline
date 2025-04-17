"""Module for combining audio segments with background audio."""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Union

from src.utils.logger import logger

def combine_audio_segments(
    tts_segments: List[Dict],
    background_audio_path: str,
    output_path: str,
    tts_output_dir: Optional[str] = None,
    original_duration: Optional[float] = None,
    adjust_volume: bool = True,
    background_volume: float = 0.2
) -> str:
    """
    Combine multiple TTS audio segments with background audio using FFmpeg.
    
    Args:
        tts_segments: List of dictionaries with segment information including start, end, and tts_output path
        background_audio_path: Path to background audio file (or None for speech-only output)
        output_path: Path to save the combined audio
        tts_output_dir: Optional directory where TTS segments are stored (if paths in tts_segments are relative)
        original_duration: Optional original video/audio duration in seconds (default: computed from segments)
        adjust_volume: Whether to adjust the background volume
        background_volume: Background volume level (0.0-1.0)
        
    Returns:
        str: Path to the combined audio file
    """
    logger.info(f"Combining {len(tts_segments)} audio segments" + 
                (f" with background audio" if background_audio_path else " (speech-only)"))
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Filter out segments that don't have a valid audio file
        valid_segments = []
        for i, segment in enumerate(tts_segments):
            # Get TTS output path
            if "tts_output" in segment:
                tts_path = segment["tts_output"]
            else:
                # If no explicit path, construct from segment index
                if tts_output_dir:
                    tts_path = os.path.join(tts_output_dir, f"segment_{i:04d}.wav")
                else:
                    logger.error(f"No TTS output path for segment {i} and no tts_output_dir provided")
                    continue
            
            # If path is relative and tts_output_dir is provided, make it absolute
            if tts_output_dir and not os.path.isabs(tts_path) and not tts_path.startswith(tts_output_dir):
                # Check if we're already within the tts_output_dir
                if os.path.commonpath([tts_output_dir, tts_path]) == tts_output_dir:
                    # Path is already relative to tts_output_dir, just normalize it
                    tts_path = os.path.normpath(tts_path)
                else:
                    # Path is not within tts_output_dir, so add it
                    tts_path = os.path.join(tts_output_dir, tts_path)
            
            # Remove any potential path duplication
            tts_path = os.path.normpath(tts_path)
            
            # Debug: show actual paths being checked
            logger.info(f"Looking for segment {i} file at: {tts_path}")
            
            # Skip if TTS file doesn't exist
            if not os.path.exists(tts_path):
                # Try a more direct path as fallback
                alternate_path = os.path.join(tts_output_dir, f"segment_{i:04d}.wav")
                alternate_path = os.path.normpath(alternate_path)
                
                logger.warning(f"TTS file not found at {tts_path}, trying alternate path: {alternate_path}")
                
                if os.path.exists(alternate_path):
                    tts_path = alternate_path
                    logger.info(f"Found file at alternate path: {tts_path}")
                else:
                    logger.warning(f"TTS file not found: {tts_path}, skipping segment {i}")
                    continue
            
            # Verify it's a valid audio file
            try:
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", tts_path
                ], capture_output=True, text=True, check=True)
                
                segment_duration = float(result.stdout.strip())
                if segment_duration < 0.1:
                    logger.warning(f"Segment {i} is too short ({segment_duration}s), may be corrupted, skipping")
                    continue
                    
                logger.info(f"Valid TTS segment found: {tts_path} with duration {segment_duration:.2f}s")
            except Exception as e:
                logger.warning(f"Error checking TTS file {tts_path}: {e}, skipping segment {i}")
                continue
            
            # Add the file path to the segment data
            segment_copy = segment.copy()
            segment_copy["file_path"] = tts_path
            
            # Add segment timing if not present
            if "start" not in segment_copy or "end" not in segment_copy:
                # Try to get from the translation segment if possible
                if "translated_segments" in segment_copy:
                    trans_segment = segment_copy["translated_segments"][i] if i < len(segment_copy["translated_segments"]) else None
                    if trans_segment and "start" in trans_segment and "end" in trans_segment:
                        segment_copy["start"] = trans_segment["start"]
                        segment_copy["end"] = trans_segment["end"]
            
            valid_segments.append(segment_copy)
        
        logger.info(f"Found {len(valid_segments)} valid audio segments to combine")
        
        # If no valid segments, return early
        if not valid_segments:
            logger.error("No valid audio segments found to combine")
            return None
        
        # Sort segments by start time
        sorted_segments = sorted(valid_segments, key=lambda x: x.get("start", 0))
        
        # Determine total duration
        if original_duration is None:
            # Calculate from segments
            if sorted_segments:
                original_duration = max(seg.get("end", 0) for seg in sorted_segments)
            else:
                # Try to get duration from background audio
                if background_audio_path and os.path.exists(background_audio_path):
                    try:
                        result = subprocess.run([
                            "ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", background_audio_path
                        ], capture_output=True, text=True, check=True)
                        original_duration = float(result.stdout.strip())
                    except Exception as e:
                        logger.error(f"Error getting background audio duration: {e}")
                        original_duration = 0
                else:
                    # No background audio, use the last segment end time
                    original_duration = 0
            
            # Add a small buffer to the end
            original_duration += 1.0
        
        logger.info(f"Total timeline duration: {original_duration:.2f} seconds")
        
        # Create a silent audio file of the total duration
        silent_path = temp_dir_path / "silent.wav"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono",
            "-t", str(original_duration),
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            silent_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Build the complex filter for FFmpeg
        filter_complex = []
        inputs = []
        
        # Add each segment at its correct timestamp
        for i, segment in enumerate(sorted_segments):
            start_time = segment.get("start", 0)
            # Add -i and file path as separate elements
            inputs.extend(["-i", segment['file_path']])
            filter_complex.append(f"[{i+1}:a]adelay={int(start_time*1000)}|{int(start_time*1000)}[s{i}]")
        
        # Mix all segments together
        mix_inputs = "".join(f"[s{i}]" for i in range(len(sorted_segments)))
        filter_complex.append(f"{mix_inputs}amix=inputs={len(sorted_segments)}:duration=longest[tts]")
        
        # Add background audio if needed
        if background_audio_path and os.path.exists(background_audio_path) and background_volume > 0.05:
            # Add -i and background path as separate elements
            inputs.extend(["-i", background_audio_path])
            filter_complex.append(f"[{len(sorted_segments)+1}:a]volume={background_volume}[bg]")
            filter_complex.append("[tts][bg]amix=inputs=2:duration=longest:weights=5 1[final]")
        else:
            filter_complex.append("[tts]volume=1.5[final]")
        
        # Build the FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", str(silent_path),
            *inputs,
            "-filter_complex", ";".join(filter_complex),
            "-map", "[final]",
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
            
            logger.info(f"Successfully combined audio segments with timing")
            
            # Verify the final output
            if os.path.exists(output_path):
                try:
                    result = subprocess.run([
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", output_path
                    ], capture_output=True, text=True, check=True)
                    final_duration = float(result.stdout.strip())
                    final_size = os.path.getsize(output_path)
                    logger.info(f"Final output created successfully: {output_path} (duration: {final_duration:.2f}s, size: {final_size} bytes)")
                    return output_path
                except Exception as e:
                    logger.error(f"Error verifying final output: {e}")
                    return None
            else:
                logger.error(f"Final output file not created: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error during audio mixing: {e}")
            return None

def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return 0.0 
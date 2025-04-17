#!/usr/bin/env python3
"""
Test script for ASR-based reference sampling.

This script demonstrates the enhanced reference sampling using ASR output.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.reference_sampling.sample_selector import extract_reference_samples_from_asr
from src.utils.logger import setup_logger

logger = setup_logger("test_asr_reference_sampling")

def main():
    parser = argparse.ArgumentParser(
        description="Test ASR-based reference sampling for voice cloning"
    )
    
    parser.add_argument(
        "--video-id", 
        type=str, 
        required=True,
        help="Video ID or full path to video/audio file"
    )
    
    parser.add_argument(
        "--audio-path", 
        type=str, 
        default=None,
        help="Path to the input audio file (default: processed_data/audio/{video_id}.wav)"
    )
    
    parser.add_argument(
        "--asr-path", 
        type=str, 
        default=None,
        help="Path to the ASR JSON output file (default: processed_data/transcriptions/{video_id}_transcription.json)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save output reference samples (default: test_output/reference_samples/{video_id})"
    )
    
    parser.add_argument(
        "--diarization-path", 
        type=str, 
        default=None,
        help="Path to diarization JSON file (default: processed_data/diarization/{video_id}.json if exists)"
    )
    
    args = parser.parse_args()
    
    # Process video_id (remove extension if present)
    video_id = args.video_id
    if video_id.endswith('.mp4') or video_id.endswith('.wav'):
        video_id = Path(video_id).stem
    
    # Generate paths following project conventions if not provided
    if args.audio_path is None:
        audio_path = Path(f"processed_data/audio/{video_id}.wav")
    else:
        audio_path = Path(args.audio_path)
    
    if args.asr_path is None:
        asr_path = Path(f"processed_data/transcriptions/{video_id}_transcription.json")
    else:
        asr_path = Path(args.asr_path)
    
    if args.output_dir is None:
        output_dir = Path(f"test_output/reference_samples/{video_id}")
    else:
        output_dir = Path(args.output_dir)
    
    if args.diarization_path is None:
        diarization_path = Path(f"processed_data/diarization/{video_id}.json")
        if not diarization_path.exists():
            diarization_path = None
    else:
        diarization_path = Path(args.diarization_path)
    
    # Validate input paths
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return 1
    
    if not asr_path.exists():
        logger.error(f"ASR file not found: {asr_path}")
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Testing ASR-based reference sampling for: {video_id}")
    logger.info(f"  - Audio: {audio_path}")
    logger.info(f"  - ASR JSON: {asr_path}")
    logger.info(f"  - Output directory: {output_dir}")
    
    try:
        # Extract reference samples using ASR output
        reference_samples = extract_reference_samples_from_asr(
            str(audio_path),
            str(asr_path),
            str(output_dir),
            str(diarization_path) if diarization_path and diarization_path.exists() else None
        )
        
        # Display results
        logger.info(f"Successfully extracted reference samples for {len(reference_samples)} speaker(s)")
        
        for speaker_id, samples in reference_samples.items():
            logger.info(f"Speaker {speaker_id}: {len(samples)} samples")
            
            # Group by emotion
            emotion_samples = {}
            for sample in samples:
                emotion = sample.get('emotion_label', 'unknown')
                if emotion not in emotion_samples:
                    emotion_samples[emotion] = []
                emotion_samples[emotion].append(sample)
            
            # Display emotion groups
            logger.info(f"  Emotion distribution:")
            for emotion, emotion_group in emotion_samples.items():
                logger.info(f"    - {emotion}: {len(emotion_group)} samples")
                
                # Show sample details for this emotion
                for i, sample in enumerate(emotion_group):
                    logger.info(f"      Sample {i+1}:")
                    logger.info(f"        Path: {os.path.basename(sample['path'])}")
                    logger.info(f"        Duration: {sample['duration']:.2f} seconds")
                    if 'text' in sample and sample['text']:
                        # Truncate text if too long
                        text = sample['text']
                        if len(text) > 60:
                            text = text[:57] + "..."
                        logger.info(f"        Text: \"{text}\"")
        
        logger.info("ASR-based reference sampling test completed successfully!")
        logger.info(f"Reference samples saved to: {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during ASR-based reference sampling test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python3
"""
Extract Reference Samples Tool

Command-line tool to extract reference audio samples for voice cloning,
analyzing audio characteristics and selecting diverse samples.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.reference_sampling.sample_selector import extract_reference_samples, extract_reference_samples_from_asr
from src.utils.logger import setup_logger

logger = setup_logger("reference_sampling")

def main():
    parser = argparse.ArgumentParser(
        description="Extract reference audio samples for voice cloning with emotion detection"
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
        "--output-dir", 
        type=str, 
        default=None,
        help="Directory to save reference samples (default: processed_data/reference_samples/{video_id})"
    )
    
    parser.add_argument(
        "--asr-path", 
        type=str, 
        default=None,
        help="Path to ASR JSON file (default: processed_data/transcriptions/{video_id}_transcription.json)"
    )
    
    parser.add_argument(
        "--diarization-path", 
        type=str, 
        default=None,
        help="Path to diarization JSON file (default: processed_data/diarization/{video_id}.json if exists)"
    )
    
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=5,
        help="Number of reference samples to extract per speaker (default: 5)"
    )
    
    parser.add_argument(
        "--min-duration", 
        type=float, 
        default=5.0,
        help="Minimum duration of each sample in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--max-duration", 
        type=float, 
        default=15.0,
        help="Maximum duration of each sample in seconds (default: 15.0)"
    )
    
    parser.add_argument(
        "--display-emotions",
        action="store_true",
        help="Display detected emotions for each sample"
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
    
    if args.output_dir is None:
        output_dir = Path(f"processed_data/reference_samples/{video_id}")
    else:
        output_dir = Path(args.output_dir)
    
    if args.asr_path is None:
        asr_path = Path(f"processed_data/transcriptions/{video_id}_transcription.json")
    else:
        asr_path = Path(args.asr_path)
    
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
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing video ID: {video_id}")
    logger.info(f"Audio path: {audio_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"ASR path: {asr_path}")
    logger.info(f"Diarization path: {diarization_path}")
    
    # Extract reference samples
    try:
        if asr_path.exists():
            logger.info(f"Using ASR data for better segmentation")
            reference_samples = extract_reference_samples_from_asr(
                str(audio_path), 
                str(asr_path),
                str(output_dir),
                str(diarization_path) if diarization_path and diarization_path.exists() else None
            )
        else:
            logger.info("Using basic segmentation (no ASR data)")
            reference_samples = extract_reference_samples(
                str(audio_path), 
                str(output_dir),
                str(diarization_path) if diarization_path and diarization_path.exists() else None
            )
        
        # Print summary
        total_samples = sum(len(samples) for samples in reference_samples.values())
        speaker_count = len(reference_samples)
        
        logger.info(f"Successfully extracted {total_samples} reference samples for {speaker_count} speaker(s)")
        
        # Display emotions if requested
        if args.display_emotions:
            for speaker_id, samples in reference_samples.items():
                logger.info(f"\nSpeaker {speaker_id}:")
                emotion_counts = {}
                
                for i, sample in enumerate(samples):
                    emotion = sample.get('emotion_label', 'unknown')
                    if emotion not in emotion_counts:
                        emotion_counts[emotion] = 0
                    emotion_counts[emotion] += 1
                    
                    logger.info(f"  Sample {i+1}: {os.path.basename(sample['path'])}")
                    logger.info(f"    - Duration: {sample['duration']:.2f}s")
                    logger.info(f"    - Emotion: {emotion}")
                    if 'text' in sample and sample['text']:
                        logger.info(f"    - Text: \"{sample['text']}\"")
                
                # Display emotion summary
                logger.info(f"  Emotion distribution for Speaker {speaker_id}:")
                for emotion, count in emotion_counts.items():
                    logger.info(f"    - {emotion}: {count} sample(s)")
        
        logger.info(f"\nReference samples saved to: {output_dir}")
        logger.info(f"Metadata saved to: {output_dir / 'reference_samples_metadata.json'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error extracting reference samples: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python3
"""
Test script for zero-shot TTS with XTTS.

This script tests the TTS functionality with a single segment or full video.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tts.zero_shot import generate_speech, process_tts
from src.tts.utils import match_reference_to_emotion, load_reference_samples
from src.utils.logger import setup_logger

logger = setup_logger("test_tts")

def test_single_segment(args):
    """
    Test TTS with a single text segment.
    """
    # Make output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference samples
    try:
        if args.reference_samples_path:
            with open(args.reference_samples_path, 'r') as f:
                reference_samples = json.load(f)
        else:
            reference_samples = load_reference_samples(args.video_id)
            
        logger.info(f"Loaded reference samples with {len(reference_samples)} speaker(s)")
    except Exception as e:
        logger.error(f"Error loading reference samples: {e}")
        return 1
    
    # Select a reference sample based on text emotion
    try:
        reference_path = match_reference_to_emotion(
            args.text, 
            reference_samples, 
            args.speaker_id
        )
        logger.info(f"Selected reference sample: {reference_path}")
    except Exception as e:
        logger.error(f"Error selecting reference sample: {e}")
        return 1
    
    # Generate output path
    output_path = output_dir / "test_output.wav"
    
    # Generate speech
    try:
        generated_path = generate_speech(
            text=args.text,
            reference_audio_path=reference_path,
            output_path=str(output_path),
            language=args.language,
            gpu=not args.no_gpu
        )
        
        logger.info(f"Generated speech saved to: {generated_path}")
        return 0
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return 1

def test_full_video(args):
    """
    Test TTS with a full video.
    """
    try:
        result = process_tts(
            video_id=args.video_id,
            translations_path=args.translations_path,
            output_dir=args.output_dir,
            reference_samples_path=args.reference_samples_path,
            language=args.language,
            gpu=not args.no_gpu
        )
        
        logger.info(f"Processed {len(result['segments'])} segments")
        return 0
    except Exception as e:
        logger.error(f"Error processing TTS: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Test zero-shot TTS with XTTS"
    )
    
    parser.add_argument(
        "--video-id", 
        required=True,
        help="Video ID or path"
    )
    
    parser.add_argument(
        "--mode",
        choices=["segment", "full"],
        default="segment",
        help="Test mode: segment (single text) or full (entire video)"
    )
    
    parser.add_argument(
        "--text",
        help="Text to synthesize (required for segment mode)"
    )
    
    parser.add_argument(
        "--speaker-id",
        default="speaker_0",
        help="Speaker ID to use (default: speaker_0)"
    )
    
    parser.add_argument(
        "--translations-path",
        help="Path to translations JSON file (for full mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./test_output/tts",
        help="Directory to save generated audio"
    )
    
    parser.add_argument(
        "--reference-samples-path",
        help="Path to reference samples metadata"
    )
    
    parser.add_argument(
        "--language",
        default="hi",
        help="Target language code (default: hi)"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "segment" and not args.text:
        logger.error("Text argument is required for segment mode")
        return 1
    
    # Process based on mode
    if args.mode == "segment":
        return test_single_segment(args)
    else:
        return test_full_video(args)

if __name__ == "__main__":
    sys.exit(main()) 
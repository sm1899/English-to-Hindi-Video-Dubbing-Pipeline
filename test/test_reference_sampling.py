#!/usr/bin/env python3
"""
Test script for reference sampling module.

This script tests the reference sampling functionality on a sample audio file.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.reference_sampling.sample_selector import extract_reference_samples
from src.utils.logger import setup_logger

logger = setup_logger("test_reference_sampling")

def main():
    parser = argparse.ArgumentParser(
        description="Test reference sampling functionality"
    )
    
    parser.add_argument(
        "--audio-path", 
        type=str, 
        required=True,
        help="Path to the input audio file for testing"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./test_output/reference_samples",
        help="Directory to save output reference samples"
    )
    
    parser.add_argument(
        "--diarization-path", 
        type=str, 
        default=None,
        help="Path to diarization JSON file (optional)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Testing reference sampling with audio file: {args.audio_path}")
    
    try:
        # Extract reference samples
        reference_samples = extract_reference_samples(
            args.audio_path,
            args.output_dir,
            args.diarization_path
        )
        
        # Display results
        logger.info(f"Successfully extracted reference samples for {len(reference_samples)} speaker(s)")
        
        for speaker_id, samples in reference_samples.items():
            logger.info(f"Speaker {speaker_id}: {len(samples)} samples")
            
            for i, sample in enumerate(samples):
                logger.info(f"  Sample {i+1}:")
                logger.info(f"    Path: {sample['path']}")
                logger.info(f"    Duration: {sample['duration']:.2f} seconds")
                logger.info(f"    Clarity score: {sample['features']['clarity']:.4f}")
                logger.info(f"    Speech rate: {sample['features']['speech_rate']:.2f}")
                logger.info(f"    Pitch variation: {sample['features']['pitch_variation']:.4f}")
        
        logger.info("Reference sampling test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during reference sampling test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
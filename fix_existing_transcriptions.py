#!/usr/bin/env python3
"""Script to fix segmentation in existing transcription files."""

import os
import argparse
from pathlib import Path

from src.utils.post_process import fix_transcription_segmentation
from src.utils.logger import logger

def main():
    """Fix segmentation in transcription files."""
    parser = argparse.ArgumentParser(description='Fix segmentation in existing transcription files')
    
    # Add arguments
    parser.add_argument('--input_dir', '-i', default='processed_data/transcriptions',
                       help='Directory containing transcription files (default: processed_data/transcriptions)')
    parser.add_argument('--output_dir', '-o', default=None,
                       help='Directory for fixed transcription files (default: same as input)')
    parser.add_argument('--pattern', '-p', default='*.json',
                       help='File pattern to match (default: *.json)')
    
    # Parse arguments
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all transcription files
    input_files = list(input_dir.glob(args.pattern))
    
    if not input_files:
        logger.warning(f"No files matching '{args.pattern}' found in {input_dir}")
        return
    
    logger.info(f"Found {len(input_files)} transcription files to process")
    
    # Process each file
    for input_file in input_files:
        output_file = output_dir / f"{input_file.stem}_fixed.json"
        logger.info(f"Processing: {input_file}")
        
        try:
            fix_transcription_segmentation(input_file, output_file)
            logger.info(f"✅ Successfully fixed: {output_file}")
        except Exception as e:
            logger.error(f"❌ Failed to fix {input_file}: {str(e)}")
    
    logger.info("Completed fixing transcription segmentation")

if __name__ == "__main__":
    main() 
"""Utility functions for post-processing transcriptions and translations."""

import json
import argparse
from pathlib import Path
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.asr.whisper_asr import fix_sentence_segmentation
from src.utils.logger import logger

def fix_transcription_segmentation(input_path, output_path=None):
    """Fix segmentation in existing transcription files.
    
    Args:
        input_path: Path to the input transcription JSON file
        output_path: Path for the output JSON file. If None, will append "_fixed" to input name
        
    Returns:
        str: Path to the fixed transcription JSON file
    """
    input_path = Path(input_path)
    
    # Generate output path if not provided
    if output_path is None:
        output_dir = input_path.parent
        output_path = output_dir / f"{input_path.stem}_fixed.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Fixing segmentation in transcription: {input_path}")
    
    try:
        # Load the original transcription
        with open(input_path, 'r', encoding='utf-8') as f:
            transcription = json.load(f)
        
        # Fix segmentation
        fixed_transcription = fix_sentence_segmentation(transcription)
        
        # Save the fixed transcription
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fixed_transcription, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Fixed transcription saved to: {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error fixing transcription: {str(e)}")
        raise

def main():
    """Command-line interface for post-processing utilities."""
    parser = argparse.ArgumentParser(description='Post-process transcriptions and translations')
    
    # Add arguments
    parser.add_argument('input_file', help='Path to the input transcription JSON file')
    parser.add_argument('--output', '-o', help='Path for the output JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Fix segmentation
    fix_transcription_segmentation(args.input_file, args.output)

if __name__ == "__main__":
    main() 
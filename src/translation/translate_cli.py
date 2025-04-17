#!/usr/bin/env python
"""Command-line interface for translating transcriptions from English to Hindi."""

import argparse
import sys
from pathlib import Path

from src.translation.nllb_translator import translate_transcription
from src.utils.logger import logger

def main():
    """Run the translation CLI."""
    parser = argparse.ArgumentParser(
        description="Translate Whisper transcription from English to Hindi using NLLB-200"
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input transcription JSON file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to save the translated JSON file (optional)"
    )
    
    parser.add_argument(
        "-m", "--model-size",
        type=str,
        choices=["600M", "3.3B"],
        default="3.3B",
        help="NLLB model size (default: 3.3B)"
    )
    
    args = parser.parse_args()
    
    try:
        # Convert to Path object
        input_path = Path(args.input)
        
        # Check if input file exists
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        # Translate transcription
        output_path = translate_transcription(
            transcription_path=input_path,
            output_path=args.output,
            model_size=args.model_size
        )
        
        logger.info(f"Translation saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""Simple test for the NLLB translator."""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.translation.nllb_translator import NLLBTranslator
from src.utils.logger import logger

def main():
    """Run a simple test of the NLLB translator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test NLLB translator")
    parser.add_argument("--model-size", choices=["600M", "3.3B"], default="3.3B",
                       help="Model size to use (default: 3.3B)")
    args = parser.parse_args()
    
    # Test texts
    test_texts = [
        "Hello, how are you? This is a test of the NLLB translator.",
        "Good morning! I hope you are doing well today.",
        "The quick brown fox jumps over the lazy dog.",
        "United Nations Chief says there is no military solution in Syria.",
        "Studies have shown that owning a dog is good for your health."
    ]
    
    logger.info(f"Testing NLLB translator with {args.model_size} model")
    
    # Initialize translator with specified model
    translator = NLLBTranslator(model_size=args.model_size)
    
    # Test all examples
    for i, test_text in enumerate(test_texts):
        logger.info(f"\nTest {i+1}: {test_text}")
        hindi_text = translator.translate(test_text)
        logger.info(f"Hindi: {hindi_text}")
    
    logger.info("\nTranslation test complete")
    
if __name__ == "__main__":
    main() 
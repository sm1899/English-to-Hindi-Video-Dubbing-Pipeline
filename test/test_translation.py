"""Test the translation module."""

import json
import sys
from pathlib import Path
import unittest

# Add the project root to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from src.translation.nllb_translator import NLLBTranslator

class TestTranslation(unittest.TestCase):
    """Test the translation module."""
    
    def test_simple_translation(self):
        """Test a simple translation from English to Hindi."""
        # Initialize translator with smaller model for testing
        translator = NLLBTranslator(model_size="600M")
        
        # Simple English text
        english_text = "Hello, how are you?"
        
        # Translate to Hindi
        hindi_text = translator.translate(english_text)
        
        # Verify translation is not empty
        self.assertTrue(len(hindi_text) > 0)
        
        # Print the translation (for manual verification)
        print(f"\nEnglish: {english_text}")
        print(f"Hindi: {hindi_text}")
        
        # Hindi typically has more characters than English due to script differences
        self.assertTrue(len(hindi_text) >= 5)  # Simple length sanity check

if __name__ == "__main__":
    unittest.main() 
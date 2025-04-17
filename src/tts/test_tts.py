import os
import time
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tts.zero_shot import generate_speech
from src.utils.logger import logger

def test_tts():
    """
    Test TTS functionality with a few Hindi phrases.
    """
    # Create test output directory
    output_dir = Path("test_data/tts_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a test reference audio
    reference_audio = "test_data/test_reference.wav"
    
    # If reference audio doesn't exist, create it
    if not os.path.exists(reference_audio):
        logger.info(f"Creating test reference audio: {reference_audio}")
        os.makedirs(os.path.dirname(reference_audio), exist_ok=True)
        
        # Create a simple sine wave as reference audio
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", 
            "-i", "sine=frequency=1000:duration=3", 
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", 
            reference_audio
        ], check=True)
    
    # Test phrases
    test_phrases = [
        "नमस्ते, आप कैसे हैं?",  # Hello, how are you?
        "मैं ठीक हूँ, धन्यवाद।",  # I am fine, thank you.
        "आज मौसम बहुत अच्छा है।",  # The weather is very nice today.
        "क्या आप हिंदी बोलते हैं?",  # Do you speak Hindi?
        "भारत एक विविधतापूर्ण देश है।"  # India is a diverse country.
    ]
    
    logger.info(f"Testing TTS with {len(test_phrases)} Hindi phrases")
    
    start_time = time.time()
    
    for i, phrase in enumerate(test_phrases):
        logger.info(f"Processing phrase {i+1}/{len(test_phrases)}: {phrase}")
        
        # Generate output path
        output_path = output_dir / f"test_phrase_{i+1}.wav"
        
        try:
            # Generate speech
            generated_path = generate_speech(
                text=phrase,
                reference_audio_path=reference_audio,
                output_path=str(output_path),
                language="hi"
            )
            
            logger.info(f"Generated speech saved to: {generated_path}")
            
        except Exception as e:
            logger.error(f"Error generating speech for phrase {i+1}: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"TTS testing completed in {total_time:.2f} seconds")

if __name__ == "__main__":
    test_tts() 
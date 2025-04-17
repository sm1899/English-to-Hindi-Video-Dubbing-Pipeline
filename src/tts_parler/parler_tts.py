#!/usr/bin/env python3
"""
Local TTS Voice Cloning Module

This module handles TTS generation using a local TTS model for high-quality voice cloning.
It provides functions to generate speech with the voice characteristics 
of reference audio samples.
"""

import os
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from src.utils.logger import logger
except ImportError:
    # Configure logging if not imported
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("local_tts")

# Check if TTS is installed
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("TTS not installed. Some features will be unavailable.")

class LocalTTS:
    """Local TTS voice cloning model wrapper."""
    
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True, gpu_index: int = 0):
        """
        Initialize local TTS model.
        
        Args:
            model_path: Path to a local model or None to use the default XTTS v2
            use_gpu: Whether to use GPU for inference
            gpu_index: Index of the GPU to use if GPU is enabled (default: 0)
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index
        self.model = None
        
        if not TTS_AVAILABLE:
            raise ImportError(
                "TTS not installed. Install with: pip install TTS"
            )
            
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the underlying TTS model."""
        import torch
        
        # Set the GPU device if using GPU
        if self.use_gpu and torch.cuda.is_available():
            if self.gpu_index >= 0 and self.gpu_index < torch.cuda.device_count():
                logger.info(f"Using GPU {self.gpu_index} for TTS")
                torch.cuda.set_device(self.gpu_index)
            else:
                logger.warning(f"GPU index {self.gpu_index} not available. Using default GPU.")
        elif self.use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA is not available. Using CPU instead.")
            self.use_gpu = False
        
        # Model initialization code
        try:
            import torch
            from TTS.api import TTS
            
            logger.info("Initializing local TTS model")
            
            # Use the specified model path or default to XTTS v2
            model_name = self.model_path or "tts_models/multilingual/multi-dataset/xtts_v2"
            
            # Initialize TTS model with appropriate device
            self.model = TTS(model_name=model_name, progress_bar=True, gpu=self.use_gpu)
            
            device = "CUDA" if self.use_gpu else "CPU"
            logger.info(f"TTS model initialized successfully on {device}")
        except Exception as e:
            logger.error(f"Error initializing TTS model: {str(e)}")
            raise
            
    def generate_speech(
        self,
        text: str,
        reference_audio: Union[str, List[str]],
        output_path: str,
        language: str = "hi",
        speaker_id: Optional[str] = None,
        voice_settings: Optional[Dict] = None
    ) -> str:
        """
        Generate speech using zero-shot voice cloning.
        
        Args:
            text: Text to synthesize
            reference_audio: Path(s) to reference audio file(s)
            output_path: Path to save the generated audio
            language: Target language code (hi for Hindi)
            speaker_id: Optional speaker ID for voice selection
            voice_settings: Optional voice settings dictionary
                - speed: Speech rate (0.5-2.0, default 1.0)
                
        Returns:
            Path to the generated audio file
        """
        if not self.model:
            self._initialize_model()
            
        logger.info(f"Generating speech for text: {text[:50]}...")
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Set default voice settings
            settings = {
                "speed": 1.0
            }
            
            # Update with user settings if provided
            if voice_settings:
                settings.update(voice_settings)
                
            # Handle multiple reference samples vs single reference
            if isinstance(reference_audio, list):
                if not reference_audio:
                    raise ValueError("No reference audio samples provided")
                    
                logger.info(f"Using {len(reference_audio)} reference samples for voice cloning")
                
                # For now, we'll use the first reference sample (XTTS doesn't directly support multiple samples)
                # In a production system, you could create a custom embedding by averaging multiple samples
                primary_reference = reference_audio[0]
                
                # Generate speech with the selected speaker reference
                self.model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=primary_reference,
                    language=language,
                    speed=settings["speed"]
                )
            else:
                # Single reference audio
                logger.info(f"Using single reference audio for voice cloning")
                
                # Generate speech with reference audio
                self.model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=reference_audio,
                    language=language,
                    speed=settings["speed"]
                )
                
            logger.info(f"Speech generated successfully and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise
            
    def create_voice_profile(
        self,
        reference_audio: Union[str, List[str]],
        speaker_id: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Create a simple voice profile from reference audio samples.
        This doesn't actually create a persistent profile but returns a dictionary with paths.
        
        Args:
            reference_audio: Path or list of paths to reference audio file(s)
            speaker_id: Optional speaker ID
            save_path: Optional path to save the voice profile
            
        Returns:
            Dictionary with reference sample paths
        """
        if not self.model:
            self._initialize_model()
            
        try:
            # Convert single reference to list if needed
            if isinstance(reference_audio, str):
                reference_audio = [reference_audio]
                
            # Generate a speaker ID if not provided
            if not speaker_id:
                speaker_id = f"speaker_{int(time.time())}"
                
            logger.info(f"Creating voice reference for speaker {speaker_id} with {len(reference_audio)} samples")
            
            # Create simple voice profile dictionary
            voice_profile = {
                "speaker_id": speaker_id,
                "reference_paths": reference_audio,
                "created_at": time.time()
            }
            
            # Save profile if save path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    import json
                    json.dump(voice_profile, f, indent=2)
                logger.info(f"Voice profile saved to {save_path}")
                
            return voice_profile
            
        except Exception as e:
            logger.error(f"Error creating voice profile: {str(e)}")
            raise
            
def initialize_tts(model_path: Optional[str] = None, use_gpu: bool = True, gpu_index: int = 0) -> LocalTTS:
    """
    Initialize and return a LocalTTS instance.
    
    Args:
        model_path: Path to a local model or None to use the default
        use_gpu: Whether to use GPU for inference
        gpu_index: Index of the GPU to use if GPU is enabled (default: 0)
        
    Returns:
        Initialized LocalTTS instance
    """
    logger.info(f"Initializing local TTS, GPU: {use_gpu}, GPU index: {gpu_index}")
    return LocalTTS(model_path=model_path, use_gpu=use_gpu, gpu_index=gpu_index)
    
def generate_speech(
    text: str,
    reference_audio: Union[str, List[str]],
    output_path: str,
    language: str = "hi",
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    gpu_index: int = 0,
    voice_settings: Optional[Dict] = None
) -> str:
    """
    Generate speech using local TTS voice cloning - wrapper function.
    
    Args:
        text: Text to synthesize
        reference_audio: Path(s) to reference audio file(s)
        output_path: Path to save the generated audio
        language: Target language code (hi for Hindi)
        model_path: Path to a local model or None to use the default
        use_gpu: Whether to use GPU for inference
        gpu_index: Index of the GPU to use if GPU is enabled (default: 0)
        voice_settings: Optional voice settings dictionary
        
    Returns:
        Path to the generated audio file
    """
    tts = initialize_tts(model_path=model_path, use_gpu=use_gpu, gpu_index=gpu_index)
    return tts.generate_speech(
        text=text,
        reference_audio=reference_audio,
        output_path=output_path,
        language=language,
        voice_settings=voice_settings
    )
    
def batch_generate_speech(
    segments: List[Dict],
    reference_audio: Union[str, List[str]],
    output_dir: str,
    language: str = "hi",
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    gpu_index: int = 0
) -> List[Dict]:
    """
    Generate speech for multiple text segments using the same voice profile.
    
    Args:
        segments: List of dictionaries with segment information including 'text' key
        reference_audio: Path(s) to reference audio file(s)
        output_dir: Directory to save generated audio files
        language: Target language code
        model_path: Path to a local model or None to use the default
        use_gpu: Whether to use GPU for inference
        gpu_index: Index of the GPU to use if GPU is enabled (default: 0)
        
    Returns:
        List of dictionaries with segment info and output paths
    """
    tts = initialize_tts(model_path=model_path, use_gpu=use_gpu, gpu_index=gpu_index)
    
    # Create a simple reference mapping
    voice_profile = tts.create_voice_profile(
        reference_audio=reference_audio,
        speaker_id=f"speaker_{int(time.time())}"
    )
    
    results = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for i, segment in enumerate(segments):
        try:
            text = segment.get("text") or segment.get("hindi")
            if not text:
                logger.warning(f"No text found for segment {i}")
                continue
                
            output_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
            
            # Generate speech using the voice profile
            tts.generate_speech(
                text=text,
                reference_audio=reference_audio,
                output_path=output_path,
                language=language
            )
            
            # Add output path to segment info
            result_segment = segment.copy()
            result_segment["tts_output"] = output_path
            results.append(result_segment)
            
            logger.info(f"Generated TTS for segment {i+1}/{len(segments)}")
            
        except Exception as e:
            logger.error(f"Error generating TTS for segment {i}: {str(e)}")
            
    return results
    
if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate speech using local TTS")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--reference", required=True, help="Path to reference audio file")
    parser.add_argument("--output", required=True, help="Path to save output audio")
    parser.add_argument("--language", default="hi", help="Target language code")
    parser.add_argument("--model", help="Path to local model (or leave empty for default XTTS v2)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    
    args = parser.parse_args()
    
    # Generate speech
    try:
        output_path = generate_speech(
            text=args.text,
            reference_audio=args.reference,
            output_path=args.output,
            language=args.language,
            model_path=args.model,
            use_gpu=not args.no_gpu
        )
        print(f"Speech generated successfully: {output_path}")
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        exit(1) 
"""
Local TTS Integration Package

This package provides integration with local TTS models for high-quality voice cloning
as part of the English-to-Hindi dubbing pipeline.
"""

from .parler_tts import LocalTTS, generate_speech, batch_generate_speech, initialize_tts
from .pipeline_integration import generate_tts_for_segments, get_tts_model

__all__ = [
    'LocalTTS',
    'generate_speech',
    'batch_generate_speech',
    'initialize_tts',
    'generate_tts_for_segments',
    'get_tts_model'
] 
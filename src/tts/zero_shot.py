"""
Zero-Shot TTS Module

This module handles the zero-shot voice cloning for Hindi TTS using XTTS.
"""

import os
import time
import json
import torch
import sys
import subprocess
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

# Handle imports differently based on whether the module is run as a script or imported
if __name__ == "__main__":
    # When run as a script, add the parent directory to sys.path to allow absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.tts.utils import (
        preprocess_hindi_text,
        match_reference_to_emotion,
        analyze_text_emotion,
        load_reference_samples,
        HINDI_FIXES
    )
    from src.utils.logger import logger
else:
    # When imported as a module, use relative imports
    from .utils import (
        preprocess_hindi_text,
        match_reference_to_emotion,
        analyze_text_emotion,
        load_reference_samples,
        HINDI_FIXES
    )
    from src.utils.logger import logger

# Import TTS library only when needed to avoid dependency issues
try:
    from TTS.api import TTS
except ImportError:
    # Direct import if not running as part of the package
    try:
        from TTS.api import TTS
    except ImportError:
        import logging
        logger = logging.getLogger(__name__)

# Import the necessary modules for low-level model loading
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Global variables
_tts_model = None
_model_type = None

def get_tts_model(
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", 
    gpu: bool = False,
    gpu_index: int = 0
) -> TTS:
    """
    Load the TTS model for voice cloning.
    
    Args:
        model_name: Name or path of the TTS model to use
        gpu: Whether to use GPU for inference
        gpu_index: Index of the GPU to use if GPU is enabled (default: 0)
        
    Returns:
        Initialized TTS model
    """
    logger.info(f"Loading TTS model: {model_name}")
    
    try:
        # If using GPU, make sure the right device is selected
        if gpu and torch.cuda.is_available():
            if gpu_index >= 0 and gpu_index < torch.cuda.device_count():
                logger.info(f"Using GPU {gpu_index} for TTS")
                torch.cuda.set_device(gpu_index)
            else:
                logger.warning(f"GPU index {gpu_index} not available. Using default GPU.")
        elif gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA is not available. Using CPU instead.")
            gpu = False
            
        # Enable/disable GPU based on flag
        tts = TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        return tts
    except Exception as e:
        logger.error(f"Error loading TTS model: {str(e)}")
        raise

def get_standard_tts_model(
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", 
    gpu: bool = False,
    gpu_index: int = 0
) -> TTS:
    """
    Load a standard Hindi TTS model (no voice cloning).
    
    Args:
        model_name: Name or path of the standard Hindi TTS model to use
        gpu: Whether to use GPU for inference
        gpu_index: Index of the GPU to use if GPU is enabled (default: 0)
        
    Returns:
        Initialized TTS model for standard Hindi TTS
    """
    logger.info(f"Loading standard Hindi TTS model: {model_name}")
    
    try:
        # If using GPU, make sure the right device is selected
        if gpu and torch.cuda.is_available():
            if gpu_index >= 0 and gpu_index < torch.cuda.device_count():
                logger.info(f"Using GPU {gpu_index} for TTS")
                torch.cuda.set_device(gpu_index)
            else:
                logger.warning(f"GPU index {gpu_index} not available. Using default GPU.")
        elif gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but CUDA is not available. Using CPU instead.")
            gpu = False
            
        # Enable/disable GPU based on flag
        tts = TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        return tts
    except Exception as e:
        logger.error(f"Error loading standard Hindi TTS model: {str(e)}")
        logger.info("Falling back to XTTS for standard Hindi TTS")
        # Just use the XTTS model but without reference audio
        return TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=gpu)

def generate_speech(
    tts_model: TTS,
    text: str,
    reference_audio: Union[str, List[str]],
    output_path: str,
    language: str = "hi"
) -> str:
    """
    Generate speech using zero-shot voice cloning.
    
    Args:
        tts_model: Initialized TTS model
        text: Text to synthesize
        reference_audio: Path(s) to reference audio file(s)
        output_path: Path to save the generated audio
        language: Target language code (hi for Hindi)
        
    Returns:
        Path to the generated audio file
    """
    logger.info(f"Generating TTS for text: {text[:50]}...")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Handle multiple reference samples
        if isinstance(reference_audio, list):
            # Ensure we have at least one reference sample
            if not reference_audio:
                raise ValueError("No reference audio samples provided")
                
            logger.info(f"Using {len(reference_audio)} reference samples for voice cloning")
            
            # Use the first sample as the primary reference
            primary_reference = reference_audio[0]
            
            # Generate the audio with the selected speaker reference
            # No post-processing - direct generation
            tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=primary_reference,
                language=language
            )
        else:
            # Single reference sample
            tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=reference_audio,
                language=language
            )
            
        logger.info(f"Generated TTS saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error generating TTS: {str(e)}")
        raise

def generate_standard_speech(
    tts_model: TTS,
    text: str,
    output_path: str,
    language: str = "hi",
    speaker: str = None
) -> str:
    """
    Generate speech using standard TTS (no voice cloning).
    
    Args:
        tts_model: Initialized TTS model for standard Hindi TTS
        text: Text to synthesize
        output_path: Path to save the generated audio
        language: Target language code (hi for Hindi)
        speaker: Speaker ID for multi-speaker models (default: None - will use first available)
        
    Returns:
        Path to the generated audio file
    """
    logger.info(f"Generating standard TTS for text: {text[:50]}...")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try using Coqui TTS directly
        try:
            # Check if it's a multi-speaker model
            is_multi_speaker = hasattr(tts_model, "synthesizer") and hasattr(tts_model.synthesizer, "tts_model") and hasattr(tts_model.synthesizer.tts_model, "num_speakers") and tts_model.synthesizer.tts_model.num_speakers > 0
            
            # If no speaker provided, always get available speakers for XTTS models
            if tts_model.model_name == "tts_models/multilingual/multi-dataset/xtts_v2" or is_multi_speaker:
                available_speakers = tts_model.speakers
                logger.info(f"Available speakers in model: {available_speakers}")
                
                # If no speaker is provided, use the first available speaker
                if not speaker and available_speakers:
                    speaker = available_speakers[0]
                    logger.info(f"Using default speaker ID: {speaker}")
                elif not speaker and not available_speakers:
                    # Fallback to a standard speaker ID for XTTS if none available
                    speaker = "default"
                    logger.info(f"No speakers found. Using fallback speaker ID: {speaker}")
            
            # Generate the audio based on model type
            if is_multi_speaker or speaker or tts_model.model_name == "tts_models/multilingual/multi-dataset/xtts_v2":
                logger.info(f"Using speaker ID: {speaker} for multi-speaker model")
                # Generate with speaker parameter
                tts_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language,
                    speaker=speaker
                )
            else:
                # Standard single-speaker model
                tts_model.tts_to_file(
                    text=text,
                    file_path=output_path,
                    language=language
                )
        except Exception as e:
            logger.warning(f"Error using TTS model directly: {str(e)}")
            logger.info("Falling back to gTTS")
            
            # Fallback to Google TTS if Coqui TTS fails
            try:
                import gtts
                import subprocess
                from gtts import gTTS
                
                # Generate MP3 with gTTS
                mp3_path = output_path.replace(".wav", ".mp3")
                tts = gTTS(text=text, lang='hi', slow=False)
                tts.save(mp3_path)
                
                # Convert to WAV
                cmd = [
                    "ffmpeg", "-y", "-i", mp3_path, 
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                    output_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Remove MP3
                if os.path.exists(mp3_path):
                    os.remove(mp3_path)
            except ImportError:
                logger.error("Cannot import gtts. Please install it with 'pip install gtts'")
                raise
            except Exception as e2:
                logger.error(f"Error generating speech with gTTS: {str(e2)}")
                raise
            
        logger.info(f"Generated standard TTS saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error generating standard TTS: {str(e)}")
        raise

def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception:
        return 0.0

def process_tts(
    video_id: str,
    translations_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    reference_samples_path: Optional[str] = None,
    language: str = "hi",
    gpu: bool = True,
    gpu_index: int = 0,
    num_reference_samples: int = 3,
    direct_reference_audio: Optional[str] = None
) -> Dict:
    """
    Process TTS for all segments in a video.
    
    Args:
        video_id: Video ID
        translations_path: Path to translations JSON file
        output_dir: Directory to save generated audio
        reference_samples_path: Path to reference samples metadata
        language: Target language code (default: Hindi)
        gpu: Whether to use GPU for inference
        gpu_index: Index of the GPU to use if GPU is enabled
        num_reference_samples: Number of reference samples to use for each segment
        direct_reference_audio: Direct path to a reference audio file (bypasses reference samples)
        
    Returns:
        Dictionary with TTS processing results
    """
    # Process video_id (remove extension if present)
    if video_id.endswith('.mp4') or video_id.endswith('.wav'):
        video_id = Path(video_id).stem
    
    logger.info(f"Processing TTS for video: {video_id}")
    
    # Generate paths if not provided
    if translations_path is None:
        translations_path = Path(f"processed_data/translations/{video_id}_transcription_translated.json")
    else:
        translations_path = Path(translations_path)
    
    if output_dir is None:
        output_dir = Path(f"processed_data/tts_output/{video_id}")
    else:
        output_dir = Path(output_dir)
    
    logger.info(f"Translations path: {translations_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference samples or use direct reference audio
    reference_samples = {}
    if direct_reference_audio:
        logger.info(f"Using direct reference audio: {direct_reference_audio}")
        if not os.path.exists(direct_reference_audio):
            raise FileNotFoundError(f"Direct reference audio not found: {direct_reference_audio}")
        
        # Create a simple reference samples dict with a single item
        reference_samples = {
            "speaker_0": [
                {
                    "path": direct_reference_audio,
                    "emotion": "neutral",
                    "quality": 10
                }
            ]
        }
    else:
        # Load reference samples from metadata
        if reference_samples_path is None:
            try:
                reference_samples = load_reference_samples(video_id)
            except FileNotFoundError:
                logger.error(f"Reference samples not found for {video_id}")
                raise
        else:
            with open(reference_samples_path, 'r') as f:
                reference_samples = json.load(f)
    
    # Log reference samples info
    logger.info(f"Loaded reference samples for {len(reference_samples)} speakers")
    for speaker_id, samples in reference_samples.items():
        logger.info(f"Speaker {speaker_id}: {len(samples)} reference samples")
    
    # Load translations
    if not translations_path.exists():
        logger.error(f"Translations file not found: {translations_path}")
        raise FileNotFoundError(f"Translations file not found: {translations_path}")
    
    logger.info(f"Loading translations from: {translations_path}")
    with open(translations_path, 'r') as f:
        translations_data = json.load(f)
    
    # Process segments
    results = {
        "video_id": video_id,
        "language": language,
        "segments": []
    }
    
    # Get list of segments
    segments = translations_data.get("segments", [])
    
    if not segments:
        logger.warning(f"No segments found in translations file: {translations_path}")
        return results
    
    logger.info(f"Processing {len(segments)} segments for TTS")
    
    total_start_time = time.time()
    
    for i, segment in enumerate(segments):
        segment_id = segment.get("id", i)
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        
        logger.info(f"\n=== Processing segment {i+1}/{len(segments)} (ID: {segment_id}) ===")
        logger.info(f"Segment timing: start={start_time:.2f}s, end={end_time:.2f}s")
        
        # Get translated text (Hindi)
        hindi_text = segment.get("text_hi", "")
        if not hindi_text:
            logger.warning(f"No Hindi text found for segment {segment_id}")
            continue
        
        logger.info(f"Hindi text: '{hindi_text[:100]}{'...' if len(hindi_text) > 100 else ''}'")
        
        # Get speaker ID (default to speaker_0 if not found)
        speaker_id = segment.get("speaker", "speaker_0")
        logger.info(f"Speaker ID: {speaker_id}")
        
        # Find the best reference samples for this text
        try:
            logger.info(f"Finding {num_reference_samples} reference samples for segment {segment_id}")
            reference_paths = match_reference_to_emotion(
                hindi_text, 
                reference_samples, 
                speaker_id,
                num_samples=num_reference_samples
            )
            
            if isinstance(reference_paths, list):
                logger.info(f"Found {len(reference_paths)} reference samples")
            else:
                logger.info(f"Found 1 reference sample: {reference_paths}")
        except Exception as e:
            logger.error(f"Error matching reference sample: {e}")
            logger.exception("Detailed traceback:")
            
            # Fall back to any available reference sample
            logger.info("Falling back to any available reference sample")
            available_speakers = list(reference_samples.keys())
            speaker_id = available_speakers[0] if available_speakers else "speaker_0"
            speaker_samples = reference_samples.get(speaker_id, [])
            if speaker_samples:
                if num_reference_samples > 1:
                    reference_paths = [s["path"] for s in speaker_samples[:num_reference_samples]]
                    logger.info(f"Using {len(reference_paths)} fallback reference samples")
                else:
                    reference_paths = speaker_samples[0]["path"]
                    logger.info(f"Using fallback reference sample: {reference_paths}")
            else:
                logger.error(f"No reference samples available for segment {segment_id}")
                continue
        
        # Generate output path
        segment_output_path = output_dir / f"segment_{segment_id:04d}.wav"
        logger.info(f"Output path: {segment_output_path}")
        
        # Calculate original segment duration
        original_duration = end_time - start_time
        logger.info(f"Original segment duration: {original_duration:.2f}s")
        
        # Generate speech
        try:
            logger.info(f"Generating speech for segment {segment_id}")
            generated_path = generate_speech(
                tts_model=get_tts_model(gpu=gpu),
                text=hindi_text,
                reference_audio=reference_paths,
                output_path=str(segment_output_path),
                language=language
            )
            
            # Get audio duration
            audio_duration = get_audio_duration(generated_path)
            logger.info(f"Generated audio duration: {audio_duration:.2f}s")
            
            # Add to results
            results["segments"].append({
                "id": segment_id,
                "start": start_time,
                "end": end_time,
                "original_duration": original_duration,
                "generated_duration": audio_duration,
                "audio_path": generated_path,
                "reference_paths": reference_paths if isinstance(reference_paths, list) else [reference_paths],
                "speaker_id": speaker_id
            })
            
            logger.info(f"Successfully processed segment {i+1}/{len(segments)}: original={original_duration:.2f}s, generated={audio_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error generating speech for segment {segment_id}: {str(e)}")
            logger.exception("Detailed traceback:")
    
    total_time = time.time() - total_start_time
    
    # Save results
    results_path = output_dir / "tts_results.json"
    logger.info(f"Saving results to: {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    success_count = len(results["segments"])
    logger.info(f"TTS processing completed in {total_time:.2f} seconds")
    logger.info(f"Generated {success_count}/{len(segments)} segments successfully")
    logger.info(f"Results saved to: {results_path}")
    
    if success_count < len(segments):
        logger.warning(f"Failed to generate {len(segments) - success_count} segments")
    
    return results

def generate_tts_for_segments(translation: Dict, reference_samples: Dict, output_dir: str, use_gpu: bool = False, gpu_index: int = 0, use_standard_tts: bool = False) -> List[Dict]:
    """
    Generate TTS output for each translated segment.
    
    Args:
        translation: Translation result dictionary
        reference_samples: Reference samples metadata
        output_dir: Directory to save TTS output files
        use_gpu: Whether to use GPU for TTS generation
        gpu_index: Index of the GPU to use for TTS generation
        use_standard_tts: Whether to use standard Hindi TTS instead of voice cloning
        
    Returns:
        List of dictionaries with segment info and TTS output paths
    """
    logger.info(f"Generating TTS for {len(translation['translated_segments'])} segments")
    
    # Initialize TTS models
    if use_standard_tts:
        logger.info("Using standard Hindi TTS model (no voice cloning)")
        from src.tts.zero_shot import get_standard_tts_model, generate_standard_speech
        tts_model = get_standard_tts_model(gpu=use_gpu, gpu_index=gpu_index)
        
        # Get available speakers if using a multi-speaker model
        default_speaker = None
        try:
            if hasattr(tts_model, "speakers") and tts_model.speakers:
                default_speaker = tts_model.speakers[0]
                logger.info(f"Using default speaker '{default_speaker}' for standard TTS")
        except Exception as e:
            logger.warning(f"Could not get default speaker: {str(e)}")
    else:
        logger.info("Using voice cloning TTS model")
        tts_model = get_tts_model(gpu=use_gpu, gpu_index=gpu_index)
    
    tts_segments = []
    
    # If using standard TTS, we don't need reference samples
    if use_standard_tts:
        for i, segment in enumerate(translation["translated_segments"]):
            hindi_text = segment["hindi"]
            output_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
            
            try:
                # Generate standard TTS
                generate_standard_speech(
                    tts_model=tts_model,
                    text=preprocess_hindi_text(segment["hindi"]),
                    output_path=output_path,
                    language="hi",
                    speaker=default_speaker  # Pass the default speaker
                )
                
                # Add TTS output info to the segment
                tts_segment = {
                    "start": segment.get("start", 0),  # Copy timing information
                    "end": segment.get("end", 0),      # Copy timing information
                    "english_text": segment["english"],
                    "hindi_text": segment["hindi"],
                    "speaker_id": default_speaker or segment.get("speaker", "standard_voice"),
                    "tts_output": output_path
                }
                
                tts_segments.append(tts_segment)
                logger.info(f"Generated standard TTS for segment {i+1}/{len(translation['translated_segments'])}")
                
            except Exception as e:
                logger.error(f"Error generating standard TTS for segment {i}: {str(e)}")
        
        return tts_segments
    
    # Voice Cloning TTS logic below (original implementation)
    # Get speakers and their reference samples
    speakers = {}
    
    # Support both formats of reference samples
    if "speakers" in reference_samples:
        # New format with speakers dictionary
        speakers = reference_samples.get("speakers", {})
    else:
        # Original format with direct speaker mapping
        speakers = reference_samples
    
    # If no speakers found, use default speaker
    if not speakers:
        logger.warning("No speakers found in reference samples, using default")
        speakers = {
            "speaker_0": {
                "reference_sample_paths": [reference_samples.get(
                    "reference_sample_paths", 
                    [os.path.join(output_dir, "..", "reference_samples", "reference_1.wav")]
                )[0]]
            }
        }
    
    # Map SPEAKER_XX to speaker_x format if needed
    speaker_mapping = {}
    for speaker_id in list(speakers.keys()):
        if speaker_id.startswith("speaker_"):
            idx = speaker_id.split("_")[1]
            speaker_mapping[f"SPEAKER_{idx.zfill(2)}"] = speaker_id
    
    logger.info(f"Found {len(speakers)} speakers with reference samples")
    
    # Get multiple reference samples for each speaker
    speaker_references = {}
    for speaker_id, speaker_data in speakers.items():
        # Handle different reference sample formats
        if isinstance(speaker_data, dict) and "reference_sample_paths" in speaker_data:
            # New format with reference_sample_paths
            speaker_references[speaker_id] = speaker_data["reference_sample_paths"]
        elif isinstance(speaker_data, list):
            # Original format with list of samples
            speaker_references[speaker_id] = [sample["path"] for sample in speaker_data]
        else:
            logger.warning(f"Unknown reference sample format for {speaker_id}")
            speaker_references[speaker_id] = []
    
    # Default reference sample (for segments without speaker info)
    default_speaker_id = next(iter(speaker_references.keys()))
    default_reference_samples = speaker_references[default_speaker_id]
    
    # Ensure we have at least one reference sample
    if not default_reference_samples:
        logger.error("No reference samples found for default speaker")
        raise ValueError("No reference samples available for TTS generation")
    
    for i, segment in enumerate(translation["translated_segments"]):
        hindi_text = segment["hindi"]
        output_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
        
        try:
            # Get speaker ID and map it if needed
            speaker_id = segment.get("speaker", default_speaker_id)
            
            # Map SPEAKER_XX format to speaker_x if needed
            if speaker_id in speaker_mapping:
                speaker_id = speaker_mapping[speaker_id]
            elif speaker_id.startswith("SPEAKER_") and speaker_id not in speaker_references:
                # Try to extract the number and convert to speaker_x format
                try:
                    num = int(speaker_id.split("_")[1])
                    alt_id = f"speaker_{num}"
                    if alt_id in speaker_references:
                        speaker_id = alt_id
                except (ValueError, IndexError):
                    pass
            
            # Get reference samples for this speaker
            if speaker_id in speaker_references and speaker_references[speaker_id]:
                reference_samples_paths = speaker_references[speaker_id]
                logger.info(f"Using {len(reference_samples_paths)} reference samples for speaker {speaker_id}")
            else:
                logger.warning(f"No reference samples found for speaker {speaker_id}, using default")
                reference_samples_paths = default_reference_samples
            
            # Generate TTS with multiple reference samples
            generate_speech(
                tts_model=tts_model,
                text=preprocess_hindi_text(segment["hindi"]),
                reference_audio=reference_samples_paths,
                output_path=output_path,
                language="hi"
            )
            
            # Add TTS output info to the segment
            tts_segment = {
                "start": segment.get("start", 0),  # Copy timing information
                "end": segment.get("end", 0),      # Copy timing information
                "english_text": segment["english"],
                "hindi_text": segment["hindi"],
                "speaker_id": speaker_id,
                "reference_samples": reference_samples_paths,
                "tts_output": output_path
            }
            
            tts_segments.append(tts_segment)
            logger.info(f"Generated TTS for segment {i+1}/{len(translation['translated_segments'])}")
            
        except Exception as e:
            logger.error(f"Error generating TTS for segment {i}: {str(e)}")
    
    return tts_segments

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Zero-shot TTS with XTTS"
    )
    
    parser.add_argument(
        "--video-id", 
        required=True,
        help="Video ID or path"
    )
    
    parser.add_argument(
        "--translations-path",
        help="Path to translations JSON file (default: processed_data/translations/{video_id}_transcription_translated.json)"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Directory to save generated audio (default: processed_data/tts_output/{video_id})"
    )
    
    parser.add_argument(
        "--reference-samples-path",
        help="Path to reference samples metadata (default: processed_data/reference_samples/{video_id}/reference_samples_metadata.json)"
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
    
    parser.add_argument(
        "--num-reference-samples",
        type=int,
        default=3,
        help="Number of reference samples to use for each segment (default: 3)"
    )
    
    parser.add_argument(
        "--direct-reference-audio",
        help="Direct path to a reference audio file (bypasses reference samples)"
    )
    
    args = parser.parse_args()
    
    process_tts(
        video_id=args.video_id,
        translations_path=args.translations_path,
        output_dir=args.output_dir,
        reference_samples_path=args.reference_samples_path,
        language=args.language,
        gpu=not args.no_gpu,
        num_reference_samples=args.num_reference_samples,
        direct_reference_audio=args.direct_reference_audio
    ) 
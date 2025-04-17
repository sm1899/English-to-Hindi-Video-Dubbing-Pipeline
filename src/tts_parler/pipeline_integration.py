#!/usr/bin/env python3
"""
Local TTS Pipeline Integration

This module provides integration functions to use local TTS models with the existing
English to Hindi dubbing pipeline. It maintains the same interface as the
existing TTS modules for easy replacement.
"""

import os
import logging
from typing import Dict, List, Optional, Union

try:
    from src.utils.logger import logger
    from src.tts_parler.parler_tts import LocalTTS, initialize_tts, batch_generate_speech
    from src.tts.utils import preprocess_hindi_text
except ImportError:
    # Handle direct imports if not running from the project root
    from tts_parler.parler_tts import LocalTTS, initialize_tts, batch_generate_speech
    try:
        from tts.utils import preprocess_hindi_text
        from utils.logger import logger
    except ImportError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("local_tts_pipeline")
        
        # Simple preprocessing function if the main one is not available
        def preprocess_hindi_text(text: str) -> str:
            """Simple preprocessing for Hindi text."""
            return text.strip()

# Global TTS model instance for reuse
_tts_model = None

def get_tts_model(model_path: Optional[str] = None, gpu: bool = True, gpu_index: int = 0) -> LocalTTS:
    """
    Get or initialize the local TTS model.
    
    Args:
        model_path: Path to local model (optional)
        gpu: Whether to use GPU for inference
        gpu_index: Index of the GPU to use if GPU is enabled (default: 0)
        
    Returns:
        Initialized LocalTTS instance
    """
    global _tts_model
    
    if _tts_model is None:
        logger.info(f"Initializing local TTS model, GPU: {gpu}, GPU index: {gpu_index}")
        _tts_model = initialize_tts(model_path=model_path, use_gpu=gpu, gpu_index=gpu_index)
    
    return _tts_model

def generate_speech(
    tts_model: Optional[LocalTTS],
    text: str,
    reference_audio: Union[str, List[str]],
    output_path: str,
    language: str = "hi",
    voice_settings: Optional[Dict] = None
) -> str:
    """
    Generate speech using local TTS with the same interface as the pipeline's TTS function.
    
    Args:
        tts_model: LocalTTS model instance (optional, will be initialized if None)
        text: Text to synthesize
        reference_audio: Path(s) to reference audio file(s)
        output_path: Path to save the generated audio
        language: Target language code (hi for Hindi)
        voice_settings: Optional voice settings dictionary
        
    Returns:
        Path to the generated audio file
    """
    # Ensure model is initialized
    model = tts_model or get_tts_model()
    
    # Preprocess text
    processed_text = preprocess_hindi_text(text)
    
    logger.info(f"Generating speech for text: {processed_text[:50]}...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate speech
        result_path = model.generate_speech(
            text=processed_text,
            reference_audio=reference_audio,
            output_path=output_path,
            language=language,
            voice_settings=voice_settings
        )
        
        logger.info(f"Speech generated successfully: {result_path}")
        return result_path
    
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        raise

def generate_tts_for_segments(
    translation: Dict, 
    reference_samples: Dict, 
    output_dir: str, 
    use_gpu: bool = True,
    gpu_index: int = 0,
    model_path: Optional[str] = None
) -> List[Dict]:
    """
    Generate TTS output for each translated segment using local TTS model.
    
    This function maintains the same interface as the pipeline's generate_tts_for_segments
    for easy drop-in replacement.
    
    Args:
        translation: Translation result dictionary
        reference_samples: Reference samples metadata
        output_dir: Directory to save TTS output files
        use_gpu: Whether to use GPU for TTS generation
        gpu_index: Index of the GPU to use if GPU is enabled (default: 0)
        model_path: Optional path to a custom TTS model
        
    Returns:
        List of dictionaries with segment info and TTS output paths
    """
    logger.info(f"Generating TTS for {len(translation['translated_segments'])} segments")
    
    # Initialize TTS model
    tts_model = get_tts_model(model_path=model_path, gpu=use_gpu, gpu_index=gpu_index)
    
    tts_segments = []
    
    # Get speakers and their reference samples
    speakers = {}
    
    # Support both reference samples formats
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
    
    # Group segments by speaker for efficient batch processing
    speaker_segments = {}
    
    for i, segment in enumerate(translation["translated_segments"]):
        hindi_text = segment["hindi"]
        
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
        
        # Initialize list for this speaker if not already done
        if speaker_id not in speaker_segments:
            speaker_segments[speaker_id] = []
        
        # Add segment data
        speaker_segments[speaker_id].append({
            "segment_index": i,
            "hindi": hindi_text,
            "english": segment.get("english", ""),
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "speaker": speaker_id
        })
    
    # Process each speaker's segments
    for speaker_id, segments in speaker_segments.items():
        # Get reference samples for this speaker
        if speaker_id in speaker_references and speaker_references[speaker_id]:
            reference_samples_paths = speaker_references[speaker_id]
            logger.info(f"Using {len(reference_samples_paths)} reference samples for speaker {speaker_id}")
        else:
            logger.warning(f"No reference samples found for speaker {speaker_id}, using default")
            reference_samples_paths = default_reference_samples
        
        # Create output paths for each segment
        for segment in segments:
            i = segment["segment_index"]
            segment["output_path"] = os.path.join(output_dir, f"segment_{i:04d}.wav")
        
        # Generate TTS for all segments of this speaker in batch
        for segment in segments:
            try:
                # Generate speech for this segment
                generate_speech(
                    tts_model=tts_model,
                    text=preprocess_hindi_text(segment["hindi"]),
                    reference_audio=reference_samples_paths,
                    output_path=segment["output_path"],
                    language="hi"
                )
                
                # Add TTS output info to the segments list
                tts_segment = {
                    "english_text": segment["english"],
                    "hindi_text": segment["hindi"],
                    "speaker_id": speaker_id,
                    "reference_samples": reference_samples_paths,
                    "tts_output": segment["output_path"]
                }
                
                tts_segments.append(tts_segment)
                logger.info(f"Generated TTS for segment {segment['segment_index']+1}/{len(translation['translated_segments'])}")
                
            except Exception as e:
                logger.error(f"Error generating TTS for segment {segment['segment_index']}: {str(e)}")
    
    # Sort segments by original index to maintain order
    tts_segments.sort(key=lambda x: int(os.path.basename(x["tts_output"]).split("_")[1].split(".")[0]))
    
    return tts_segments

# Create an __init__.py to make the directory a proper Python package
if __name__ == "__main__":
    # Example usage
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Generate TTS using local model for a translation JSON")
    parser.add_argument("--translation", required=True, help="Path to translation JSON file")
    parser.add_argument("--references", required=True, help="Path to reference samples metadata JSON")
    parser.add_argument("--output-dir", required=True, help="Directory to save TTS output")
    parser.add_argument("--model", help="Path to a custom TTS model (optional)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    
    args = parser.parse_args()
    
    try:
        # Load translation data
        with open(args.translation, 'r', encoding='utf-8') as f:
            translation = json.load(f)
        
        # Load reference samples metadata
        with open(args.references, 'r', encoding='utf-8') as f:
            reference_samples = json.load(f)
        
        # Generate TTS
        tts_segments = generate_tts_for_segments(
            translation=translation,
            reference_samples=reference_samples,
            output_dir=args.output_dir,
            use_gpu=not args.no_gpu,
            gpu_index=0,
            model_path=args.model
        )
        
        # Save results
        result_path = os.path.join(args.output_dir, "tts_segments.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(tts_segments, f, ensure_ascii=False, indent=2)
        
        print(f"TTS generation completed. Generated {len(tts_segments)} segments.")
        print(f"Results saved to {result_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1) 
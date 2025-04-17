#!/usr/bin/env python3
"""
English to Hindi Video Dubbing Pipeline

This script orchestrates the process of dubbing a video from English to Hindi:
1. Extracts audio from the input video
2. Separates speech from background music/noise
3. Transcribes the speech audio to English text
4. Translates the English text to Hindi
5. Generates Hindi speech using the speaker's voice characteristics
6. Combines speech segments with background audio
7. Outputs combined Hindi audio file

Usage:
    python pipeline.py --video_path path/to/video.mp4 --output_dir output/ --use_gpu
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import datetime
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules from src
try:
    from src.tts.zero_shot import generate_speech, get_tts_model
    from src.tts.utils import preprocess_hindi_text
    from src.audio_processing.separation import separate_audio_from_video, cleanup_separation_temp
    from src.audio_processing.combine import combine_audio_segments
    from src.diarization.pyannotate import perform_diarization
    from src.reference_sampling.pipeline_integration import process_reference_sampling
except ImportError:
    # Handle direct imports if not running from the project root
    from tts.zero_shot import generate_speech, get_tts_model
    from tts.utils import preprocess_hindi_text
    try:
        from audio_processing.separation import separate_audio_from_video, cleanup_separation_temp
        from audio_processing.combine import combine_audio_segments
        from diarization.pyannotate import perform_diarization
        from reference_sampling.pipeline_integration import process_reference_sampling
    except ImportError:
        # If audio_processing module isn't found, we'll handle this later
        pass

# Conditionally import Mistral translator
try:
    from src.translation_llm.use_mistral import translate_mistral
except ImportError:
    logger.warning("Mistral translation module not found. --use_llm flag will not work.")
    translate_mistral = None

# Define the LatentSync directory relative to the pipeline script
LATENTSYNC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LatentSync")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dubbing_pipeline.log")
    ]
)
logger = logging.getLogger("dubbing_pipeline")

def setup_directories(output_dir: str) -> Dict[str, str]:
    """
    Create necessary directories for the pipeline outputs.
    
    Args:
        output_dir: Base directory for all outputs
        
    Returns:
        Dictionary with paths to each subdirectory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(output_dir, timestamp)
    
    dirs = {
        "base": base_dir,
        "audio": os.path.join(base_dir, "audio"),
        "separated_audio": os.path.join(base_dir, "separated_audio"),
        "transcriptions": os.path.join(base_dir, "transcriptions"),
        "translations": os.path.join(base_dir, "translations"),
        "diarization": os.path.join(base_dir, "diarization"),
        "reference_samples": os.path.join(base_dir, "reference_samples"),
        "tts_output": os.path.join(base_dir, "tts_output"),
        "results": os.path.join(base_dir, "results")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return dirs

def extract_and_separate_audio(video_path: str, output_dir: str, use_gpu: bool = False, gpu_index: int = 0) -> Dict:
    """
    Extract audio from video and separate speech from background.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the extracted and separated audio
        use_gpu: Whether to use GPU for audio separation
        gpu_index: Index of the GPU to use for audio separation (not used by separation function)
        
    Returns:
        Dictionary with paths to separated audio files
    """
    logger.info(f"Extracting and separating audio from {video_path}")
    
    try:
        # Use the Demucs model for audio separation
        # Note: separate_audio_from_video doesn't accept gpu_index parameter
        # If GPU is used, we'll use CUDA_VISIBLE_DEVICES to select the GPU
        if use_gpu and gpu_index is not None:
            # Store the original setting
            original_cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES")
            # Set the GPU index
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            logger.info(f"Set CUDA_VISIBLE_DEVICES to {gpu_index} for audio separation")
            
        try:
            separation_result = separate_audio_from_video(
                video_path=video_path,
                output_dir=output_dir,
                use_gpu=use_gpu
            )
        finally:
            # Restore the original setting if it existed
            if use_gpu and gpu_index is not None:
                if original_cuda_device is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_device
                    logger.info(f"Restored CUDA_VISIBLE_DEVICES to {original_cuda_device}")
                else:
                    # If there was no original setting, remove it
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    logger.info("Removed CUDA_VISIBLE_DEVICES setting")
        
        logger.info(f"Audio separation completed")
        logger.info(f"Speech audio: {separation_result['speech']}")
        logger.info(f"Background audio: {separation_result['background']}")
        
        return separation_result
    except Exception as e:
        logger.error(f"Error during audio extraction and separation: {str(e)}")
        raise

def perform_speaker_diarization(audio_path: str, output_path: str, auth_token: Optional[str] = None) -> str:
    """
    Perform speaker diarization to identify different speakers.
    
    Args:
        audio_path: Path to the speech audio file
        output_path: Path to save the diarization results
        auth_token: Optional authentication token for PyAnnote
        
    Returns:
        Path to the diarization results file
    """
    logger.info(f"Performing speaker diarization on {audio_path}")
    
    try:
        # Call the diarization module
        diarization_path = perform_diarization(
            audio_path=audio_path,
            output_path=output_path,
            auth_token=auth_token
        )
        
        logger.info(f"Diarization completed and saved to {diarization_path}")
        return diarization_path
    except ImportError:
        logger.warning("Speaker diarization module not available. Skipping diarization.")
        return None
    except Exception as e:
        logger.error(f"Error during speaker diarization: {str(e)}")
        # Continue without diarization
        return None

def extract_reference_samples(audio_path: str, output_dir: str, transcription_path: Optional[str] = None, 
                             diarization_path: Optional[str] = None) -> Dict:
    """
    Extract reference audio samples for voice cloning.
    
    Args:
        audio_path: Path to the speech audio file
        output_dir: Directory to save reference samples
        transcription_path: Optional path to transcription JSON
        diarization_path: Optional path to diarization results
        
    Returns:
        Dictionary with reference sample information
    """
    logger.info(f"Extracting reference samples from {audio_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load transcription if available
    transcription = None
    if transcription_path and os.path.exists(transcription_path):
        try:
            with open(transcription_path, 'r', encoding='utf-8') as f:
                transcription = json.load(f)
            logger.info(f"Loaded transcription from {transcription_path}")
        except Exception as e:
            logger.error(f"Error loading transcription: {str(e)}")
    
    # Load diarization if available
    diarization = None
    if diarization_path and os.path.exists(diarization_path):
        try:
            with open(diarization_path, 'r', encoding='utf-8') as f:
                diarization = json.load(f)
            logger.info(f"Loaded diarization from {diarization_path}")
        except Exception as e:
            logger.error(f"Error loading diarization: {str(e)}")
    
    # Get list of unique speakers from diarization
    speakers = set()
    if diarization:
        for segment in diarization:
            speakers.add(segment.get("speaker", "UNKNOWN"))
    
    # If no speakers found, use default speaker
    if not speakers:
        speakers = {"speaker_0"}
    
    logger.info(f"Found {len(speakers)} speakers")
    
    # Create a structure to hold reference samples for each speaker
    reference_samples = {}
    
    # Process each speaker
    for speaker in speakers:
        speaker_id = speaker.lower().replace("speaker_", "speaker_")
        
        # Get segments from diarization for this speaker
        speaker_segments = []
        if diarization:
            speaker_segments = [
                segment for segment in diarization 
                if segment.get("speaker", "").lower() == speaker.lower()
            ]
        
        # Extract samples for this speaker
        speaker_samples = []
        
        if speaker_segments and transcription:
            # Get the longest segments for this speaker (top 3)
            longest_segments = sorted(
                speaker_segments, 
                key=lambda x: x["end"] - x["start"], 
                reverse=True
            )[:3]
            
            for i, segment in enumerate(longest_segments):
                start_time = segment["start"]
                end_time = segment["end"]
                
                # Ensure at least 3 seconds of audio
                if end_time - start_time < 3.0:
                    continue
                
                # Extract the audio segment
                sample_path = os.path.join(output_dir, f"{speaker_id}_sample_{i+1}.wav")
                
                try:
                    # Use ffmpeg to extract the segment
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", audio_path,
                        "-ss", str(start_time),
                        "-to", str(end_time),
                        "-acodec", "pcm_s16le",
                        "-ar", "16000",
                        "-ac", "1",
                        sample_path
                    ]
                    
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Find matching transcription text
                    segment_text = ""
                    if transcription and "segments" in transcription:
                        # Find segments that overlap with this time range
                        for trans_segment in transcription["segments"]:
                            t_start = trans_segment["start"]
                            t_end = trans_segment["end"]
                            
                            # Check for overlap
                            if (t_start <= end_time and t_end >= start_time):
                                segment_text += trans_segment.get("text", "") + " "
                    
                    # Create sample metadata
                    sample_metadata = {
                        "path": sample_path,
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "text": segment_text.strip(),
                        "emotion_label": "neutral"  # Default emotion label
                    }
                    
                    # Add features if available
                    features = {
                        "speech_rate": len(segment_text.split()) / (end_time - start_time) if segment_text else 0,
                        "duration": end_time - start_time
                    }
                    
                    sample_metadata["features"] = features
                    speaker_samples.append(sample_metadata)
                    
                    logger.info(f"Extracted reference sample for {speaker_id}: {sample_path}")
                    
                except Exception as e:
                    logger.error(f"Error extracting sample for {speaker_id}: {str(e)}")
        
        # If no samples were extracted for this speaker, create a basic one
        if not speaker_samples:
            try:
                # Create a basic sample from the first 5-10 seconds
                sample_path = os.path.join(output_dir, f"{speaker_id}_reference.wav")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", audio_path,
                    "-ss", "5",  # Start at 5 seconds
                    "-t", "5",   # 5 seconds duration
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    sample_path
                ]
                
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                sample_metadata = {
                    "path": sample_path,
                    "start": 5.0,
                    "end": 10.0,
                    "duration": 5.0,
                    "text": "",
                    "features": {
                        "duration": 5.0,
                    },
                    "emotion_label": "neutral"
                }
                
                speaker_samples.append(sample_metadata)
                logger.info(f"Created basic reference sample for {speaker_id}: {sample_path}")
                
            except Exception as e:
                logger.error(f"Error creating basic sample for {speaker_id}: {str(e)}")
        
        # Add this speaker's samples to the result
        if speaker_samples:
            # Convert speaker ID format if needed 
            normalized_speaker_id = speaker_id
            if speaker_id.startswith("SPEAKER_"):
                try:
                    speaker_num = int(speaker_id.split("_")[1])
                    normalized_speaker_id = f"speaker_{speaker_num}"
                except (ValueError, IndexError):
                    pass
            
            reference_samples[normalized_speaker_id] = speaker_samples
    
    # Save reference samples metadata
    metadata_path = os.path.join(output_dir, "reference_samples_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(reference_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Reference samples metadata saved to {metadata_path}")
    
    # Return the reference samples structure with metadata path
    result = {
        "video_id": os.path.basename(audio_path),
        "reference_metadata_path": metadata_path,
        "speakers": reference_samples
    }
    
    return result

def transcribe_audio(audio_path: str, output_path: str, diarization_path: Optional[str] = None, model_name: str = "large-v3", max_segment_length: int = 30, use_gpu: bool = False, gpu_index: int = 0) -> Dict:
    """
    Transcribe audio to text using Whisper.
    
    Args:
        audio_path: Path to the speech audio file
        output_path: Path to save the transcription JSON
        diarization_path: Optional path to diarization results to incorporate speaker information
        model_name: Whisper model to use (tiny, base, small, medium, large)
        max_segment_length: Maximum number of words per segment (default: 30)
        use_gpu: Whether to use GPU for ASR
        gpu_index: Index of the GPU to use for ASR
        
    Returns:
        Transcription result dictionary
    """
    logger.info(f"Transcribing audio using Whisper {model_name} model")
    
    try:
        # Use the improved asr module's transcribe_audio function which includes proper segmentation
        try:
            from src.asr.whisper_asr import transcribe_audio as asr_transcribe
            
            # Call the module's transcribe_audio function
            asr_result_path = asr_transcribe(
                audio_path=audio_path,
                output_path=output_path,
                model_name=model_name,
                language="en",
                word_timestamps=True,
                diarization_path=diarization_path,
                fix_segmentation=True,
                max_segment_length=max_segment_length,
                use_gpu=use_gpu,
                gpu_index=gpu_index
            )
            
            # Load the result
            with open(asr_result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
            logger.info(f"Transcription completed using improved ASR module")
            return result
        except ImportError:
            # Fall back to the original implementation if the ASR module is not available
            logger.warning("Could not import from src.asr.whisper_asr. Falling back to basic whisper implementation.")
            import whisper
            
            # Load the Whisper model
            model = whisper.load_model(model_name)
            
            # Transcribe the audio
            logger.info("Running fallback whisper transcription...")
            result = model.transcribe(audio_path, language="en")
            logger.info(f"Fallback whisper transcription produced {len(result.get('segments', []))} initial segments.")
            
            # Load diarization data if available to add speaker information
            if diarization_path and os.path.exists(diarization_path):
                logger.info(f"Incorporating speaker information from {diarization_path}")
                try:
                    with open(diarization_path, 'r', encoding='utf-8') as f:
                        diarization_data = json.load(f)
                    
                    # Add speaker information to each segment
                    for segment in result["segments"]:
                        segment_start = segment["start"]
                        segment_end = segment["end"]
                        
                        # Find the overlapping diarization segment
                        for diar_segment in diarization_data:
                            diar_start = diar_segment["start"]
                            diar_end = diar_segment["end"]
                            
                            # Check if segments overlap
                            if (segment_start <= diar_end and segment_end >= diar_start):
                                # If they overlap by at least 50% of the transcription segment
                                overlap = min(segment_end, diar_end) - max(segment_start, diar_start)
                                segment_duration = segment_end - segment_start
                                
                                if segment_duration > 0 and overlap / segment_duration >= 0.5:
                                    segment["speaker"] = diar_segment["speaker"]
                                    break
                        
                        # If no speaker found, assign a default
                        if "speaker" not in segment:
                            segment["speaker"] = "UNKNOWN"
                    
                    logger.info("Speaker information added to transcription segments")
                except Exception as e:
                    logger.error(f"Error incorporating speaker data: {str(e)}")
            
            # Manual segmentation logic to approximate the improved segmentation
            # (simplified version of fix_sentence_segmentation)
            try:
                # Create improved segments based on punctuation and speaker changes
                old_segments = result["segments"]
                new_segments = []
                current_segment = None
                current_text = ""
                #word_count = 0 # Word count is no longer the primary driver for splitting
                current_speaker = None
                
                for segment in old_segments:
                    if not segment.get("text", "").strip():
                        continue
                    
                    segment_text = segment.get("text", "").strip()
                    segment_speaker = segment.get("speaker", "UNKNOWN")
                    
                    # Check for segment boundaries (sentence-ending punctuation or speaker change)
                    ends_with_boundary = any(segment_text.rstrip().endswith(mark) for mark in [".", "!", "?", "।", "॥"]) 
                    speaker_changed = current_speaker is not None and current_speaker != segment_speaker
                    #segment_too_long = word_count + len(segment_text.split()) > max_segment_length # De-prioritize splitting by length
                    
                    # If we need to create a new segment (only on sentence boundary or speaker change)
                    if current_segment is not None and (ends_with_boundary or speaker_changed):
                        # Finalize current segment
                        current_segment["text"] = current_text
                        new_segments.append(current_segment)
                        
                        # Reset for next segment
                        current_segment = None
                        current_text = ""
                        #word_count = 0
                    
                    # Start a new segment if needed
                    if current_segment is None:
                        current_segment = segment.copy()
                        current_text = segment_text
                        #word_count = len(segment_text.split())
                        current_speaker = segment_speaker
                    else:
                        # Add to current segment
                        current_text += " " + segment_text
                        #word_count += len(segment_text.split())
                        # Update end time only if extending the segment
                        current_segment["end"] = segment["end"]
                
                # Add final segment if there is one
                if current_segment is not None:
                    current_segment["text"] = current_text
                    new_segments.append(current_segment)
                
                # Replace segments in result
                result["segments"] = new_segments
                
                # Update the full text
                result["text"] = " ".join(segment["text"] for segment in new_segments)
                
                logger.info(f"Fallback segmentation applied: {len(old_segments)} original segments → {len(new_segments)} new segments")
            except Exception as e:
                logger.warning(f"Error during manual segmentation improvement: {str(e)}")
            
            # Save the transcription result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Transcription completed and saved to {output_path}")
            return result
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        raise

def translate_text(transcription: Dict, output_path: str, model_name: str = "facebook/nllb-200-distilled-600M", use_llm: bool = False) -> Dict | None:
    """
    Translate English transcription to Hindi using either NLLB or Mistral.
    
    Args:
        transcription: Dictionary containing transcription results from ASR
        output_path: Path to save the translation JSON
        model_name: NLLB Translation model name (used if use_llm is False)
        use_llm: Whether to use Mistral LLM for translation instead of NLLB
        
    Returns:
        Dictionary with original transcription and translated segments, or None on failure.
    """
    logger.info(f"Starting translation. Using LLM: {use_llm}")
    
    if use_llm and translate_mistral is None:
        logger.error("Mistral translation requested (--use_llm) but module failed to import.")
        return None

    segments = transcription.get("segments", [])
    if not segments:
        logger.warning("No segments found in transcription for translation.")
        # Return structure with empty list if no segments
        output_data = {
            "original": transcription,
            "translated_segments": []
        }
        return output_data

    translated_segments = []

    if use_llm:
        # --- Use Mistral LLM for Translation ---
        logger.info("Using Mistral API for translation.")
        for segment in tqdm(segments, desc="Translating Segments (Mistral)"):
            english_text = segment.get("text", "")
            start_time = segment.get("start")
            end_time = segment.get("end")
            speaker = segment.get("speaker", "SPEAKER_00") # Default if not present

            if not english_text:
                logger.warning(f"Segment {segment.get('id', '?')} has empty text. Skipping translation.")
                hindi_text = ""
            else:
                hindi_text = translate_mistral(english_text, target_language_code="hi")
                if hindi_text is None:
                    logger.error(f"Mistral translation failed for segment: '{english_text[:50]}...'")
                    hindi_text = "[Translation Failed]" # Placeholder
            
            translated_segments.append({
                "start": start_time,
                "end": end_time,
                "english": english_text,
                "hindi": hindi_text,
                "speaker": speaker
            })
    else:
        # --- Use NLLB Model for Translation ---
        logger.info(f"Using NLLB model '{model_name}' for translation.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info(f"NLLB model loaded onto device: {device}")
        except Exception as e:
            logger.error(f"Failed to load NLLB model '{model_name}': {e}")
            return None

        # Group segments by speaker for potentially better context (optional)
        # For now, translate segment by segment
        for segment in tqdm(segments, desc="Translating Segments (NLLB)"):
            english_text = segment.get("text", "")
            start_time = segment.get("start")
            end_time = segment.get("end")
            speaker = segment.get("speaker", "SPEAKER_00")

            if not english_text:
                logger.warning(f"Segment {segment.get('id', '?')} has empty text. Skipping translation.")
                hindi_text = ""
            else:
                try:
                    # NLLB expects source language code
                    tokenizer.src_lang = "eng_Latn"
                    inputs = tokenizer(english_text, return_tensors="pt").to(device)
                    
                    # Translate to Hindi
                    translated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"]
                    )
                    hindi_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    logger.debug(f"NLLB Translated: '{english_text[:30]}...' -> '{hindi_text[:30]}...'")
                except Exception as e:
                    logger.error(f"NLLB translation failed for segment '{english_text[:50]}...': {e}")
                    hindi_text = "[Translation Failed]"
            
            translated_segments.append({
                "start": start_time,
                "end": end_time,
                "english": english_text,
                "hindi": hindi_text,
                "speaker": speaker
            })

    # Final output structure
    output_data = {
        "original": transcription,
        "translated_segments": translated_segments
    }
    
    # Save the results
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "translation.json" 
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Translation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save translation results: {e}")
        return None
        
    return output_data

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

def run_latentsync(video_path: str, audio_path: str, output_video_path: str, latentsync_script_path: str, unet_config_path: str, ckpt_path: str, use_gpu: bool = False, gpu_index: int = 0) -> str:
    """
    Run LatentSync inference to create a lip-synced video.

    Args:
        video_path: Path to the original input video.
        audio_path: Path to the target audio (dubbed audio).
        output_video_path: Path to save the lip-synced video.
        latentsync_script_path: Path to the LatentSync inference script.
        unet_config_path: Path to the UNet config file.
        ckpt_path: Path to the LatentSync checkpoint file.
        use_gpu: Whether to use GPU.
        gpu_index: Index of the GPU to use.

    Returns:
        Path to the generated lip-synced video.
    """
    logger.info("Step 9: Running LatentSync for lip-syncing")
    logger.info(f"  Input Video: {video_path}")
    logger.info(f"  Input Audio: {audio_path}")
    logger.info(f"  Output Video: {output_video_path}")

    # Ensure LatentSync script exists
    if not os.path.exists(latentsync_script_path):
        logger.error(f"LatentSync script not found at: {latentsync_script_path}")
        raise FileNotFoundError(f"LatentSync script not found: {latentsync_script_path}")

    # Construct the command
    # Note: LatentSync might need to be run from its own directory or have paths adjusted
    # Assuming the script can handle paths relative to the project root or absolute paths.
    cmd = [
        sys.executable, # Use the same python executable
        latentsync_script_path,
        "--unet_config_path", unet_config_path,
        "--inference_ckpt_path", ckpt_path,
        "--inference_steps", "20", # Default from inference.sh
        "--guidance_scale", "2.0", # Default from inference.sh
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_video_path
    ]

    # Handle GPU selection using CUDA_VISIBLE_DEVICES if requested
    env = os.environ.copy()
    original_cuda_device = env.get("CUDA_VISIBLE_DEVICES")
    
    # Add LatentSync directory to PYTHONPATH for the subprocess
    latentsync_root_dir = os.path.abspath(LATENTSYNC_DIR)
    original_pythonpath = env.get("PYTHONPATH")
    if original_pythonpath:
        env["PYTHONPATH"] = f"{latentsync_root_dir}:{original_pythonpath}"
    else:
        env["PYTHONPATH"] = latentsync_root_dir
    logger.info(f"Set PYTHONPATH for subprocess: {env['PYTHONPATH']}")

    try:
        # Set CUDA_VISIBLE_DEVICES
        if use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            logger.info(f"Set CUDA_VISIBLE_DEVICES to {gpu_index} for LatentSync")
        else:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.info("Disabling GPU for LatentSync (CUDA_VISIBLE_DEVICES=-1)")
            
        # Run the LatentSync script
        latentsync_base_dir = os.path.dirname(latentsync_script_path)
        if not latentsync_base_dir: # Handle case where script path is just the filename
            latentsync_base_dir = "."
        
        logger.info(f"Running LatentSync command: {' '.join(cmd)}")
        logger.info(f"Working directory for LatentSync: {LATENTSYNC_DIR}")

        # Convert relevant paths to be absolute
        abs_video_path = os.path.abspath(video_path)
        abs_audio_path = os.path.abspath(audio_path)
        abs_output_video_path = os.path.abspath(output_video_path)
        abs_unet_config_path = os.path.abspath(unet_config_path)
        abs_ckpt_path = os.path.abspath(ckpt_path)

        # Update command with absolute paths
        cmd = [
            sys.executable, latentsync_script_path,
            "--unet_config_path", abs_unet_config_path,
            "--inference_ckpt_path", abs_ckpt_path,
            "--inference_steps", "20",
            "--guidance_scale", "2.0",
            "--video_path", abs_video_path,
            "--audio_path", abs_audio_path,
            "--video_out_path", abs_output_video_path
        ]

        process = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            env=env,
            cwd=LATENTSYNC_DIR # Run from LatentSync directory
        )
        logger.info("LatentSync stdout:")
        logger.info(process.stdout)
        if process.stderr:
            logger.warning("LatentSync stderr:")
            logger.warning(process.stderr)
            
        logger.info(f"LatentSync lip-sync completed. Output video: {output_video_path}")
        return output_video_path
        
    except FileNotFoundError as e:
        logger.error(f"Error running LatentSync: {e}. Ensure Python executable and script path are correct.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"LatentSync script failed with exit code {e.returncode}")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error("stdout:")
        logger.error(e.stdout)
        logger.error("stderr:")
        logger.error(e.stderr)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during LatentSync execution: {str(e)}")
        raise
    finally:
        # Restore original CUDA_VISIBLE_DEVICES setting
        # This block MUST be at the same indentation level as the 'try' and 'except' blocks
        if original_cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_device
            logger.info(f"Restored CUDA_VISIBLE_DEVICES to {original_cuda_device}")
        else:
            # If it wasn't set before, remove it
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            logger.info("Removed CUDA_VISIBLE_DEVICES setting")

        # Restore original PYTHONPATH
        if original_pythonpath is not None:
            os.environ["PYTHONPATH"] = original_pythonpath
            logger.info(f"Restored PYTHONPATH to: {original_pythonpath}")
        elif "PYTHONPATH" in os.environ: # Check if we added it
            del os.environ["PYTHONPATH"]
            logger.info("Removed PYTHONPATH setting")

def run_pipeline(args) -> Dict:
    """
    Run the complete pipeline from video to Hindi TTS segments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with pipeline results
    """
    # Setup GPU device if needed
    if args.use_gpu and hasattr(args, 'gpu_index'):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        logger.info(f"Using GPU index: {args.gpu_index}")
    
    # Setup output directories
    dirs = setup_directories(args.output_dir)
    logger.info(f"Pipeline run started with video: {args.video_path}")
    logger.info(f"Output directory: {dirs['base']}")
    
    # File paths for various outputs
    audio_path = os.path.join(dirs["audio"], "audio.wav")
    speech_path = os.path.join(dirs["separated_audio"], "speech.wav")
    background_path = os.path.join(dirs["separated_audio"], "background.wav")
    transcription_path = os.path.join(dirs["transcriptions"], "transcription.json")
    translation_path = os.path.join(dirs["translations"], "translation.json")
    diarization_path = os.path.join(dirs["diarization"], "diarization.json")
    result_path = os.path.join(dirs["results"], "pipeline_result.json")
    combined_audio_path = os.path.join(dirs["tts_output"], f"{Path(args.video_path).stem}_final_audio.wav")
    
    # Step 1: Extract audio from the video
    logger.info("Step 1: Extracting audio from video")
    try:
        # Extract audio using FFmpeg
        extract_audio_cmd = [
            "ffmpeg", "-y", "-i", args.video_path, 
            "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", audio_path
        ]
        subprocess.run(extract_audio_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Audio extracted to {audio_path}")
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise
    
    # Step 2: Separate speech from background music
    logger.info("Step 2: Separating speech from background music")
    try:
        # Use the audio separation function
        separation_result = extract_and_separate_audio(
            video_path=args.video_path,
            output_dir=dirs["separated_audio"],
            use_gpu=args.use_gpu
        )
        
        # Get paths to separated audio
        speech_path = separation_result["speech"]
        background_path = separation_result["background"]
        logger.info(f"Speech audio saved to: {speech_path}")
        logger.info(f"Background audio saved to: {background_path}")
    except ImportError:
        logger.warning("Audio separation module not available. Using full audio for processing.")
        speech_path = audio_path
    except Exception as e:
        logger.error(f"Error during audio separation: {str(e)}")
        logger.warning("Continuing with full audio for processing")
        speech_path = audio_path
        
    # Step 3: Perform speaker diarization (optional)
    logger.info("Step 3: Performing speaker diarization")
    if not args.skip_diarization:
        # Get auth token from environment variable if not provided in args
        auth_token = args.pyannote_token
        if not auth_token:
            auth_token = os.environ.get('auth_token')
            if auth_token:
                logger.info("Using auth token from environment variable")
            
        diarization_result = perform_speaker_diarization(
            audio_path=speech_path, 
            output_path=diarization_path,
            auth_token=auth_token
        )
    else:
        logger.info("Speaker diarization skipped")
        diarization_path = None
        
    # Step 4: Transcribe the audio
    logger.info("Step 4: Transcribing audio")
    transcription = transcribe_audio(
        speech_path, 
        transcription_path, 
        diarization_path if not args.skip_diarization else None,
        args.asr_model,
        max_segment_length=args.max_segment_length if hasattr(args, 'max_segment_length') else 30,
        use_gpu=args.use_gpu,
        gpu_index=args.gpu_index if hasattr(args, 'gpu_index') else 0
    )
    
    # Step 5: Extract reference samples for voice cloning OR use provided directory
    logger.info("Step 5: Preparing reference samples")
    if args.reference_audio_dir:
        logger.info(f"Using provided reference audio directory: {args.reference_audio_dir}")
        reference_samples_dir = args.reference_audio_dir
        if not os.path.isdir(reference_samples_dir):
            logger.error(f"Provided reference audio directory does not exist: {reference_samples_dir}")
            raise FileNotFoundError(f"Reference audio directory not found: {reference_samples_dir}")

        reference_files = sorted([
            os.path.join(reference_samples_dir, f) 
            for f in os.listdir(reference_samples_dir) 
            if f.lower().endswith(".wav")
        ])
        
        if not reference_files:
            logger.error(f"No .wav files found in the provided reference audio directory: {reference_samples_dir}")
            raise ValueError(f"No .wav files found in {reference_samples_dir}")

        # Assume single speaker 'speaker_0' when using a directory
        speaker_id = "speaker_0" 
        reference_samples = {
            "speakers": {
                speaker_id: []
            },
            "video_id": os.path.basename(args.video_path),
            "reference_metadata_path": None # No metadata file when provided directly
        }
        
        for ref_file in reference_files:
            # Create basic metadata. Duration could be calculated using ffprobe if needed later.
            sample_metadata = {
                "path": ref_file,
                "start": 0.0, "end": 0.0, "duration": 0.0, # Placeholders
                "text": "", "features": {}, "emotion_label": "neutral"
            }
            reference_samples["speakers"][speaker_id].append(sample_metadata)

        logger.info(f"Loaded {len(reference_files)} reference samples for speaker '{speaker_id}' from {reference_samples_dir}")
        
        # Save basic metadata to the reference_samples directory for consistency
        metadata_path = os.path.join(dirs["reference_samples"], "reference_samples_metadata.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(reference_samples["speakers"], f, ensure_ascii=False, indent=2)
            reference_samples["reference_metadata_path"] = metadata_path
            logger.info(f"Saved basic reference metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save reference metadata: {e}")

    else:
        logger.info("Extracting reference samples automatically")
        try:
            reference_samples = extract_reference_samples(
                audio_path=speech_path,
                output_dir=dirs["reference_samples"],
                transcription_path=transcription_path,
                diarization_path=diarization_path if not args.skip_diarization else None
            )
        except Exception as e:
            logger.error(f"Error extracting reference samples: {str(e)}")
            # Fallback or raise error depending on desired behavior
            # For now, let's raise to make the failure explicit
            raise RuntimeError(f"Failed to extract reference samples: {e}")
            # Alternatively, create basic reference samples:
            # reference_samples = create_basic_reference_samples(speech_path, dirs["reference_samples"])
        
    # Step 6: Translate English text to Hindi
    logger.info("Step 6: Translating transcription to Hindi")
    translation_result = translate_text(
        transcription=transcription,
        output_path=translation_path,
        model_name=args.translation_model,
        use_llm=args.use_llm
    )
    
    # Ensure translated segments have timing information from transcription
    if "segments" in transcription and "translated_segments" in translation_result:
        logger.info("Ensuring timing information is preserved in translated segments")
        for i, trans_segment in enumerate(translation_result["translated_segments"]):
            if i < len(transcription["segments"]):
                src_segment = transcription["segments"][i]
                if "start" in src_segment and "end" in src_segment:
                    if "start" not in trans_segment:
                        trans_segment["start"] = src_segment["start"]
                        logger.info(f"Adding start time {src_segment['start']} to translated segment {i}")
                    if "end" not in trans_segment:
                        trans_segment["end"] = src_segment["end"]
                        logger.info(f"Adding end time {src_segment['end']} to translated segment {i}")
    
    # Step 7: Generate TTS output for each segment
    logger.info("Step 7: Generating TTS output for translated segments")
    try:
        tts_segments = generate_tts_for_segments(
            translation=translation_result,
            reference_samples=reference_samples,
            output_dir=dirs["tts_output"],
            use_gpu=args.use_gpu,
            gpu_index=args.gpu_index if hasattr(args, 'gpu_index') else 0,
            use_standard_tts=args.use_standard_tts
        )
        
        # Double-check that TTS segments have timing information
        logger.info("Verifying timing information in TTS segments")
        for i, segment in enumerate(tts_segments):
            if "start" not in segment or "end" not in segment:
                # Try to get timing from translated segments
                if i < len(translation_result["translated_segments"]):
                    trans_segment = translation_result["translated_segments"][i]
                    if "start" in trans_segment:
                        segment["start"] = trans_segment["start"]
                        logger.info(f"Added missing start time {trans_segment['start']} to TTS segment {i}")
                    if "end" in trans_segment:
                        segment["end"] = trans_segment["end"]
                        logger.info(f"Added missing end time {trans_segment['end']} to TTS segment {i}")
        
        logger.info(f"Generated TTS for {len(tts_segments)} segments")
    except Exception as e:
        logger.error(f"Error during TTS generation: {str(e)}")
        raise
    
    # Step 8: Combine TTS segments with background audio
    logger.info("Step 8: Combining TTS segments")
    speech_only_output_path = os.path.join(dirs["tts_output"], f"{Path(args.video_path).stem}_speech_only.wav")
    try:
        # Verify we have segments with timing information
        if not tts_segments:
            logger.error("No TTS segments to combine")
        else:
            logger.info(f"Combining {len(tts_segments)} TTS segments")
            # Check if segments have timing info
            has_timing = all("start" in segment and "end" in segment for segment in tts_segments)
            if not has_timing:
                logger.warning("Some segments missing timing information - will use sequential placement")
            
            # Print first few segments for debugging
            for i, segment in enumerate(tts_segments[:3]):
                logger.info(f"Segment {i}: start={segment.get('start', 'N/A')}, end={segment.get('end', 'N/A')}")
                
                # Check for tts_output key or construct path
                if "tts_output" not in segment:
                    segment["tts_output"] = os.path.join(dirs["tts_output"], f"segment_{i:04d}.wav")
                    logger.info(f"Added tts_output path: {segment['tts_output']}")
                
                # Make sure translations are added to the segments too
                if "translated_segments" in translation_result and i < len(translation_result["translated_segments"]):
                    trans_segment = translation_result["translated_segments"][i]
                    if "start" in trans_segment and "end" in trans_segment:
                        segment["start"] = trans_segment["start"]
                        segment["end"] = trans_segment["end"]
                    
                    # Add Hindi text for debugging
                    if "hindi" in trans_segment:
                        segment["hindi_text"] = trans_segment["hindi"]
        
            # Get original audio duration if needed
            original_duration = None
            try:
                result = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", audio_path
                ], capture_output=True, text=True, check=True)
                original_duration = float(result.stdout.strip())
                logger.info(f"Original audio duration: {original_duration} seconds")
            except Exception as e:
                logger.warning(f"Error getting original audio duration: {str(e)}")
                logger.warning("Will calculate duration from segments")
            
            # If mixing with background, use background audio
            if args.mix_background:
                # Use a very low background volume to ensure Hindi speech is clear
                background_vol = 0.05  # Very low background volume
                if hasattr(args, 'background_volume'):
                    background_vol = min(args.background_volume, 0.1)  # Cap at 0.1 to ensure Hindi is clear
                    
                logger.info(f"Using background volume level: {background_vol} to ensure Hindi speech is clear")
                
                # Ensure the background path exists
                if not os.path.exists(background_path):
                    logger.warning(f"Background audio file not found: {background_path}")
                    logger.warning("Will try to use the original audio instead")
                    background_path = audio_path
                
                # Combine segments with background
                mixed_audio_output_path = os.path.join(dirs["tts_output"], f"{Path(args.video_path).stem}_final_audio.wav")
                combined_audio = combine_audio_segments(
                    tts_segments=tts_segments,
                    background_audio_path=background_path,
                    output_path=mixed_audio_output_path,
                    tts_output_dir=dirs["tts_output"],
                    original_duration=original_duration,
                    adjust_volume=True,
                    background_volume=background_vol
                )
                
                if combined_audio:
                    logger.info(f"Final audio with background generated: {combined_audio}")
                    combined_audio_path = combined_audio
                else:
                    logger.error("Failed to generate combined audio with background")
                    # Keep speech-only output as fallback
            else:
                logger.info(f"Skipping background audio mixing (generating speech-only output)")
                
                # Combine segments without background
                speech_only_output_path = os.path.join(dirs["tts_output"], f"{Path(args.video_path).stem}_speech_only.wav")
                combined_audio = combine_audio_segments(
                    tts_segments=tts_segments,
                    background_audio_path=None,  # No background
                    output_path=speech_only_output_path,
                    tts_output_dir=dirs["tts_output"],
                    original_duration=original_duration,
                    adjust_volume=False,
                    background_volume=0.0
                )
                
                if combined_audio:
                    logger.info(f"Speech-only audio generated: {combined_audio}")
                    combined_audio_path = combined_audio
                else:
                    logger.error("Failed to generate combined speech-only audio")
    except Exception as e:
        logger.error(f"Error combining audio segments: {str(e)}")
        logger.warning("Continuing without combined audio")
        combined_audio_path = None
        speech_only_output_path = None
    
    # Step 9: Run LatentSync for lip-sync (if combined audio was generated)
    final_lip_synced_video_path = None
    if combined_audio_path and os.path.exists(combined_audio_path):
        # Run LatentSync unless skipped
        if not args.skip_latentsync: 
            # Construct output path for the lip-synced video
            video_basename = Path(args.video_path).stem
            lip_sync_output_path = os.path.join(dirs["results"], f"{video_basename}_lipsync.mp4")
            
            try:
                final_lip_synced_video_path = run_latentsync(
                    video_path=args.video_path, 
                    audio_path=combined_audio_path, 
                    output_video_path=lip_sync_output_path,
                    latentsync_script_path=args.latentsync_script_path,
                    unet_config_path=args.latentsync_unet_config_path,
                    ckpt_path=args.latentsync_ckpt_path,
                    use_gpu=args.use_gpu, 
                    gpu_index=args.gpu_index
                )
            except Exception as e:
                logger.error(f"LatentSync step failed: {str(e)}")
                # Continue without lip-sync video
        else:
            logger.info("Skipping LatentSync step as --skip_latentsync flag was provided.")
    else:
        logger.warning("Skipping LatentSync step because combined audio was not generated.")
    
    # Save final results
    result = {
        "video_path": args.video_path,
        "audio_path": audio_path,
        "speech_path": speech_path,
        "background_path": background_path,
        "transcription_path": transcription_path,
        "translation_path": translation_path,
        "diarization_path": diarization_path,
        "reference_metadata_path": reference_samples.get("reference_metadata_path"),
        "speakers": reference_samples.get("speakers", {}),
        "tts_output_dir": dirs["tts_output"],
        "tts_segments": tts_segments,
        "combined_audio_path": combined_audio_path,
        "final_lip_synced_video_path": final_lip_synced_video_path,
        "output_base_dir": dirs["base"],
        "completed_at": datetime.datetime.now().isoformat()
    }
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Pipeline completed successfully")
    logger.info(f"Results saved to {result_path}")
    
    return result

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="English to Hindi Video Dubbing Pipeline")
    
    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", default="processed_data", help="Directory to save all outputs")
    parser.add_argument("--asr_model", default="large-v3", help="Whisper ASR model name (e.g., tiny, base, small, medium, large, large-v2, large-v3)")
    parser.add_argument("--translation_model", default="facebook/nllb-200-distilled-600M", help="NLLB translation model name")
    parser.add_argument("--pyannote_token", help="PyAnnotate API token for speaker diarization")
    parser.add_argument("--reference_audio_dir", default=None, help="Optional path to a directory containing reference .wav files for TTS.")
    parser.add_argument("--skip_diarization", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--mix_background", action="store_true", help="Mix background audio into final output (default: False)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for processing (if available)")
    parser.add_argument("--gpu_index", type=int, default=0, help="Index of the GPU to use if multiple GPUs are available")
    parser.add_argument("--max_segment_length", type=int, default=30, help="Maximum words per segment")
    parser.add_argument("--background_volume", type=float, default=0.2, help="Background audio volume level (0.0-1.0)")
    parser.add_argument("--use_standard_tts", action="store_true", help="Use standard TTS instead of voice cloning.")
    parser.add_argument("--use_llm", action="store_true", help="Use Mistral LLM for translation instead of NLLB.")
    
    # --- LatentSync Arguments ---
    # Changed to --skip_latentsync, default is to run
    parser.add_argument("--skip_latentsync", action="store_true", help="Skip the LatentSync lip-syncing step.")
    parser.add_argument("--latentsync_script_path", default=os.path.join(LATENTSYNC_DIR, "scripts", "inference.py"), help="Path to the LatentSync inference script.")
    parser.add_argument("--latentsync_unet_config_path", default=os.path.join(LATENTSYNC_DIR, "configs", "unet", "stage2.yaml"), help="Path to the LatentSync UNet config.")
    parser.add_argument("--latentsync_ckpt_path", default=os.path.join(LATENTSYNC_DIR, "checkpoints", "latentsync_unet.pt"), help="Path to the LatentSync checkpoint file.")
    
    args = parser.parse_args()
    
    # Configure logging to file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Load tokens from environment variables if not provided
    if not args.pyannote_token:
        args.pyannote_token = os.environ.get('auth_token')
        if args.pyannote_token:
            logger.info("Using PyAnnote token from environment variable")
    
    logger.info(f"Starting pipeline with arguments: {args}")
    
    try:
        result = run_pipeline(args)
        logger.info("Pipeline completed successfully")
        
        # Print a nice summary
        print("\n" + "="*50)
        print("English-to-Hindi Dubbing Pipeline: Complete")
        print("="*50)
        print(f"Input video: {args.video_path}")
        print(f"Generated TTS segments: {len(result['tts_segments'])}")
        
        if result.get("combined_audio_path") and os.path.exists(result["combined_audio_path"]):
            print(f"Final audio output: {result['combined_audio_path']}")
        
        if result.get("final_lip_synced_video_path") and os.path.exists(result["final_lip_synced_video_path"]):
            print(f"Final lip-synced video output: {result['final_lip_synced_video_path']}")
        
        print(f"All outputs saved in: {result['output_base_dir']}")
        print("="*50 + "\n")
        
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
"""ASR module using OpenAI Whisper."""

import json
import time
from pathlib import Path

from src.utils.logger import logger

try:
    import torch
    import whisper
except ImportError:
    logger.warning("Whisper not installed. ASR will not be available.")

# Define sentence ending punctuation globally for reuse
SENTENCE_END_CHARS = {".", "!", "?", "।", "॥"}

def transcribe_audio(
    audio_path, 
    output_path=None, 
    model_name="large-v3", 
    language="en",
    word_timestamps=True,
    diarization_path=None,
    fix_segmentation=True,
    max_segment_length=30,
    use_gpu=True,
    gpu_index=0
):
    """Transcribe audio using OpenAI Whisper.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path for the output JSON file. If None, a path is generated.
        model_name: Whisper model to use (tiny, base, small, medium, large-v3)
        language: Language code (default: en)
        word_timestamps: Whether to include word-level timestamps
        diarization_path: Optional path to diarization results to add speaker labels
        fix_segmentation: Whether to fix segmentation at sentence boundaries
        max_segment_length: Maximum number of words per segment (default: 30)
        use_gpu: Whether to use GPU for transcription
        gpu_index: Index of the GPU to use for transcription (default: 0)
        
    Returns:
        str: Path to the transcription results JSON file
    """
    if 'whisper' not in globals():
        raise ImportError("Whisper not installed. Please install it with: "
                         "pip install openai-whisper")
    
    audio_path = Path(audio_path)
    
    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("processed_data/transcriptions")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{audio_path.stem}_transcription.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Transcribing audio {audio_path} using Whisper {model_name}")
    start_time = time.time()
    
    try:
        # Check if CUDA is available and gpu is requested
        if torch.cuda.is_available() and use_gpu:
            # Set the device to the specified GPU index
            if gpu_index >= 0 and gpu_index < torch.cuda.device_count():
                device = f"cuda:{gpu_index}"
                logger.info(f"Using GPU {gpu_index} for ASR")
            else:
                # Fall back to the first GPU if the specified one is not available
                device = "cuda:0"
                logger.warning(f"GPU index {gpu_index} not available. Using GPU 0 instead.")
        else:
            device = "cpu"
            if use_gpu and not torch.cuda.is_available():
                logger.warning("GPU requested but CUDA is not available. Using CPU instead.")
            
        logger.info(f"Using device: {device} for ASR")
        
        # Load Whisper model
        model = whisper.load_model(model_name, device=device)
        
        # Transcribe audio
        result = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            verbose=False
        )
        
        # Post-process segmentation if requested
        if fix_segmentation:
            result = fix_sentence_segmentation(result, max_segment_length=max_segment_length)
        
        # If diarization results are provided, merge them with transcription
        if diarization_path:
            result = add_speaker_labels(result, diarization_path)
            
            # Re-apply segmentation after adding speaker labels to ensure speaker changes are respected
            if fix_segmentation:
                result = fix_sentence_segmentation(result, max_segment_length=max_segment_length)
        
        # Save transcription results to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Free up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
        
        duration = time.time() - start_time
        logger.info(f"Transcription completed in {duration:.2f} seconds")
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise

def add_speaker_labels(transcription_result, diarization_path):
    """Add speaker labels to transcription results using diarization data.
    
    Args:
        transcription_result: Whisper transcription result
        diarization_path: Path to diarization JSON file
        
    Returns:
        dict: Transcription result with speaker labels added
    """
    try:
        # Load diarization results
        diarization_path = Path(diarization_path)
        with open(diarization_path, 'r') as f:
            diarization = json.load(f)
        
        # Create a copy of the transcription result
        result = transcription_result.copy()
        
        # Add speaker labels to segments
        for segment in result.get("segments", []):
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Find the speaker who speaks the most during this segment
            speaker_durations = {}
            
            for diar_segment in diarization:
                diar_start = diar_segment["start"]
                diar_end = diar_segment["end"]
                diar_speaker = diar_segment["speaker"]
                
                # Calculate overlap
                overlap_start = max(start_time, diar_start)
                overlap_end = min(end_time, diar_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > 0:
                    speaker_durations[diar_speaker] = \
                        speaker_durations.get(diar_speaker, 0) + overlap_duration
            
            # Assign the speaker with the longest duration
            if speaker_durations:
                segment["speaker"] = max(speaker_durations, key=speaker_durations.get)
            else:
                segment["speaker"] = "unknown"
        
        logger.info(f"Added speaker labels to {len(result.get('segments', []))} segments")
        return result
    
    except Exception as e:
        logger.warning(f"Failed to add speaker labels: {str(e)}")
        return transcription_result 

def fix_sentence_segmentation(transcription_result, max_segment_length=30, soft_max_length=20):
    """Fix segmentation at sentence boundaries in Whisper transcription.
    
    This function re-segments the transcription to ensure segments end at:
    1. Sentence-ending punctuation marks (., !, ?, ।, ॥)
    2. Speaker changes
    3. Commas, if segment length exceeds soft_max_length
    4. Maximum segment length (in words)
    
    Args:
        transcription_result: Whisper transcription result dict
        max_segment_length: Maximum number of words in a segment (hard limit)
        soft_max_length: Preferred maximum number of words before splitting at a comma
        
    Returns:
        dict: Transcription with improved segmentation
    """
    logger.info(f"Fixing sentence segmentation with max_segment_length={max_segment_length}, soft_max_length={soft_max_length}")
    
    try:
        result = transcription_result.copy()
        original_segments = result.get("segments", [])
        
        if not original_segments:
            logger.warning("No segments found in transcription")
            return result
            
        new_segments = []
        current_words = []
        current_start = None
        current_speaker = None
        current_text = ""
        
        for i, segment in enumerate(original_segments):
            segment_speaker = segment.get("speaker", "unknown")
            segment_text = segment.get("text", "").strip()
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            segment_words = segment.get("words", [])
            
            # Skip empty segments
            if not segment_text:
                continue
                
            # Initialize if first segment or after a reset
            if current_start is None:
                current_start = segment_start
                current_speaker = segment_speaker
                
            # Check for speaker change
            speaker_changed = current_speaker != segment_speaker
            
            # If speaker changed, finalize the previous segment
            if speaker_changed and current_text:
                new_segment = {
                    "text": current_text.strip(),
                    "start": current_start,
                    # Use the start time of the *current* segment as the end of the previous one
                    "end": segment_start,
                    "speaker": current_speaker,
                    "words": current_words
                }
                new_segments.append(new_segment)
                logger.debug(f"Created segment at speaker change: {new_segment['text'][:50]}...")
                
                # Reset for new speaker
                current_text = ""
                current_words = []
                current_start = segment_start
                current_speaker = segment_speaker
                
            # Process words with their timestamps
            for word_info in segment_words:
                word_text = word_info.get("word", "").strip()
                # Handle potential missing start/end in word_info robustly
                word_start = word_info.get("start", current_start if current_start is not None else segment_start)
                word_end = word_info.get("end", word_start + 0.1) # Estimate end if missing
                
                # Update current_start if this is the very first word being processed
                if current_start is None:
                    current_start = word_start
                
                # Append word info
                current_text += " " + word_text
                current_words.append(word_info)
                
                # Determine the end time for potential segment creation
                # Use the word's end time if available, otherwise estimate based on the segment
                segment_end_time = word_end if word_end > word_start else (current_words[-2].get("end", word_start) if len(current_words) > 1 else word_start)
                
                # Check conditions for creating a new segment
                split_reason = None
                if any(word_text.endswith(char) for char in SENTENCE_END_CHARS):
                    split_reason = "sentence end"
                elif word_text.endswith(",") and len(current_words) > soft_max_length:
                     split_reason = "comma soft limit"
                elif len(current_words) >= max_segment_length:
                    split_reason = "max length"
                
                if split_reason:
                    new_segment = {
                        "text": current_text.strip(),
                        "start": current_start,
                        "end": segment_end_time, # Use calculated end time
                        "speaker": current_speaker,
                        "words": current_words
                    }
                    new_segments.append(new_segment)
                    logger.debug(f"Created segment at {split_reason}: {new_segment['text'][:50]}...")
                    
                    # Reset for next segment
                    current_text = ""
                    current_words = []
                    # Start next segment after the current word's end time
                    current_start = segment_end_time
            
        # Add any remaining text as the last segment
        if current_text:
            # Use the end time of the last word or segment
            last_end = current_words[-1].get("end", segment_end if 'segment_end' in locals() else current_start + 1.0) if current_words else (segment_end if 'segment_end' in locals() else current_start + 1.0)
            
            new_segment = {
                "text": current_text.strip(),
                "start": current_start,
                "end": last_end,
                "speaker": current_speaker,
                "words": current_words
            }
            new_segments.append(new_segment)
            logger.debug(f"Created final segment: {new_segment['text'][:50]}...")
            
        # Replace the original segments and update the full text
        result["segments"] = new_segments
        result["text"] = " ".join(s["text"] for s in new_segments if s["text"])
        
        logger.info(f"Segmentation complete: {len(original_segments)} original segments -> {len(new_segments)} new segments")
        return result
        
    except Exception as e:
        logger.error(f"Error during sentence segmentation: {str(e)}", exc_info=True)
        # Return original result on error to avoid breaking the pipeline
        return transcription_result 
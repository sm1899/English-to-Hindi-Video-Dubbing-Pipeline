"""
Pipeline Integration for Reference Sampling

This module integrates the reference sampling component into the main pipeline,
acting as a step between ASR/Diarization and TTS.
"""

import os
import json
import logging
from pathlib import Path

from .sample_selector import extract_reference_samples, extract_reference_samples_from_asr

# Configure logging
from src.utils.logger import logger

def process_reference_sampling(
    video_id: str,
    audio_path=None,
    output_path=None,
    asr_path=None,
    diarization_path=None
):
    """
    Process a video's audio to extract reference samples for voice cloning.
    Uses ASR output when available for improved segmentation.
    
    Args:
        video_id: Unique identifier for the video
        audio_path: Path to the extracted audio file. If None, a path is generated.
        output_path: Base directory for reference samples. If None, a path is generated.
        asr_path: Path to ASR output JSON file. If None, a path is generated.
        diarization_path: Path to diarization JSON file. If None, a path is generated if available.
        
    Returns:
        Dictionary with paths to reference samples
    """
    # Convert strings to Path objects
    if video_id.endswith('.mp4') or video_id.endswith('.wav'):
        video_id = Path(video_id).stem
    
    # Generate paths if not provided
    if audio_path is None:
        audio_path = Path(f"processed_data/audio/{video_id}.wav")
    else:
        audio_path = Path(audio_path)
    
    if asr_path is None:
        asr_path = Path(f"processed_data/transcriptions/{video_id}_transcription.json")
    else:
        asr_path = Path(asr_path)
    
    if diarization_path is None:
        potential_diarization_path = Path(f"processed_data/diarization/{video_id}.json")
        if potential_diarization_path.exists():
            diarization_path = potential_diarization_path
    elif diarization_path:
        diarization_path = Path(diarization_path)
    
    # Generate output directory if not provided
    if output_path is None:
        output_dir = Path(f"processed_data/reference_samples/{video_id}")
    else:
        output_dir = Path(output_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting reference samples for video: {video_id}")
    logger.info(f"Audio path: {audio_path}")
    logger.info(f"ASR path: {asr_path}")
    logger.info(f"Diarization path: {diarization_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Verify input files exist
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check if we have ASR output to use
    if asr_path.exists():
        logger.info(f"Using ASR output for reference sampling")
        
        # Extract reference samples using ASR output
        reference_samples = extract_reference_samples_from_asr(
            str(audio_path),
            str(asr_path),
            str(output_dir),
            str(diarization_path) if diarization_path and diarization_path.exists() else None
        )
    else:
        logger.info("No ASR output found. Using basic segmentation for reference sampling.")
        
        # Extract reference samples using basic approach
        reference_samples = extract_reference_samples(
            str(audio_path),
            str(output_dir),
            str(diarization_path) if diarization_path and diarization_path.exists() else None
        )
    
    # Create summary for pipeline integration
    pipeline_summary = {
        'video_id': video_id,
        'reference_metadata_path': str(output_dir / "reference_samples_metadata.json"),
        'speakers': {}
    }
    
    # Add each speaker's samples
    for speaker_id, samples in reference_samples.items():
        # Group samples by emotion
        emotion_groups = {}
        for sample in samples:
            emotion = sample.get('emotion_label', 'unknown')
            if emotion not in emotion_groups:
                emotion_groups[emotion] = []
            emotion_groups[emotion].append(sample['path'])
            
        speaker_sample_paths = [sample['path'] for sample in samples]
        pipeline_summary['speakers'][speaker_id] = {
            'reference_sample_paths': speaker_sample_paths,
            'sample_count': len(speaker_sample_paths),
            'emotion_groups': emotion_groups
        }
    
    # Save pipeline summary
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(pipeline_summary, f, indent=2)
    
    logger.info(f"Reference sampling complete. Found {len(reference_samples)} speaker(s)")
    logger.info(f"Pipeline summary saved to: {summary_path}")
    
    return pipeline_summary 
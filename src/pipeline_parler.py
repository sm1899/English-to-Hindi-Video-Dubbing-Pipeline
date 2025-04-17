#!/usr/bin/env python3
"""
English to Hindi Video Dubbing Pipeline with Local TTS

This script orchestrates the process of dubbing a video from English to Hindi using a local TTS model:
1. Extracts audio from the input video
2. Separates speech from background music/noise
3. Transcribes the speech audio to English text
4. Translates the English text to Hindi
5. Generates Hindi speech using a local TTS model with the speaker's voice characteristics
6. Combines speech segments with background audio
7. Outputs combined Hindi audio file

Usage:
    python pipeline_local_tts.py --video_path path/to/video.mp4 --output_dir output/ --use_gpu --model_path path/to/model
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

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules from src
try:
    # First try to import from src directly (when running from the root directory)
    from src.tts_parler.pipeline_integration import generate_tts_for_segments, get_tts_model
    from src.tts.utils import preprocess_hindi_text
    from src.audio_processing.separation import separate_audio_from_video, cleanup_separation_temp
    from src.diarization.pyannotate import perform_diarization
    from src.reference_sampling.pipeline_integration import process_reference_sampling
    from src.pipeline import (
        setup_directories, 
        extract_and_separate_audio, 
        perform_speaker_diarization, 
        extract_reference_samples, 
        transcribe_audio, 
        translate_text,
        create_basic_reference_samples
    )
    from src.audio_processing.combine import combine_audio_segments
except ImportError:
    # Handle direct imports if not running from the project root
    from tts_parler.pipeline_integration import generate_tts_for_segments, get_tts_model
    from tts.utils import preprocess_hindi_text
    try:
        from audio_processing.separation import separate_audio_from_video, cleanup_separation_temp
        from diarization.pyannotate import perform_diarization
        from reference_sampling.pipeline_integration import process_reference_sampling
        from pipeline import (
            setup_directories, 
            extract_and_separate_audio, 
            perform_speaker_diarization, 
            extract_reference_samples, 
            transcribe_audio, 
            translate_text,
            create_basic_reference_samples
        )
        from audio_processing.combine import combine_audio_segments
    except ImportError:
        # If modules aren't found, we'll handle this later
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dubbing_pipeline_local_tts.log")
    ]
)
logger = logging.getLogger("dubbing_pipeline_local_tts")

def run_pipeline(args) -> Dict:
    """
    Run the complete pipeline from video to Hindi TTS segments using a local TTS model.
    
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
            use_gpu=args.use_gpu,
            gpu_index=args.gpu_index if hasattr(args, 'gpu_index') else 0
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
        diarization_result = perform_speaker_diarization(
            audio_path=speech_path, 
            output_path=diarization_path,
            auth_token=args.pyannote_token
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
    
    # Step 5: Extract reference samples for voice cloning
    logger.info("Step 5: Extracting reference samples")
    try:
        reference_samples = extract_reference_samples(
            audio_path=speech_path,
            output_dir=dirs["reference_samples"],
            transcription_path=transcription_path,
            diarization_path=diarization_path if not args.skip_diarization else None
        )
    except Exception as e:
        logger.error(f"Error extracting reference samples: {str(e)}")
        # Create basic reference samples if extraction fails
        reference_samples = create_basic_reference_samples(speech_path, dirs["reference_samples"])
        
    # Step 6: Translate transcription to Hindi
    logger.info("Step 6: Translating transcription to Hindi")
    translation = translate_text(
        transcription=transcription,
        output_path=translation_path,
        model_name=args.translation_model
    )
    
    # Step 7: Generate TTS with local model for each translated segment
    logger.info("Step 7: Generating TTS for translated segments")
    tts_segments = generate_tts_for_segments(
        translation=translation,
        reference_samples=reference_samples,
        output_dir=dirs["tts_output"],
        use_gpu=args.use_gpu,
        gpu_index=args.gpu_index if hasattr(args, 'gpu_index') else 0,
        model_path=args.model_path
    )
    
    # Step 8: Combine TTS segments with background audio
    logger.info("Step 8: Combining TTS segments with background audio")
    try:
        # Get original audio duration
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
        
        # Ensure the background path exists
        if not os.path.exists(background_path):
            logger.warning(f"Background audio file not found: {background_path}")
            logger.warning("Will try to use the original audio instead")
            background_path = audio_path
        
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
                if "translated_segments" in translation and i < len(translation["translated_segments"]):
                    trans_segment = translation["translated_segments"][i]
                    if "start" in trans_segment and "end" in trans_segment:
                        segment["start"] = trans_segment["start"]
                        segment["end"] = trans_segment["end"]
                    
                    # Add Hindi text for debugging
                    if "hindi" in trans_segment:
                        segment["hindi_text"] = trans_segment["hindi"]
        
        # Use a very low background volume to ensure Hindi speech is clear
        background_vol = 0.05  # Very low background volume
        if hasattr(args, 'background_volume'):
            background_vol = min(args.background_volume, 0.1)  # Cap at 0.1 to ensure Hindi is clear
            
        logger.info(f"Using background volume level: {background_vol} to ensure Hindi speech is clear")
        
        # Combine segments with background
        combined_audio = combine_audio_segments(
            tts_segments=tts_segments,
            background_audio_path=background_path,
            output_path=combined_audio_path,
            tts_output_dir=dirs["tts_output"],
            original_duration=original_duration,
            adjust_volume=True,
            background_volume=background_vol
        )
        
        if combined_audio:
            logger.info(f"Final audio generated: {combined_audio}")
        else:
            logger.error("Failed to generate combined audio")
            combined_audio_path = None
    except Exception as e:
        logger.error(f"Error combining audio segments: {str(e)}")
        logger.warning("Continuing without combined audio")
        combined_audio_path = None
    
    # Save final results
    result = {
        "video_path": args.video_path,
        "audio_path": audio_path,
        "speech_path": speech_path,
        "background_path": background_path,
        "transcription_path": transcription_path,
        "translation_path": translation_path,
        "diarization_path": diarization_path,
        "reference_metadata_path": os.path.join(dirs["reference_samples"], "reference_samples_metadata.json"),
        "speakers": reference_samples.get("speakers", {}),
        "tts_output_dir": dirs["tts_output"],
        "tts_segments": tts_segments,
        "combined_audio_path": combined_audio_path,
        "tts_engine": "local_tts",
        "tts_model": args.model_path or "xtts_v2",
        "completed_at": datetime.datetime.now().isoformat()
    }
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Pipeline completed. Results saved to {result_path}")
    logger.info(f"Total segments processed: {len(tts_segments)}")
    
    return result

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="English to Hindi Video Dubbing Pipeline with Local TTS")
    
    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", default="processed_data", help="Directory to save all outputs")
    parser.add_argument("--asr_model", default="large-v3", help="Whisper model to use for ASR")
    parser.add_argument("--translation_model", default="facebook/nllb-200-3.3B", help="Translation model to use")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for intensive tasks")
    parser.add_argument("--gpu_index", type=int, default=0, help="Specify which GPU to use (default: 0)")
    parser.add_argument("--skip_diarization", action="store_true", help="Skip speaker diarization")
    parser.add_argument("--pyannote_token", help="Authentication token for PyAnnote")
    parser.add_argument("--model_path", help="Path to the local TTS model (uses default XTTS if not provided)")
    
    # TTS-specific options
    parser.add_argument("--voice_settings", help="JSON string with voice settings (speed)")
    
    parser.add_argument("--max_segment_length", type=int, default=30, 
                      help="Maximum number of words per segment during transcription (default: 30)")
    
    parser.add_argument("--background_volume", type=float, default=0.2, help="Background audio volume level (0.0-1.0)")
    
    args = parser.parse_args()
    
    try:
        result = run_pipeline(args)
        print(f"Pipeline completed successfully.")
        print(f"Generated {len(result['tts_segments'])} Hindi TTS segments in: {result['tts_output_dir']}")
        return 0
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
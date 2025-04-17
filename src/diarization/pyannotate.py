"""Speaker diarization using PyAnnote."""

import json
import time
from pathlib import Path

from src.utils.logger import logger

try:
    import torch
    from pyannote.audio import Pipeline
except ImportError:
    logger.warning("PyAnnote not installed. Speaker diarization will not be available.")

def perform_diarization(
    audio_path, 
    output_path=None, 
    auth_token=None,
    min_duration=0.5
):
    """Perform speaker diarization on an audio file using PyAnnote.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path for the output JSON file. If None, a path is generated.
        auth_token: HuggingFace authentication token for PyAnnote access
        min_duration: Minimum duration in seconds for a valid segment
        
    Returns:
        str: Path to the diarization results JSON file
    """
    if 'Pipeline' not in globals():
        raise ImportError("PyAnnote not installed. Please install it with: "
                         "pip install pyannote.audio")
    
    if not auth_token:
        logger.warning("No auth token provided for PyAnnote. Using default model.")
    
    audio_path = Path(audio_path)
    
    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("processed_data/diarization")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{audio_path.stem}_diarization.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Performing speaker diarization on {audio_path}")
    start_time = time.time()
    
    try:
        # Initialize diarization pipeline
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f"Using device: {device} for diarization")
        
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        ).to(device)
        
        # Run diarization
        diarization = diarization_pipeline(audio_path)
        
        # Convert to a more manageable format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Filter out segments that are too short
            duration = turn.end - turn.start
            if duration >= min_duration:
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                })
        
        # Write results to JSON file
        with open(output_path, 'w') as f:
            json.dump(segments, f, indent=2)
        
        # Free up GPU memory
        if use_cuda:
            torch.cuda.empty_cache()
        
        duration = time.time() - start_time
        logger.info(f"Diarization completed in {duration:.2f} seconds, found {len(segments)} segments")
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error during diarization: {str(e)}")
        raise 
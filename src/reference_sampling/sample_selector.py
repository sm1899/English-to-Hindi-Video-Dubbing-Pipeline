"""
Reference Sample Selector Module

This module analyzes audio to extract diverse reference samples for voice cloning,
considering speech characteristics and emotional qualities.
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
MIN_SEGMENT_DURATION = 5  # Minimum duration in seconds for reference samples
MAX_SEGMENT_DURATION = 15  # Maximum duration in seconds for reference samples
NUM_REFERENCE_SAMPLES = 5  # Number of reference samples to select per speaker
FEATURE_WEIGHTS = {
    'clarity': 0.4,
    'speech_rate': 0.2,
    'pitch_variation': 0.2,
    'intensity': 0.2
}

def analyze_audio_segment(audio_path: str, start_time: float, end_time: float) -> Dict:
    """
    Analyzes an audio segment to extract speech features.
    
    Args:
        audio_path: Path to the full audio file
        start_time: Start time of the segment (seconds)
        end_time: End time of the segment (seconds)
        
    Returns:
        Dictionary of audio features
    """
    # Load the audio segment
    y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=(end_time-start_time))
    
    # Calculate speech rate (syllables per second approximation)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    speech_rate = tempo / 60.0  # Convert BPM to syllables per second (approximation)
    
    # Calculate pitch variation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if pitch_values:
        pitch_mean = np.mean(pitch_values)
        pitch_std = np.std(pitch_values)
        pitch_variation = pitch_std / pitch_mean if pitch_mean > 0 else 0
    else:
        pitch_variation = 0
    
    # Calculate intensity (volume)
    intensity = np.mean(librosa.feature.rms(y=y)[0])
    
    # Calculate clarity (signal-to-noise ratio approximation)
    # Higher value means cleaner speech
    spec = np.abs(librosa.stft(y))
    spec_mean = np.mean(spec, axis=1)
    spec_std = np.std(spec, axis=1)
    signal_power = np.sum(spec_mean**2)
    noise_power = np.sum(spec_std**2)
    clarity = signal_power / noise_power if noise_power > 0 else 0
    
    # Detect emotional characteristics
    # Higher energy and pitch variation often indicate more emotional speech
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    # Emotional intensity score (combination of energy, pitch variation, and timbre)
    emotional_intensity = (intensity * 0.4) + (pitch_variation * 0.4) + (np.std(mfcc_mean) * 0.2)
    
    # Valence approximation (positive vs negative emotion)
    # Higher spectral centroid often correlates with more "bright" or positive speech
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)[0])
    
    # Normalize to a -1 to 1 scale (very rough approximation)
    valence = (spectral_centroid / 5000) - 0.5  # Normalized to roughly -0.5 to 0.5
    
    return {
        'speech_rate': float(speech_rate),
        'pitch_variation': float(pitch_variation),
        'intensity': float(intensity),
        'clarity': float(clarity),
        'emotional_intensity': float(emotional_intensity),
        'valence': float(valence),
        'duration': float(end_time - start_time)
    }

def calculate_diversity_score(existing_samples: List[Dict], new_sample: Dict) -> float:
    """
    Calculate how diverse a new sample is compared to existing samples.
    
    Args:
        existing_samples: List of dictionaries containing features of existing samples
        new_sample: Dictionary containing features of the new sample
        
    Returns:
        Diversity score (higher is more diverse)
    """
    if not existing_samples:
        return 1.0  # First sample is always diverse
        
    # Calculate normalized Euclidean distance for each feature
    distances = []
    features = ['speech_rate', 'pitch_variation', 'intensity', 'emotional_intensity', 'valence']
    
    for sample in existing_samples:
        feature_dists = []
        for feature in features:
            # Skip if either sample is missing this feature
            if feature not in sample or feature not in new_sample:
                continue
                
            # Calculate normalized distance for this feature
            feature_dists.append(abs(sample[feature] - new_sample[feature]))
            
        if feature_dists:
            # Average distance across all features
            distances.append(np.mean(feature_dists))
    
    # Return average distance to all existing samples
    return np.mean(distances) if distances else 0.0

def select_diverse_samples(segments: List[Dict], audio_path: str, 
                         output_dir: str, num_samples: int = NUM_REFERENCE_SAMPLES) -> List[Dict]:
    """
    Select diverse reference samples from a list of segments.
    
    Args:
        segments: List of segment dictionaries with start and end times
        audio_path: Path to the full audio file
        output_dir: Directory to save reference samples
        num_samples: Number of reference samples to select
        
    Returns:
        List of selected reference sample metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze all segments and filter out those that are too short
    analyzed_segments = []
    for i, segment in enumerate(segments):
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        
        # Skip segments that are too short or too long
        duration = end_time - start_time
        if duration < MIN_SEGMENT_DURATION or duration > MAX_SEGMENT_DURATION:
            continue
            
        # Analyze the segment
        try:
            features = analyze_audio_segment(audio_path, start_time, end_time)
            analyzed_segments.append({
                'index': i,
                'start': start_time,
                'end': end_time,
                'text': segment.get('text', ''),
                'features': features
            })
        except Exception as e:
            logger.error(f"Error analyzing segment {i}: {e}")
    
    # Sort by clarity (most important feature)
    analyzed_segments.sort(key=lambda x: x['features']['clarity'], reverse=True)
    
    # Always include the clearest segment
    selected_samples = [analyzed_segments[0]] if analyzed_segments else []
    remaining_segments = analyzed_segments[1:] if len(analyzed_segments) > 1 else []
    
    # Select additional samples based on diversity
    while len(selected_samples) < num_samples and remaining_segments:
        # Calculate diversity score for each remaining segment
        diversity_scores = []
        for segment in remaining_segments:
            score = calculate_diversity_score(
                [s['features'] for s in selected_samples], 
                segment['features']
            )
            diversity_scores.append((segment, score))
        
        # Sort by diversity score (highest first)
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add the most diverse segment
        selected_samples.append(diversity_scores[0][0])
        remaining_segments.remove(diversity_scores[0][0])
    
    # Extract and save the selected segments
    reference_samples = []
    for i, sample in enumerate(selected_samples):
        segment_index = sample['index']
        start_time = sample['start']
        end_time = sample['end']
        
        # Generate output filename
        output_filename = f"reference_{i+1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Load the full audio segment
            y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=(end_time-start_time))
            
            # Save the segment
            sf.write(output_path, y, sr)
            
            # Add to reference samples
            reference_samples.append({
                'path': output_path,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'text': sample.get('text', ''),
                'features': sample['features'],
                'emotion_label': get_emotion_label(sample['features'])
            })
            
            logger.info(f"Saved reference sample {i+1} to {output_path}")
        except Exception as e:
            logger.error(f"Error saving segment {segment_index}: {e}")
    
    return reference_samples

def get_emotion_label(features: Dict) -> str:
    """
    Get a human-readable emotion label based on audio features.
    Very simplified approximation.
    
    Args:
        features: Dictionary of audio features
        
    Returns:
        String label for the emotion
    """
    emotional_intensity = features.get('emotional_intensity', 0)
    valence = features.get('valence', 0)
    speech_rate = features.get('speech_rate', 0)
    
    # Simple rule-based emotion classification
    if emotional_intensity < 0.3:
        if valence < 0:
            return "calm_serious"
        else:
            return "calm_neutral"
    elif emotional_intensity < 0.6:
        if valence < -0.2:
            return "concerned"
        elif valence < 0.2:
            return "neutral_engaged"
        else:
            return "positive"
    else:  # High emotional intensity
        if speech_rate > 3.5:
            if valence < 0:
                return "urgent_intense"
            else:
                return "excited_passionate"
        else:
            if valence < 0:
                return "emphatic_serious"
            else:
                return "inspiring_powerful"

def extract_reference_samples(audio_path: str, output_dir: str, 
                            diarization_path: Optional[str] = None) -> Dict:
    """
    Main function to extract reference samples for voice cloning.
    
    Args:
        audio_path: Path to the full audio file
        output_dir: Base directory to save reference samples
        diarization_path: Path to diarization JSON file (optional)
        
    Returns:
        Dictionary mapping speaker IDs to reference samples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio file info
    y, sr = librosa.load(audio_path, sr=None, duration=10)  # Just load a bit to get info
    audio_duration = librosa.get_duration(y=y, sr=sr)
    
    reference_samples = {}
    
    # Check if diarization file exists
    if diarization_path and os.path.exists(diarization_path):
        # Multiple speakers case
        try:
            with open(diarization_path, 'r') as f:
                diarization_data = json.load(f)
            
            # Process each speaker
            for speaker_id, speaker_segments in diarization_data.items():
                # Create speaker directory
                speaker_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
                os.makedirs(speaker_dir, exist_ok=True)
                
                # Select diverse samples for this speaker
                speaker_samples = select_diverse_samples(
                    speaker_segments, audio_path, speaker_dir
                )
                
                reference_samples[speaker_id] = speaker_samples
                
        except Exception as e:
            logger.error(f"Error processing diarization data: {e}")
            # Fall back to single speaker case
            single_speaker_segments = [{'start': 0, 'end': audio_duration}]
            reference_samples['speaker_0'] = select_diverse_samples(
                single_speaker_segments, audio_path, output_dir
            )
    else:
        # Single speaker case
        logger.info("No diarization file found, treating as single speaker")
        
        # Create dummy segments by splitting the audio into equal parts
        num_segments = min(20, int(audio_duration / MIN_SEGMENT_DURATION))
        segment_duration = audio_duration / num_segments
        
        segments = []
        for i in range(num_segments):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, audio_duration)
            segments.append({'start': start, 'end': end})
        
        reference_samples['speaker_0'] = select_diverse_samples(
            segments, audio_path, output_dir
        )
    
    # Save reference samples metadata
    metadata_path = os.path.join(output_dir, "reference_samples_metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(reference_samples, f, indent=2)
        logger.info(f"Saved reference samples metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
    
    return reference_samples

def extract_reference_samples_from_asr(
    audio_path: str, 
    asr_json_path: str,
    output_dir: str, 
    diarization_path: Optional[str] = None
) -> Dict:
    """
    Extract reference samples using ASR segments, which provide better
    linguistic boundaries and quality indicators.
    
    Args:
        audio_path: Path to the full audio file
        asr_json_path: Path to the ASR JSON output from Whisper
        output_dir: Base directory to save reference samples
        diarization_path: Path to diarization JSON file (optional)
        
    Returns:
        Dictionary mapping speaker IDs to reference samples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ASR output
    try:
        with open(asr_json_path, 'r') as f:
            asr_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading ASR data: {e}")
        return extract_reference_samples(audio_path, output_dir, diarization_path)
    
    # Extract segments from ASR output
    asr_segments = []
    for segment in asr_data.get('segments', []):
        # Extract segment information
        asr_segments.append({
            'start': segment.get('start', 0),
            'end': segment.get('end', 0),
            'text': segment.get('text', ''),
            'no_speech_prob': segment.get('no_speech_prob', 0)
        })
    
    # Filter out segments with high no_speech_prob or that are too short/long
    filtered_segments = []
    for segment in asr_segments:
        duration = segment['end'] - segment['start']
        if duration >= MIN_SEGMENT_DURATION and duration <= MAX_SEGMENT_DURATION and segment['no_speech_prob'] < 0.5:
            filtered_segments.append(segment)
    
    reference_samples = {}
    
    # Check if diarization file exists
    if diarization_path and os.path.exists(diarization_path):
        # Multiple speakers case - need to map ASR segments to speakers
        try:
            with open(diarization_path, 'r') as f:
                diarization_data = json.load(f)
            
            # Map each ASR segment to a speaker based on overlapping time
            speaker_segments = {speaker_id: [] for speaker_id in diarization_data.keys()}
            
            for asr_segment in filtered_segments:
                asr_start = asr_segment['start']
                asr_end = asr_segment['end']
                assigned_speaker = None
                max_overlap = 0
                
                # Find speaker with maximum time overlap
                for speaker_id, speaker_segs in diarization_data.items():
                    for speaker_seg in speaker_segs:
                        speaker_start = speaker_seg.get('start', 0)
                        speaker_end = speaker_seg.get('end', 0)
                        
                        # Calculate overlap
                        overlap_start = max(asr_start, speaker_start)
                        overlap_end = min(asr_end, speaker_end)
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > max_overlap:
                            max_overlap = overlap
                            assigned_speaker = speaker_id
                
                # Assign segment to speaker with most overlap
                if assigned_speaker and max_overlap > 0:
                    speaker_segments[assigned_speaker].append(asr_segment)
            
            # Process each speaker separately
            for speaker_id, segments in speaker_segments.items():
                if segments:
                    # Create speaker directory
                    speaker_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
                    os.makedirs(speaker_dir, exist_ok=True)
                    
                    # Select diverse samples for this speaker
                    speaker_samples = select_diverse_samples(
                        segments, audio_path, speaker_dir
                    )
                    
                    reference_samples[speaker_id] = speaker_samples
                    
        except Exception as e:
            logger.error(f"Error mapping ASR segments to speakers: {e}")
            # Fall back to treating all segments as one speaker
            reference_samples['speaker_0'] = select_diverse_samples(
                filtered_segments, audio_path, output_dir
            )
    else:
        # Single speaker case - no need to map segments
        logger.info("No diarization file found, treating as single speaker")
        
        reference_samples['speaker_0'] = select_diverse_samples(
            filtered_segments, audio_path, output_dir
        )
    
    # Save reference samples metadata
    metadata_path = os.path.join(output_dir, "reference_samples_metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(reference_samples, f, indent=2)
        logger.info(f"Saved reference samples metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
    
    return reference_samples

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract reference samples for voice cloning")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--output", required=True, help="Path to save reference samples")
    parser.add_argument("--diarization", help="Path to diarization JSON file (optional)")
    parser.add_argument("--asr", help="Path to ASR JSON file (optional)")
    
    args = parser.parse_args()
    
    if args.asr and os.path.exists(args.asr):
        extract_reference_samples_from_asr(args.audio, args.asr, args.output, args.diarization)
    else:
        extract_reference_samples(args.audio, args.output, args.diarization) 
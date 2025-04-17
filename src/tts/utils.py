"""
TTS Utilities

Utility functions for TTS processing, including reference sample selection and text preprocessing.
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.utils.logger import logger

# Emotion categories and their mapping to speech patterns
EMOTION_CATEGORIES = {
    "calm_neutral": ["calm", "neutral", "relaxed", "gentle"],
    "calm_serious": ["serious", "formal", "informative", "thoughtful"],
    "concerned": ["concerned", "worried", "cautious", "sad"], 
    "neutral_engaged": ["engaged", "interested", "attentive", "conversational"],
    "positive": ["happy", "pleased", "optimistic", "confident"],
    "urgent_intense": ["urgent", "intense", "warning", "alarmed"],
    "excited_passionate": ["excited", "passionate", "enthusiastic", "energetic"],
    "emphatic_serious": ["emphatic", "stern", "assertive", "firm"],
    "inspiring_powerful": ["inspiring", "powerful", "motivational", "strong"]
}

# Hindi text preprocessing patterns
HINDI_FIXES = [
    # Add space after punctuation if not present
    (r'([।॥?!,;:])([^\s])', r'\1 \2'),
    
    # Fix common spacing issues
    (r'\s+', ' '),
    
    # Normalize quotes
    (r'[""]', '"'),
    (r'[\'\']', "'"),
    
    # Make sure there's proper spacing around punctuation
    (r'\s*([।॥?!,;:])\s*', r' \1 ')
]

def analyze_text_emotion(text: str) -> Dict[str, float]:
    """
    Analyze text to detect emotional tone based on keywords and patterns.
    This is a simplified version - in production, a proper NLP model would be used.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary with emotion category scores
    """
    text = text.lower()
    
    # Initialize scores for each emotion category
    emotion_scores = {category: 0.0 for category in EMOTION_CATEGORIES.keys()}
    
    # Simple keyword-based scoring
    for category, keywords in EMOTION_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                emotion_scores[category] += 1.0
    
    # Check for punctuation patterns
    if "!" in text:
        punctuation_count = text.count("!")
        emotion_scores["excited_passionate"] += punctuation_count * 0.5
        emotion_scores["urgent_intense"] += punctuation_count * 0.5
    
    if "?" in text:
        emotion_scores["concerned"] += text.count("?") * 0.3
    
    # If no strong emotion is detected, default to neutral
    total_score = sum(emotion_scores.values())
    if total_score < 1.0:
        emotion_scores["neutral_engaged"] += 1.0
    
    # Normalize scores
    if total_score > 0:
        for category in emotion_scores:
            emotion_scores[category] /= total_score
    
    return emotion_scores

def match_reference_to_emotion(
    text: str, 
    reference_samples: Dict[str, List[Dict]], 
    speaker_id: str = "speaker_0",
    num_samples: int = 3
) -> Union[str, List[str]]:
    """
    Match text to appropriate reference samples based on emotional content.
    
    Args:
        text: Text content to match
        reference_samples: Dictionary of reference samples
        speaker_id: Speaker ID to use for matching
        num_samples: Number of reference samples to return (if 1, returns string path)
        
    Returns:
        Path to best matching reference sample or list of paths if num_samples > 1
    """
    # Get emotion of the text
    text_emotion = analyze_text_emotion(text)
    
    # Get reference samples for the speaker
    if speaker_id not in reference_samples:
        # Try to convert from different format (SPEAKER_XX to speaker_x)
        if speaker_id.startswith("SPEAKER_"):
            try:
                speaker_num = int(speaker_id.split("_")[1])
                alt_speaker_id = f"speaker_{speaker_num}"
                if alt_speaker_id in reference_samples:
                    logger.info(f"Mapped {speaker_id} to {alt_speaker_id}")
                    speaker_id = alt_speaker_id
            except (ValueError, IndexError):
                pass
    
    # If still not found, use first available speaker
    if speaker_id not in reference_samples:
        available_speakers = list(reference_samples.keys())
        if not available_speakers:
            raise ValueError("No reference samples available")
        speaker_id = available_speakers[0]
        
    speaker_samples = reference_samples[speaker_id]
    
    if not speaker_samples:
        raise ValueError(f"No reference samples found for speaker {speaker_id}")
    
    # Sort reference samples by emotion match
    scored_samples = []
    
    for sample in speaker_samples:
        sample_emotion = sample.get("emotion_label", "neutral")
        
        # Calculate score based on emotional match
        emotion_score = 0
        for emotion, score in text_emotion.items():
            if emotion in sample_emotion:
                emotion_score += score * 5
        
        # Add sample quality factors
        clarity = sample.get("features", {}).get("clarity", 0.5)
        duration = sample.get("features", {}).get("duration", 1.0)
        
        # Prefer samples with good clarity and reasonable duration (2-10 seconds)
        quality_score = clarity * 2
        if 2.0 <= duration <= 10.0:
            quality_score += 1.0
            
        # Add a slight randomness to avoid always picking the same sample
        variation = random.uniform(0, 0.5)
        
        final_score = emotion_score + quality_score + variation
        
        scored_samples.append((sample, final_score))
    
    # Sort by score (descending)
    scored_samples.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top N paths based on num_samples
    if num_samples == 1:
        return scored_samples[0][0]["path"]
    else:
        # Limit to number of available samples, up to num_samples
        num_to_return = min(num_samples, len(scored_samples))
        return [scored_samples[i][0]["path"] for i in range(num_to_return)]

def preprocess_hindi_text(text: str) -> str:
    """
    Preprocess Hindi text for better TTS results.
    
    Args:
        text: Hindi text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Apply all regex fixes
    processed_text = text
    for pattern, replacement in HINDI_FIXES:
        processed_text = re.sub(pattern, replacement, processed_text)
    
    # Trim extra whitespace
    processed_text = processed_text.strip()
    
    return processed_text

def load_reference_samples(video_id: str) -> Dict:
    """
    Load reference samples metadata for a given video.
    
    Args:
        video_id: Video ID
        
    Returns:
        Dictionary with reference samples metadata
    """
    reference_dir = Path(f"processed_data/reference_samples/{video_id}")
    metadata_path = reference_dir / "reference_samples_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Reference samples metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        reference_samples = json.load(f)
    
    return reference_samples 
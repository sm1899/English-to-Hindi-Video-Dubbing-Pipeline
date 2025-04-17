"""English to Hindi translation using Meta AI NLLB-200 model."""

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from dotenv import load_dotenv

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.logger import logger

# Load environment variables from .env file
load_dotenv()

class NLLBTranslator:
    """Translator using Meta AI's NLLB-200 model for English to Hindi translation."""
    
    def __init__(self, model_size="3.3B", device=None):
        """Initialize the NLLB translator.
        
        Args:
            model_size: Size of the NLLB model ("600M" or "3.3B")
            device: Device to use for inference ("cuda" or "cpu")
        """
        self.model_size = model_size
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing NLLB-200-{model_size} on {self.device}")
        
        # Get HuggingFace auth token from environment variables
        auth_token = os.environ.get('auth_token')
        if not auth_token:
            logger.warning("No HuggingFace auth_token found in .env file. This may lead to download timeouts.")
        
        # Map model size to correct model name
        if model_size == "600M":
            model_name = "facebook/nllb-200-distilled-600M"
        elif model_size == "3.3B":
            model_name = "facebook/nllb-200-3.3B"
        else:
            raise ValueError(f"Unsupported model size: {model_size}. Choose '600M' or '3.3B'.")
        
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer with correct source language
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=auth_token,
            src_lang="eng_Latn"  # Default source language is English
        )
        
        # Load model with low-level parameters for better memory management
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            token=auth_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        # Get special token IDs for Hindi
        # Use the correct method according to NLLB documentation
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            self.hindi_token_id = self.tokenizer.lang_code_to_id["hin_Deva"]
            logger.info(f"Hindi token ID (lang_code_to_id): {self.hindi_token_id}")
        else:
            try:
                self.hindi_token_id = self.tokenizer.convert_tokens_to_ids("hin_Deva")
                logger.info(f"Hindi token ID (convert_tokens_to_ids): {self.hindi_token_id}")
            except:
                logger.warning("Could not determine Hindi token ID properly. Translations may be affected.")
                self.hindi_token_id = None
        
        logger.info(f"NLLB-200-{model_size} loaded successfully")
    
    def translate(self, text, src_lang="eng_Latn", tgt_lang="hin_Deva", max_length=500):
        """Translate text from source language to target language.
        
        Args:
            text: Text to translate
            src_lang: Source language code (default: English)
            tgt_lang: Target language code (default: Hindi)
            max_length: Maximum length for text chunks
            
        Returns:
            str: Translated text
        """
        # Skip empty text
        if not text or text.isspace():
            return ""
        
        try:
            # Break long text into smaller chunks if needed
            if len(text.split()) > 100:  # Approximate threshold based on word count
                chunks = self._split_text(text)
                translations = []
                
                for i, chunk in enumerate(chunks):
                    logger.info(f"Translating chunk {i+1}/{len(chunks)}")
                    translations.append(self._translate_chunk(chunk, src_lang, tgt_lang))
                
                return " ".join(translations)
            else:
                return self._translate_chunk(text, src_lang, tgt_lang)
        
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return ""
    
    def _translate_chunk(self, text, src_lang, tgt_lang):
        """Translate a single chunk of text."""
        try:
            # Ensure the source language is set properly for the tokenizer
            if self.tokenizer.src_lang != src_lang:
                logger.debug(f"Setting source language to {src_lang}")
                self.tokenizer.src_lang = src_lang
            
            # Tokenize the input text - explicit padding and truncation settings
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512  # Maximum size NLLB can handle
            ).to(self.device)
            
            # Define generation parameters for better translation quality
            generate_kwargs = {
                "max_length": min(512, len(text.split()) * 2),  # Dynamic max length based on input
                "min_length": max(10, len(text.split()) // 2),  # Ensure reasonable length output
                "num_beams": 5,  # Beam search for better quality
                "length_penalty": 1.0,  # Balanced length penalty
                "no_repeat_ngram_size": 3,  # Avoid repetition
                "early_stopping": True,  # Stop when all beams have finished
            }
            
            # Use the Hindi token ID for forced_bos_token_id if available
            if self.hindi_token_id is not None:
                generate_kwargs["forced_bos_token_id"] = self.hindi_token_id
            
            # Generate translation with torch.no_grad() for memory efficiency
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    **generate_kwargs
                )
            
            # Decode the generated tokens with skip_special_tokens=True
            translation = self.tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True
            )[0]
            
            # Log token counts for debugging
            logger.debug(f"Input tokens: {len(inputs.input_ids[0])}, Output tokens: {len(translated_tokens[0])}")
            
            return translation
        
        except Exception as e:
            logger.error(f"Error in _translate_chunk: {str(e)}")
            # Return an empty string as fallback
            return ""
    
    def _split_text(self, text, max_words=100):
        """Split text into chunks at sentence boundaries."""
        # More robust sentence splitting based on punctuation
        sentences = []
        current = ""
        
        # Define sentence ending punctuation and handle edge cases
        sentence_endings = [".", "!", "?", "।", "॥"]  # Include Hindi sentence markers
        
        # Handle cases where punctuation might have a space before it
        normalized_text = text
        for ending in sentence_endings:
            normalized_text = normalized_text.replace(f" {ending}", ending)
        
        # Split text by common sentence-ending punctuation
        parts = []
        current_part = ""
        i = 0
        while i < len(normalized_text):
            current_part += normalized_text[i]
            
            # Check if current character is a sentence ending
            if normalized_text[i] in sentence_endings:
                # Check if this is really a sentence end (not part of an abbreviation or number)
                is_sentence_end = True
                
                # Look behind to check if this is part of an abbreviation (e.g., "Mr.", "Dr.", "U.S.")
                if normalized_text[i] == "." and i > 1:
                    # Simple check for common abbreviations
                    if i >= 3 and normalized_text[i-3:i] in ["Mr", "Dr", "Ms", "St"]:
                        is_sentence_end = False
                    # Check for initials (single letter followed by period)
                    elif i >= 2 and normalized_text[i-2].isalpha() and normalized_text[i-1] == " ":
                        is_sentence_end = False
                
                # If it's a sentence end, add to parts and reset current_part
                if is_sentence_end:
                    parts.append(current_part)
                    current_part = ""
            
            i += 1
        
        # Add any remaining text
        if current_part:
            parts.append(current_part)
        
        # Process the parts into sentences
        for part in parts:
            if part.strip():
                current += part
                # Check if we have a complete sentence
                if any(current.endswith(ending) for ending in sentence_endings):
                    sentences.append(current.strip())
                    current = ""
        
        # Add any remaining text as a final sentence
        if current.strip():
            sentences.append(current.strip())
        
        # Group sentences into chunks based on word count
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            word_count = len(sentence.split())
            
            # If this sentence alone exceeds max_words, we need to split it
            if word_count > max_words:
                # If we have accumulated sentences, add them as a chunk first
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                
                # Split this long sentence into smaller parts
                words = sentence.split()
                for i in range(0, len(words), max_words):
                    sub_sentence = " ".join(words[i:i+max_words])
                    chunks.append(sub_sentence)
            
            # If adding this sentence doesn't exceed max_words
            elif current_word_count + word_count <= max_words:
                current_chunk.append(sentence)
                current_word_count += word_count
            else:
                # Save the current chunk and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
        
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.device == "cuda":
            # Free up GPU memory
            del self.model
            torch.cuda.empty_cache()


def translate_transcription(
    transcription_path, 
    output_path=None, 
    model_size="3.3B"
):
    """Translate a Whisper transcription from English to Hindi.
    
    Args:
        transcription_path: Path to the Whisper transcription JSON
        output_path: Path for the output JSON with translations. If None, a path is generated.
        model_size: NLLB model size ("600M" or "3.3B")
        
    Returns:
        str: Path to the translated JSON file
    """
    # Initialize translator
    translator = NLLBTranslator(model_size=model_size)
    
    transcription_path = Path(transcription_path)
    
    # Generate output path if not provided
    if output_path is None:
        output_dir = Path("processed_data/translations")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{transcription_path.stem}_translated.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Translating transcription from {transcription_path}")
    start_time = time.time()
    
    try:
        # Load the transcription
        with open(transcription_path, 'r', encoding='utf-8') as f:
            transcription = json.load(f)
        
        # Create a deep copy to avoid modifying the original
        translated = deepcopy(transcription)
        
        # Translate the full text
        if "text" in transcription:
            logger.info("Translating full transcript...")
            translated["text_hi"] = translator.translate(transcription["text"])
        
        # Translate each segment while preserving all metadata
        if "segments" in transcription:
            total_segments = len(transcription["segments"])
            logger.info(f"Translating {total_segments} segments...")
            
            # Initialize translated_segments list
            translated["translated_segments"] = []
            
            for i, segment in enumerate(transcription["segments"]):
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(f"Translating segment {i+1}/{total_segments}")
                
                if "text" in segment:
                    # Create a new segment with all original metadata
                    translated_segment = {
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "english": segment["text"],
                        "hindi": translator.translate(segment["text"]),
                        "speaker": segment.get("speaker", "unknown"),
                        "words": segment.get("words", [])  # Preserve word-level timestamps
                    }
                    
                    # Log timing information
                    logger.debug(f"Segment {i}: start={translated_segment['start']}, end={translated_segment['end']}")
                    
                    translated["translated_segments"].append(translated_segment)
        
        # Save translated results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated, f, indent=2, ensure_ascii=False)
        
        duration = time.time() - start_time
        logger.info(f"Translation completed in {duration:.2f} seconds")
        
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        raise 
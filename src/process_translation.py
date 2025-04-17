import json
import argparse
import os
import logging
from pathlib import Path
from tqdm import tqdm  # Import tqdm for progress bar

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Make sure we can import from the parent directory (src)
import sys
# Add the parent directory (dubline) to the Python path
# This assumes the script is run from the workspace root (dubline)
# Or adjust as necessary based on your execution context
sys.path.append(str(Path(__file__).resolve().parents[1])) 

try:
    # Dynamically import based on whether running from src or root
    if Path.cwd().name == 'src':
         from translation_llm.use_mistral import translate_mistral
    else: # Assuming run from workspace root
         from src.translation_llm.use_mistral import translate_mistral
except ImportError as e:
    logging.error(f"Could not import translate_mistral. Ensure you are running from the workspace root or src directory, and check PYTHONPATH. Error: {e}")
    sys.exit(1)

# TODO: Add import for NLLB translation function when needed
# from src.translation_nllb.use_nllb import translate_nllb # Example

def process_translations(input_json_path: str, output_json_path: str, use_llm: bool):
    """
    Reads an ASR transcription JSON, translates segments, and saves in the target format.
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        logging.info(f"Successfully loaded transcription from: {input_json_path}")
    except FileNotFoundError:
        logging.error(f"Input transcription file not found: {input_json_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from: {input_json_path}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the input file: {e}")
        return

    translated_segments = []
    segments = transcription_data.get("segments", [])

    if not segments:
        logging.warning("No segments found in the transcription data.")
        # Decide if we should still write an output file
        # For now, let's write the original structure with empty translated_segments
    else:
        logging.info(f"Starting translation for {len(segments)} segments...")
        # Wrap segments iteration with tqdm for a progress bar
        for segment in tqdm(segments, desc="Translating Segments"):
            english_text = segment.get("text", "")
            start_time = segment.get("start")
            end_time = segment.get("end")
            speaker = segment.get("speaker", "UNKNOWN_SPEAKER") # Handle missing speaker

            if not english_text:
                logging.warning(f"Segment at {start_time}-{end_time} has empty text. Skipping translation.")
                hindi_text = ""
            elif use_llm:
                logging.debug(f"Translating using Mistral: '{english_text[:50]}...'")
                hindi_text = translate_mistral(english_text, target_language_code="hi")
                if hindi_text is None:
                    logging.error(f"Mistral translation failed for segment: {english_text}")
                    hindi_text = "[Translation Failed]" # Placeholder for failed translation
            else:
                # Placeholder for NLLB or other default translation
                logging.info("NLLB/Default translation not implemented yet in this script.")
                # hindi_text = translate_nllb(english_text, target_language="hi") # Example
                hindi_text = "[NLLB Translation Placeholder]"

            translated_segments.append({
                "start": start_time,
                "end": end_time,
                "english": english_text,
                "hindi": hindi_text,
                "speaker": speaker
            })
        logging.info("Finished translation.")

    output_data = {
        "original": transcription_data,
        "translated_segments": translated_segments
    }

    # Ensure the output directory exists
    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved translations to: {output_json_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing the output file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate ASR transcription segments using Mistral or another model.")
    parser.add_argument("input_json", help="Path to the input transcription JSON file (e.g., output/transcriptions/transcription.json)")
    parser.add_argument("output_json", help="Path to save the output translation JSON file (e.g., output/translations/translation.json)")
    parser.add_argument("--use_llm", action="store_true", help="Use Mistral LLM for translation instead of the default.")

    args = parser.parse_args()

    process_translations(args.input_json, args.output_json, args.use_llm) 
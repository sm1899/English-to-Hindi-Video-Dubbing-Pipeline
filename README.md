# English-to-Hindi Video Dubbing Pipeline

An end-to-end solution for dubbing English videos to Hindi, preserving the original speaker's voice and optionally synchronizing lip movements. This pipeline utilizes state-of-the-art open-source models for high-fidelity results.

## Overview

This pipeline automates the following steps:
1.  **Audio Extraction & Separation:** Extracts audio from the input video and separates speech from background noise/music using `Demucs`.
2.  **(Optional) Speaker Diarization:** Identifies different speakers and timestamps their speech segments using `PyAnnote`.
3.  **Automatic Speech Recognition (ASR):** Transcribes the English speech segments into time-aligned text using `Whisper` (default: `large-v3`).
4.  **Reference Sample Selection:** Selects diverse audio samples from the original speaker(s) for voice cloning.
5.  **Translation:** Translates the English text segments into Hindi using `NLLB-200` (default: `facebook/nllb-200-distilled-600M`) or optionally `Mistral` (via `--use_llm`).
6.  **Voice Cloning & Hindi TTS:** Generates Hindi speech in the original speaker's voice using `Coqui XTTS v2` zero-shot capabilities.
7.  **(Optional) Lip Synchronization:** Modifies the original video's lip movements to match the generated Hindi speech using `LatentSync v1.5`.
8.  **Final Assembly:** Combines the generated Hindi speech (potentially with original background audio) and the lip-synced video (if generated) into the final output file.

## Features

*   **High-Quality Voice Cloning:** Leverages Coqui XTTS v2 for zero-shot cross-lingual voice cloning.
*   **Accurate Transcription:** Uses OpenAI Whisper for robust ASR.
*   **Natural Translation:** Employs NLLB-200 or Mistral for English-to-Hindi translation.
*   **Speaker Diarization:** Supports multi-speaker videos by identifying individual speakers.
*   **Audio Separation:** Isolates speech from background noise/music for cleaner TTS input and final output mixing.
*   **Lip Synchronization (Optional):** Integrates LatentSync for realistic lip movements matching the Hindi audio.
*   **Modular Design:** Easily adaptable and extensible components.
*   **GPU Acceleration:** Supports GPU usage for faster processing of key steps (Separation, ASR, TTS, Lip Sync).

## Installation

1.  Clone this repository and the LatentSync submodule (if using lip-sync):
    ```bash
    git clone https://github.com/your-repo/dubline.git # Replace with your repo URL
    cd dubline
    # If using lip-sync, initialize/update submodules if LatentSync is one,
    # or follow LatentSync's specific installation instructions in its directory.
    # git submodule update --init --recursive # Example if it's a submodule
    ```
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Install FFmpeg:
    *   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS:** `brew install ffmpeg`
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
4.  Install Git LFS (required for downloading large model files):
    ```bash
    git lfs install
    ```
5.  (Optional) Set up Hugging Face Hub token for PyAnnote diarization:
    *   Create a `.env` file in the project root.
    *   Add your token: `HF_AUTH_TOKEN=your_huggingface_auth_token`

## Pre-trained / Fine-tuned Weights 

To  benefit from models already fine-tuned for Hindi performance (as discussed in the fine-tuning sections), pre-trained weights for certain components (like XTTS or LatentSync) is  available.

*   **Download Link:** [https://drive.google.com/drive/folders/1B0EGZBUhwEje3fdcXPtDi74PlqxXSd7A?usp=sharing]

## Usage

### Basic Usage (Single Speaker, No Lip Sync)

```bash
python src/pipeline.py --video_path path/to/your_video.mp4 --output_dir output/ --use_gpu
```

### Multi-Speaker Usage (Requires Diarization)

Ensure you have set `HF_AUTH_TOKEN` in `.env`.

```bash
python src/pipeline.py \\
  --video_path path/to/multi_speaker_video.mp4 \\
  --output_dir output_multi/ \\
  --diarize \\
  --use_gpu
```

### With Lip Synchronization

This requires LatentSync to be set up correctly. The pipeline will attempt to run the `inference.sh` script within the `LatentSync` directory.

```bash
python src/pipeline.py \\
  --video_path path/to/video.mp4 \\
  --output_dir output_lipsync/ \\
  --run_lip_sync \\
  --use_gpu
```
*Note: You might need to configure paths within `src/pipeline.py` or the LatentSync script itself depending on your setup.*

### Advanced Options

```bash
python src/pipeline.py \\
  --video_path path/to/video.mp4 \\
  --output_dir output_advanced/ \\
  --reference_audio path/to/specific_reference.wav \\
  --asr_model large-v3 \\
  --translation_model facebook/nllb-200-3.3B \\
  --use_llm \\ # Use Mistral for translation instead of NLLB
  --diarize \\
  --run_lip_sync \\
  --use_gpu \\
  --gpu_index 0 \\ # Specify GPU index
  --batch_size_tts 8 # Example TTS batch size
```

## Command Line Arguments

*(Run `python src/pipeline.py --help` for a full, up-to-date list)*

| Argument                 | Description                                                                 | Default                            |
| ------------------------ | --------------------------------------------------------------------------- | ---------------------------------- |
| `--video_path`           | Path to the input video file                                                | (Required)                         |
| `--output_dir`           | Directory to store output files (timestamped subfolder created within)    | `output`                           |
| `--reference_audio`      | Path to specific reference audio for voice cloning (overrides sampling)   | (Auto-sampled if not provided)   |
| `--hf_auth_token`        | Hugging Face auth token (alternative to .env file)                          | `None`                             |
| `--asr_model`            | Whisper model size (e.g., tiny, base, small, medium, large-v1/v2/v3)      | `large-v3`                         |
| `--translation_model`    | NLLB translation model to use                                               | `facebook/nllb-200-distilled-600M` |
| `--use_llm`              | Use Mistral LLM for translation instead of NLLB                             | `False`                            |
| `--diarize`              | Enable speaker diarization for multi-speaker videos                       | `False`                            |
| `--run_lip_sync`         | Enable lip synchronization using LatentSync                                 | `False`                            |
| `--use_gpu`              | Use GPU for supported steps if available                                      | `False`                            |
| `--gpu_index`            | Specify which GPU index to use (if multiple GPUs)                           | `0`                                |
| `--batch_size_asr`       | Batch size for ASR inference                                                | `16`                               |
| `--batch_size_tts`       | Batch size for TTS inference                                                | `4`                                |
| `--max_segment_length`   | Max length (seconds) for audio segments passed to ASR                       | `30`                               |
| `--cleanup_temp`         | Remove temporary files after pipeline completion                          | `True`                             |

## Output Structure

A timestamped directory (e.g., `output/20231027_103000/`) is created containing:

```
output/<timestamp>/
├── audio/                 # Original extracted audio (.wav)
├── separated_audio/       # Separated speech and background audio
├── transcriptions/        # English transcriptions with timestamps (JSON, SRT, TXT)
├── translations/          # Hindi translations (JSON)
├── diarization/           # Speaker diarization results (JSON/RTTM)
├── reference_samples/     # Selected reference audio samples per speaker
├── tts_output/            # Generated Hindi TTS audio segments
├── final_audio/           # Combined final Hindi audio
├── final_video/           # Final lip-synced video (if lip-sync enabled)
└── results/               # Log files, final metadata
```

## Requirements

### Hardware Recommendations
*   **GPU**: NVIDIA GPU with CUDA support is highly recommended.
    *   ASR (Whisper large-v3): ~10GB VRAM
    *   TTS (XTTS v2): ~6GB VRAM
    *   Translation (NLLB 3.3B): ~7GB VRAM (smaller models require less)
    *   Lip Sync (LatentSync): ~8GB VRAM
    *   Audio Separation (Demucs): VRAM usage varies, GPU helps speed.
*   **CPU**: Modern multi-core CPU (8+ cores recommended)
*   **RAM**: Minimum 16GB, 32GB+ recommended, especially without a high-VRAM GPU.
*   **Storage**: SSD recommended. ~20-30GB for models + space for intermediate files and output.

### Software Dependencies
*   Python 3.10+
*   PyTorch 2.0+
*   FFmpeg
*   CUDA 11.8+ (for NVIDIA GPU acceleration)
*   Git LFS
*   See `requirements.txt` for specific Python package dependencies.

### Model Storage
*   Whisper (large-v3): ~3GB
*   XTTS v2: ~1.5GB
*   NLLB-200 (600M): ~2.4GB
*   NLLB-200 (3.3B): ~13GB
*   LatentSync: ~2GB
*   Demucs: around ~100MB 

## XTTS Fine-Tuning

For higher voice cloning fidelity for specific speakers, especially to improve naturalness and accuracy with Hindi speech where zero-shot cloning might struggle, you can fine-tune an XTTS model:

1.  **Prepare Data:** Use the `reference_sampling` module or manually select 5-10 minutes of high-quality, clean audio samples for the target speaker. Place them in a directory like `data/finetune_samples/speaker_name/`.
2.  **Run Fine-tuning:** Use the provided `src/finetune_xtts_indic.py` script (adapt parameters as needed):
    ```bash
    python src/finetune_xtts_indic.py \\
      --model_name "tts_models/multilingual/multi-dataset/xtts_v2" \\
      --dataset_path data/finetune_samples/ \\
      --output_base_path finetuned_models/ \\
      --language "hi" \\ # Language of your training data
      --epochs 100 \\
      --batch_size 4 \\
      --use_gpu True
    ```
3.  **Use Fine-tuned Model:** Reference the output path of the fine-tuned model during TTS generation in the main pipeline (pass the saved sweights path).

## LatentSync Fine-Tuning

For improved lip-sync accuracy, especially for specific speakers or challenging video conditions, LatentSync can also be fine-tuned. This is particularly relevant for Hindi dubbing, as Hindi phonemes differ significantly from English, and fine-tuning can help the model adapt better. This typically involves training the SyncNet and UNet components on speaker-specific video data.

1.  **Prepare Data:** You will need suitable video footage paired with corresponding audio for the target speaker(s).
2.  **Run Training:** Explore the scripts within the `LatentSync` directory, such as `train_syncnet.sh` and `train_unet.sh`, as starting points.
3.  **Consult LatentSync Documentation:** For detailed instructions on data preparation, training procedures, and using fine-tuned LatentSync models, please refer to the documentation within the `LatentSync` directory itself (e.g., its `README.md` file).

## Future Development / Considerations

*   Integration with different TTS models (e.g., tried with parler).
*   Improved post-processing for translation naturalness.
*   Batch processing capabilities.
*   Web interface for easier management.
*   More sophisticated audio processing options.

## License


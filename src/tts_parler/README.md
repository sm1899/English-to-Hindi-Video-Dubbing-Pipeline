# Local TTS for Hindi Voice Cloning

This module provides integration with local TTS models for high-quality voice cloning in the English-to-Hindi dubbing pipeline. It uses XTTS v2 by default for zero-shot voice cloning capabilities, but supports custom models as well.

## Features

- High-quality voice cloning using XTTS v2 or other local TTS models
- Support for multiple reference samples per speaker
- Consistent speaker voice across segments
- Full integration with the existing dubbing pipeline
- Proper handling of speaker diarization data
- Background music separation support

## Installation

Install the required dependencies:

```bash
pip install TTS
```

## Usage

### Standalone Usage

You can use the Local TTS module directly for simple text-to-speech generation:

```python
from src.tts_parler.parler_tts import generate_speech

output_path = generate_speech(
    text="हैलो दुनिया",  # Hindi text
    reference_audio="path/to/reference_audio.wav",
    output_path="output.wav",
    language="hi",
    use_gpu=True
)
```

For multiple reference samples:

```python
output_path = generate_speech(
    text="हैलो दुनिया",
    reference_audio=["sample1.wav", "sample2.wav", "sample3.wav"],
    output_path="output.wav",
    language="hi"
)
```

### Pipeline Integration

To use the local TTS with the full dubbing pipeline, use the `pipeline_parler.py` script (despite the name, it now uses local TTS):

```bash
python src/pipeline_parler.py \
  --video_path path/to/video.mp4 \
  --output_dir output/ \
  --use_gpu
```

Additional options:

```bash
  --asr_model {tiny,base,small,medium,large}
                        Whisper model size to use for transcription
  --translation_model TRANSLATION_MODEL
                        Translation model to use
  --skip_diarization    Skip speaker diarization
  --pyannote_token PYANNOTE_TOKEN
                        Authentication token for PyAnnote
  --model_path MODEL_PATH
                        Path to a custom TTS model (leave empty for default XTTS v2)
  --voice_settings VOICE_SETTINGS
                        JSON string with voice settings (speed)
```

Example with voice settings:

```bash
python src/pipeline_parler.py \
  --video_path video.mp4 \
  --output_dir output/ \
  --use_gpu \
  --voice_settings '{"speed": 0.95}'
```

## Voice Settings

You can customize the voice output with the following parameters:

- `speed`: Adjust the speaking rate (0.5-2.0, default 1.0)

## Integration API

For developers who want to integrate local TTS into their own scripts:

```python
from src.tts_parler.pipeline_integration import generate_tts_for_segments

# Generate TTS for all segments with the same interface as the original pipeline
tts_segments = generate_tts_for_segments(
    translation=translation_data,
    reference_samples=reference_samples,
    output_dir="output/tts",
    use_gpu=True,
    model_path=None  # Use default XTTS v2 model
)
```

## Using Custom Models

You can use your own trained TTS models by providing the model path:

```python
from src.tts_parler.parler_tts import generate_speech

output_path = generate_speech(
    text="हैलो दुनिया",
    reference_audio="reference.wav",
    output_path="output.wav",
    language="hi",
    model_path="path/to/your/model"
)
```

In the pipeline:

```bash
python src/pipeline_parler.py \
  --video_path video.mp4 \
  --output_dir output/ \
  --use_gpu \
  --model_path path/to/your/model
```

## Performance Considerations

- For best results, use a GPU with at least 4GB VRAM
- XTTS v2 is resource-intensive but produces high-quality speech
- The first initialization may take some time as models are loaded

## Troubleshooting

- If you encounter `ImportError`, make sure you've installed the TTS library
- Verify that reference audio samples are clean and contain a single speaker
- For best results, use multiple reference samples for each speaker
- If you're getting GPU out-of-memory errors, try using a smaller model or switch to CPU with `--no-gpu` 
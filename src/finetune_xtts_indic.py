import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.utils.audio import AudioProcessor

# --- Configuration ---

# Path to your Indic TTS dataset
DATASET_PATH = "../dataset" # Assuming 'dataset' is one level up from 'src'
METADATA_FILE = "metadata.csv" # Standard metadata file name
LANGUAGE_CODE = "hi"          # Language code for Hindi
OUTPUT_PATH = "xtts_v2_indic_finetuned" # Directory to save the fine-tuned model
CHECKPOINT_DIR = os.path.join(OUTPUT_PATH, "checkpoints") # Directory for checkpoints
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Training hyperparameters (adjust as needed)
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
GRAD_ACCUM_STEPS = 2 # Accumulate gradients over 2 steps to simulate a larger batch size

# --- 1. Load Dataset ---

# Check if metadata file exists
metadata_path = os.path.join(DATASET_PATH, METADATA_FILE)
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found at: {metadata_path}. "
                          f"Please ensure '{METADATA_FILE}' exists in '{DATASET_PATH}'.")

# Load dataset samples
# This assumes the standard LJSpeech-like format: wav_file|transcription
# Adjust the formatter if your dataset has a different structure
train_samples, eval_samples = load_tts_samples(
    datasets=[{
        "formatter": "lj", # Use 'lj' formatter for wav_file|text format
        "dataset_name": "indic_tts",
        "path": DATASET_PATH,
        "meta_file_train": METADATA_FILE,
        "meta_file_val": None, # Use a portion of the training data for validation if no separate val file
        "language": LANGUAGE_CODE,
    }],
    eval_split=True,
    eval_split_max_size=256, # Max samples for evaluation
    eval_split_size=0.1,     # Use 10% of data for evaluation
)

if not train_samples:
    raise ValueError(f"No training samples found. Check dataset path and '{METADATA_FILE}' content.")
if not eval_samples:
    print("Warning: No evaluation samples created. Splitting might have failed or dataset is too small.")
    eval_samples = train_samples[:EVAL_BATCH_SIZE] # Use a small part of training data for eval

print(f"Found {len(train_samples)} training samples and {len(eval_samples)} evaluation samples.")

# --- 2. Load Model Configuration ---

# Download model if not present
ModelManager().download_model(MODEL_NAME)
model_path = os.path.join(ModelManager().output_prefix, MODEL_NAME.replace("/", "--"))

# Load the base XTTS v2 config
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))

# Update config for fine-tuning
config.output_path = OUTPUT_PATH
config.model_args.num_chars = None # Let the model determine char count from data
config.audio.sample_rate = 24000   # XTTS v2 preferred sample rate
config.batch_size = BATCH_SIZE
config.eval_batch_size = EVAL_BATCH_SIZE
config.num_loader_workers = 4      # Adjust based on your system cores
config.num_eval_loader_workers = 2
config.run_eval = True             # Evaluate during training
config.test_delay_epochs = -1      # No testing during training
config.epochs = NUM_EPOCHS
config.text_cleaner = "multilingual_cleaners"
config.use_phonemes = False        # XTTS works directly with characters
config.phoneme_language = None
config.compute_input_seq_lens = False
config.print_step = 25
config.print_eval = True
config.mixed_precision = True      # Use mixed precision for faster training (requires compatible GPU)
config.save_step = 1000            # Save checkpoint every 1000 steps
config.save_n_checkpoints = 5      # Keep the last 5 checkpoints
config.save_best_after = 1000      # Start saving best model after 1000 steps
config.gradient_accumulation_steps = GRAD_ACCUM_STEPS

# --- 3. Initialize Audio Processor ---
# Required for tokenization during data loading
ap = AudioProcessor.init_from_config(config)

# --- 4. Initialize Model ---
model = Xtts.init_from_config(config, custom_model_path=model_path)

# --- 5. Initialize Trainer ---
trainer_args = TrainerArgs(
    restore_path=None, # Set to a checkpoint path to resume training
    skip_train_epoch=False,
)

trainer = Trainer(
    args=trainer_args,
    config=config,
    output_path=OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap}
)

# --- 6. Start Fine-tuning ---
print("Starting fine-tuning...")
trainer.fit()

print(f"Fine-tuning complete. Model saved to: {OUTPUT_PATH}") 
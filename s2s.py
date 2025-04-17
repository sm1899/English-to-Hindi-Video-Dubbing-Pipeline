from transformers import Wav2Vec2Processor, NllbTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torchaudio
import torch
import soundfile as sf
import numpy as np
import os
from gtts import gTTS

def load_and_resample_audio(audio_path, target_sr=16000):
    # Check if file exists
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Use soundfile to load audio
    audio, orig_freq = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if orig_freq != target_sr:
        audio = torch.tensor(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=target_sr)
        audio = audio.squeeze(0).numpy()
    
    return audio

# Load processors and tokenizers
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Load ZeroSwot Encoder
commit_hash = "30d17145fd8e040430bbfcf74a011070fa83debd"
zeroswot_encoder = AutoModel.from_pretrained(
    "johntsi/ZeroSwot-Medium_asr-mustc_en-to-200", trust_remote_code=True, revision=commit_hash,
)
zeroswot_encoder.eval()
zeroswot_encoder.to("cuda")

# Load NLLB Model
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model.eval()
nllb_model.to("cuda")

path_to_audio_file = "trim_audio_60sec.wav"


# Load audio file
audio = load_and_resample_audio(path_to_audio_file) # you can use "resources/sample.wav" for testing
input_values = processor(audio, sampling_rate=16000, return_tensors="pt").to("cuda")

# translation to German
compressed_embeds, attention_mask = zeroswot_encoder(**input_values)
predicted_ids = nllb_model.generate(
    inputs_embeds=compressed_embeds,
    attention_mask=attention_mask,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("deu_Latn"),
    num_beams=5,
)
translation = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
print("German translation:")
print(translation)

# Generate German speech from the translation
output_audio_file = "output_german_speech.mp3"
tts = gTTS(text=translation, lang='de')
tts.save(output_audio_file)
print(f"\nAudio output saved to: {output_audio_file}")

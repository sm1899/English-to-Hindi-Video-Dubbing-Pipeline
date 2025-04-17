#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/stage2.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 2.0 \
    --video_path "assets/srk_demo_2.webm" \
    --audio_path "assets/srk_demo_2_speech_only.wav" \
    --video_out_path "video_out.mp4"

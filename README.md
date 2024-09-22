# Implementation of Soundstream, An Audio Codec for Speech Tasks

This repository contains an implementation of **Soundstream**, an audio codec designed for speech tasks. The model is built using **PyTorch Lightning** and supports multi-GPU training for faster and more efficient training.
This verison of Soundstream should output audio in 24kHz

## Current Features

1. **Model Code**: The core Soundstream model architecture is implemented.
2. **Training Script**: Supports multi-GPU training through PyTorch Lightning.
3. **Hugging Face Dataset Support**: Uses the **GPT OMNI Audio Dataset**, available [here](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K), for training on speech tasks.

## Coming Soon

1. **Argparse Support**: Adding `argparse` for finer control over training settings, including the ability to choose custom datasets.
2. **More Discriminators**: Plans to implement **MultiScale** and **MultiFrequency** discriminators to enhance model performance.
3. **Transformer Blocks**: Adding transformer layers to the encoder and decoder modules, similar to the **Mimi Codec** architecture.

---

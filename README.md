# Machine Learning Playground

A collection of machine learning experiments implemented in Rust using the [Burn](https://burn.dev/) framework and [Ollama](https://ollama.com/).

## Projects

### 1. Classification
A neural network for classification tasks using Burn. Includes training, validation, checkpointing, and inference.

### 2. Diffusion
A diffusion model implementation in Rust/Burn with noise scheduling, training, and image generation.

### 3. Image Captioning
Generates captions for images using a local Ollama server.

### 4. Chat
A simple chat interface powered by a local Ollama instance.

## Prerequisites

- **Rust** (edition 2024) — [rustup.rs](https://rustup.rs/)
- **Ollama** (for projects 3 and 4) — [ollama.com](https://ollama.com/)

## Getting Started

Each project is a standalone Rust crate. To build and run:

```bash
cd 1_classification
cargo run
```

## Project Structure

```
├── 1_classification/    # Neural network classification with Burn
├── 2_diffusion/         # Diffusion model with Burn
├── 3_image_captioning/  # Image captioning via Ollama
├── 4_chat/              # Chat interface via Ollama
└── pytorch_setup_guide.md
```

## License

MIT

# Machine Learning Playground

A hands-on collection of ML projects in Rust — from training neural networks to generating images to chatting with a local LLM. Built with [Burn](https://burn.dev/) and [Ollama](https://ollama.com/).

## Prerequisites

| Requirement | Projects | Install |
|-------------|----------|---------|
| **Rust** (edition 2024) | All | [rustup.rs](https://rustup.rs/) |
| **Ollama** | 3, 4 | [ollama.com/download](https://ollama.com/download) |
| **gemma4:e4b model** | 3, 4 | `ollama pull gemma4:e4b` |

> The MNIST dataset (used by projects 1–3) is downloaded automatically by Burn on first run.

---

## Project 1 — MNIST Classification

Train a CNN to recognize handwritten digits (0–9), then test it.

**Architecture:** Two conv layers (1→8→16 channels) → adaptive pooling → dropout (0.5) → two linear layers (1024→512→10).

### Train

```bash
cd 1_classification
cargo run -- train
```

Trains for 10 epochs with batch size 64 and lr 1e-4. Saves the model to `artifacts/`.

### Run inference

```bash
cargo run -- infer 20       # classify 20 test samples
```

Output:

```
Sample 1: predicted=7, actual=7 ✓
Sample 2: predicted=2, actual=2 ✓
Sample 3: predicted=1, actual=8 ✗
...
Accuracy: 18/20 (90.00%)
```

### Customize training

Edit the defaults in `src/training.rs`:

```rust
num_epochs: 10,
batch_size: 64,
learning_rate: 1e-4,
dropout: 0.5,
hidden_size: 512,
```

---

## Project 2 — Diffusion (Image Generation)

A DDPM (Denoising Diffusion Probabilistic Model) that learns to generate synthetic handwritten digits.

**Architecture:** UNet with time + class conditioning, 1000 diffusion timesteps, linear beta schedule (1e-4 → 0.02).

### Train

```bash
cd 2_diffusion
cargo run -- train
```

Trains for 10 epochs. Outputs per-epoch loss:

```
Epoch 1/10: avg_loss = 0.432156
Epoch 2/10: avg_loss = 0.198743
...
```

### Generate digits

```bash
cargo run -- generate 5 3     # generate 3 images of the digit "5"
cargo run -- generate 0 1     # generate 1 image of "0"
```

Saves 28×28 PNGs to `artifacts/generated/`:

```
artifacts/generated/digit_5_sample_0.png
artifacts/generated/digit_5_sample_1.png
artifacts/generated/digit_5_sample_2.png
```

---

## Project 3 — Image Captioning

Uses Ollama's vision model to describe images. Works with any PNG or with MNIST test samples.

### Setup

Make sure Ollama is running:

```bash
ollama serve              # start the server (if not already running)
ollama pull gemma4:e4b    # download the model (~5 GB)
```

### Caption any image

```bash
cd 3_image_captioning
cargo run -- caption photo.png
```

### Benchmark on MNIST

```bash
cargo run -- caption-mnist 10    # test on 10 MNIST samples (default: 5)
```

Output:

```
Sample 1 (idx=997, label=6): 6  [correct]
Sample 2 (idx=1994, label=3): 3  [correct]
Sample 3 (idx=2991, label=9): 8  [WRONG]
...
Result: 8/10 correctly identified
```

MNIST images are upscaled from 28×28 to 112×112 before sending to the vision model.

---

## Project 4 — Chat

An interactive multi-turn chat with a local LLM. Maintains full conversation history so the model can reference earlier messages.

### Run

```bash
cd 4_chat
cargo run
```

```
Chat with Gemma4 (type 'exit' or 'quit' to end)
---
You: What is a transformer in machine learning?
Gemma4: A transformer is a neural network architecture...
You: How does attention work in that context?
Gemma4: Attention allows the model to weigh the importance...
You: exit
Goodbye!
```

---

## Project Structure

```
├── 1_classification/         # CNN digit classifier (Burn)
│   └── src/
│       ├── main.rs           # CLI: train / infer
│       ├── model.rs          # CNN architecture
│       ├── training.rs       # Training loop + config
│       ├── data.rs           # MNIST data loading
│       └── inference.rs      # Inference logic
│
├── 2_diffusion/              # DDPM image generator (Burn)
│   └── src/
│       ├── main.rs           # CLI: train / generate
│       ├── model/
│       │   ├── mod.rs        # UNet architecture
│       │   ├── blocks.rs     # ResBlock, DownBlock, UpBlock
│       │   └── embeddings.rs # Time + class embeddings
│       ├── scheduler.rs      # DDPM noise schedule
│       ├── training.rs       # Training loop
│       ├── data.rs           # MNIST data loading
│       └── inference.rs      # Image generation
│
├── 3_image_captioning/       # Vision captioning (Ollama)
│   └── src/
│       ├── main.rs           # CLI: caption / caption-mnist
│       └── ollama.rs         # Ollama vision API client
│
├── 4_chat/                   # Chat interface (Ollama)
│   └── src/
│       ├── main.rs           # Interactive chat loop
│       └── ollama.rs         # Ollama chat API client
│
└── pytorch_setup_guide.md    # Reference: PyTorch setup from scratch
```

## Troubleshooting

**"connection refused" on projects 3 or 4** — Ollama isn't running. Start it with `ollama serve`.

**"model not found"** — Pull the model first: `ollama pull gemma4:e4b`.

**Slow first run** — Burn downloads the MNIST dataset on first use. Subsequent runs use the cached data.

**GPU issues** — Projects 1 and 2 use Burn's `wgpu` backend which auto-selects your GPU. If you hit driver issues, check that your GPU drivers are up to date.

## License

MIT

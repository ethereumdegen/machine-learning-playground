# Monocular Depth Estimation

A from-scratch UNet trained on the [NYU Depth V2](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2) dataset to predict per-pixel depth from a single RGB image. Built entirely in Rust using the [Burn](https://burn.dev) deep learning framework with GPU acceleration via wgpu.

## How it works

Given a single RGB photo, the model outputs a dense depth map — a grayscale image where each pixel's value represents how far that point is from the camera (in meters). This is a challenging problem because a single 2D image is inherently ambiguous about 3D structure; the network learns monocular depth cues like texture gradients, relative size, occlusion, and perspective.

### Architecture

The model is a **UNet** — an encoder-decoder with skip connections:

```
RGB Image [B, 3, 128, 160]
       |
    Conv 3→16
       |
  DownBlock 16→32  ───skip1───┐
       |                      |
  DownBlock 32→64  ───skip2───┤
       |                      |
  DownBlock 64→128 ───skip3───┤
       |                      |
    MidBlock 128              |
       |                      |
  UpBlock 128→64  ←───skip3───┘
       |                      |
  UpBlock 64→32   ←───skip2───┘
       |                      |
  UpBlock 32→16   ←───skip1───┘
       |
  GroupNorm → SiLU → Conv 16→1
       |
     ReLU (non-negative depth)
       |
  Depth Map [B, 1, 128, 160]
```

Each **DownBlock** contains a residual block (GroupNorm → SiLU → Conv3x3, twice, with a skip addition) followed by a stride-2 convolution that halves the spatial resolution. Each **UpBlock** uses a transposed convolution to upsample, concatenates the matching skip connection from the encoder, then applies a residual block to fuse the features. The **MidBlock** is two residual blocks at the bottleneck resolution.

Skip connections are crucial — they let the decoder recover fine spatial detail (edges, object boundaries) that would otherwise be lost during downsampling.

### Training

- **Dataset**: ~3,100 training / ~650 validation indoor scenes from NYU Depth V2, downloaded automatically from HuggingFace
- **Resolution**: Images resized to 128x160 (from the original 640x480)
- **Loss**: Mean squared error between predicted and ground-truth depth
- **Optimizer**: Adam (lr=1e-4, weight decay=1e-5)
- **Epochs**: 25
- **Batch size**: 2 (tuned for 6GB GPUs)

The dataset consists of indoor RGB-D scenes captured with a Kinect sensor. Depth maps are stored as TIFF float images in meters, with values typically ranging from 0 to ~10m.

## Usage

### Train

```bash
cargo run --release -- train
```

This downloads the NYU Depth V2 dataset from HuggingFace on first run (~1.5 GB), trains for 25 epochs, and saves the model to `artifacts/depth_model`.

### Inference

```bash
cargo run --release -- infer path/to/photo.png
```

Loads the trained model and produces a viridis-colorized depth map saved as `path/to/photo_depth.png`. Closer surfaces appear purple/blue, farther surfaces appear yellow/green.

## Project structure

```
src/
├── main.rs           # CLI entry point (train / infer)
├── training.rs       # Training loop with epoch logging
├── inference.rs      # Single-image inference + viridis colormap output
├── data.rs           # HuggingFace dataset download, parquet parsing, image decoding
└── model/
    ├── mod.rs        # DepthUNet definition and channel plan
    └── blocks.rs     # ResBlock, DownBlock, UpBlock, MidBlock
```

## Requirements

- Rust (edition 2024)
- A Vulkan/Metal/DX12-capable GPU (wgpu backend)
- ~3 GB GPU memory minimum (tested on GTX 1060 6GB)

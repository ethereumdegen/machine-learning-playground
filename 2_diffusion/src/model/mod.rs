pub mod blocks;
pub mod embeddings;

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::GroupNormConfig;
use burn::prelude::*;

use blocks::{DownBlock, DownBlockConfig, MidBlock, MidBlockConfig, UpBlock, UpBlockConfig};
use embeddings::{
    ClassEmbedding, ClassEmbeddingConfig, TimeEmbedding, TimeEmbeddingConfig,
};

/// Minimal UNet for DDPM on MNIST.
/// Channel plan: 1 -> 32 -> 64 -> 128 (down), 128 -> 64 -> 32 -> 1 (up)
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    // Input projection
    conv_in: Conv2d<B>,

    // Conditioning
    time_embed: TimeEmbedding<B>,
    class_embed: ClassEmbedding<B>,

    // Encoder
    down1: DownBlock<B>,  // 32 -> 32, spatial 28->14
    down2: DownBlock<B>,  // 32 -> 64, spatial 14->7
    down3: DownBlock<B>,  // 64 -> 128, spatial 7->3 (7/2 = 3 with stride 2)

    // Middle
    mid: MidBlock<B>,

    // Decoder
    up3: UpBlock<B>,  // 128+128 -> 64, spatial 3->6 (but we need 7)
    up2: UpBlock<B>,  // 64+64 -> 32, spatial 7->14
    up1: UpBlock<B>,  // 32+32 -> 32, spatial 14->28

    // Output
    norm_out: burn::nn::GroupNorm<B>,
    conv_out: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct UNetConfig {
    #[config(default = 128)]
    cond_dim: usize,
    #[config(default = 10)]
    num_classes: usize,
}

impl UNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        let cond_dim = self.cond_dim;

        UNet {
            conv_in: Conv2dConfig::new([1, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),

            time_embed: TimeEmbeddingConfig::new(64, cond_dim).init(device),
            class_embed: ClassEmbeddingConfig::new(self.num_classes, cond_dim).init(device),

            down1: DownBlockConfig::new(32, 32, cond_dim).init(device),
            down2: DownBlockConfig::new(32, 64, cond_dim).init(device),
            down3: DownBlockConfig::new(64, 128, cond_dim).init(device),

            mid: MidBlockConfig::new(128, cond_dim).init(device),

            // up3 input: 128 (from mid) + 128 (skip from down3) = 256
            up3: UpBlockConfig::new(256, 64, cond_dim).init(device),
            // up2 input: 64 (from up3 after crop) + 64 (skip from down2) = 128
            up2: UpBlockConfig::new(128, 32, cond_dim).init(device),
            // up1 input: 32 (from up2) + 32 (skip from down1) = 64
            up1: UpBlockConfig::new(64, 32, cond_dim).init(device),

            norm_out: GroupNormConfig::new(8, 32).init(device),
            conv_out: Conv2dConfig::new([32, 1], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
        }
    }
}

impl<B: Backend> UNet<B> {
    /// Forward pass.
    /// - `x`: noisy image [B, 1, 28, 28]
    /// - `t`: timestep as float [B]
    /// - `labels`: class labels [B]
    /// Returns predicted noise [B, 1, 28, 28]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        t: Tensor<B, 1>,
        labels: Tensor<B, 1, Int>,
    ) -> Tensor<B, 4> {
        // Conditioning
        let t_emb = self.time_embed.forward(t);
        let c_emb = self.class_embed.forward(labels);
        let cond = t_emb + c_emb; // [B, cond_dim]

        // Input
        let h = self.conv_in.forward(x); // [B, 32, 28, 28]

        // Encoder
        let (h, skip1) = self.down1.forward(h, cond.clone()); // h=[B,32,14,14], skip1=[B,32,28,28]
        let (h, skip2) = self.down2.forward(h, cond.clone()); // h=[B,64,7,7], skip2=[B,64,14,14]
        let (h, skip3) = self.down3.forward(h, cond.clone()); // h=[B,128,3,3], skip3=[B,128,7,7]

        // Middle
        let h = self.mid.forward(h, cond.clone()); // [B, 128, 3, 3]

        // Decoder - need to handle spatial dimension mismatches
        // up3: ConvTranspose2d 3->6, but skip3 is 7x7, so we pad up3 output to 7
        let h = self.up3.forward(h, skip3, cond.clone()); // ConvTranspose: 3->6, then we need 7
        let h = pad_to_match::<B>(h, 7, 7);

        let h = self.up2.forward(h, skip2, cond.clone()); // ConvTranspose: 7->14
        let h = self.up1.forward(h, skip1, cond); // ConvTranspose: 14->28

        // Output
        let h = self.norm_out.forward(h);
        let h = embeddings::silu(h);
        self.conv_out.forward(h)
    }
}

/// Pad tensor to target spatial dimensions using zero padding on the right/bottom.
fn pad_to_match<B: Backend>(x: Tensor<B, 4>, target_h: usize, target_w: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    if h == target_h && w == target_w {
        return x;
    }
    let device = x.device();
    let padded = Tensor::<B, 4>::zeros([b, c, target_h, target_w], &device);
    // Slice-assign: place x into top-left corner of padded
    padded.slice_assign(
        [0..b, 0..c, 0..h, 0..w],
        x,
    )
}

pub mod blocks;

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::GroupNormConfig;
use burn::prelude::*;

use blocks::{
    silu, DownBlock, DownBlockConfig, MidBlock, MidBlockConfig, UpBlock, UpBlockConfig,
};

/// UNet for monocular depth estimation.
/// Channel plan: 3 -> 16 -> 32 -> 64 -> 128 (down), 128 -> 64 -> 32 -> 16 -> 1 (up)
#[derive(Module, Debug)]
pub struct DepthUNet<B: Backend> {
    conv_in: Conv2d<B>,

    down1: DownBlock<B>,
    down2: DownBlock<B>,
    down3: DownBlock<B>,

    mid: MidBlock<B>,

    up3: UpBlock<B>,
    up2: UpBlock<B>,
    up1: UpBlock<B>,

    norm_out: burn::nn::GroupNorm<B>,
    conv_out: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct DepthUNetConfig {}

impl DepthUNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DepthUNet<B> {
        DepthUNet {
            conv_in: Conv2dConfig::new([3, 16], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),

            down1: DownBlockConfig::new(16, 32).init(device),
            down2: DownBlockConfig::new(32, 64).init(device),
            down3: DownBlockConfig::new(64, 128).init(device),

            mid: MidBlockConfig::new(128).init(device),

            up3: UpBlockConfig::new(128, 128, 64).init(device),
            up2: UpBlockConfig::new(64, 64, 32).init(device),
            up1: UpBlockConfig::new(32, 32, 16).init(device),

            norm_out: GroupNormConfig::new(8, 16).init(device),
            conv_out: Conv2dConfig::new([16, 1], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
        }
    }
}

impl<B: Backend> DepthUNet<B> {
    /// Forward pass.
    /// - `x`: RGB image [B, 3, 128, 160]
    /// Returns predicted depth [B, 1, 128, 160]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.conv_in.forward(x);

        let (h, skip1) = self.down1.forward(h);
        let (h, skip2) = self.down2.forward(h);
        let (h, skip3) = self.down3.forward(h);

        let h = self.mid.forward(h);

        let h = self.up3.forward(h, skip3);
        let h = self.up2.forward(h, skip2);
        let h = self.up1.forward(h, skip1);

        let h = self.norm_out.forward(h);
        let h = silu(h);
        let h = self.conv_out.forward(h);

        // ReLU to ensure non-negative depth
        burn::tensor::activation::relu(h)
    }
}

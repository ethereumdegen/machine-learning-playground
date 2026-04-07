use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use super::embeddings::silu;

/// Residual block with time+class conditioning.
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    cond_proj: Linear<B>,
    residual_conv: Option<Conv2d<B>>,
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    in_channels: usize,
    out_channels: usize,
    cond_dim: usize,
}

fn num_groups(channels: usize) -> usize {
    // Use groups that evenly divide channels; fall back to 1
    for g in [32, 16, 8, 4] {
        if channels % g == 0 && channels >= g {
            return g;
        }
    }
    1
}

impl ResBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        let residual_conv = if self.in_channels != self.out_channels {
            Some(Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1]).init(device))
        } else {
            None
        };

        ResBlock {
            norm1: GroupNormConfig::new(num_groups(self.in_channels), self.in_channels).init(device),
            conv1: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            norm2: GroupNormConfig::new(num_groups(self.out_channels), self.out_channels)
                .init(device),
            conv2: Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            cond_proj: LinearConfig::new(self.cond_dim, self.out_channels).init(device),
            residual_conv,
        }
    }
}

impl<B: Backend> ResBlock<B> {
    /// Forward pass. `cond` is the combined time+class embedding [B, cond_dim].
    pub fn forward(&self, x: Tensor<B, 4>, cond: Tensor<B, 2>) -> Tensor<B, 4> {
        let residual = match &self.residual_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        let h = self.norm1.forward(x);
        let h = silu(h);
        let h = self.conv1.forward(h);

        // Add conditioning: project cond to [B, out_channels, 1, 1] and add
        let cond = self.cond_proj.forward(cond);
        let cond = silu(cond);
        let [b, c] = cond.dims();
        let cond = cond.reshape([b, c, 1, 1]);
        let h = h + cond;

        let h = self.norm2.forward(h);
        let h = silu(h);
        let h = self.conv2.forward(h);

        h + residual
    }
}

/// Downsample block: ResBlock + 2x2 stride-2 convolution
#[derive(Module, Debug)]
pub struct DownBlock<B: Backend> {
    res_block: ResBlock<B>,
    downsample: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct DownBlockConfig {
    in_channels: usize,
    out_channels: usize,
    cond_dim: usize,
}

impl DownBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DownBlock<B> {
        DownBlock {
            res_block: ResBlockConfig::new(self.in_channels, self.out_channels, self.cond_dim)
                .init(device),
            downsample: Conv2dConfig::new([self.out_channels, self.out_channels], [2, 2])
                .with_stride([2, 2])
                .init(device),
        }
    }
}

impl<B: Backend> DownBlock<B> {
    /// Returns (downsampled output, skip connection before downsampling)
    pub fn forward(&self, x: Tensor<B, 4>, cond: Tensor<B, 2>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let h = self.res_block.forward(x, cond);
        let down = self.downsample.forward(h.clone());
        (down, h)
    }
}

/// Upsample block: concat skip + ResBlock + 2x nearest upsample via ConvTranspose
#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    res_block: ResBlock<B>,
    upsample: burn::nn::conv::ConvTranspose2d<B>,
}

#[derive(Config, Debug)]
pub struct UpBlockConfig {
    /// in_channels includes skip connection, so it's typically 2 * skip_channels
    in_channels: usize,
    out_channels: usize,
    cond_dim: usize,
}

impl UpBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpBlock<B> {
        UpBlock {
            res_block: ResBlockConfig::new(self.in_channels, self.out_channels, self.cond_dim)
                .init(device),
            upsample: burn::nn::conv::ConvTranspose2dConfig::new(
                [self.out_channels, self.out_channels],
                [2, 2],
            )
            .with_stride([2, 2])
            .init(device),
        }
    }
}

impl<B: Backend> UpBlock<B> {
    /// Takes current features and skip connection, returns upsampled features.
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        skip: Tensor<B, 4>,
        cond: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        // Concat along channel dim
        let h = Tensor::cat(vec![x, skip], 1);
        let h = self.res_block.forward(h, cond);
        self.upsample.forward(h)
    }
}

/// Middle block: ResBlock -> ResBlock
#[derive(Module, Debug)]
pub struct MidBlock<B: Backend> {
    res1: ResBlock<B>,
    res2: ResBlock<B>,
}

#[derive(Config, Debug)]
pub struct MidBlockConfig {
    channels: usize,
    cond_dim: usize,
}

impl MidBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MidBlock<B> {
        MidBlock {
            res1: ResBlockConfig::new(self.channels, self.channels, self.cond_dim).init(device),
            res2: ResBlockConfig::new(self.channels, self.channels, self.cond_dim).init(device),
        }
    }
}

impl<B: Backend> MidBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>, cond: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.res1.forward(x, cond.clone());
        self.res2.forward(h, cond)
    }
}

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig};
use burn::prelude::*;

pub fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sigmoid = burn::tensor::activation::sigmoid(x.clone());
    x * sigmoid
}

fn num_groups(channels: usize) -> usize {
    for g in [32, 16, 8, 4] {
        if channels % g == 0 && channels >= g {
            return g;
        }
    }
    1
}

/// Residual block (no conditioning).
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    residual_conv: Option<Conv2d<B>>,
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    in_channels: usize,
    out_channels: usize,
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
            residual_conv,
        }
    }
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match &self.residual_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        let h = self.norm1.forward(x);
        let h = silu(h);
        let h = self.conv1.forward(h);

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
}

impl DownBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DownBlock<B> {
        DownBlock {
            res_block: ResBlockConfig::new(self.in_channels, self.out_channels).init(device),
            downsample: Conv2dConfig::new([self.out_channels, self.out_channels], [2, 2])
                .with_stride([2, 2])
                .init(device),
        }
    }
}

impl<B: Backend> DownBlock<B> {
    /// Returns (downsampled output, skip connection before downsampling)
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let h = self.res_block.forward(x);
        let down = self.downsample.forward(h.clone());
        (down, h)
    }
}

/// Upsample block: ConvTranspose2d upsample + concat skip + ResBlock
#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    res_block: ResBlock<B>,
    upsample: burn::nn::conv::ConvTranspose2d<B>,
}

#[derive(Config, Debug)]
pub struct UpBlockConfig {
    x_channels: usize,
    skip_channels: usize,
    out_channels: usize,
}

impl UpBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UpBlock<B> {
        UpBlock {
            upsample: burn::nn::conv::ConvTranspose2dConfig::new(
                [self.x_channels, self.x_channels],
                [2, 2],
            )
            .with_stride([2, 2])
            .init(device),
            res_block: ResBlockConfig::new(
                self.x_channels + self.skip_channels,
                self.out_channels,
            )
            .init(device),
        }
    }
}

impl<B: Backend> UpBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>, skip: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.upsample.forward(x);
        let [_b, _c, sh, sw] = skip.dims();
        let [_, _, hh, hw] = h.dims();
        let h = if hh != sh || hw != sw {
            let [b, c, _, _] = h.dims();
            let device = h.device();
            let padded = Tensor::<B, 4>::zeros([b, c, sh, sw], &device);
            padded.slice_assign([0..b, 0..c, 0..hh, 0..hw], h)
        } else {
            h
        };
        let h = Tensor::cat(vec![h, skip], 1);
        self.res_block.forward(h)
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
}

impl MidBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MidBlock<B> {
        MidBlock {
            res1: ResBlockConfig::new(self.channels, self.channels).init(device),
            res2: ResBlockConfig::new(self.channels, self.channels).init(device),
        }
    }
}

impl<B: Backend> MidBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.res1.forward(x);
        self.res2.forward(h)
    }
}

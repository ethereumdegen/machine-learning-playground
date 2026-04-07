use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

#[derive(Clone)]
pub struct DiffusionBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DiffusionBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionBatch<B: Backend> {
    /// Images normalized to [-1, 1], shape [B, 1, 28, 28]
    pub images: Tensor<B, 4>,
    /// Class labels, shape [B]
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MnistItem, DiffusionBatch<B>> for DiffusionBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> DiffusionBatch<B> {
        let images: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|item| {
                let data = TensorData::from(item.image).convert::<f32>();
                let tensor = Tensor::<B, 2>::from_data(data, &self.device);
                // Normalize from [0,255] to [-1,1]
                let tensor = tensor / 255.0 * 2.0 - 1.0;
                tensor.reshape([1, 1, 28, 28])
            })
            .collect();

        let labels: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(item.label as i64).elem::<i64>()]),
                    &self.device,
                )
            })
            .collect();

        DiffusionBatch {
            images: Tensor::cat(images, 0),
            labels: Tensor::cat(labels, 0),
        }
    }
}

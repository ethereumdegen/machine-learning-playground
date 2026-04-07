use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use image::{GrayImage, Luma};

use crate::model::{UNet, UNetConfig};
use crate::scheduler::{DDPMScheduler, NUM_TIMESTEPS};

pub fn generate<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    digit: u8,
    num_samples: usize,
) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/diffusion_model").into(), &device)
        .expect("Trained model should exist at artifacts/diffusion_model");

    let model: UNet<B> = UNetConfig::new().init::<B>(&device).load_record(record);
    let scheduler = DDPMScheduler::new();

    std::fs::create_dir_all(format!("{artifact_dir}/generated")).ok();

    for sample_idx in 0..num_samples {
        // Start from pure noise
        let mut x = Tensor::<B, 4>::random(
            [1, 1, 28, 28],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Class label
        let label = Tensor::<B, 1, Int>::from_ints([digit as i32], &device);

        // Reverse diffusion: t from T-1 down to 0
        for t in (0..NUM_TIMESTEPS).rev() {
            let t_tensor = Tensor::<B, 1>::from_floats([t as f32], &device);
            let noise_pred = model.forward(x.clone(), t_tensor, label.clone());
            x = scheduler.step(x, noise_pred, t);
        }

        // Convert to image
        // Clip to [-1, 1] and rescale to [0, 255]
        let x = x.clamp(-1.0, 1.0);
        let x = (x + 1.0) / 2.0 * 255.0;

        let pixels: Vec<f32> = x.reshape([28 * 28]).into_data().to_vec().unwrap();

        let img = GrayImage::from_fn(28, 28, |col, row| {
            let idx = (row * 28 + col) as usize;
            Luma([pixels[idx] as u8])
        });

        let path = format!("{artifact_dir}/generated/digit_{digit}_sample_{sample_idx}.png");
        img.save(&path).expect("Failed to save image");
        println!("Saved {path}");
    }
}

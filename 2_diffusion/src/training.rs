use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::Optimizer,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};
use rand::Rng;

use crate::data::DiffusionBatcher;
use crate::model::UNetConfig;
use crate::scheduler::{DDPMScheduler, NUM_TIMESTEPS};

const NUM_EPOCHS: usize = 10;
const BATCH_SIZE: usize = 64;
const LEARNING_RATE: f64 = 1e-4;
const SEED: u64 = 42;

pub fn train<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();

    B::seed(SEED);

    let scheduler = DDPMScheduler::new();
    let model = UNetConfig::new().init::<B>(&device);
    let optim_config = burn::optim::AdamConfig::new().with_weight_decay(Some(
        burn::optim::decay::WeightDecayConfig::new(1e-5),
    ));
    let mut optim = optim_config.init::<B, _>();

    let batcher = DiffusionBatcher::<B>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(BATCH_SIZE)
        .shuffle(SEED)
        .num_workers(4)
        .build(MnistDataset::train());

    let mut rng = rand::thread_rng();
    let mut model = model;

    for epoch in 0..NUM_EPOCHS {
        let mut total_loss = 0.0f64;
        let mut num_batches = 0usize;

        for batch in dataloader.iter() {
            let [batch_size, _, _, _] = batch.images.dims();

            // Sample random timesteps for each sample
            let timesteps: Vec<usize> = (0..batch_size)
                .map(|_| rng.gen_range(0..NUM_TIMESTEPS))
                .collect();

            // Create timestep tensor for the model (as float)
            let t_tensor = Tensor::<B, 1>::from_floats(
                timesteps.iter().map(|&t| t as f32).collect::<Vec<f32>>().as_slice(),
                &device,
            );

            // Sample noise
            let noise = Tensor::<B, 4>::random(
                batch.images.shape(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );

            // Forward diffusion
            let x_t = scheduler.add_noise(batch.images.clone(), noise.clone(), &timesteps);

            // Predict noise
            let noise_pred = model.forward(x_t, t_tensor, batch.labels.clone());

            // MSE loss
            let diff = noise_pred - noise;
            let loss = (diff.clone() * diff).mean();

            // Track loss
            let loss_val: f64 = loss.clone().into_scalar().elem();
            total_loss += loss_val;
            num_batches += 1;

            // Backprop
            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grads);
        }

        let avg_loss = total_loss / num_batches as f64;
        println!("Epoch {}/{}: avg_loss = {:.6}", epoch + 1, NUM_EPOCHS, avg_loss);
    }

    // Save model
    model
        .save_file(format!("{artifact_dir}/diffusion_model"), &CompactRecorder::new())
        .expect("Model should be saved successfully");

    println!("Model saved to {artifact_dir}/diffusion_model");
}

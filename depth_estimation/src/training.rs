use burn::{
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    optim::Optimizer,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::data::{load_dataset, DepthBatcher, DepthItem};
use crate::model::DepthUNetConfig;

const NUM_EPOCHS: usize = 25;
const BATCH_SIZE: usize = 8;
const LEARNING_RATE: f64 = 1e-4;
const SEED: u64 = 42;

pub fn train<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();

    B::seed(SEED);

    let model = DepthUNetConfig::new().init::<B>(&device);
    let optim_config = burn::optim::AdamConfig::new().with_weight_decay(Some(
        burn::optim::decay::WeightDecayConfig::new(1e-5),
    ));
    let mut optim = optim_config.init::<B, _>();

    let train_data = load_dataset("train");
    let val_data = load_dataset("validation");

    let train_dataset = InMemDataset::new(train_data);
    let val_dataset = InMemDataset::new(val_data);

    let batcher_train = DepthBatcher::<B>::new(device.clone());
    let batcher_val = DepthBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(BATCH_SIZE)
        .shuffle(SEED)
        .num_workers(4)
        .build(train_dataset);

    let dataloader_val = DataLoaderBuilder::new(batcher_val)
        .batch_size(BATCH_SIZE)
        .build(val_dataset);

    let mut model = model;

    for epoch in 0..NUM_EPOCHS {
        // Training
        let mut total_loss = 0.0f64;
        let mut num_batches = 0usize;

        for batch in dataloader_train.iter() {
            let predicted = model.forward(batch.images);
            let diff = predicted - batch.depths;
            let loss = (diff.clone() * diff).mean();

            let loss_val: f64 = loss.clone().into_scalar().elem();
            total_loss += loss_val;
            num_batches += 1;

            let grads = loss.backward();
            let grads = burn::optim::GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grads);

            if num_batches % 50 == 0 {
                println!(
                    "  Epoch {}/{}, batch {}: loss = {:.6}",
                    epoch + 1,
                    NUM_EPOCHS,
                    num_batches,
                    loss_val
                );
            }
        }

        let avg_train_loss = total_loss / num_batches as f64;

        // Validation
        let mut val_loss = 0.0f64;
        let mut val_batches = 0usize;
        let model_valid = model.valid();

        for batch in dataloader_val.iter() {
            let predicted = model_valid.forward(batch.images);
            let diff = predicted - batch.depths;
            let loss = (diff.clone() * diff).mean();
            val_loss += loss.into_scalar().elem::<f64>();
            val_batches += 1;
        }

        let avg_val_loss = val_loss / val_batches as f64;
        println!(
            "Epoch {}/{}: train_loss = {:.6}, val_loss = {:.6}",
            epoch + 1,
            NUM_EPOCHS,
            avg_train_loss,
            avg_val_loss
        );
    }

    model
        .save_file(format!("{artifact_dir}/depth_model"), &CompactRecorder::new())
        .expect("Model should be saved successfully");

    println!("Model saved to {artifact_dir}/depth_model");
}

/// Simple in-memory dataset.
struct InMemDataset {
    items: Vec<DepthItem>,
}

impl InMemDataset {
    fn new(items: Vec<DepthItem>) -> Self {
        Self { items }
    }
}

impl burn::data::dataset::Dataset<DepthItem> for InMemDataset {
    fn get(&self, index: usize) -> Option<DepthItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

mod data;
mod inference;
mod model;
mod training;

use burn::backend::{Autodiff, Wgpu};
use training::{train, TrainingConfig};
use model::MnistModelConfig;
use burn::optim::AdamConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("train");

    let device = burn::backend::wgpu::WgpuDevice::default();

    match command {
        "train" => {
            type MyBackend = Wgpu;
            type MyAutodiffBackend = Autodiff<MyBackend>;

            let config = TrainingConfig::new(
                MnistModelConfig::new(10, 512),
                AdamConfig::new(),
            );

            train::<MyAutodiffBackend>("artifacts", config, device);
        }
        "infer" => {
            let num_samples = args.get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);

            inference::infer::<Wgpu>("artifacts", device, num_samples);
        }
        _ => {
            eprintln!("Usage: cargo run -- <train|infer> [num_samples]");
            std::process::exit(1);
        }
    }
}

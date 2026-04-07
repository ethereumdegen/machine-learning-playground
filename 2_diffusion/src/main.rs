mod data;
mod inference;
mod model;
mod scheduler;
mod training;

use burn::backend::{Autodiff, Wgpu};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("train");

    let device = burn::backend::wgpu::WgpuDevice::default();

    match command {
        "train" => {
            type B = Autodiff<Wgpu>;
            training::train::<B>("artifacts", device);
        }
        "generate" => {
            let digit: u8 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .expect("Usage: cargo run -- generate <digit> [count]");
            let count: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
            inference::generate::<Wgpu>("artifacts", device, digit, count);
        }
        _ => {
            eprintln!("Usage: cargo run -- <train|generate <digit> [count]>");
            std::process::exit(1);
        }
    }
}

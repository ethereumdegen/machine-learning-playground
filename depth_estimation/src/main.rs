mod data;
mod inference;
mod model;
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
        "infer" => {
            let image_path = args
                .get(2)
                .expect("Usage: cargo run -- infer <image.png>");
            inference::infer::<Wgpu>("artifacts", device, image_path);
        }
        _ => {
            eprintln!("Usage: cargo run -- <train|infer <image.png>>");
            std::process::exit(1);
        }
    }
}

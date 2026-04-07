mod ollama;

use burn::data::dataset::{vision::MnistDataset, Dataset};
use image::GrayImage;
use std::io::Cursor;

fn mnist_item_to_png(image_data: &[[f32; 28]; 28]) -> Vec<u8> {
    let img = GrayImage::from_fn(28, 28, |x, y| {
        let pixel = image_data[y as usize][x as usize];
        image::Luma([pixel as u8])
    });

    // Scale up to 112x112 so the model can see it better
    let resized = image::imageops::resize(&img, 112, 112, image::imageops::FilterType::Nearest);

    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    resized
        .write_to(&mut cursor, image::ImageFormat::Png)
        .expect("Failed to encode PNG");
    buf
}

fn caption_file(path: &str) {
    let png_bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Failed to read {path}: {e}");
        std::process::exit(1);
    });

    let prompt = "What digit (0-9) is shown in this image? Reply with just the digit and a brief description.";
    match ollama::caption_image(&png_bytes, prompt) {
        Ok(response) => println!("{response}"),
        Err(e) => eprintln!("Ollama error: {e}"),
    }
}

fn caption_mnist(count: usize) {
    println!("Loading MNIST test dataset...");
    let dataset = MnistDataset::test();
    let total = dataset.len();

    let prompt = "What single digit (0-9) is shown in this image? Reply with just the digit.";
    let mut correct = 0;

    for i in 0..count {
        // Pick a random-ish sample spread across the dataset
        let idx = (i * 997) % total;
        let item = dataset.get(idx).unwrap();
        let label = item.label;

        let png_bytes = mnist_item_to_png(&item.image);

        print!("Sample {i} (idx={idx}, label={label}): ");
        match ollama::caption_image(&png_bytes, prompt) {
            Ok(response) => {
                let response_trimmed = response.trim();
                // Check if the response contains the correct digit
                let predicted_correct = response_trimmed.contains(&label.to_string());
                if predicted_correct {
                    correct += 1;
                }
                let status = if predicted_correct { "correct" } else { "WRONG" };
                println!("{response_trimmed}  [{status}]");
            }
            Err(e) => println!("error: {e}"),
        }
    }

    println!("\nResult: {correct}/{count} correctly identified");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("caption") => {
            let path = args.get(2).unwrap_or_else(|| {
                eprintln!("Usage: cargo run -- caption <image.png>");
                std::process::exit(1);
            });
            caption_file(path);
        }
        Some("caption-mnist") => {
            let count: usize = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            caption_mnist(count);
        }
        _ => {
            println!("Usage:");
            println!("  cargo run -- caption <image.png>       Caption any image");
            println!("  cargo run -- caption-mnist [count]     Caption MNIST test samples (default: 5)");
        }
    }
}

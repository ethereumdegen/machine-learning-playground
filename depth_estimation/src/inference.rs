use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use image::RgbImage;

use crate::model::{DepthUNet, DepthUNetConfig};

const HEIGHT: usize = 64;
const WIDTH: usize = 80;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, image_path: &str) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/depth_model").into(), &device)
        .expect("Trained model should exist at artifacts/depth_model");

    let model: DepthUNet<B> = DepthUNetConfig::new().init::<B>(&device).load_record(record);

    // Load and preprocess input image
    let img = image::open(image_path).expect("Failed to open input image");
    let img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = img.to_rgb8();

    // Convert to CHW [0,1] tensor
    let mut data = vec![0.0f32; 3 * HEIGHT * WIDTH];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * HEIGHT * WIDTH + y * WIDTH + x] = pixel[c] as f32 / 255.0;
            }
        }
    }

    let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device);
    let input = input.reshape([1, 3, HEIGHT, WIDTH]);

    let depth = model.forward(input);
    let depth_data: Vec<f32> = depth.reshape([HEIGHT * WIDTH]).into_data().to_vec().unwrap();

    // Find min/max for normalization
    let min_d = depth_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_d = depth_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_d - min_d).max(1e-6);

    // Colorize using viridis-like palette
    let mut output = RgbImage::new(WIDTH as u32, HEIGHT as u32);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let val = (depth_data[y * WIDTH + x] - min_d) / range;
            let (r, g, b) = viridis(val);
            output.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    let output_path = format!(
        "{}_depth.png",
        image_path.strip_suffix(".png").or_else(|| image_path.strip_suffix(".jpg")).or_else(|| image_path.strip_suffix(".jpeg")).unwrap_or(image_path)
    );
    output.save(&output_path).expect("Failed to save output");
    println!("Depth map saved to {output_path}");
}

/// Simple viridis-like colormap: maps [0,1] to RGB.
fn viridis(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    // Approximate viridis with a few interpolation points
    // purple -> blue -> teal -> green -> yellow
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (
            68.0 + s * (49.0 - 68.0),
            1.0 + s * (54.0 - 1.0),
            84.0 + s * (149.0 - 84.0),
        )
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (
            49.0 + s * (33.0 - 49.0),
            54.0 + s * (145.0 - 54.0),
            149.0 + s * (140.0 - 149.0),
        )
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (
            33.0 + s * (94.0 - 33.0),
            145.0 + s * (201.0 - 145.0),
            140.0 + s * (98.0 - 140.0),
        )
    } else {
        let s = (t - 0.75) / 0.25;
        (
            94.0 + s * (253.0 - 94.0),
            201.0 + s * (231.0 - 201.0),
            98.0 + s * (37.0 - 98.0),
        )
    };
    (r as u8, g as u8, b as u8)
}

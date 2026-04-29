use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
};
use std::io::Cursor;

const HEIGHT: usize = 64;
const WIDTH: usize = 80;

#[derive(Clone, Debug)]
pub struct DepthItem {
    pub image: Vec<f32>, // [3 * H * W] CHW, [0,1]
    pub depth: Vec<f32>, // [H * W] meters
}

#[derive(Clone, Debug)]
pub struct DepthBatch<B: Backend> {
    pub images: Tensor<B, 4>, // [B, 3, H, W]
    pub depths: Tensor<B, 4>, // [B, 1, H, W]
}

#[derive(Clone)]
pub struct DepthBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DepthBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<DepthItem, DepthBatch<B>> for DepthBatcher<B> {
    fn batch(&self, items: Vec<DepthItem>) -> DepthBatch<B> {
        let images: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|item| {
                let data = TensorData::from(item.image.as_slice()).convert::<f32>();
                let tensor = Tensor::<B, 1>::from_data(data, &self.device);
                tensor.reshape([1, 3, HEIGHT, WIDTH])
            })
            .collect();

        let depths: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|item| {
                let data = TensorData::from(item.depth.as_slice()).convert::<f32>();
                let tensor = Tensor::<B, 1>::from_data(data, &self.device);
                tensor.reshape([1, 1, HEIGHT, WIDTH])
            })
            .collect();

        DepthBatch {
            images: Tensor::cat(images, 0),
            depths: Tensor::cat(depths, 0),
        }
    }
}

/// Download and parse NYU Depth V2 dataset from HuggingFace.
pub fn load_dataset(split: &str) -> Vec<DepthItem> {
    println!("Loading {split} split from sayakpaul/nyu_depth_v2...");

    let api = hf_hub::api::sync::Api::new().expect("Failed to create HF API");
    let repo = api.repo(hf_hub::Repo::with_revision(
        "sayakpaul/nyu_depth_v2".to_string(),
        hf_hub::RepoType::Dataset,
        "refs/convert/parquet".to_string(),
    ));

    // Splits are named partial-train / partial-validation on this branch
    let hf_split = format!("partial-{split}");

    // Download all shards (0000.parquet, 0001.parquet, ...)
    let mut items = Vec::new();
    for shard in 0.. {
        let filename = format!("default/{hf_split}/{shard:04}.parquet");
        match repo.get(&filename) {
            Ok(parquet_path) => {
                println!("Parsing {filename}...");
                items.extend(parse_parquet(&parquet_path));
            }
            Err(_) => break, // No more shards
        }
    }

    println!("Loaded {} samples for {split}", items.len());
    items
}

fn parse_parquet(path: &std::path::Path) -> Vec<DepthItem> {
    use arrow::array::{Array, BinaryArray, StructArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let file = File::open(path).expect("Failed to open parquet file");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("Failed to read parquet");
    let reader = builder.build().expect("Failed to build reader");

    let mut items = Vec::new();

    for batch in reader {
        let batch = batch.expect("Failed to read batch");
        let num_rows = batch.num_rows();

        let image_col = batch
            .column_by_name("image")
            .expect("No 'image' column");
        let depth_col = batch
            .column_by_name("depth_map")
            .expect("No 'depth_map' column");

        // image column is a struct with a "bytes" field
        let image_struct = image_col
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("image column should be struct");
        let image_bytes_col = image_struct
            .column_by_name("bytes")
            .expect("No 'bytes' in image struct");
        let image_bytes = image_bytes_col
            .as_any()
            .downcast_ref::<BinaryArray>()
            .or_else(|| None)
            .unwrap_or_else(|| {
                // Try LargeBinaryArray
                panic!("image bytes not BinaryArray - check arrow schema");
            });

        let depth_struct = depth_col
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("depth_map column should be struct");
        let depth_bytes_col = depth_struct
            .column_by_name("bytes")
            .expect("No 'bytes' in depth_map struct");
        let depth_bytes = depth_bytes_col
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap_or_else(|| {
                panic!("depth bytes not BinaryArray - check arrow schema");
            });

        for i in 0..num_rows {
            let img_data = image_bytes.value(i);
            let dep_data = depth_bytes.value(i);

            match (decode_rgb(img_data), decode_depth(dep_data)) {
                (Some(image), Some(depth)) => {
                    items.push(DepthItem { image, depth });
                }
                _ => {
                    eprintln!("Warning: failed to decode sample {i}, skipping");
                }
            }
        }
    }

    println!("Loaded {} samples", items.len());
    items
}

/// Decode PNG image bytes -> resize to HxW -> [0,1] CHW f32 vec
fn decode_rgb(bytes: &[u8]) -> Option<Vec<f32>> {
    let img = image::load_from_memory(bytes).ok()?;
    let img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = img.to_rgb8();

    let mut data = vec![0.0f32; 3 * HEIGHT * WIDTH];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                data[c * HEIGHT * WIDTH + y * WIDTH + x] = pixel[c] as f32 / 255.0;
            }
        }
    }
    Some(data)
}

/// Decode TIFF depth bytes -> resize to HxW -> f32 vec in meters
fn decode_depth(bytes: &[u8]) -> Option<Vec<f32>> {
    let cursor = Cursor::new(bytes);
    let mut decoder = tiff::decoder::Decoder::new(cursor).ok()?;
    let (orig_w, orig_h) = decoder.dimensions().ok()?;
    let orig_w = orig_w as usize;
    let orig_h = orig_h as usize;

    let result = decoder.read_image().ok()?;
    let float_data = match result {
        tiff::decoder::DecodingResult::F32(data) => data,
        tiff::decoder::DecodingResult::U16(data) => {
            // Some depth maps might be u16 encoded as millimeters
            data.iter().map(|&v| v as f32 / 1000.0).collect()
        }
        _ => return None,
    };

    // Bilinear resize to target dims
    let mut resized = vec![0.0f32; HEIGHT * WIDTH];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let src_y = y as f32 * (orig_h as f32 - 1.0) / (HEIGHT as f32 - 1.0);
            let src_x = x as f32 * (orig_w as f32 - 1.0) / (WIDTH as f32 - 1.0);

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(orig_h - 1);
            let x1 = (x0 + 1).min(orig_w - 1);

            let fy = src_y - y0 as f32;
            let fx = src_x - x0 as f32;

            let v00 = float_data[y0 * orig_w + x0];
            let v01 = float_data[y0 * orig_w + x1];
            let v10 = float_data[y1 * orig_w + x0];
            let v11 = float_data[y1 * orig_w + x1];

            resized[y * WIDTH + x] =
                v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy;
        }
    }

    Some(resized)
}

use burn::{
    data::dataset::{vision::MnistDataset, Dataset},
    prelude::*,
    record::{CompactRecorder, Recorder},
};

use crate::model::MnistModel;
use crate::training::TrainingConfig;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, num_samples: usize) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist at artifacts/model.mpk");

    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist");

    let model: MnistModel<B> = config.model.init::<B>(&device).load_record(record);

    let dataset = MnistDataset::test();

    let mut correct = 0;
    for i in 0..num_samples {
        let item = dataset.get(i).unwrap();
        let label = item.label;

        let image = TensorData::from(item.image).convert::<f32>();
        let image = Tensor::<B, 2>::from_data(image, &device).reshape([1, 28, 28]);
        let image = ((image / 255) - 0.1307) / 0.3081;

        let output = model.forward(image);
        let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar().elem::<i64>();

        let status = if predicted as u8 == label { correct += 1; "✓" } else { "✗" };
        println!("Sample {i}: predicted={predicted}, actual={label} {status}");
    }

    println!("\nAccuracy: {correct}/{num_samples} ({:.1}%)", 100.0 * correct as f64 / num_samples as f64);
}

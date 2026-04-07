use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

/// Sinusoidal positional embedding for timesteps.
pub fn sinusoidal_embedding<B: Backend>(
    timesteps: Tensor<B, 1>,
    embed_dim: usize,
) -> Tensor<B, 2> {
    let device = timesteps.device();
    let half_dim = embed_dim / 2;

    // log(10000) / (half_dim - 1)
    let exponent = -(10000.0f32.ln()) / (half_dim as f32 - 1.0);

    // Create frequency values
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| (exponent * i as f32).exp())
        .collect();
    let freqs = Tensor::<B, 1>::from_floats(freqs.as_slice(), &device);

    // [B, 1] * [1, half_dim] -> [B, half_dim]
    let timesteps = timesteps.reshape([-1, 1]);
    let freqs = freqs.reshape([1, -1]);
    let args = timesteps * freqs;

    // Concat sin and cos
    let sin = args.clone().sin();
    let cos = args.cos();

    Tensor::cat(vec![sin, cos], 1) // [B, embed_dim]
}

/// Time embedding: sinusoidal -> Linear -> SiLU -> Linear
#[derive(Module, Debug)]
pub struct TimeEmbedding<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    embed_dim: usize,
}

#[derive(Config, Debug)]
pub struct TimeEmbeddingConfig {
    embed_dim: usize,
    hidden_dim: usize,
}

impl TimeEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TimeEmbedding<B> {
        TimeEmbedding {
            linear1: LinearConfig::new(self.embed_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            embed_dim: self.embed_dim,
        }
    }
}

impl<B: Backend> TimeEmbedding<B> {
    /// Takes timestep tensor [B] and returns embedding [B, hidden_dim]
    pub fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let emb = sinusoidal_embedding(t, self.embed_dim);
        let emb = self.linear1.forward(emb);
        let emb = silu(emb);
        self.linear2.forward(emb)
    }
}

/// Class embedding: Embedding lookup -> Linear
#[derive(Module, Debug)]
pub struct ClassEmbedding<B: Backend> {
    embedding: burn::nn::Embedding<B>,
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ClassEmbeddingConfig {
    num_classes: usize,
    hidden_dim: usize,
}

impl ClassEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ClassEmbedding<B> {
        ClassEmbedding {
            embedding: burn::nn::EmbeddingConfig::new(self.num_classes, self.hidden_dim)
                .init(device),
            linear: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
        }
    }
}

impl<B: Backend> ClassEmbedding<B> {
    /// Takes class labels [B] and returns embedding [B, hidden_dim]
    pub fn forward(&self, labels: Tensor<B, 1, Int>) -> Tensor<B, 2> {
        // Embedding expects [B, seq_len], so unsqueeze to [B, 1]
        let labels = labels.unsqueeze_dim(1); // [B, 1]
        let emb = self.embedding.forward(labels); // [B, 1, hidden_dim]
        let [b, _, h] = emb.dims();
        let emb = emb.reshape([b, h]); // [B, hidden_dim]
        self.linear.forward(emb)
    }
}

/// SiLU activation: x * sigmoid(x)
pub fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * burn::tensor::activation::sigmoid(x)
}

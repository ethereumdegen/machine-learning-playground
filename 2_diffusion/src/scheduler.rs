use burn::prelude::*;

/// Number of diffusion timesteps.
pub const NUM_TIMESTEPS: usize = 1000;

/// DDPM noise scheduler with linear beta schedule.
pub struct DDPMScheduler {
    /// Beta values for each timestep
    pub betas: Vec<f32>,
    /// Alpha = 1 - beta
    pub alphas: Vec<f32>,
    /// Cumulative product of alphas (alpha_bar)
    pub alpha_bars: Vec<f32>,
}

impl DDPMScheduler {
    pub fn new() -> Self {
        let beta_start: f32 = 1e-4;
        let beta_end: f32 = 0.02;
        let mut betas = Vec::with_capacity(NUM_TIMESTEPS);
        let mut alphas = Vec::with_capacity(NUM_TIMESTEPS);
        let mut alpha_bars = Vec::with_capacity(NUM_TIMESTEPS);

        for i in 0..NUM_TIMESTEPS {
            let beta = beta_start + (beta_end - beta_start) * (i as f32) / (NUM_TIMESTEPS as f32 - 1.0);
            betas.push(beta);
            alphas.push(1.0 - beta);
        }

        let mut alpha_bar = 1.0f32;
        for &a in &alphas {
            alpha_bar *= a;
            alpha_bars.push(alpha_bar);
        }

        Self {
            betas,
            alphas,
            alpha_bars,
        }
    }

    /// Forward diffusion: add noise to x_0 at timestep t.
    /// Returns (x_t, noise) where noise is the epsilon that was added.
    pub fn add_noise<B: Backend>(
        &self,
        x_0: Tensor<B, 4>,
        noise: Tensor<B, 4>,
        timesteps: &[usize],
    ) -> Tensor<B, 4> {
        let device = x_0.device();
        let batch_size = timesteps.len();

        // Build sqrt_alpha_bar and sqrt_one_minus_alpha_bar per sample
        let sqrt_ab: Vec<f32> = timesteps
            .iter()
            .map(|&t| self.alpha_bars[t].sqrt())
            .collect();
        let sqrt_one_minus_ab: Vec<f32> = timesteps
            .iter()
            .map(|&t| (1.0 - self.alpha_bars[t]).sqrt())
            .collect();

        let sqrt_ab = Tensor::<B, 1>::from_floats(sqrt_ab.as_slice(), &device)
            .reshape([batch_size, 1, 1, 1]);
        let sqrt_one_minus_ab =
            Tensor::<B, 1>::from_floats(sqrt_one_minus_ab.as_slice(), &device)
                .reshape([batch_size, 1, 1, 1]);

        // x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_ab * x_0 + sqrt_one_minus_ab * noise
    }

    /// Reverse step: given x_t and predicted noise, compute x_{t-1}.
    pub fn step<B: Backend>(
        &self,
        x_t: Tensor<B, 4>,
        noise_pred: Tensor<B, 4>,
        t: usize,
    ) -> Tensor<B, 4> {
        let device = x_t.device();

        let alpha = self.alphas[t];
        let alpha_bar = self.alpha_bars[t];
        let beta = self.betas[t];

        // Predicted x_0 direction coefficient: 1/sqrt(alpha_t)
        let coeff1 = 1.0 / alpha.sqrt();
        // Noise coefficient: beta_t / sqrt(1 - alpha_bar_t)
        let coeff2 = beta / (1.0 - alpha_bar).sqrt();

        // Mean of p(x_{t-1} | x_t)
        let mean = (x_t - noise_pred * coeff2) * coeff1;

        if t == 0 {
            mean
        } else {
            // Add noise with variance = beta_t
            let noise = Tensor::random(mean.shape(), burn::tensor::Distribution::Normal(0.0, 1.0), &device);
            mean + noise * beta.sqrt()
        }
    }
}

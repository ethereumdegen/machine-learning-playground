use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::{Deserialize, Serialize};

const OLLAMA_URL: &str = "http://localhost:11434/api/generate";
const MODEL: &str = "gemma4:e4b";

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    images: Vec<String>,
    stream: bool,
}

#[derive(Deserialize)]
pub struct GenerateResponse {
    pub response: String,
}

pub fn caption_image(png_bytes: &[u8], prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let image_b64 = BASE64.encode(png_bytes);

    let request = GenerateRequest {
        model: MODEL.to_string(),
        prompt: prompt.to_string(),
        images: vec![image_b64],
        stream: false,
    };

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(OLLAMA_URL)
        .json(&request)
        .send()?
        .error_for_status()?;

    let result: GenerateResponse = resp.json()?;
    Ok(result.response)
}

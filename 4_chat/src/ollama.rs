use serde::{Deserialize, Serialize};

const OLLAMA_URL: &str = "http://localhost:11434/api/chat";
const MODEL: &str = "gemma4:e4b";

#[derive(Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: Message,
}

pub fn chat(messages: &[Message]) -> Result<String, Box<dyn std::error::Error>> {
    let request = ChatRequest {
        model: MODEL.to_string(),
        messages: messages.to_vec(),
        stream: false,
    };

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(OLLAMA_URL)
        .json(&request)
        .send()?
        .error_for_status()?;

    let chat_resp: ChatResponse = resp.json()?;
    Ok(chat_resp.message.content)
}

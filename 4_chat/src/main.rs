mod ollama;

use ollama::Message;
use std::io::{self, BufRead, Write};

fn main() {
    println!("Chat with Gemma4 (type 'exit' or 'quit' to end)");
    println!("---");

    let mut history: Vec<Message> = Vec::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\nYou: ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            break; // EOF
        }

        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            println!("Goodbye!");
            break;
        }

        history.push(Message {
            role: "user".to_string(),
            content: input,
        });

        match ollama::chat(&history) {
            Ok(response) => {
                println!("\nGemma4: {response}");
                history.push(Message {
                    role: "assistant".to_string(),
                    content: response,
                });
            }
            Err(e) => {
                eprintln!("\nError: {e}");
                // Remove the failed user message so history stays consistent
                history.pop();
            }
        }
    }
}

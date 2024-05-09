use langchain_rust::llm::OpenAIConfig;
use langchain_rust::{language_models::llm::LLM, llm::openai::OpenAI};

#[derive(Clone)]
pub struct LocalLLM {
    pub(crate) model: OpenAI<OpenAIConfig>,
}

impl LocalLLM {
    pub fn new(model_name: String) -> Self {
        let ollama = OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model_name);
        Self {
            model: ollama
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        let response = self.model.invoke(prompt).await;
        match response {
            Ok(res) => Ok(res),
            Err(e) => Err(e.to_string()),
        }
    }
}

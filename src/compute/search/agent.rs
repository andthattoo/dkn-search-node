use crate::compute::search::config::{PersonaPool, Persona};
use crate::compute::search::utils::prompt::create_system_prompt;
use ollama_rs::{
    generation::functions::tools::{Scraper, DDGSearcher, StockScraper},
    generation::functions::{FunctionCallRequest, NousFunctionCall},
    generation::chat::ChatMessage,
    generation::chat::request::ChatMessageRequest,
    Ollama,
};
use std::sync::Arc;
use parking_lot::RwLock;

#[derive(Debug, Clone)]
pub struct Agent{
    persona: Persona,
    react: Arc<RwLock<Ollama>>,
    observer: Ollama,
}

impl Default for Agent {
    fn default() -> Self {
        Self::new()
    }
}

impl Agent {
    pub fn new() -> Self {

        let observer = Ollama::default();
        let react = Arc::new(RwLock::new(Ollama::new_default_with_history(30)));
        let pool = PersonaPool::new_from_file("data/personas.json");
        let persona: Persona = match pool.get_random_agent() {
            Some(agent) => agent.clone(),
            None => Persona::default()
        };

        let system_prompt = create_system_prompt(
            "Search for information on the topic of 'Machine Learning'.",
            None,
            None,
            &Persona::default(),
        );

        react.write().set_system_response(
            "default".to_string(),
            system_prompt,
        );

        Self {
            persona,
            react,
            observer
        }
    }

    pub fn new_from_persona(query: &str) -> Self {
        let decide = Ollama::default();
        let react = Arc::new(RwLock::new(Ollama::new_default_with_history(30)));
        Self {
            persona,
            react,
            observer
        }
    }

    pub fn get_persona(&self) -> &Persona {
        &self.persona
    }

    pub fn set_persona(&mut self, persona: Persona) {
        self.persona = persona;
    }

    pub async fn react_response(&self, message: String) -> String {

        let user_message = ChatMessage::user(message.clone());

        let result = self.react.write().send_chat_messages_with_history(
            ChatMessageRequest::new(
                "adrienbrault/nous-hermes2theta-llama3-8b:q8_0".to_string(),
                vec![user_message]
            ),
            "default".to_string()
        ).await.unwrap();

        result.message.unwrap().content
    }

    pub async fn function_call(&self, query: &str)-> String{
        let parser = Arc::new(NousFunctionCall {});
        let scraper_tool = Arc::new(Scraper {});
        let ddg_search_tool = Arc::new(DDGSearcher::new());
        let stock_scraper = Arc::new(StockScraper::new());
        let result  = self.observer.send_function_call(
            FunctionCallRequest::new(
                "adrienbrault/nous-hermes2theta-llama3-8b:q8_0".to_string(),
            vec![stock_scraper.clone(), ddg_search_tool.clone(), scraper_tool.clone()],
            vec![ChatMessage::user(query.to_string())]
        ),
            parser.clone()).await.unwrap();

        result.message.unwrap().content
    }
}
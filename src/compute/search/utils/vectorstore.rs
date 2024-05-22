use langchain_rust::embedding::{embedder_trait::Embedder, EmbedderError, ollama::ollama_embedder::OllamaEmbedder};


#[derive(Debug)]
pub struct Embeddings {
    pub(crate) embedder: OllamaEmbedder,
}

impl Embeddings {
    pub fn new() -> Self {
        let ollama = OllamaEmbedder::default().with_model("nomic-embed-text");
        Self { embedder: ollama }
    }

    pub async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        self.embedder.embed_documents(documents).await
    }

    pub async fn embed_query(&self, query: &str) -> Result<Vec<f64>, EmbedderError> {
        self.embedder.embed_query(query).await
    }
}

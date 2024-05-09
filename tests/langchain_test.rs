#[cfg_attr(test, cfg(feature = "langchain_test"))]
mod langchain_test {
    use langchain_rust::tools::Tool;
    use serde_json::json;
    use dkn_compute::compute::search::vectorstore::Embeddings;
    use dkn_compute::compute::search::local_llm::LocalLLM;
    use dkn_compute::compute::search::tools::scraper::Scraper;
    use dkn_compute::compute::search::tools::search_ddg::DDGSearcher;


    #[tokio::test]
    async fn test_llm_prompt() {
        //Since Ollmama is OpenAi compatible
        //You can call Ollama this way:
        let model = LocalLLM::new("llama3".to_string());
        let ans = model.generate("Who built you?").await.unwrap();
        println!("{}", ans);
    }

    #[tokio::test]
    async fn test_ollama_embeddings() {
        let embeddings = Embeddings::new();
        let res = embeddings.embed_query("Who built you?").await.unwrap();
        println!("{:?}", res);
    }

    #[tokio::test]
    async fn test_scraping_tool(){
        let scraper = Scraper {};
        let sentences = scraper.run(json!({"website":"http://example.com"})).await.unwrap();
        println!("{}", sentences);

    }

    #[tokio::test]
    async fn test_web_search_tool(){
        let searcher = DDGSearcher::new();
        let res = searcher.search("Who built Llama3?").await.unwrap();
        for result in res {
            println!("{:?}", result);
        }
    }

}

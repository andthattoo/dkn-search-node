use std::env;
use std::time::Duration;
use std::sync::Arc;
use serde_json::Value;

use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::{options::ChainCallOptions, Chain},
    llm::openai::OpenAI,
    llm::OpenAIConfig,
    memory::SimpleMemory,
    prompt_args,
    tools::{CommandExecutor, Tool},
};

use crate::{
    compute::{ollama::OllamaClient, payload::TaskRequestPayload},
    compute::search::tools::{StockScraper, Scraper, DDGSearcher},
    node::DriaComputeNode,
    utils::get_current_time_nanos,
    waku::message::WakuMessage,
};

use crate::compute::constants::{
    DEFAULT_DKN_OLLAMA_HOST, DEFAULT_DKN_OLLAMA_MODEL, DEFAULT_DKN_OLLAMA_PORT,
};


/// # Search Payload
///
/// A synthesis task is the task of putting a prompt to an LLM and obtaining many results, essentially growing the number of data points in a dataset,
/// hence creating synthetic data.
type SearchPayload = TaskRequestPayload<String>;

pub fn search_worker(
    node: Arc<DriaComputeNode>,
    topic: &'static str,
    tools: (Arc<Scraper>, Arc<StockScraper>, Arc<DDGSearcher>),
    sleep_amount: Duration,
) -> tokio::task::JoinHandle<()> {

    let ollama = OllamaClient::new(None, None, None);

    let host =  env::var("DKN_OLLAMA_HOST").unwrap_or(DEFAULT_DKN_OLLAMA_HOST.to_string());

    let port = env::var("DKN_OLLAMA_PORT")
        .and_then(|port_str| {
            port_str
                .parse::<u16>()
                .map_err(|_| env::VarError::NotPresent)
        })
        .unwrap_or(DEFAULT_DKN_OLLAMA_PORT);

    let ollama_url = format!("http://{}:{}/v1", host, port);

    let llm = OpenAI::default()
        .with_config(
            OpenAIConfig::default()
                .with_api_base(ollama_url)
                .with_api_key("ollama"),
        )
        .with_model(env::var("DKN_OLLAMA_MODEL").unwrap_or("llama3".to_string()));

    let memory = SimpleMemory::new();

    //let command_executor = CommandExecutor::default();
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[
            tools.0.clone(),
            tools.1.clone(),
            tools.2.clone(),
        ])
        .options(ChainCallOptions::new().with_max_tokens(4000))
        .build(llm)
        .unwrap();

    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());

    tokio::spawn(async move {
        if let Err(e) = ollama.setup(node.cancellation.clone()).await {
            log::error!("Could not setup Ollama: {}", e);
        }

        node.subscribe_topic(topic).await;

        loop {
            tokio::select! {
                _ = node.cancellation.cancelled() => {
                    if let Err(e) = node.unsubscribe_topic(topic).await {
                        log::error!("Error unsubscribing from {}: {}\nContinuing anyway.", topic, e);
                    }
                    break;
                }
                _ = tokio::time::sleep(sleep_amount) => {
                    let mut tasks = Vec::new();
                    if let Ok(messages) = node.process_topic(topic, true).await {
                        if messages.is_empty() {
                            continue;
                        }
                        log::info!("Received {} synthesis tasks.", messages.len());

                        for message in messages {
                            match message.parse_payload::<SearchPayload>(true) {
                                Ok(task) => {
                                    // check deadline
                                    if get_current_time_nanos() >= task.deadline {
                                        log::debug!("{}", format!("Skipping {} due to deadline.", task.task_id));
                                        continue;
                                    }

                                    // check task inclusion
                                    match node.is_tasked(&task.filter) {
                                        Ok(is_tasked) => {
                                            if is_tasked {
                                                log::debug!("{}", format!("Skipping {} due to filter.", task.task_id));
                                                continue;
                                            }
                                        },
                                        Err(e) => {
                                            log::error!("Error checking task inclusion: {}", e);
                                            continue;
                                        }
                                    }

                                    tasks.push(task);
                                },
                                Err(e) => {
                                    log::error!("Error parsing payload: {}", e);
                                    continue;
                                }
                            }
                        }
                    }
                    // Set node to busy
                    node.set_busy(true);

                    for task in tasks {
                        // parse public key
                        let task_public_key = match hex::decode(&task.public_key) {
                            Ok(public_key) => public_key,
                            Err(e) => {
                                log::error!("Error parsing public key: {}", e);
                                continue;
                            }
                        };

                        let input_variables = prompt_args! {
                                "input" => &task.input,
                            };

                        let search_result = match executor.invoke(input_variables).await {
                                Ok(result) => result.replace("\n", " "),
                                Err(e) => {
                                log::error!("Error invoking LLMChain: {:?}", e);
                                continue
                                }
                            };

                        // create h||s||e payload
                        let payload = match node.create_payload(search_result, &task_public_key) {
                            Ok(payload) => payload,
                            Err(e) => {
                                log::error!("Error creating payload: {}", e);
                                continue;
                            }
                        };

                        // stringify payload
                        let payload_str = match payload.to_string() {
                            Ok(payload_str) => payload_str,
                            Err(e) => {
                                log::error!("Error stringifying payload: {}", e);
                                continue;
                            }
                        };

                        // send result to Waku network
                        let message = WakuMessage::new(payload_str, &task.task_id);
                        if let Err(e) = node.send_message_once(message)
                            .await {
                                log::error!("Error sending message: {}", e);
                                continue;
                            }
                    }

                    // Set node to not busy
                    node.set_busy(false);
                }
            }
        }
    })
}

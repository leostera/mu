use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    process::Stdio,
};

use futures_util::StreamExt;
use mu_session::{
    AssistantToolResponse, AssistantTurnResponse, MessageCost, MessageUsage, SessionDocument,
    ToolCallRecord,
};
use reqwest::{Client, Url, header};
use serde_json::{Value, json};
use thiserror::Error;

const ORIGIN: &str = "https://opencode.cloudflare.dev";

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("configuration: {0}")]
    Configuration(String),
    #[error("provider: {0}")]
    Provider(String),
    #[error("session: {0}")]
    Session(String),
    #[error("I/O: {0}")]
    Io(#[from] io::Error),
}

struct Config {
    agent_dir: PathBuf,
    cwd: PathBuf,
    provider: String,
    model: String,
    model_choices: Vec<String>,
    hide_thinking_block: bool,
}
struct Provider {
    transport: ProviderTransport,
    model: String,
    instructions: String,
    supports_reasoning: bool,
}

enum ProviderTransport {
    OpenCode {
        http: Client,
        endpoint: Url,
        token: String,
    },
    Demo,
}
struct FunctionCall {
    call_id: String,
    name: String,
    arguments: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolStarted {
        call_id: String,
        name: String,
        arguments: Value,
    },
    ToolFinished {
        call_id: String,
        output: String,
        is_error: bool,
    },
}
struct Reply {
    text: String,
    thinking: String,
    thinking_signature: Option<String>,
    output_items: Vec<Value>,
    tool_calls: Vec<FunctionCall>,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    reasoning: Option<u64>,
    total_tokens: u64,
}

fn config() -> Result<Config, AgentError> {
    let cwd = std::env::current_dir()?;
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| AgentError::Configuration("HOME is not set".into()))?;
    let agent_dir = std::env::var_os("PI_CODING_AGENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".pi/agent"));
    let mut provider = None;
    let mut model = None;
    let mut hide_thinking_block = false;
    for path in [
        agent_dir.join("settings.json"),
        cwd.join(".pi/settings.json"),
    ] {
        if !path.exists() {
            continue;
        }
        let value: Value = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .map_err(|e| AgentError::Configuration(format!("{}: {e}", path.display())))?,
        )
        .map_err(|e| AgentError::Configuration(format!("{}: {e}", path.display())))?;
        if let Some(v) = value.get("defaultProvider").and_then(Value::as_str) {
            provider = Some(v.to_owned());
        }
        if let Some(v) = value.get("defaultModel").and_then(Value::as_str) {
            model = Some(v.to_owned());
        }
        if let Some(v) = value.get("hideThinkingBlock").and_then(Value::as_bool) {
            hide_thinking_block = v;
        }
    }
    let provider = provider
        .ok_or_else(|| AgentError::Configuration("Pi settings have no defaultProvider".into()))?;
    let model = model
        .ok_or_else(|| AgentError::Configuration("Pi settings have no defaultModel".into()))?;
    let model_choices = configured_models(&agent_dir, &provider, &model)?;
    Ok(Config {
        agent_dir,
        cwd,
        provider,
        model,
        model_choices,
        hide_thinking_block,
    })
}

fn configured_models(
    agent_dir: &Path,
    provider: &str,
    default_model: &str,
) -> Result<Vec<String>, AgentError> {
    let path = agent_dir.join("models.json");
    let mut models = if path.exists() {
        let value: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).map_err(|error| {
                AgentError::Configuration(format!("{}: {error}", path.display()))
            })?)
            .map_err(|error| AgentError::Configuration(format!("{}: {error}", path.display())))?;
        value
            .get("providers")
            .and_then(|providers| providers.get(provider))
            .and_then(|provider| provider.get("models"))
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|model| model.get("id").and_then(Value::as_str))
            .map(str::to_owned)
            .collect()
    } else {
        Vec::new()
    };
    if !models.iter().any(|model| model == default_model) {
        models.insert(0, default_model.to_owned());
    }
    Ok(models)
}

async fn token(provider: &str) -> Result<String, AgentError> {
    let env = if provider == "opencode-cf" {
        "OPENCODE_CF_TOKEN".into()
    } else {
        format!("{}_TOKEN", provider.replace('-', "_").to_ascii_uppercase())
    };
    if let Ok(value) = std::env::var(&env) {
        if !value.trim().is_empty() {
            return Ok(value.trim().to_owned());
        }
    }
    let bin = std::env::var_os("CLOUDFLARED_PATH").unwrap_or_else(|| "cloudflared".into());
    let output = tokio::process::Command::new(bin)
        .args(["access", "token", "-app=https://opencode.cloudflare.dev"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .map_err(|e| AgentError::Provider(format!("could not run cloudflared: {e}")))?;
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !output.status.success() || value.is_empty() || value.to_lowercase().contains("login") {
        return Err(AgentError::Provider(format!(
            "no Cloudflare Access token; run `cloudflared access login -app={ORIGIN}` or set {env}"
        )));
    }
    Ok(value)
}

fn trusted_url(value: &str) -> Result<Url, AgentError> {
    let url =
        Url::parse(value).map_err(|e| AgentError::Provider(format!("invalid gateway URL: {e}")))?;
    let host = url.host_str().unwrap_or_default();
    if url.scheme() != "https"
        || !(host == "opencode.cloudflare.dev" || host.ends_with(".opencode.cloudflare.dev"))
    {
        return Err(AgentError::Provider(format!(
            "refusing untrusted gateway URL `{value}`"
        )));
    }
    Ok(url)
}

async fn connect(config: &Config, skill_prompt: String) -> Result<Provider, AgentError> {
    if !config.provider.starts_with("opencode-cf") {
        return Err(AgentError::Configuration(format!(
            "Pi provider `{}` is unsupported (expected opencode-cf)",
            config.provider
        )));
    }
    let token = token(&config.provider).await?;
    let http = Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| AgentError::Provider(format!("HTTP client: {e}")))?;
    let well: Value = http
        .get(format!("{ORIGIN}/.well-known/opencode"))
        .send()
        .await
        .map_err(|e| AgentError::Provider(format!("gateway discovery failed: {e}")))?
        .error_for_status()
        .map_err(|e| AgentError::Provider(format!("gateway discovery failed: {e}")))?
        .json()
        .await
        .map_err(|e| AgentError::Provider(format!("invalid gateway discovery: {e}")))?;
    let remote = well
        .get("remote_config")
        .ok_or_else(|| AgentError::Provider("discovery has no remote_config".into()))?;
    let config_url = remote
        .get("url")
        .and_then(Value::as_str)
        .ok_or_else(|| AgentError::Provider("discovery has no config URL".into()))?;
    let mut request = http.get(trusted_url(config_url)?);
    if let Some(headers) = remote.get("headers").and_then(Value::as_object) {
        for (name, value) in headers {
            if let Some(value) = value.as_str() {
                let value = value
                    .replace("{env:TOKEN}", &token)
                    .replace("${TOKEN}", &token);
                request = request.header(name, value);
            }
        }
    }
    let gateway: Value = request
        .send()
        .await
        .map_err(|e| AgentError::Provider(format!("gateway config fetch failed: {e}")))?
        .error_for_status()
        .map_err(|e| AgentError::Provider(format!("gateway config fetch failed: {e}")))?
        .json()
        .await
        .map_err(|e| AgentError::Provider(format!("invalid gateway config: {e}")))?;
    let providers = gateway
        .get("provider")
        .and_then(Value::as_object)
        .ok_or_else(|| AgentError::Provider("config has no provider catalog".into()))?;
    let mut endpoint = None;
    let mut supports_reasoning = false;
    for (id, p) in providers {
        let Some(m) = p.pointer(&format!("/models/{}", config.model)) else {
            continue;
        };
        let base = m
            .pointer("/provider/api")
            .and_then(Value::as_str)
            .or_else(|| p.pointer("/options/baseURL").and_then(Value::as_str))
            .or_else(|| p.pointer("/options/baseUrl").and_then(Value::as_str))
            .or_else(|| p.get("api").and_then(Value::as_str))
            .ok_or_else(|| AgentError::Provider(format!("provider `{id}` has no API URL")))?;
        if id != "openai"
            && !Url::parse(base)
                .map(|u| u.path().contains("/openai"))
                .unwrap_or(false)
        {
            return Err(AgentError::Provider(format!(
                "model uses unsupported provider `{id}`"
            )));
        }
        let mut url = trusted_url(base)?;
        url.set_path(&format!("{}/responses", url.path().trim_end_matches('/')));
        supports_reasoning = m.get("reasoning").and_then(Value::as_bool).unwrap_or(false);
        endpoint = Some(url);
        break;
    }
    let endpoint = endpoint.ok_or_else(|| {
        AgentError::Provider(format!(
            "model `{}` not found in gateway catalog",
            config.model
        ))
    })?;
    Ok(Provider {
        transport: ProviderTransport::OpenCode {
            http,
            endpoint,
            token,
        },
        model: config.model.clone(),
        instructions: format!(
            "You are Mu, a helpful coding assistant. Be clear, practical, and concise.{skill_prompt}"
        ),
        supports_reasoning,
    })
}

impl Provider {
    async fn complete(
        &self,
        history: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        if matches!(&self.transport, ProviderTransport::Demo) {
            let prompt = history
                .last()
                .and_then(|item| item.pointer("/content/0/text"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let text = format!("Demo backend received: {prompt}");
            for character in text.chars() {
                on_event(AgentEvent::TextDelta(character.to_string()));
            }
            return Ok(Reply {
                text,
                thinking: String::new(),
                thinking_signature: None,
                output_items: Vec::new(),
                tool_calls: Vec::new(),
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
                reasoning: None,
                total_tokens: 0,
            });
        }
        let ProviderTransport::OpenCode {
            http,
            endpoint,
            token,
        } = &self.transport
        else {
            unreachable!("demo backend returned above")
        };
        let input = history;
        let mut request_body = json!({"model":self.model,"instructions":self.instructions,"input":input,"tools":crate::tools::definitions(),"stream":true,"store":false});
        if self.supports_reasoning {
            request_body["reasoning"] = json!({"effort":"medium","summary":"auto"});
            request_body["include"] = json!(["reasoning.encrypted_content"]);
        }
        let response = http
            .post(endpoint.clone())
            .bearer_auth(token)
            .header("cf-access-token", token)
            .header("x-requested-with", "xmlhttprequest")
            .header(header::ACCEPT, "text/event-stream")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(AgentError::Provider(format!(
                "model request returned HTTP {}",
                response.status()
            )));
        }
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        let mut text = String::new();
        let mut thinking = String::new();
        let mut thinking_signature = None;
        let mut output_items = Vec::new();
        let mut tool_calls = Vec::new();
        let (mut input_tokens, mut output_tokens) = (0, 0);
        let (mut cache_read, mut cache_write, mut reasoning, mut total_tokens) = (0, 0, None, 0);
        while let Some(chunk) = stream.next().await {
            bytes.extend_from_slice(
                &chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?,
            );
            while let Some(pos) = bytes.windows(2).position(|w| w == b"\n\n") {
                let frame: Vec<_> = bytes.drain(..pos + 2).collect();
                let frame = String::from_utf8_lossy(&frame);
                let Some(data) = frame.lines().find_map(|line| line.strip_prefix("data:")) else {
                    continue;
                };
                if let Ok(event) = serde_json::from_str::<Value>(data.trim()) {
                    match event.get("type").and_then(Value::as_str).unwrap_or("") {
                        "response.output_text.delta" => {
                            if let Some(s) = event.get("delta").and_then(Value::as_str) {
                                text.push_str(s);
                                on_event(AgentEvent::TextDelta(s.to_owned()));
                            }
                        }
                        "response.reasoning_summary_text.delta"
                        | "response.reasoning_text.delta" => {
                            if let Some(s) = event.get("delta").and_then(Value::as_str) {
                                thinking.push_str(s);
                                on_event(AgentEvent::ThinkingDelta(s.to_owned()));
                            }
                        }
                        "response.output_item.done" => {
                            if event.pointer("/item/type").and_then(Value::as_str)
                                == Some("reasoning")
                            {
                                let item = event.get("item").unwrap_or(&Value::Null);
                                thinking_signature = Some(item.to_string());
                                let final_thinking = response_reasoning(item);
                                if !final_thinking.is_empty() {
                                    if thinking.is_empty() {
                                        thinking = final_thinking.clone();
                                        on_event(AgentEvent::ThinkingDelta(final_thinking));
                                    } else if let Some(suffix) =
                                        final_thinking.strip_prefix(&thinking)
                                    {
                                        if !suffix.is_empty() {
                                            thinking.push_str(suffix);
                                            on_event(AgentEvent::ThinkingDelta(suffix.to_owned()));
                                        }
                                    }
                                }
                            }
                        }
                        "response.completed" => {
                            if let Some(items) =
                                event.pointer("/response/output").and_then(Value::as_array)
                            {
                                output_items = items.clone();
                            }
                            if let Some(output) =
                                event.pointer("/response/output").and_then(Value::as_array)
                            {
                                if let Some(item) = output.iter().find(|item| {
                                    item.get("type").and_then(Value::as_str) == Some("reasoning")
                                }) {
                                    if thinking_signature.is_none()
                                        || item.get("encrypted_content").is_some()
                                    {
                                        thinking_signature = Some(item.to_string());
                                    }
                                    if thinking.is_empty() {
                                        thinking = response_reasoning(item);
                                        if !thinking.is_empty() {
                                            on_event(AgentEvent::ThinkingDelta(thinking.clone()));
                                        }
                                    }
                                }
                                for item in output.iter().filter(|item| {
                                    item.get("type").and_then(Value::as_str)
                                        == Some("function_call")
                                }) {
                                    let (Some(call_id), Some(name), Some(raw_arguments)) = (
                                        item.get("call_id").and_then(Value::as_str),
                                        item.get("name").and_then(Value::as_str),
                                        item.get("arguments").and_then(Value::as_str),
                                    ) else {
                                        continue;
                                    };
                                    if let Ok(arguments) = serde_json::from_str(raw_arguments) {
                                        tool_calls.push(FunctionCall {
                                            call_id: call_id.to_owned(),
                                            name: name.to_owned(),
                                            arguments,
                                        });
                                    }
                                }
                            }
                            let usage = event.pointer("/response/usage");
                            let raw_input = usage
                                .and_then(|value| value.get("input_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            cache_read = usage
                                .and_then(|value| {
                                    value.pointer("/input_tokens_details/cached_tokens")
                                })
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            cache_write = usage
                                .and_then(|value| {
                                    value.pointer("/input_tokens_details/cache_write_tokens")
                                })
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            input_tokens = raw_input.saturating_sub(cache_read + cache_write);
                            output_tokens = usage
                                .and_then(|value| value.get("output_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(0);
                            reasoning = usage
                                .and_then(|value| {
                                    value.pointer("/output_tokens_details/reasoning_tokens")
                                })
                                .and_then(Value::as_u64);
                            total_tokens = usage
                                .and_then(|value| value.get("total_tokens"))
                                .and_then(Value::as_u64)
                                .unwrap_or(raw_input + output_tokens);
                        }
                        "response.failed" | "error" => {
                            return Err(AgentError::Provider(
                                event
                                    .pointer("/response/error/message")
                                    .or_else(|| event.pointer("/error/message"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("response failed")
                                    .to_owned(),
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        if text.is_empty() && tool_calls.is_empty() {
            return Err(AgentError::Provider(
                "stream ended without response text or tool calls".into(),
            ));
        }
        Ok(Reply {
            text,
            thinking,
            thinking_signature,
            output_items,
            tool_calls,
            input: input_tokens,
            output: output_tokens,
            cache_read,
            cache_write,
            reasoning,
            total_tokens,
        })
    }
}

fn response_reasoning(item: &Value) -> String {
    let parts = |key: &str| {
        item.get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    let summary = parts("summary");
    if summary.is_empty() {
        parts("content")
    } else {
        summary
    }
}

fn api_name(transport: &ProviderTransport) -> &'static str {
    if matches!(transport, ProviderTransport::Demo) {
        "mu-demo"
    } else {
        "openai-responses"
    }
}

fn session_path(c: &Config) -> PathBuf {
    let cwd = c
        .cwd
        .to_string_lossy()
        .trim_start_matches(['/', '\\'])
        .replace(['/', '\\', ':'], "-");
    let dir = c.agent_dir.join("sessions").join(format!("--{cwd}--"));
    let ts = chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        .replace([':', '.'], "-");
    dir.join(format!("{ts}_{}.jsonl", uuid::Uuid::new_v4()))
}

fn usage(r: &Reply) -> MessageUsage {
    MessageUsage {
        input: r.input,
        output: r.output,
        cache_read: r.cache_read,
        cache_write: r.cache_write,
        reasoning: r.reasoning,
        total_tokens: r.total_tokens,
        cost: MessageCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

pub struct MuAgent {
    provider: Provider,
    provider_id: String,
    model_choices: Vec<String>,
    session: SessionDocument,
    history: Vec<Value>,
    session_path: PathBuf,
    cwd: PathBuf,
    skills: crate::skills::SkillCatalog,
    hide_thinking_block: bool,
}

impl MuAgent {
    pub async fn connect() -> Result<Self, AgentError> {
        let config = config()?;
        let skills = crate::skills::SkillCatalog::discover(&config.cwd, &config.agent_dir);
        let provider = connect(&config, skills.system_prompt()).await?;
        let session_path = session_path(&config);
        let session = SessionDocument::new(config.cwd.to_string_lossy().into_owned());
        Ok(Self {
            provider,
            provider_id: config.provider,
            model_choices: config.model_choices,
            session,
            history: Vec::new(),
            session_path,
            cwd: config.cwd.clone(),
            skills,
            hide_thinking_block: config.hide_thinking_block,
        })
    }

    /// Create a deterministic no-network agent for terminal previews and tests.
    pub fn demo() -> Result<Self, AgentError> {
        let cwd = std::env::current_dir()?;
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or_else(|| AgentError::Configuration("HOME is not set".into()))?;
        let config = Config {
            agent_dir: std::env::var_os("PI_CODING_AGENT_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".pi/agent")),
            cwd,
            provider: "mu-demo".into(),
            model: "mu-demo".into(),
            model_choices: vec!["mu-demo".into(), "mu-demo-alt".into()],
            hide_thinking_block: false,
        };
        let path = session_path(&config);
        Ok(Self {
            provider: Provider {
                transport: ProviderTransport::Demo,
                model: config.model.clone(),
                instructions: "You are Mu, a deterministic demo assistant.".into(),
                supports_reasoning: false,
            },
            provider_id: config.provider,
            model_choices: config.model_choices,
            session: SessionDocument::new(config.cwd.to_string_lossy().into_owned()),
            history: Vec::new(),
            session_path: path,
            cwd: config.cwd.clone(),
            skills: crate::skills::SkillCatalog::discover(&config.cwd, &config.agent_dir),
            hide_thinking_block: config.hide_thinking_block,
        })
    }

    pub fn model(&self) -> &str {
        &self.provider.model
    }

    pub fn provider(&self) -> &str {
        &self.provider_id
    }

    pub fn model_choices(&self) -> &[String] {
        &self.model_choices
    }

    pub fn select_model(&mut self, model: &str) -> Result<bool, AgentError> {
        if !self.model_choices.iter().any(|choice| choice == model) {
            return Ok(false);
        }
        if self.provider.model == model {
            return Ok(true);
        }
        self.provider.model = model.to_owned();
        self.session.append_model_change(&self.provider_id, model);
        self.session
            .write_atomic(&self.session_path)
            .map_err(|error| AgentError::Session(error.to_string()))?;
        Ok(true)
    }

    pub fn session_path(&self) -> &Path {
        &self.session_path
    }

    /// Skills discovered from Pi's global and project skill directories.
    pub fn skills(&self) -> &crate::skills::SkillCatalog {
        &self.skills
    }

    pub fn hide_thinking_block(&self) -> bool {
        self.hide_thinking_block
    }

    /// Execute one of Mu's Pi-compatible built-in tools in the project directory.
    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<crate::tools::ToolResult, crate::tools::ToolError> {
        crate::tools::execute(&self.cwd, name, arguments).await
    }

    pub async fn submit(
        &mut self,
        prompt: String,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), AgentError> {
        self.session.append_user_text(prompt.clone());
        self.session
            .write_atomic(&self.session_path)
            .map_err(|error| AgentError::Session(error.to_string()))?;
        self.history
            .push(json!({"role":"user","content":[{"type":"input_text","text":prompt}]}));
        let mut reply = self.provider.complete(&self.history, on_event).await?;
        for _ in 0..8 {
            if reply.tool_calls.is_empty() {
                break;
            }
            let api = api_name(&self.provider.transport);
            let calls = std::mem::take(&mut reply.tool_calls);
            let records = calls
                .iter()
                .map(|call| ToolCallRecord {
                    id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect::<Vec<_>>();
            self.session.append_assistant_tool_calls(
                api,
                &self.provider_id,
                &self.provider.model,
                AssistantToolResponse {
                    thinking: reply.thinking.clone(),
                    thinking_signature: reply.thinking_signature.clone(),
                    text: reply.text.clone(),
                    calls: records,
                    usage: usage(&reply),
                },
            );
            self.session
                .write_atomic(&self.session_path)
                .map_err(|error| AgentError::Session(error.to_string()))?;

            // Preserve the provider's full assistant output (including reasoning
            // signatures and all sibling calls) for stateless Responses replay.
            self.history.extend(reply.output_items.iter().cloned());
            // The Responses API expects all assistant calls before their outputs.
            for call in &calls {
                on_event(AgentEvent::ToolStarted {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
            }
            for call in calls {
                let result = self.execute_tool(&call.name, &call.arguments).await;
                let (output, is_error) = match result {
                    Ok(result) => (result.content, false),
                    Err(error) => (error.to_string(), true),
                };
                on_event(AgentEvent::ToolFinished {
                    call_id: call.call_id.clone(),
                    output: output.clone(),
                    is_error,
                });
                self.history.push(json!({
                    "type":"function_call_output", "call_id":call.call_id, "output":output,
                }));
                self.session
                    .append_tool_result(&call.call_id, &call.name, output, is_error);
                self.session
                    .write_atomic(&self.session_path)
                    .map_err(|error| AgentError::Session(error.to_string()))?;
            }
            reply = self.provider.complete(&self.history, on_event).await?;
        }
        if !reply.tool_calls.is_empty() {
            return Err(AgentError::Provider(
                "tool call limit exceeded (8 rounds)".into(),
            ));
        }
        let api = api_name(&self.provider.transport);
        self.session.append_assistant_response(
            api,
            &self.provider_id,
            &self.provider.model,
            AssistantTurnResponse {
                thinking: reply.thinking.clone(),
                thinking_signature: reply.thinking_signature.clone(),
                text: reply.text.clone(),
                usage: usage(&reply),
            },
        );
        self.session
            .write_atomic(&self.session_path)
            .map_err(|error| AgentError::Session(error.to_string()))?;
        if reply.output_items.is_empty() {
            self.history.push(
                json!({"role":"assistant","content":[{"type":"output_text","text":reply.text}]}),
            );
        } else {
            self.history.extend(reply.output_items);
        }
        Ok(())
    }
}

pub async fn run_prompt(prompt: String) -> Result<(), AgentError> {
    let mut agent = MuAgent::connect().await?;
    agent
        .submit(prompt, &mut |event| {
            if let AgentEvent::TextDelta(text) = event {
                print!("{text}");
            }
            let _ = io::stdout().flush();
        })
        .await?;
    println!();
    Ok(())
}

pub async fn run_cli() -> Result<(), AgentError> {
    let mut agent = MuAgent::connect().await?;
    println!(
        "Mu — {} via {} (session {})",
        agent.model(),
        agent.provider(),
        agent.session_path().display()
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        return agent
            .submit(args.join(" "), &mut |event| {
                if let AgentEvent::TextDelta(text) = event {
                    print!("{text}")
                }
            })
            .await;
    }
    let mut line = String::new();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        line.clear();
        if io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let prompt = line.trim();
        if prompt == "/exit" || prompt == "/quit" {
            break;
        }
        if prompt.is_empty() {
            continue;
        }
        agent
            .submit(prompt.to_owned(), &mut |event| {
                if let AgentEvent::TextDelta(text) = event {
                    print!("{text}");
                }
                let _ = io::stdout().flush();
            })
            .await?;
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{configured_models, trusted_url};
    use std::path::Path;
    #[test]
    fn reads_final_openai_responses_reasoning_when_no_reasoning_delta_arrived() {
        let item = serde_json::json!({
            "type":"reasoning",
            "summary":[{"type":"summary_text","text":"Plan the change."},{"type":"summary_text","text":"Then implement it."}],
            "content":[{"type":"reasoning_text","text":"private detail"}]
        });
        assert_eq!(
            super::response_reasoning(&item),
            "Plan the change.\n\nThen implement it."
        );
    }

    #[test]
    fn allows_access_gateway_and_rejects_untrusted_hosts() {
        assert!(trusted_url("https://gateway.opencode.cloudflare.dev/openai").is_ok());
        assert!(trusted_url("https://attacker.example/openai").is_err());
    }

    #[test]
    fn reads_custom_model_choices_and_keeps_the_default_model_available() {
        let dir = std::env::temp_dir().join(format!("mu-model-config-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("create model config directory");
        std::fs::write(
            dir.join("models.json"),
            r#"{"providers":{"oracle":{"models":[{"id":"alpha"},{"id":"beta"}]}}}"#,
        )
        .expect("write model config");

        let models = configured_models(&dir, "oracle", "beta").expect("read model choices");
        assert_eq!(models, ["alpha", "beta"]);
        let includes_unlisted_default =
            configured_models(&dir, "oracle", "gamma").expect("include configured default");
        assert_eq!(includes_unlisted_default, ["gamma", "alpha", "beta"]);
        std::fs::remove_dir_all(dir).expect("remove model config directory");

        let absent = configured_models(Path::new("/no/such/mu/config"), "oracle", "default")
            .expect("fall back to default model");
        assert_eq!(absent, ["default"]);
    }
}

use std::{
    collections::{HashMap, HashSet},
    io::{self, Write},
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
};

use futures_util::StreamExt;
use mu_session::{MessageCost, MessageUsage, SessionDocument};
use reqwest::{Client, Url, header};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::sync::OnceCell;

mod anthropic;

/// `(mime type, base64 data)` of an `input_image` part whose URL is a `data:` URL.
fn data_url_parts(part: &Value) -> Option<(String, String)> {
    let url = part.get("image_url")?.as_str()?;
    let (header, data) = url.strip_prefix("data:")?.split_once(";base64,")?;
    Some((header.to_owned(), data.to_owned()))
}

/// A tool output as plain text: strings pass through; image-bearing outputs keep their text parts
/// and note each omitted image (providers without tool-result image support).
fn output_text(item: &Value) -> String {
    match &item["output"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("input_image") => {
                    "[image omitted: this provider does not accept images in tool results]"
                        .to_owned()
                }
                _ => part["text"].as_str().unwrap_or_default().to_owned(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}
mod bedrock;
mod completions;
mod export_html;
mod google;
mod prompt_queue;
pub(crate) use prompt_queue::QueueMode;
use prompt_queue::configured_queue_mode;
pub use prompt_queue::{AgentPromptQueue, PromptBehavior};
mod retry;
use retry::{
    RetryPolicy, http_failure, retry_delay_ms, retryable_provider_error, update_retry_policy,
};
mod runtime;
mod tool_prompts;
mod tool_selection;
use tool_prompts::instructions_for_tools;
use tool_selection::{
    ToolSelection, apply_tool_selection, filter_tool_definitions, merge_default_tools,
    resolve_default_tools,
};
mod strict_schema;
use strict_schema::{mark_strict_builtin_tool, model_supports_strict_sampling};
mod sessions;
pub use export_html::export_session_file;
use export_html::render_session_html;
pub use sessions::{
    SessionInfo, SessionTranscriptMessage, SessionTranscriptRole, SessionTranscriptTool,
    latest_session_for_current_cwd, list_sessions_for_current_cwd,
};
use sessions::{
    branch_entry_values, find_session_id_path, latest_session_path, load_session, message_blocks,
    missing_session_cwd_error, new_session_path, normalize_session_name, session_active_mcp_tools,
    session_codemode_store, session_path, session_preview, session_transcript, validate_session_id,
    write_session,
};
mod sigv4;

use crate::agent_loop::AgentMessage;
use tokio_util::sync::CancellationToken;

const ORIGIN: &str = "https://opencode.cloudflare.dev";
const MCP_STARTUP_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionOptions {
    pub path: Option<PathBuf>,
    pub directory: Option<PathBuf>,
    pub id: Option<String>,
    pub id_prefix: Option<String>,
    pub continue_latest: bool,
    pub no_session: bool,
    pub fork_from: Option<PathBuf>,
    pub name: Option<String>,
    pub tools: Option<Vec<String>>,
    pub no_tools: bool,
    pub no_builtin_tools: bool,
    pub skill_paths: Vec<PathBuf>,
    pub no_skills: bool,
    /// Use the deterministic offline backend instead of the configured provider.
    pub demo: bool,
    /// `--system-prompt`: replaces the default prompt.
    pub system_prompt: Option<String>,
    /// `--append-system-prompt`: each entry is literal text or a file path whose contents are used.
    pub append_system_prompt: Vec<String>,
    /// `--no-context-files`: skip `AGENTS.md`/`CLAUDE.md` discovery.
    pub no_context_files: bool,
    /// `--exclude-tools`: names or `*` patterns to disable, applied after selection.
    pub exclude_tools: Vec<String>,
    /// `--thinking`: requested reasoning level.
    pub thinking: ThinkingSetting,
    /// `--provider`: overrides `defaultProvider`.
    pub provider: Option<String>,
    /// `--model`: model id or `provider/id`, overrides `defaultModel`.
    pub model: Option<String>,
    /// `--no-mcp`: do not connect MCP servers.
    pub no_mcp: bool,
    /// `--prompt-template`: extra template files or directories.
    pub prompt_templates: Vec<PathBuf>,
    /// `--no-prompt-templates`: skip default template directories.
    pub no_prompt_templates: bool,
    /// `--api-key`: runtime API key for the selected provider (never persisted).
    pub api_key: Option<String>,
    /// `--approve` (`Some(true)`) / `--no-approve` (`Some(false)`): project trust for this run.
    pub project_trust: Option<bool>,
    /// `--theme`: theme name for this run (overrides the `theme` setting).
    pub theme: Option<String>,
    /// `--models`: comma-separated patterns scoping Ctrl+P cycling (overrides `enabledModels`).
    pub models: Vec<String>,
    /// `--verbose`: print the resolved startup configuration to stderr (print mode only).
    pub verbose: bool,
    /// Interactive runs continue in the current directory when the session's stored cwd is gone
    /// (Pi asks first; print and RPC runs refuse).
    pub continue_in_current_cwd: bool,
}

/// Pi's thinking levels (`off` … `max`); Mu's default is `medium`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ThinkingSetting {
    Off,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    Xhigh,
    Max,
}

impl std::str::FromStr for ThinkingSetting {
    type Err = String;

    fn from_str(level: &str) -> Result<Self, String> {
        Ok(match level {
            "off" => Self::Off,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::Xhigh,
            "max" => Self::Max,
            other => {
                return Err(format!(
                    "Invalid thinking level \"{other}\". Valid values: off, minimal, low, medium, high, xhigh, max"
                ));
            }
        })
    }
}

impl ThinkingSetting {
    /// The next level in Pi's cycle (`off → minimal → … → max → off`).
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Minimal,
            Self::Minimal => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::Xhigh,
            Self::Xhigh => Self::Max,
            Self::Max => Self::Off,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }

    /// OpenAI Responses `reasoning.effort`, or `None` when thinking is off.
    fn effort(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Minimal => Some("minimal"),
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
            Self::Xhigh | Self::Max => Some("xhigh"),
        }
    }
}

/// Resolved prompt customizations.
#[derive(Clone, Debug, Default)]
struct PromptOptions {
    system_prompt: Option<String>,
    appended: Vec<String>,
    context: String,
    thinking: ThinkingSetting,
    api_key: Option<String>,
    theme: Option<String>,
    models: Vec<String>,
}

impl PromptOptions {
    fn from_session_options(options: &SessionOptions) -> Self {
        Self {
            system_prompt: options.system_prompt.as_deref().map(read_prompt_input),
            appended: options
                .append_system_prompt
                .iter()
                .map(|text| read_prompt_input(text))
                .collect(),
            context: String::new(),
            thinking: options.thinking,
            api_key: options.api_key.clone().filter(|key| !key.is_empty()),
            theme: options.theme.clone(),
            models: options.models.clone(),
        }
    }

    /// Load `AGENTS.md`-style files unless disabled.
    fn with_context_files(mut self, cwd: &Path, agent_dir: &Path, disabled: bool) -> Self {
        if !disabled {
            let files = crate::context_files::load_project_context_files(cwd, agent_dir);
            if !files.is_empty() {
                self.context = crate::context_files::render_project_context(&files);
            }
        }
        self
    }

    fn appendix(&self) -> String {
        let mut sections = self.appended.clone();
        if !self.context.is_empty() {
            sections.push(self.context.clone());
        }
        sections.join("\n\n")
    }
}

/// Pi accepts either literal text or a path to a file whose contents are the prompt.
fn read_prompt_input(input: &str) -> String {
    let path = Path::new(input);
    if path.is_file() {
        if let Ok(contents) = std::fs::read_to_string(path) {
            return contents;
        }
    }
    input.to_owned()
}

struct Config {
    agent_dir: PathBuf,
    cwd: PathBuf,
    provider: String,
    model: String,
    model_choices: Vec<String>,
    hide_thinking_block: bool,
    retry: RetryPolicy,
    enabled_tools: Vec<String>,
}
#[derive(Clone)]
struct Provider {
    transport: ProviderTransport,
    model: String,
    instructions: String,
    /// A custom `--system-prompt` replaces the default prompt, including the tool section.
    describes_tools: bool,
    /// `--append-system-prompt` text placed after everything else.
    appendix: String,
    supports_reasoning: bool,
    /// The model accepts image input (`input` includes `image` in `models.json`; default yes).
    supports_images: bool,
    /// Requested reasoning effort (`--thinking`); `None` means off.
    reasoning_effort: Option<&'static str>,
    supports_strict_sampling: bool,
}

/// How Bedrock requests authenticate.
#[derive(Clone)]
enum BedrockAuth {
    /// A Bedrock API key sent as a bearer token.
    Bearer(String),
    /// AWS SigV4 signing with environment credentials.
    Sigv4 {
        credentials: sigv4::AwsCredentials,
        region: String,
    },
}

/// `bedrock-runtime.<region>.amazonaws.com` → `<region>`.
fn bedrock_region_from_host(host: &str) -> Option<String> {
    let rest = host.strip_prefix("bedrock-runtime.")?;
    rest.strip_suffix(".amazonaws.com").map(str::to_owned)
}

/// How a Gemini request authenticates: Generative AI takes an API key header, Vertex AI an OAuth
/// access token.
#[derive(Clone)]
enum GoogleAuth {
    ApiKey(String),
    Bearer(String),
}

#[derive(Clone)]
enum ProviderTransport {
    OpenCode {
        http: Client,
        endpoint: Url,
        token: String,
        gateway_headers: bool,
    },
    /// Bedrock Converse (`api: "bedrock-converse-stream"`), bearer-token auth only.
    Bedrock {
        http: Client,
        endpoint: Url,
        auth: BedrockAuth,
    },
    /// Anthropic Messages API (`api: "anthropic-messages"` in `models.json`).
    Anthropic {
        http: Client,
        endpoint: Url,
        key: String,
    },
    /// OpenAI Chat Completions (`api: "openai-completions"`).
    Completions {
        http: Client,
        endpoint: Url,
        key: String,
    },
    /// Google Generative AI (`api: "google-generative-ai"`); the endpoint includes the model.
    Google {
        http: Client,
        endpoint: Url,
        auth: GoogleAuth,
    },
    Demo,
    #[cfg(test)]
    Scripted(Arc<std::sync::Mutex<std::collections::VecDeque<Result<Reply, String>>>>),
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
        /// Structured tool details (truncation info, diffs) for renderers.
        details: Option<Value>,
        duration_ms: Option<u64>,
    },
    QueuedPromptStarted {
        prompt: String,
        behavior: PromptBehavior,
    },
    RetryScheduled {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        error: String,
    },
    RetryAttemptStarted {
        attempt: u32,
    },
    SkillCommandWarning(String),
    /// The context passed the compaction threshold; a summary is being generated.
    AutoCompactionStarted,
    /// Automatic compaction finished: the summary now in the session and the tokens it replaced.
    AutoCompactionFinished {
        summary: String,
        tokens_before: u64,
    },
}
enum TerminalResponse {
    Completed,
    Incomplete { reason: Option<String> },
}

#[derive(Debug)]
enum SkillCommandError {
    ReadFile { skill: String, error: String },
}

impl std::fmt::Display for SkillCommandError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadFile { skill, error } => {
                write!(formatter, "Could not read skill '{skill}': {error}")
            }
        }
    }
}

fn response_failed_error(event: &Value) -> AgentError {
    let error = event.pointer("/response/error");
    let message = if let Some(error) = error {
        format!(
            "{}: {}",
            error
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("no message")
        )
    } else if let Some(reason) = event
        .pointer("/response/incomplete_details/reason")
        .and_then(Value::as_str)
    {
        format!("incomplete: {reason}")
    } else {
        "Unknown error (no error details in response)".into()
    };
    AgentError::Provider(message)
}

fn format_mcp_tool_summary(definition: &Value) -> Option<String> {
    let name = definition.get("name")?.as_str()?;
    let description = definition
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .lines()
        .next()
        .unwrap_or_default();
    Some(if description.is_empty() {
        name.to_owned()
    } else {
        format!("{name}: {description}")
    })
}

fn response_error_event(event: &Value) -> AgentError {
    // Pi interpolates absent JavaScript event fields as the literal `undefined`.
    let code = event
        .get("code")
        .and_then(Value::as_str)
        .unwrap_or("undefined");
    let message = event
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("undefined");
    AgentError::Provider(format!("Error Code {code}: {message}"))
}

impl TerminalResponse {
    fn error_message(&self) -> Option<String> {
        match self {
            Self::Completed => None,
            Self::Incomplete {
                reason: Some(reason),
            } if reason == "max_output_tokens" => None,
            Self::Incomplete {
                reason: Some(reason),
            } => Some(format!("Response incomplete: {reason}")),
            Self::Incomplete { reason: None } => {
                Some("Response incomplete without a provider reason".into())
            }
        }
    }

    fn incomplete_reason(&self) -> Option<&str> {
        match self {
            Self::Completed => None,
            Self::Incomplete { reason } => reason.as_deref(),
        }
    }
}

/// Feeds Anthropic stream events through the decoder; returns text bytes plus delta events.
#[cfg(feature = "bench-internals")]
pub fn decode_anthropic_events(events: &[Value]) -> usize {
    let mut decoder = anthropic::Decoder::default();
    let mut deltas = 0;
    for event in events {
        decoder
            .feed(event, &mut |_| deltas += 1)
            .expect("benchmark events decode");
    }
    decoder.into_reply().text.len() + deltas
}

/// Feeds Chat Completions chunks through the decoder; returns text bytes plus delta events.
#[cfg(feature = "bench-internals")]
pub fn decode_completions_chunks(chunks: &[Value]) -> usize {
    let mut decoder = completions::Decoder::default();
    let mut deltas = 0;
    for chunk in chunks {
        decoder
            .feed(chunk, &mut |_| deltas += 1)
            .expect("benchmark chunks decode");
    }
    decoder.into_reply().text.len() + deltas
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
    incomplete_reason: Option<String>,
}

fn expand_skill_prompt(
    prompt: String,
    skills: &crate::skills::SkillCatalog,
    on_event: &mut impl FnMut(AgentEvent),
) -> String {
    match expand_skill_command(prompt.clone(), skills) {
        Ok(expanded) => expanded,
        Err(error) => {
            on_event(AgentEvent::SkillCommandWarning(error.to_string()));
            prompt
        }
    }
}

fn expand_skill_command(
    prompt: String,
    skills: &crate::skills::SkillCatalog,
) -> Result<String, SkillCommandError> {
    let Some(command) = prompt.strip_prefix("/skill:") else {
        return Ok(prompt);
    };
    let (name, arguments) = command
        .split_once(' ')
        .map_or((command, ""), |(name, arguments)| (name, arguments.trim()));
    let Some(skill) = skills.skills.iter().find(|skill| skill.name == name) else {
        return Ok(prompt);
    };
    let content =
        std::fs::read_to_string(&skill.file_path).map_err(|error| SkillCommandError::ReadFile {
            skill: name.to_owned(),
            error: error.to_string(),
        })?;
    let body = crate::skills::strip_frontmatter(&content);
    let base_dir = skill.file_path.parent().unwrap_or_else(|| Path::new(""));
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path.display(),
        base_dir.display(),
        body,
    );
    Ok(if arguments.is_empty() {
        block
    } else {
        format!("{block}\n\n{arguments}")
    })
}

fn configured_setting<T>(
    agent_dir: &Path,
    cwd: &Path,
    key: &str,
    parse: impl Fn(&Value) -> Option<T>,
) -> Option<T> {
    let settings = crate::settings::Settings::load_for_project(agent_dir, cwd);
    settings.merged().get(key).and_then(parse)
}

fn shell_setting(agent_dir: &Path, cwd: &Path, key: &str) -> Option<String> {
    configured_setting(agent_dir, cwd, key, |value| {
        value.as_str().map(str::to_owned)
    })
    .filter(|value| !value.is_empty())
}

fn skill_commands_enabled(agent_dir: &Path, cwd: &Path) -> bool {
    configured_setting(agent_dir, cwd, "enableSkillCommands", Value::as_bool).unwrap_or(true)
}

fn configured_bash_shell_path(agent_dir: &Path, cwd: &Path) -> Option<String> {
    let configured = shell_setting(agent_dir, cwd, "shellPath")?;
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    let path = match home {
        Some(home) if configured == "~" => home,
        Some(home) if configured.starts_with("~/") || configured.starts_with("~\\") => {
            home.join(&configured[2..])
        }
        _ => PathBuf::from(configured),
    };
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    Some(path.to_string_lossy().into_owned())
}

fn config() -> Result<Config, AgentError> {
    config_for(false)
}

/// Pi's `formatNoApiKeyFoundMessage` (`core/auth-guidance.ts`).
fn no_api_key_message(provider: &str) -> String {
    format!(
        "No API key found for {provider}.\n\nUse /login to log into a provider via OAuth or API key, or set apiKey for it in models.json (a literal, $ENV_VAR, or !command)."
    )
}

/// Pi's `formatNoModelSelectedMessage` (`core/auth-guidance.ts`), with the cause first.
fn no_model_selected_message(cause: &str) -> String {
    format!(
        "{cause}. No model selected.\n\nUse /login to log into a provider via OAuth or API key, or add one to models.json.\n\nThen use /model to select a model."
    )
}

/// Read Pi settings; with `demo`, a missing provider or model falls back to the demo backend.
fn config_for(demo: bool) -> Result<Config, AgentError> {
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
    let mut retry = RetryPolicy::default();
    let mut default_tool_entries = None;
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
        update_retry_policy(&mut retry, &value);
        merge_default_tools(&mut default_tool_entries, &value)?;
    }
    let provider = match provider {
        Some(provider) => provider,
        None if demo => "mu-demo".to_owned(),
        None => {
            return Err(AgentError::Configuration(no_model_selected_message(
                "Pi settings have no defaultProvider",
            )));
        }
    };
    let model = match model {
        Some(model) => model,
        None if demo => "mu-demo".to_owned(),
        None => {
            return Err(AgentError::Configuration(no_model_selected_message(
                "Pi settings have no defaultModel",
            )));
        }
    };
    let mut model_choices = configured_models(&agent_dir, &provider, &model)?;
    if demo && provider == "mu-demo" && model_choices.len() == 1 {
        model_choices.push("mu-demo-alt".into());
    }
    Ok(Config {
        agent_dir,
        cwd,
        provider,
        model,
        model_choices,
        hide_thinking_block,
        retry,
        enabled_tools: resolve_default_tools(default_tool_entries),
    })
}

/// Apply `--provider` / `--model` (`provider/id` allowed) over the settings defaults.
/// Resolve `enabledModels` patterns to model ids, in pattern order without duplicates.
///
/// A pattern matches a model's id or `provider/id` case-insensitively and may use `*`/`?` globs;
/// an optional `:<thinking>` suffix is ignored. With no patterns, or none matching, every
/// choice is returned.
fn scope_models(patterns: &[String], choices: &[String], provider: &str) -> Vec<String> {
    let mut scoped: Vec<String> = Vec::new();
    for pattern in patterns {
        let base = pattern
            .rsplit_once(':')
            .filter(|(_, level)| level.parse::<ThinkingSetting>().is_ok())
            .map_or(pattern.as_str(), |(base, _)| base);
        let Ok(glob) = globset::GlobBuilder::new(&base.to_lowercase()).build() else {
            continue;
        };
        let matcher = glob.compile_matcher();
        for choice in choices {
            let id = choice.to_lowercase();
            if (matcher.is_match(&id)
                || matcher.is_match(format!("{}/{id}", provider.to_lowercase())))
                && !scoped.contains(choice)
            {
                scoped.push(choice.clone());
            }
        }
    }
    if scoped.is_empty() {
        choices.to_vec()
    } else {
        scoped
    }
}

fn apply_model_override(
    config: &mut Config,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<(), AgentError> {
    if let Some(provider) = provider {
        config.provider = provider.to_owned();
    }
    if let Some(model) = model {
        match model.split_once('/') {
            Some((provider, id)) if provider_is_known(config, provider) => {
                config.provider = provider.to_owned();
                config.model = id.to_owned();
            }
            _ => config.model = model.to_owned(),
        }
    }
    if provider.is_some() || model.is_some() {
        config.model_choices =
            configured_models(&config.agent_dir, &config.provider, &config.model)?;
    }
    Ok(())
}

/// A prefix before `/` names a provider only if it is `opencode-cf*` or listed in `models.json`.
fn provider_is_known(config: &Config, name: &str) -> bool {
    name.starts_with("opencode-cf")
        || std::fs::read_to_string(config.agent_dir.join("models.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .is_some_and(|json| json["providers"].get(name).is_some())
}

/// A `models.json` model accepts images unless it lists `input` without `image`.
fn model_accepts_images(model: &Value) -> bool {
    model
        .get("input")
        .and_then(Value::as_array)
        .is_none_or(|input| input.iter().any(|kind| kind == "image"))
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

/// Refresh a stored Anthropic OAuth credential that has expired and save the new tokens.
async fn refresh_expired_oauth(config: &Config) -> Result<(), AgentError> {
    let path = config.agent_dir.join("auth.json");
    let Some((_, refresh, expires)) = crate::auth::oauth_access(&path, &config.provider) else {
        return Ok(());
    };
    let client_id = crate::auth::load(&path)
        .ok()
        .and_then(|mut all| all.remove(&config.provider))
        .and_then(|credential| match credential {
            crate::auth::Credential::Oauth { extra, .. } => extra
                .get("clientId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            crate::auth::Credential::ApiKey { .. } => None,
        });
    if !crate::oauth::is_expired(expires) {
        return Ok(());
    }
    let refreshed = match config.provider.as_str() {
        "anthropic" => crate::oauth::anthropic::refresh_token(&refresh).await,
        "xai" => crate::oauth::xai::refresh_token(&refresh).await,
        "openai-codex" => {
            crate::oauth::openai_codex::refresh_token(&refresh, crate::oauth::openai_codex::TOKEN_URL)
                .await
        }
        "github-copilot" => {
            let enterprise = crate::auth::load(&path)
                .ok()
                .and_then(|mut all| all.remove(&config.provider))
                .and_then(|credential| match credential {
                    crate::auth::Credential::Oauth { extra, .. } => extra
                        .get("enterpriseUrl")
                        .and_then(Value::as_str)
                        .and_then(crate::oauth::github_copilot::normalize_domain),
                    crate::auth::Credential::ApiKey { .. } => None,
                });
            crate::oauth::github_copilot::copilot_token(&refresh, enterprise.as_deref()).await
        }
        "meta" => {
            crate::oauth::meta::mint_api_key(&refresh, crate::oauth::meta::API_KEY_MINT_URL).await
        }
        "openai-chatgpt" => match client_id {
            Some(client_id) => {
                crate::oauth::openai_chatgpt::refresh_token(
                    &refresh,
                    &client_id,
                    crate::oauth::openai_chatgpt::TOKEN_URL,
                )
                .await
            }
            None => Err(
                "Stored OpenAI OAuth credential does not contain an issued client ID; reconnect ChatGPT"
                    .to_owned(),
            ),
        },
        "kimi-coding" => {
            let host = std::env::var("KIMI_CODE_OAUTH_HOST")
                .or_else(|_| std::env::var("KIMI_OAUTH_HOST"))
                .unwrap_or_else(|_| crate::oauth::kimi::DEFAULT_HOST.to_owned());
            crate::oauth::kimi::refresh_token(host.trim_end_matches('/'), &refresh).await
        }
        _ => return Ok(()),
    };
    let credential = refreshed.map_err(|error| {
        AgentError::Provider(format!(
            "OAuth refresh failed ({error}); run `mu auth login --provider {}`",
            config.provider
        ))
    })?;
    crate::auth::set_credential(&path, &config.provider, credential)
        .map_err(|error| AgentError::Configuration(error.to_string()))
}

async fn connect(
    config: &Config,
    skill_prompt: String,
    prompt: &PromptOptions,
) -> Result<Provider, AgentError> {
    if !config.provider.starts_with("opencode-cf") {
        refresh_expired_oauth(config).await?;
        return connect_custom_provider(config, skill_prompt, prompt);
    }
    let token = match &prompt.api_key {
        Some(key) => key.clone(),
        None => match crate::auth::api_key(&config.agent_dir.join("auth.json"), &config.provider) {
            Some(key) => key,
            None => token(&config.provider).await?,
        },
    };
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
    let mut supports_strict_sampling = false;
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
        supports_strict_sampling = model_supports_strict_sampling(m);
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
            gateway_headers: true,
        },
        model: config.model.clone(),
        instructions: match &prompt.system_prompt {
            Some(custom) => format!("{custom}{skill_prompt}"),
            None => format!(
                "You are Mu, a helpful coding assistant. Be clear, practical, and concise.{skill_prompt}"
            ),
        },
        describes_tools: prompt.system_prompt.is_none(),
        appendix: prompt.appendix(),
        supports_reasoning,
        supports_images: true,
        reasoning_effort: prompt.thinking.effort(),
        supports_strict_sampling,
    })
}

impl Provider {
    async fn complete_google(
        &self,
        http: &Client,
        endpoint: &Url,
        auth: &GoogleAuth,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut system = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            system = format!("{system}\n\n{}", self.appendix);
        }
        let request = http.post(endpoint.clone());
        let request = match auth {
            GoogleAuth::Bearer(token) => request.bearer_auth(token),
            GoogleAuth::ApiKey(key) => request.header("x-goog-api-key", key),
        };
        let response = request
            .json(&{
                let mut body = google::request_body(&system, history, tools);
                if self.supports_reasoning {
                    body["generationConfig"] = json!({"thinkingConfig": google::thinking_config(&self.model, self.reasoning_effort)});
                }
                body
            })
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
        }
        let mut stream = response.bytes_stream();
        let mut frames = crate::sse::SseFrames::default();
        let mut decoder = google::Decoder::default();
        while let Some(chunk) = stream.next().await {
            frames.push(
                &chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?,
            );
            while let Some(data) = frames.next_data() {
                if let Ok(event) = serde_json::from_str::<Value>(&data) {
                    decoder
                        .feed(&event, on_event)
                        .map_err(AgentError::Provider)?;
                }
            }
        }
        if !decoder.finished() {
            return Err(AgentError::Provider(
                "Google stream ended before a finish reason".into(),
            ));
        }
        Ok(decoder.into_reply())
    }

    async fn complete_completions(
        &self,
        http: &Client,
        endpoint: &Url,
        key: &str,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut system = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            system = format!("{system}\n\n{}", self.appendix);
        }
        let effort = self.reasoning_effort.filter(|_| self.supports_reasoning);
        let body = completions::request_body(&self.model, &system, history, tools, effort);
        let response = http
            .post(endpoint.clone())
            .bearer_auth(key)
            .header(header::ACCEPT, "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
        }
        let mut stream = response.bytes_stream();
        let mut frames = crate::sse::SseFrames::default();
        let mut decoder = completions::Decoder::default();
        while let Some(chunk) = stream.next().await {
            frames.push(
                &chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?,
            );
            while let Some(data) = frames.next_data() {
                if data == "[DONE]" {
                    decoder.feed_done();
                } else if let Ok(event) = serde_json::from_str::<Value>(&data) {
                    decoder
                        .feed(&event, on_event)
                        .map_err(AgentError::Provider)?;
                }
            }
        }
        if !decoder.finished() {
            return Err(AgentError::Provider(
                "Completions stream ended before a finish reason".into(),
            ));
        }
        Ok(decoder.into_reply())
    }

    async fn complete_bedrock(
        &self,
        http: &Client,
        endpoint: &Url,
        auth: &BedrockAuth,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut system = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            system = format!("{system}\n\n{}", self.appendix);
        }
        let mut request_json = bedrock::request_body(&system, history, tools);
        if self.supports_reasoning
            && let Some((fields, budget)) =
                bedrock::thinking_fields(&self.model, self.reasoning_effort)
        {
            request_json["additionalModelRequestFields"] = fields;
            // Claude requires max tokens above the thinking budget.
            if budget > 0 {
                request_json["inferenceConfig"]["maxTokens"] = json!(budget + 16_384);
            }
        }
        let body = serde_json::to_vec(&request_json).expect("request serializes");
        let mut request = http
            .post(endpoint.clone())
            .header(header::CONTENT_TYPE, "application/json");
        match auth {
            BedrockAuth::Bearer(key) => request = request.bearer_auth(key),
            BedrockAuth::Sigv4 {
                credentials,
                region,
            } => {
                let host = endpoint.host_str().unwrap_or_default().to_owned();
                let amz_date = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
                let signed = sigv4::sign(
                    "POST",
                    endpoint.path(),
                    endpoint.query().unwrap_or_default(),
                    &[
                        ("host".into(), host),
                        ("content-type".into(), "application/json".into()),
                    ],
                    &body,
                    credentials,
                    region,
                    "bedrock",
                    &amz_date,
                );
                request = request
                    .header("x-amz-date", signed.amz_date)
                    .header(header::AUTHORIZATION, signed.authorization);
                if let Some(token) = &credentials.session_token {
                    request = request.header("x-amz-security-token", token);
                }
            }
        }
        let response = request
            .body(body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            let body: Value = response.json().await.unwrap_or(Value::Null);
            let detail = body["message"].as_str().unwrap_or("no details");
            return Err(AgentError::Provider(format!(
                "model request returned HTTP {status}: {detail}"
            )));
        }
        // `converse-stream` answers with AWS's binary event stream, decoded as it arrives.
        let mut stream = response.bytes_stream();
        let mut decoder = bedrock::EventStreamDecoder::default();
        let mut accumulator = bedrock::StreamAccumulator::default();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?;
            for message in decoder.push(&chunk).map_err(AgentError::Provider)? {
                accumulator
                    .feed(&message, on_event)
                    .map_err(AgentError::Provider)?;
            }
        }
        Ok(accumulator.into_reply())
    }

    async fn complete_anthropic(
        &self,
        http: &Client,
        endpoint: &Url,
        key: &str,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let mut system = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            system = format!("{system}\n\n{}", self.appendix);
        }
        let budget = self
            .reasoning_effort
            .filter(|_| self.supports_reasoning)
            .map(|effort| match effort {
                "minimal" => 1_024,
                "low" => 2_048,
                "medium" => 8_192,
                "high" => 16_384,
                _ => 32_768,
            });
        let oauth = anthropic::is_oauth_token(key);
        if oauth {
            system = format!("{}\n\n{system}", anthropic::CLAUDE_CODE_IDENTITY);
        }
        let (history, wire_tools) = if oauth {
            anthropic::rename_for_claude_code(history, tools)
        } else {
            (history.to_vec(), tools.to_vec())
        };
        let mut body = anthropic::request_body(&self.model, &system, &history, &wire_tools, budget);
        if let Some(budget) = budget {
            body["max_tokens"] = json!(budget + 16_384);
        }
        let request = http.post(endpoint.clone());
        let request = if oauth {
            // Pi: OAuth tokens use Bearer auth plus Claude Code identity headers.
            request
                .bearer_auth(key)
                .header("anthropic-beta", "claude-code-20250219,oauth-2025-04-20")
                .header("user-agent", "claude-cli/2.1.280")
                .header("x-app", "cli")
        } else {
            request.header("x-api-key", key)
        };
        let response = request
            .header("anthropic-version", "2023-06-01")
            .header(header::ACCEPT, "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
        }
        let mut stream = response.bytes_stream();
        let mut frames = crate::sse::SseFrames::default();
        let mut decoder = anthropic::Decoder::default();
        while let Some(chunk) = stream.next().await {
            frames.push(
                &chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?,
            );
            while let Some(data) = frames.next_data() {
                if let Ok(event) = serde_json::from_str::<Value>(&data) {
                    decoder
                        .feed(&event, on_event)
                        .map_err(AgentError::Provider)?;
                }
            }
        }
        if !decoder.finished() {
            return Err(AgentError::Provider(
                "Anthropic stream ended before message_stop".into(),
            ));
        }
        let mut reply = decoder.into_reply();
        if oauth {
            for call in &mut reply.tool_calls {
                call.name = anthropic::from_claude_code_name(&call.name, tools);
            }
        }
        Ok(reply)
    }

    async fn complete(
        &self,
        history: &[Value],
        tools: &[Value],
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<Reply, AgentError> {
        let downgraded;
        let history = if self.supports_images {
            history
        } else {
            downgraded = runtime::downgrade_images(history);
            &downgraded
        };
        #[cfg(test)]
        if let ProviderTransport::Scripted(replies) = &self.transport {
            let reply = replies
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .pop_front()
                .ok_or_else(|| AgentError::Provider("scripted provider exhausted".into()))?
                .map_err(AgentError::Provider)?;
            for character in reply.thinking.chars() {
                on_event(AgentEvent::ThinkingDelta(character.to_string()));
            }
            for character in reply.text.chars() {
                on_event(AgentEvent::TextDelta(character.to_string()));
            }
            return Ok(reply);
        }
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
                incomplete_reason: None,
            });
        }
        if let ProviderTransport::Google {
            http,
            endpoint,
            auth,
        } = &self.transport
        {
            return self
                .complete_google(http, endpoint, auth, history, tools, on_event)
                .await;
        }
        if let ProviderTransport::Completions {
            http,
            endpoint,
            key,
        } = &self.transport
        {
            return self
                .complete_completions(http, endpoint, key, history, tools, on_event)
                .await;
        }
        if let ProviderTransport::Bedrock {
            http,
            endpoint,
            auth,
        } = &self.transport
        {
            return self
                .complete_bedrock(http, endpoint, auth, history, tools, on_event)
                .await;
        }
        if let ProviderTransport::Anthropic {
            http,
            endpoint,
            key,
        } = &self.transport
        {
            return self
                .complete_anthropic(http, endpoint, key, history, tools, on_event)
                .await;
        }
        let ProviderTransport::OpenCode {
            http,
            endpoint,
            token,
            gateway_headers,
        } = &self.transport
        else {
            unreachable!("demo backend returned above")
        };
        let input = history;
        let mut instructions = if self.describes_tools {
            instructions_for_tools(&self.instructions, tools)
        } else {
            self.instructions.clone()
        };
        if !self.appendix.is_empty() {
            instructions = format!("{instructions}\n\n{}", self.appendix);
        }
        let mut request_body = json!({"model":self.model,"instructions":instructions,"input":input,"tools":tools,"stream":true,"store":false});
        if let Some(effort) = self.reasoning_effort.filter(|_| self.supports_reasoning) {
            request_body["reasoning"] = json!({"effort":effort,"summary":"auto"});
            request_body["include"] = json!(["reasoning.encrypted_content"]);
        }
        let mut request = http.post(endpoint.clone()).bearer_auth(token);
        if *gateway_headers {
            // Cloudflare Access headers must never be sent to third-party providers.
            request = request
                .header("cf-access-token", token)
                .header("x-requested-with", "xmlhttprequest");
        }
        let response = request
            .header(header::ACCEPT, "text/event-stream")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AgentError::Provider(format!("model request failed: {e}")))?;
        if !response.status().is_success() {
            return Err(http_failure(response).await);
        }
        let mut stream = response.bytes_stream();
        let mut frames = crate::sse::SseFrames::default();
        let mut text = String::new();
        let mut thinking = String::new();
        let mut thinking_signature = None;
        let mut output_items = Vec::new();
        let mut tool_calls = Vec::new();
        let mut terminal_response = None;
        let (mut input_tokens, mut output_tokens) = (0, 0);
        let (mut cache_read, mut cache_write, mut reasoning, mut total_tokens) = (0, 0, None, 0);
        while let Some(chunk) = stream.next().await {
            frames.push(
                &chunk.map_err(|e| AgentError::Provider(format!("model stream failed: {e}")))?,
            );
            while let Some(data) = frames.next_data() {
                if let Ok(event) = serde_json::from_str::<Value>(&data) {
                    match event.get("type").and_then(Value::as_str).unwrap_or("") {
                        "response.output_text.delta" | "response.refusal.delta" => {
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
                        "response.completed" | "response.incomplete" => {
                            let incomplete_reason = event
                                .pointer("/response/incomplete_details/reason")
                                .and_then(Value::as_str)
                                .map(str::to_owned);
                            terminal_response = Some(
                                if event.get("type").and_then(Value::as_str)
                                    == Some("response.completed")
                                {
                                    TerminalResponse::Completed
                                } else {
                                    TerminalResponse::Incomplete {
                                        reason: incomplete_reason.clone(),
                                    }
                                },
                            );
                            if let Some(items) =
                                event.pointer("/response/output").and_then(Value::as_array)
                            {
                                output_items = items.clone();
                                if let Some(delta) =
                                    merge_final_response_text(&mut text, &output_items)
                                {
                                    on_event(AgentEvent::TextDelta(delta));
                                }
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
                                    let arguments = serde_json::from_str(raw_arguments)
                                        .unwrap_or_else(|_| {
                                            if incomplete_reason.as_deref()
                                                == Some("max_output_tokens")
                                            {
                                                Value::String(raw_arguments.to_owned())
                                            } else {
                                                Value::Null
                                            }
                                        });
                                    if !arguments.is_null() {
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
                        "response.failed" => return Err(response_failed_error(&event)),
                        "error" => return Err(response_error_event(&event)),
                        _ => {}
                    }
                }
            }
        }
        let terminal_response = terminal_response.ok_or_else(|| {
            AgentError::Provider(
                "OpenAI Responses stream ended before a terminal response event".into(),
            )
        })?;
        let incomplete_reason = terminal_response.incomplete_reason().map(str::to_owned);
        if let Some(message) = terminal_response.error_message() {
            return Err(AgentError::Provider(message));
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
            incomplete_reason,
        })
    }
}

fn response_message_text(output_items: &[Value]) -> String {
    output_items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter_map(|block| match block.get("type").and_then(Value::as_str) {
            Some("output_text") => block.get("text").and_then(Value::as_str),
            Some("refusal") => block.get("refusal").and_then(Value::as_str),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn merge_final_response_text(streamed_text: &mut String, output_items: &[Value]) -> Option<String> {
    let final_text = response_message_text(output_items);
    if final_text.is_empty() || final_text == *streamed_text {
        return None;
    }
    if let Some(delta) = final_text.strip_prefix(streamed_text.as_str()) {
        streamed_text.push_str(delta);
        return Some(delta.to_owned());
    }
    *streamed_text = final_text;
    None
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

/// A provider declared in `models.json` with `api: "openai-responses"`, `baseUrl` and `apiKey`
/// (an environment variable name, else the literal key).
fn connect_custom_provider(
    config: &Config,
    skill_prompt: String,
    prompt: &PromptOptions,
) -> Result<Provider, AgentError> {
    let unsupported = || {
        AgentError::Configuration(format!(
            "Provider `{}` is not configured: add it to models.json with api \"openai-responses\", baseUrl and apiKey",
            config.provider
        ))
    };
    let models: Value = std::fs::read_to_string(config.agent_dir.join("models.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .ok_or_else(unsupported)?;
    let entry = models["providers"]
        .get(&config.provider)
        .ok_or_else(unsupported)?;
    match entry.get("api").and_then(Value::as_str) {
        Some(
            "openai-responses"
            | "anthropic-messages"
            | "openai-completions"
            | "google-generative-ai"
            | "google-vertex"
            | "bedrock-converse-stream",
        ) => {}
        other => {
            return Err(AgentError::Configuration(format!(
                "Provider `{}` uses unsupported api `{}` (supported: openai-responses, openai-completions, anthropic-messages, google-generative-ai, google-vertex, bedrock-converse-stream)",
                config.provider,
                other.unwrap_or("none")
            )));
        }
    }
    let base = match (
        entry.get("baseUrl").and_then(Value::as_str),
        entry.get("api").and_then(Value::as_str),
    ) {
        (Some(base), _) => base.to_owned(),
        (None, Some("google-vertex")) => vertex_base_url(entry)?,
        // A GitHub Copilot login names its API host inside the stored token.
        (None, _) if config.provider == "github-copilot" => {
            let (access, _, _) =
                crate::auth::oauth_access(&config.agent_dir.join("auth.json"), &config.provider)
                    .ok_or_else(unsupported)?;
            let enterprise = crate::auth::load(&config.agent_dir.join("auth.json"))
                .ok()
                .and_then(|mut all| all.remove(&config.provider))
                .and_then(|credential| match credential {
                    crate::auth::Credential::Oauth { extra, .. } => extra
                        .get("enterpriseUrl")
                        .and_then(Value::as_str)
                        .and_then(crate::oauth::github_copilot::normalize_domain),
                    crate::auth::Credential::ApiKey { .. } => None,
                });
            crate::oauth::github_copilot::base_url(&access, enterprise.as_deref())
        }
        (None, _) => return Err(unsupported()),
    };
    let base = base.as_str();
    let mut endpoint = Url::parse(base)
        .map_err(|error| AgentError::Configuration(format!("invalid baseUrl: {error}")))?;
    let anthropic = entry.get("api").and_then(Value::as_str) == Some("anthropic-messages");
    let completions = entry.get("api").and_then(Value::as_str) == Some("openai-completions");
    let vertex = entry.get("api").and_then(Value::as_str) == Some("google-vertex");
    let google = vertex || entry.get("api").and_then(Value::as_str) == Some("google-generative-ai");
    let bedrock = entry.get("api").and_then(Value::as_str) == Some("bedrock-converse-stream");
    let suffix = if bedrock {
        format!("model/{}/converse-stream", config.model)
    } else if anthropic {
        "messages".to_owned()
    } else if completions {
        "chat/completions".to_owned()
    } else if google {
        format!("models/{}:streamGenerateContent", config.model)
    } else {
        "responses".to_owned()
    };
    endpoint.set_path(&format!(
        "{}/{suffix}",
        endpoint.path().trim_end_matches('/')
    ));
    if google {
        endpoint.set_query(Some("alt=sse"));
    }
    // Bedrock may omit `apiKey` and sign with AWS credentials from the environment instead.
    // `apiKey` may be omitted when a stored login (OAuth or `auth.json`) supplies the token; a
    // missing credential is reported below.
    let key = entry
        .get("apiKey")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let token = prompt
        .api_key
        .clone()
        .or_else(|| crate::auth::api_key(&config.agent_dir.join("auth.json"), &config.provider))
        .or_else(|| {
            crate::auth::oauth_access(&config.agent_dir.join("auth.json"), &config.provider)
                .map(|(access, _, _)| access)
        })
        .or_else(|| crate::config_value::resolve(key, &HashMap::new()))
        .filter(|v| !v.is_empty())
        .unwrap_or_default();
    if token.is_empty() && !bedrock {
        return Err(AgentError::Configuration(no_api_key_message(
            &config.provider,
        )));
    }
    let bedrock_auth = if !bedrock {
        None
    } else if !token.is_empty() {
        Some(BedrockAuth::Bearer(token.clone()))
    } else {
        let credentials = sigv4::AwsCredentials::from_env().ok_or_else(|| {
            AgentError::Configuration(
                "Bedrock needs an apiKey (Bedrock API key) or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
                    .into(),
            )
        })?;
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .ok()
            .or_else(|| bedrock_region_from_host(endpoint.host_str().unwrap_or_default()))
            .ok_or_else(|| {
                AgentError::Configuration("Bedrock region unknown: set AWS_REGION".into())
            })?;
        Some(BedrockAuth::Sigv4 {
            credentials,
            region,
        })
    };
    let http = provider_http_client(entry.get("headers"), &config.provider)?;
    let model = configured_model_entry(&config.agent_dir, &config.provider, &config.model);
    let flag = |name: &str| {
        model
            .as_ref()
            .and_then(|m| m.get(name))
            .and_then(Value::as_bool)
    };
    Ok(Provider {
        transport: if bedrock {
            ProviderTransport::Bedrock {
                http: http.clone(),
                endpoint,
                auth: bedrock_auth.expect("bedrock auth resolved above"),
            }
        } else if anthropic {
            ProviderTransport::Anthropic {
                http: http.clone(),
                endpoint,
                key: token,
            }
        } else if google {
            ProviderTransport::Google {
                http: http.clone(),
                endpoint,
                auth: if vertex {
                    GoogleAuth::Bearer(token)
                } else {
                    GoogleAuth::ApiKey(token)
                },
            }
        } else if completions {
            ProviderTransport::Completions {
                http: http.clone(),
                endpoint,
                key: token,
            }
        } else {
            ProviderTransport::OpenCode {
                http: http.clone(),
                endpoint,
                token,
                gateway_headers: false,
            }
        },
        model: config.model.clone(),
        instructions: match &prompt.system_prompt {
            Some(custom) => format!("{custom}{skill_prompt}"),
            None => format!(
                "You are Mu, a helpful coding assistant. Be clear, practical, and concise.{skill_prompt}"
            ),
        },
        describes_tools: prompt.system_prompt.is_none(),
        appendix: prompt.appendix(),
        supports_reasoning: flag("reasoning").unwrap_or(false),
        supports_images: model.as_ref().is_none_or(model_accepts_images),
        reasoning_effort: prompt.thinking.effort(),
        supports_strict_sampling: model.as_ref().is_some_and(model_supports_strict_sampling),
    })
}

/// An HTTP client sending the provider's `headers` from `models.json` on every request; values
/// resolve like other config values (`$VAR`, `!command`) and unresolved ones are skipped.
/// Pi's static `COPILOT_HEADERS` plus the `Openai-Intent` of its dynamic ones. `X-Initiator` and
/// `Copilot-Vision-Request` depend on the request and are not sent.
const COPILOT_HEADERS: &[(&str, &str)] = &[
    ("user-agent", "GitHubCopilotChat/0.35.0"),
    ("editor-version", "vscode/1.107.0"),
    ("editor-plugin-version", "copilot-chat/0.35.0"),
    ("copilot-integration-id", "vscode-chat"),
    ("openai-intent", "conversation-edits"),
];

fn provider_http_client(headers: Option<&Value>, provider: &str) -> Result<Client, AgentError> {
    let mut defaults = header::HeaderMap::new();
    if provider == "github-copilot" {
        for (name, value) in COPILOT_HEADERS {
            defaults.insert(
                header::HeaderName::from_static(name),
                header::HeaderValue::from_static(value),
            );
        }
    }
    for (name, value) in headers.and_then(Value::as_object).into_iter().flatten() {
        let Some(resolved) = value
            .as_str()
            .and_then(|v| crate::config_value::resolve(v, &HashMap::new()))
        else {
            continue;
        };
        let invalid =
            |what: &str| AgentError::Configuration(format!("invalid header {what} `{name}`"));
        defaults.insert(
            header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| invalid("name"))?,
            header::HeaderValue::from_str(&resolved).map_err(|_| invalid("value"))?,
        );
    }
    Client::builder()
        .default_headers(defaults)
        .build()
        .map_err(|error| AgentError::Configuration(format!("cannot build HTTP client: {error}")))
}

/// Vertex AI's regional endpoint from `project` and `location` (default `us-central1`) when
/// `models.json` gives no `baseUrl`.
fn vertex_base_url(entry: &Value) -> Result<String, AgentError> {
    let project = entry
        .get("project")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
        .filter(|p| !p.is_empty())
        .ok_or_else(|| {
            AgentError::Configuration(
                "Vertex AI needs `baseUrl`, or `project` (or GOOGLE_CLOUD_PROJECT) in models.json"
                    .into(),
            )
        })?;
    let location = entry
        .get("location")
        .and_then(Value::as_str)
        .unwrap_or("us-central1");
    let host = if location == "global" {
        "aiplatform.googleapis.com".to_owned()
    } else {
        format!("{location}-aiplatform.googleapis.com")
    };
    Ok(format!(
        "https://{host}/v1/projects/{project}/locations/{location}/publishers/google"
    ))
}

fn api_name(transport: &ProviderTransport) -> &'static str {
    if matches!(transport, ProviderTransport::Demo) {
        "mu-demo"
    } else if matches!(
        transport,
        ProviderTransport::Google {
            auth: GoogleAuth::Bearer(_),
            ..
        }
    ) {
        "google-vertex"
    } else if matches!(transport, ProviderTransport::Google { .. }) {
        "google-generative-ai"
    } else if matches!(transport, ProviderTransport::Completions { .. }) {
        "openai-completions"
    } else if matches!(transport, ProviderTransport::Bedrock { .. }) {
        "bedrock-converse-stream"
    } else if matches!(transport, ProviderTransport::Anthropic { .. }) {
        "anthropic-messages"
    } else {
        "openai-responses"
    }
}

fn resolve_session_options(
    config: &Config,
    options: SessionOptions,
) -> Result<(PathBuf, SessionDocument, Vec<AgentMessage>), AgentError> {
    if let Some(id) = options.id.as_deref() {
        validate_session_id(id)?;
    }
    let SessionOptions {
        path,
        directory,
        id,
        id_prefix,
        continue_latest,
        no_session,
        fork_from,
        name,
        tools: _,
        no_tools: _,
        no_builtin_tools: _,
        skill_paths: _,
        no_skills: _,
        demo: _,
        system_prompt: _,
        append_system_prompt: _,
        no_context_files: _,
        exclude_tools: _,
        thinking: _,
        provider: _,
        model: _,
        no_mcp: _,
        prompt_templates: _,
        no_prompt_templates: _,
        api_key: _,
        project_trust: _,
        theme: _,
        models: _,
        verbose: _,
        continue_in_current_cwd,
    } = options;
    let name = normalize_session_name(name)?;
    if let Some(source_path) = fork_from {
        let (source, _) = load_session(&source_path)?;
        let id = id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        validate_session_id(&id)?;
        if find_session_id_path(
            &config.agent_dir,
            &config.cwd,
            directory.as_deref(),
            &id,
            false,
        )
        .is_some()
        {
            return Err(AgentError::Session(format!(
                "Session already exists with id '{id}'"
            )));
        }
        let parent_session = source_path.to_string_lossy().into_owned();
        let mut session = source
            .fork_with_id(
                config.cwd.to_string_lossy().into_owned(),
                id,
                parent_session,
            )
            .ok_or_else(|| AgentError::Session("source session has no header".into()))?;
        if let Some(name) = name {
            session.append_session_name(&name);
        }
        let path = session_path(config, directory.as_deref());
        if !no_session {
            write_session(&session, &path)?;
        }
        let history = runtime::session_messages(&session);
        return Ok((path, session, history));
    }
    let requested_path = path
        .or_else(|| {
            continue_latest.then(|| {
                latest_session_path(&config.agent_dir, &config.cwd, directory.as_deref())
            })?
        })
        .or_else(|| {
            id.as_deref().and_then(|id| {
                find_session_id_path(
                    &config.agent_dir,
                    &config.cwd,
                    directory.as_deref(),
                    id,
                    false,
                )
            })
        })
        .or_else(|| {
            id_prefix.as_deref().and_then(|prefix| {
                find_session_id_path(
                    &config.agent_dir,
                    &config.cwd,
                    directory.as_deref(),
                    prefix,
                    true,
                )
            })
        });
    if id_prefix.is_some() && requested_path.is_none() {
        return Err(AgentError::Session(format!(
            "No session found matching '{}' in this working directory",
            id_prefix.as_deref().unwrap_or_default()
        )));
    }
    let (path, mut session, history) = match requested_path {
        Some(path) => {
            let (session, history) = load_session(&path)?;
            if let Some(message) = missing_session_cwd_error(&session, &path, &config.cwd) {
                if continue_in_current_cwd {
                    eprintln!("mu: {message}\nmu: continuing in the current working directory");
                } else {
                    return Err(AgentError::Session(message));
                }
            }
            (path, session, history)
        }
        None => {
            let session = match id {
                Some(id) => {
                    SessionDocument::new_with_id(config.cwd.to_string_lossy().into_owned(), id)
                }
                None => SessionDocument::new(config.cwd.to_string_lossy().into_owned()),
            };
            (
                session_path(config, directory.as_deref()),
                session,
                Vec::new(),
            )
        }
    };
    if let Some(name) = name {
        session.append_session_name(&name);
        if !no_session {
            write_session(&session, &path)?;
        }
    }
    Ok((path, session, history))
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

/// One row of `/tree`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeRow {
    pub id: String,
    pub depth: usize,
    pub label: String,
    /// On the path from the root to the current tip.
    pub on_branch: bool,
    pub is_leaf: bool,
}

/// Result of [`MuAgent::compact`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionOutcome {
    pub summary: String,
    pub tokens_before: u64,
}

/// Plain-text rendering of messages for the summarizer.
/// Token, cost, and context totals shown in Pi's footer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FooterStats {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub cost: f64,
    /// Tokens in the most recent successful assistant response (the context size estimate).
    pub context_tokens: Option<u64>,
    pub context_window: u64,
}

fn session_footer_stats(session: &SessionDocument) -> FooterStats {
    let mut stats = FooterStats::default();
    for entry in session.entries() {
        let Some(message) = entry.extra_fields().get("message") else {
            continue;
        };
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(usage) = message.get("usage") else {
            continue;
        };
        let field = |name: &str| usage.get(name).and_then(Value::as_u64).unwrap_or(0);
        stats.input += field("input");
        stats.output += field("output");
        stats.cache_read += field("cacheRead");
        stats.cache_write += field("cacheWrite");
        stats.cost += usage
            .pointer("/cost/total")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let failed = matches!(
            message.get("stopReason").and_then(Value::as_str),
            Some("error" | "aborted")
        );
        if !failed {
            let total = field("totalTokens");
            stats.context_tokens = Some(if total > 0 {
                total
            } else {
                field("input") + field("output") + field("cacheRead") + field("cacheWrite")
            });
        }
    }
    stats
}

/// A model's entry in `models.json` for `provider`.
fn configured_model_entry(agent_dir: &Path, provider: &str, model: &str) -> Option<Value> {
    let contents = std::fs::read_to_string(agent_dir.join("models.json")).ok()?;
    let value: Value = serde_json::from_str(&contents).ok()?;
    value
        .get("providers")?
        .get(provider)?
        .get("models")?
        .as_array()?
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model))
        .cloned()
}

/// Everything both startup paths derive from the configuration and session options.
struct Prepared {
    config: Config,
    tool_selection: Option<ToolSelection>,
    no_tools: bool,
    persist_session: bool,
    skills: crate::skills::SkillCatalog,
    session_path: PathBuf,
    session: SessionDocument,
    transcript: Vec<AgentMessage>,
    prompt: PromptOptions,
    excluded_tools: Vec<String>,
    templates: Vec<crate::prompt_templates::PromptTemplate>,
    resources: ResourceOptions,
}

/// Flags that decide which skills and prompt templates are discovered (kept for `/reload`).
#[derive(Clone, Debug, Default)]
struct ResourceOptions {
    skill_paths: Vec<PathBuf>,
    no_skills: bool,
    template_paths: Vec<PathBuf>,
    no_templates: bool,
}

impl Prepared {
    fn new(mut config: Config, options: SessionOptions) -> Result<Self, AgentError> {
        let tool_selection = options
            .tools
            .clone()
            .map(ToolSelection::parse)
            .transpose()?;
        if let Some(selection) = &tool_selection {
            config.enabled_tools = apply_tool_selection(&config.enabled_tools, selection);
        }
        if options.no_builtin_tools || options.no_tools {
            config.enabled_tools.clear();
        }
        let no_tools = options.no_tools;
        let persist_session = !options.no_session;
        let excluded_tools = options.exclude_tools.clone();
        let resources = ResourceOptions {
            skill_paths: options.skill_paths.clone(),
            no_skills: options.no_skills,
            template_paths: options.prompt_templates.clone(),
            no_templates: options.no_prompt_templates,
        };
        let templates = crate::prompt_templates::load_prompt_templates(
            &config.cwd,
            &config.agent_dir,
            &options.prompt_templates,
            !options.no_prompt_templates,
        );
        let prompt = PromptOptions::from_session_options(&options).with_context_files(
            &config.cwd,
            &config.agent_dir,
            options.no_context_files,
        );
        let skills = crate::skills::SkillCatalog::discover_with_paths(
            &config.cwd,
            &config.agent_dir,
            &options.skill_paths,
            !options.no_skills,
        );
        let (session_path, session, transcript) = resolve_session_options(&config, options)?;
        Ok(Self {
            config,
            tool_selection,
            no_tools,
            persist_session,
            skills,
            session_path,
            session,
            transcript,
            prompt,
            excluded_tools,
            templates,
            resources,
        })
    }

    fn into_agent(
        self,
        provider: Provider,
        mcp_startup: Option<Arc<OnceCell<crate::mcp::McpRegistry>>>,
    ) -> MuAgent {
        let active_mcp_tools = session_active_mcp_tools(&self.session);
        MuAgent {
            provider,
            provider_id: self.config.provider,
            model_choices: self.config.model_choices,
            session: self.session,
            transcript: self.transcript,
            session_path: self.session_path,
            persist_session: self.persist_session,
            agent_dir: self.config.agent_dir,
            cwd: self.config.cwd,
            skills: self.skills,
            hide_thinking_block: self.config.hide_thinking_block,
            retry: self.config.retry,
            enabled_tools: self.config.enabled_tools,
            no_tools: self.no_tools,
            tool_selection: self.tool_selection,
            excluded_tools: self.excluded_tools,
            templates: self.templates,
            resources: self.resources,
            thinking: self.prompt.thinking,
            no_mcp: false,
            theme_override: self.prompt.theme,
            model_scope_override: self.prompt.models,
            mcp: crate::mcp::McpRegistry::default(),
            mcp_startup,
            active_mcp_tools,
            prompt_queue: AgentPromptQueue::default(),
            event_tap: None,
        }
    }
}

pub struct MuAgent {
    provider: Provider,
    provider_id: String,
    model_choices: Vec<String>,
    session: SessionDocument,
    transcript: Vec<AgentMessage>,
    session_path: PathBuf,
    persist_session: bool,
    agent_dir: PathBuf,
    cwd: PathBuf,
    skills: crate::skills::SkillCatalog,
    templates: Vec<crate::prompt_templates::PromptTemplate>,
    resources: ResourceOptions,
    hide_thinking_block: bool,
    retry: RetryPolicy,
    enabled_tools: Vec<String>,
    no_tools: bool,
    tool_selection: Option<ToolSelection>,
    excluded_tools: Vec<String>,
    thinking: ThinkingSetting,
    no_mcp: bool,
    theme_override: Option<String>,
    model_scope_override: Vec<String>,
    mcp: crate::mcp::McpRegistry,
    mcp_startup: Option<Arc<OnceCell<crate::mcp::McpRegistry>>>,
    active_mcp_tools: HashSet<String>,
    prompt_queue: AgentPromptQueue,
    event_tap: Option<EventTap>,
}

/// Observer of raw agent-loop events (used by `--mode json`).
pub type EventTap = Arc<dyn Fn(&crate::agent_loop::AgentEvent) + Send + Sync>;

impl MuAgent {
    pub async fn connect() -> Result<Self, AgentError> {
        Self::connect_with_session(None).await
    }

    pub async fn connect_with_session(session_file: Option<PathBuf>) -> Result<Self, AgentError> {
        Self::connect_with_options(SessionOptions {
            path: session_file,
            ..SessionOptions::default()
        })
        .await
    }

    pub async fn connect_with_options(options: SessionOptions) -> Result<Self, AgentError> {
        if options.demo {
            return Self::open_demo(options);
        }
        let mut config = config_for(false)?;
        apply_model_override(
            &mut config,
            options.provider.as_deref(),
            options.model.as_deref(),
        )?;
        let no_mcp = options.no_mcp;
        crate::mcp::set_project_trust_override(options.project_trust);
        let prepared = Prepared::new(config, options)?;
        let mcp_startup = Arc::new(OnceCell::new());
        let background_mcp = Arc::clone(&mcp_startup);
        let mcp_agent_dir = prepared.config.agent_dir.clone();
        let mcp_cwd = prepared.config.cwd.clone();
        let mcp_startup_task = tokio::spawn(async move {
            background_mcp
                .get_or_init(|| async move {
                    if no_mcp {
                        crate::mcp::McpRegistry::default()
                    } else {
                        crate::mcp::McpRegistry::connect(&mcp_agent_dir, &mcp_cwd).await
                    }
                })
                .await;
        });
        let provider = match connect(
            &prepared.config,
            prepared.skills.system_prompt(),
            &prepared.prompt,
        )
        .await
        {
            Ok(provider) => provider,
            Err(error) => {
                mcp_startup_task.abort();
                return Err(error);
            }
        };
        drop(mcp_startup_task);
        let mut agent = prepared.into_agent(provider, Some(mcp_startup));
        agent.no_mcp = no_mcp;
        Ok(agent)
    }

    /// Create a deterministic no-network agent for terminal previews and tests.
    pub fn demo() -> Result<Self, AgentError> {
        Self::open_demo(SessionOptions {
            demo: true,
            ..SessionOptions::default()
        })
    }

    fn open_demo(options: SessionOptions) -> Result<Self, AgentError> {
        let config = config_for(true)?;
        let prepared = Prepared::new(config, options)?;
        let provider = Provider {
            transport: ProviderTransport::Demo,
            model: prepared.config.model.clone(),
            instructions: "You are Mu, a deterministic demo assistant.".into(),
            describes_tools: true,
            appendix: String::new(),
            supports_reasoning: false,
            supports_images: true,
            reasoning_effort: None,
            supports_strict_sampling: true,
        };
        Ok(prepared.into_agent(provider, None))
    }

    #[cfg(test)]
    fn demo_from_config(config: Config) -> Result<Self, AgentError> {
        let path = session_path(&config, None);
        let session = SessionDocument::new(config.cwd.to_string_lossy().into_owned());
        let active_mcp_tools = session_active_mcp_tools(&session);
        Ok(Self {
            provider: Provider {
                transport: ProviderTransport::Demo,
                model: config.model.clone(),
                instructions: "You are Mu, a deterministic demo assistant.".into(),
                describes_tools: true,
                appendix: String::new(),
                supports_reasoning: false,
                supports_images: true,
                reasoning_effort: None,
                supports_strict_sampling: true,
            },
            provider_id: config.provider,
            model_choices: config.model_choices,
            session,
            transcript: Vec::new(),
            session_path: path,
            persist_session: true,
            agent_dir: config.agent_dir.clone(),
            cwd: config.cwd.clone(),
            skills: crate::skills::SkillCatalog::discover(&config.cwd, &config.agent_dir),
            hide_thinking_block: config.hide_thinking_block,
            retry: config.retry,
            enabled_tools: config.enabled_tools,
            no_tools: false,
            tool_selection: None,
            excluded_tools: Vec::new(),
            templates: Vec::new(),
            resources: ResourceOptions::default(),
            thinking: ThinkingSetting::default(),
            no_mcp: false,
            theme_override: None,
            model_scope_override: Vec::new(),
            mcp: crate::mcp::McpRegistry::default(),
            mcp_startup: None,
            active_mcp_tools,
            prompt_queue: AgentPromptQueue::default(),
            event_tap: None,
        })
    }

    /// Observe every raw agent-loop event as it happens.
    pub fn set_event_tap(&mut self, tap: EventTap) {
        self.event_tap = Some(tap);
    }

    pub fn model(&self) -> &str {
        &self.provider.model
    }

    pub fn provider(&self) -> &str {
        &self.provider_id
    }

    pub fn mcp_tools(&self) -> Vec<String> {
        self.mcp
            .codemode_definitions()
            .iter()
            .filter_map(format_mcp_tool_summary)
            .collect()
    }

    pub fn mcp_status(&self) -> String {
        self.mcp_startup
            .as_ref()
            .and_then(|startup| startup.get())
            .map_or_else(
                || {
                    if self.mcp_startup.is_some() {
                        "MCP servers are connecting in the background.".to_owned()
                    } else {
                        self.mcp.status_text()
                    }
                },
                crate::mcp::McpRegistry::status_text,
            )
    }

    async fn ensure_mcp_ready(&mut self) {
        let Some(startup) = self.mcp_startup.clone() else {
            return;
        };
        let agent_dir = self.agent_dir.clone();
        let cwd = self.cwd.clone();
        let no_mcp = self.no_mcp;
        let connected = tokio::time::timeout(
            MCP_STARTUP_WAIT,
            startup.get_or_init(|| async move {
                if no_mcp {
                    crate::mcp::McpRegistry::default()
                } else {
                    crate::mcp::McpRegistry::connect(&agent_dir, &cwd).await
                }
            }),
        )
        .await;
        if let Ok(registry) = connected {
            self.mcp = registry.clone();
            self.mcp_startup = None;
        }
    }

    /// `/mcp reload`: drop the connected servers and connect again from the current config
    /// (Pi's MCP `reload`). Previously loaded deferred tools are forgotten, as the tools may differ.
    pub async fn reload_mcp(&mut self) -> String {
        self.mcp_startup = None;
        self.active_mcp_tools.clear();
        self.mcp = if self.no_mcp {
            crate::mcp::McpRegistry::default()
        } else {
            crate::mcp::McpRegistry::connect(&self.agent_dir, &self.cwd).await
        };
        self.mcp.status_text()
    }

    pub fn prompt_queue(&self) -> AgentPromptQueue {
        self.prompt_queue.clone()
    }

    /// Models Ctrl+P cycles through: `enabledModels` patterns applied to the choices.
    pub fn cycle_models(&self) -> Vec<String> {
        let patterns = if self.model_scope_override.is_empty() {
            configured_setting(&self.agent_dir, &self.cwd, "enabledModels", |value| {
                value.as_array().map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
            })
            .unwrap_or_default()
        } else {
            self.model_scope_override.clone()
        };
        scope_models(&patterns, &self.model_choices, &self.provider_id)
    }

    pub fn model_choices(&self) -> &[String] {
        &self.model_choices
    }

    fn save_session(&self) -> Result<(), AgentError> {
        if self.persist_session {
            write_session(&self.session, &self.session_path)?;
        }
        Ok(())
    }

    /// Set the reasoning level for later requests (`/thinking`).
    pub fn set_thinking(&mut self, level: ThinkingSetting) {
        self.provider.reasoning_effort = level.effort();
        self.thinking = level;
    }

    pub fn thinking(&self) -> ThinkingSetting {
        self.thinking
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
        self.save_session()?;
        Ok(true)
    }

    /// Text of the newest assistant message on the active branch (empty when none).
    pub fn last_assistant_text(&self) -> String {
        branch_entry_values(&self.session)
            .iter()
            .rev()
            .filter_map(|entry| entry.get("message")?.as_object().cloned())
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .map(|message| {
                message_blocks(&message)
                    .iter()
                    .filter_map(|block| block.get("text")?.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .find(|text| !text.trim().is_empty())
            .unwrap_or_default()
    }

    /// Text block for `/session`: identity plus message counts and token/cost totals.
    pub fn session_summary(&self) -> String {
        let info = self.session_info();
        let (mut user, mut assistant, mut tool_results, mut tool_calls) = (0, 0, 0, 0);
        for entry in self.session.entries() {
            let Some(message) = entry.extra_fields().get("message") else {
                continue;
            };
            match message["role"].as_str() {
                Some("user") => user += 1,
                Some("assistant") => {
                    assistant += 1;
                    tool_calls += message["content"].as_array().map_or(0, |blocks| {
                        blocks.iter().filter(|b| b["type"] == "toolCall").count()
                    });
                }
                Some("toolResult") => tool_results += 1,
                _ => {}
            }
        }
        let stats = self.footer_stats();
        format!(
            "Session: {}\nName: {}\nFile: {}\nCreated: {}\nMessages: {user} user, {assistant} assistant, {tool_calls} tool calls, {tool_results} tool results\nTokens: {} input, {} output, {} cache read, {} cache write\nCost: ${:.3}",
            info.id,
            info.name.as_deref().unwrap_or("(unnamed)"),
            info.path.display(),
            info.timestamp,
            stats.input,
            stats.output,
            stats.cache_read,
            stats.cache_write,
            stats.cost,
        )
    }

    /// Resolved startup configuration, one `key: value` per line (`--verbose`).
    pub fn startup_summary(&self) -> String {
        format!(
            "provider: {}\nmodel: {}\nthinking: {}\nsession: {}\ncwd: {}\ntools: {}\nskills: {}\nprompt templates: {}",
            self.provider_id,
            self.provider.model,
            format!("{:?}", self.thinking).to_lowercase(),
            self.session_path.display(),
            self.cwd.display(),
            if self.no_tools {
                "(none)".to_owned()
            } else {
                self.enabled_tools.join(", ")
            },
            self.skills.skills.len(),
            self.templates.len(),
        )
    }

    /// Display name for a model id (`name` in `models.json`, else the id).
    pub fn model_display_name(&self, model: &str) -> String {
        configured_model_entry(&self.agent_dir, &self.provider_id, model)
            .and_then(|entry| entry.get("name").and_then(Value::as_str).map(str::to_owned))
            .unwrap_or_else(|| model.to_owned())
    }

    /// Usage totals and context window for the footer.
    pub fn footer_stats(&self) -> FooterStats {
        let mut stats = session_footer_stats(&self.session);
        stats.context_window =
            configured_model_entry(&self.agent_dir, &self.provider_id, &self.provider.model)
                .and_then(|entry| entry.get("contextWindow").and_then(Value::as_u64))
                .unwrap_or(0);
        stats
    }

    /// Save a project-trust decision for the working directory (`/trust`).
    pub fn set_project_trust(&self, trusted: bool) -> Result<(), AgentError> {
        crate::mcp::save_project_trust(&self.agent_dir, &self.cwd, trusted)
            .map_err(AgentError::Configuration)
    }

    /// Store an API key for `provider` in `auth.json` (0600, atomic).
    pub fn login_api_key(&self, provider: &str, key: &str) -> Result<(), AgentError> {
        if provider.trim().is_empty() || key.trim().is_empty() {
            return Err(AgentError::Configuration(
                "provider and API key must not be empty".into(),
            ));
        }
        crate::auth::set_api_key(
            &self.agent_dir.join("auth.json"),
            provider.trim(),
            key.trim(),
        )
        .map_err(|error| AgentError::Configuration(error.to_string()))
    }

    /// Remove a provider's stored credential from `auth.json`; returns whether one existed.
    pub fn logout(&self, provider: &str) -> Result<bool, AgentError> {
        crate::auth::remove(&self.agent_dir.join("auth.json"), provider)
            .map_err(|error| AgentError::Configuration(error.to_string()))
    }

    /// Effective layered settings, read fresh from disk.
    pub fn settings(&self) -> crate::settings::Settings {
        crate::settings::Settings::load_for_project(&self.agent_dir, &self.cwd)
    }

    /// Flip a boolean setting in global `settings.json`; returns the new value.
    pub fn toggle_setting(&self, key: &str) -> Result<bool, AgentError> {
        self.toggle_setting_in_scope(crate::settings::Scope::Global, key)
    }

    /// Flip a boolean setting in the requested settings layer.
    pub fn toggle_setting_in_scope(
        &self,
        scope: crate::settings::Scope,
        key: &str,
    ) -> Result<bool, AgentError> {
        let mut settings = self.settings();
        let next = !settings
            .get_for_scope(scope, key)
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        settings
            .set(scope, key, Value::Bool(next))
            .map_err(|error| AgentError::Configuration(error.to_string()))?;
        Ok(next)
    }

    /// Set an enum or number setting from text (`/settings <key> <value>`), validated like Pi's
    /// schema for the keys Mu exposes. Returns the stored value.
    pub fn set_setting_text(&self, key: &str, text: &str) -> Result<Value, AgentError> {
        self.set_setting_text_in_scope(crate::settings::Scope::Global, key, text)
    }

    /// Set and validate a value in the requested settings layer.
    pub fn set_setting_text_in_scope(
        &self,
        scope: crate::settings::Scope,
        key: &str,
        text: &str,
    ) -> Result<Value, AgentError> {
        let value =
            crate::settings::parse_setting_value(key, text).map_err(AgentError::Configuration)?;
        self.settings()
            .set(scope, key, value.clone())
            .map_err(|error| AgentError::Configuration(error.to_string()))?;
        Ok(value)
    }

    /// The `codemode.inlineBudget` setting (Pi default 3000 estimated tokens).
    pub(crate) fn codemode_inline_budget(&self) -> Option<usize> {
        self.settings()
            .get("codemode.inlineBudget")
            .and_then(|value| value.as_u64())
            .map(|budget| budget as usize)
    }

    /// `codemode.mode` is `only` (Pi's default is `on`).
    pub(crate) fn codemode_only(&self) -> bool {
        self.settings()
            .get("codemode.mode")
            .as_ref()
            .and_then(Value::as_str)
            == Some("only")
    }

    pub fn agent_dir(&self) -> &Path {
        &self.agent_dir
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    pub fn session_name(&self) -> Option<&str> {
        self.session.session_name()
    }

    /// `compaction.{enabled,reserveTokens,keepRecentTokens}` from settings (project overrides agent).
    pub fn compaction_settings(&self) -> crate::compaction::CompactionSettings {
        let mut settings = crate::compaction::CompactionSettings::default();
        let read = |key: &str| {
            configured_setting(&self.agent_dir, &self.cwd, "compaction", |value| {
                value.get(key).cloned()
            })
        };
        if let Some(enabled) = read("enabled").and_then(|v| v.as_bool()) {
            settings.enabled = enabled;
        }
        if let Some(tokens) = read("reserveTokens").and_then(|v| v.as_u64()) {
            settings.reserve_tokens = tokens;
        }
        if let Some(tokens) = read("keepRecentTokens").and_then(|v| v.as_u64()) {
            settings.keep_recent_tokens = tokens;
        }
        settings
    }

    /// Whether the last response left the context past the auto-compaction threshold.
    pub fn needs_auto_compaction(&self) -> bool {
        let stats = self.footer_stats();
        match stats.context_tokens {
            Some(tokens) if stats.context_window > 0 => crate::compaction::should_compact(
                tokens,
                stats.context_window,
                &self.compaction_settings(),
            ),
            _ => false,
        }
    }

    /// Summarize older history, append a `compaction` entry, and rebuild the transcript.
    ///
    /// Returns `Ok(None)` when there is nothing before the cut point to compact.
    pub async fn compact(&mut self) -> Result<Option<CompactionOutcome>, AgentError> {
        self.compact_with(None).await
    }

    /// Compact with optional custom focus instructions (`/compact <instructions>`).
    pub async fn compact_with(
        &mut self,
        custom_instructions: Option<&str>,
    ) -> Result<Option<CompactionOutcome>, AgentError> {
        let settings = self.compaction_settings();
        let branch = branch_entry_values(&self.session);
        // Pi summarizes the projected context: a previous compaction's summary is updated with
        // the entries that followed it rather than re-summarizing from the start.
        let path = crate::compaction::build_context_entries(&branch);
        let previous_summary = path
            .first()
            .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("compaction"))
            .and_then(|entry| entry.get("summary")?.as_str().map(str::to_owned));
        let boundary_start = usize::from(previous_summary.is_some());
        let cut = crate::compaction::find_cut_point(
            &path,
            boundary_start,
            path.len(),
            settings.keep_recent_tokens,
        );
        let Some(first_kept_id) = path
            .get(cut.first_kept_entry_index)
            .and_then(|entry| entry.get("id")?.as_str().map(str::to_owned))
        else {
            return Ok(None);
        };
        // A cut inside a turn summarizes the earlier turns and the turn's prefix separately.
        let (history_end, prefix) = match (cut.is_split_turn, cut.turn_start_index) {
            (true, Some(turn_start)) => (
                turn_start.max(boundary_start),
                &path[turn_start.max(boundary_start)..cut.first_kept_entry_index],
            ),
            _ => (cut.first_kept_entry_index.max(boundary_start), &path[0..0]),
        };
        let summarized = &path[boundary_start..history_end];
        let has_messages = |entries: &[Value]| entries.iter().any(|e| e.get("message").is_some());
        if !has_messages(summarized) && !has_messages(prefix) {
            return Ok(None);
        }
        let tokens_before = path
            .iter()
            .filter_map(|entry| entry.get("message"))
            .map(crate::compaction::estimate_tokens)
            .sum();
        let summary = if has_messages(prefix) {
            let history_text = if has_messages(summarized) {
                self.summarize(
                    &crate::compaction::serialize_conversation(summarized),
                    previous_summary.as_deref(),
                    custom_instructions,
                )
                .await?
            } else {
                previous_summary
                    .clone()
                    .unwrap_or_else(|| "No prior history.".to_owned())
            };
            let prefix_text = self
                .summarize_text(format!(
                    "# Conversation\n{}\n\n# Instructions\n{}",
                    crate::compaction::serialize_conversation(prefix),
                    crate::compaction::TURN_PREFIX_SUMMARIZATION_PROMPT
                ))
                .await?;
            format!("{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}")
        } else {
            self.summarize(
                &crate::compaction::serialize_conversation(summarized),
                previous_summary.as_deref(),
                custom_instructions,
            )
            .await?
        };
        let mut file_ops = crate::compaction::FileOperations::default();
        if let Some(previous) = &previous_summary {
            file_ops.add_previous_summary(previous);
        }
        file_ops.add_entries(summarized);
        file_ops.add_entries(prefix);
        let summary = format!("{summary}{}", file_ops.format());
        self.session
            .append_compaction(&summary, &first_kept_id, tokens_before);
        self.save_session()?;
        self.transcript = runtime::session_messages(&self.session);
        Ok(Some(CompactionOutcome {
            summary,
            tokens_before,
        }))
    }

    /// One summarization call over `conversation` (Pi's `generateSummary`).
    async fn summarize(
        &mut self,
        conversation: &str,
        previous_summary: Option<&str>,
        custom_instructions: Option<&str>,
    ) -> Result<String, AgentError> {
        let previous = previous_summary
            .map(|summary| format!("<previous-summary>\n{summary}\n</previous-summary>\n\n"))
            .unwrap_or_default();
        self.summarize_text(format!(
            "<conversation>\n{conversation}\n</conversation>\n\n{previous}{}",
            crate::compaction::summarization_prompt_for(
                custom_instructions,
                previous_summary.is_some()
            )
        ))
        .await
    }

    async fn summarize_text(&mut self, prompt: String) -> Result<String, AgentError> {
        let history = vec![json!({
            "role":"user",
            "content":[{"type":"input_text","text":prompt}],
        })];
        let reply = self.provider.complete(&history, &[], &mut |_| {}).await?;
        if reply.incomplete_reason.is_some() || reply.text.trim().is_empty() {
            return Err(AgentError::Provider(
                "Summarization did not complete; the session was not changed".into(),
            ));
        }
        Ok(reply.text)
    }

    pub fn session_path(&self) -> &Path {
        &self.session_path
    }

    pub fn session_transcript(&self) -> Vec<SessionTranscriptMessage> {
        session_transcript(&self.session)
    }

    pub fn session_info(&self) -> SessionInfo {
        let header = self.session.header();
        SessionInfo {
            id: header
                .map(|header| header.id.as_str().to_owned())
                .unwrap_or_default(),
            name: self.session.session_name().map(str::to_owned),
            preview: session_preview(&self.session),
            path: self.session_path.clone(),
            timestamp: header
                .map(|header| header.timestamp.as_str().to_owned())
                .unwrap_or_default(),
        }
    }

    pub fn name_session(&mut self, name: &str) -> Result<(), AgentError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(AgentError::Session("Session name must not be empty".into()));
        }
        let mut session = self.session.clone();
        session.append_session_name(name);
        if self.persist_session {
            write_session(&session, &self.session_path)?;
        }
        self.session = session;
        Ok(())
    }

    /// Export the current active branch as Pi JSONL or a standalone HTML transcript.
    pub fn export_session(&self, path: &Path) -> Result<(), AgentError> {
        match path.extension().and_then(|extension| extension.to_str()) {
            Some("jsonl") => write_session(&self.session, path),
            Some("html") => std::fs::write(path, self.render_session_html())
                .map_err(|error| AgentError::Session(error.to_string())),
            _ => Err(AgentError::Session(
                "Session export path must end in .jsonl or .html".into(),
            )),
        }
    }

    fn render_session_html(&self) -> String {
        render_session_html(&self.session)
    }

    pub fn import_session(&mut self, path: &Path) -> Result<(), AgentError> {
        let (session, history) = load_session(path)?;
        if let Some(message) = missing_session_cwd_error(&session, path, &self.cwd) {
            return Err(AgentError::Session(message));
        }
        self.session = session;
        self.transcript = history;
        self.session_path = path.to_owned();
        self.persist_session = true;
        self.active_mcp_tools = session_active_mcp_tools(&self.session);
        self.prompt_queue.clear();
        Ok(())
    }

    pub fn clone_session(&mut self) -> Result<(), AgentError> {
        if !self.persist_session {
            return Err(AgentError::Session(
                "Cannot clone a session that is not persisted to disk".into(),
            ));
        }
        let new_id = uuid::Uuid::new_v4().to_string();
        let new_session = self
            .session
            .fork_with_id(
                self.cwd.to_string_lossy().into_owned(),
                new_id,
                self.session_path.to_string_lossy().into_owned(),
            )
            .ok_or_else(|| AgentError::Session("Current session has no header".into()))?;
        let new_path = new_session_path(&self.agent_dir, &self.cwd, None);
        write_session(&new_session, &new_path)?;

        self.session = new_session;
        self.transcript = runtime::session_messages(&self.session);
        self.session_path = new_path;
        self.active_mcp_tools = session_active_mcp_tools(&self.session);
        self.prompt_queue.clear();
        Ok(())
    }

    /// Every message entry as a tree row (`depth` = distance from the root), oldest first.
    pub fn tree_rows(&self) -> Vec<TreeRow> {
        let active: HashSet<String> = self
            .session
            .active_branch_entries()
            .iter()
            .filter_map(|entry| entry.id.as_ref().map(|id| id.as_str().to_owned()))
            .collect();
        let leaf = self.session.leaf_id().map(|id| id.as_str().to_owned());
        let mut depths: HashMap<String, usize> = HashMap::new();
        let mut rows = Vec::new();
        for entry in self.session.entries() {
            let Some(id) = entry.id.as_ref().map(|id| id.as_str().to_owned()) else {
                continue;
            };
            let depth = match &entry.parent_id {
                mu_session::ParentReference::Entry(parent) => {
                    depths.get(parent.as_str()).map_or(0, |d| d + 1)
                }
                _ => 0,
            };
            depths.insert(id.clone(), depth);
            let Some(message) = entry
                .extra_fields()
                .get("message")
                .and_then(Value::as_object)
            else {
                continue;
            };
            let role = message.get("role").and_then(Value::as_str).unwrap_or("?");
            let text = message_blocks(message)
                .iter()
                .filter_map(|block| block.get("text")?.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
                .join(" ");
            let summary = text.split_whitespace().collect::<Vec<_>>().join(" ");
            rows.push(TreeRow {
                on_branch: active.contains(&id),
                is_leaf: leaf.as_deref() == Some(id.as_str()),
                label: format!("{role}: {}", summary.chars().take(60).collect::<String>()),
                id,
                depth,
            });
        }
        rows
    }

    /// Move the session tip to `entry_id`; the next message branches from there (Pi's `/tree`).
    pub fn navigate_to(&mut self, entry_id: &str) -> Result<(), AgentError> {
        if !self.session.set_leaf(entry_id) {
            return Err(AgentError::Session("No such entry".into()));
        }
        self.transcript = runtime::session_messages(&self.session);
        self.prompt_queue.clear();
        Ok(())
    }

    /// Switch to `entry_id` and leave a summary of the branch being abandoned (Pi's
    /// `navigateTree` with `summarize`). With nothing to summarize it just navigates.
    pub async fn navigate_with_summary(
        &mut self,
        entry_id: &str,
        custom_instructions: Option<&str>,
    ) -> Result<Option<String>, AgentError> {
        let old_leaf = self.session.leaf_id().map(|id| id.as_str().to_owned());
        let old_branch = branch_entry_values(&self.session);
        self.navigate_to(entry_id)?;
        let kept = branch_entry_values(&self.session)
            .iter()
            .filter_map(|e| e.get("id")?.as_str().map(str::to_owned))
            .collect::<std::collections::HashSet<_>>();
        let abandoned = old_branch
            .into_iter()
            .filter(|e| {
                e.get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| !kept.contains(id))
            })
            .collect::<Vec<_>>();
        let Some(from_id) =
            old_leaf.filter(|_| abandoned.iter().any(|e| e.get("message").is_some()))
        else {
            return Ok(None);
        };
        let instructions = match custom_instructions.map(str::trim).filter(|c| !c.is_empty()) {
            Some(custom) => format!(
                "{}\n\nAdditional focus: {custom}",
                crate::compaction::BRANCH_SUMMARY_PROMPT
            ),
            None => crate::compaction::BRANCH_SUMMARY_PROMPT.to_owned(),
        };
        let prompt = format!(
            "<conversation>\n{}\n</conversation>\n\n{instructions}",
            crate::compaction::serialize_conversation(&abandoned)
        );
        let text = match self.summarize_text(prompt).await {
            Ok(text) => text,
            Err(error) => return Err(error),
        };
        let mut file_ops = crate::compaction::FileOperations::default();
        file_ops.add_entries(&abandoned);
        let summary = format!(
            "{}{text}{}",
            crate::compaction::BRANCH_SUMMARY_PREAMBLE,
            file_ops.format()
        );
        self.session.append_branch_summary(&summary, &from_id);
        self.save_session()?;
        self.transcript = runtime::session_messages(&self.session);
        Ok(Some(summary))
    }

    /// User messages on the active branch, oldest first, as `(entry id, text)`.
    pub fn user_messages(&self) -> Vec<(String, String)> {
        branch_entry_values(&self.session)
            .into_iter()
            .filter_map(|entry| {
                let id = entry.get("id")?.as_str()?.to_owned();
                let message = entry.get("message")?.as_object()?;
                if message.get("role")?.as_str()? != "user" {
                    return None;
                }
                let text = message_blocks(message)
                    .iter()
                    .filter_map(|b| b.get("text")?.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
                    .join("\n");
                (!text.is_empty()).then_some((id, text))
            })
            .collect()
    }

    /// Fork into a new session holding everything before the user message `entry_id`.
    /// Returns that message's text so it can be edited and resubmitted (Pi's `/fork`).
    pub fn fork_before_user_message(&mut self, entry_id: &str) -> Result<String, AgentError> {
        let text = self
            .user_messages()
            .into_iter()
            .find(|(id, _)| id == entry_id)
            .map(|(_, text)| text)
            .ok_or_else(|| AgentError::Session("No such user message".into()))?;
        let truncated = self
            .session
            .truncated_before(entry_id)
            .ok_or_else(|| AgentError::Session("No such user message".into()))?;
        let fork = truncated
            .fork_with_id(
                self.cwd.to_string_lossy().into_owned(),
                uuid::Uuid::new_v4().to_string(),
                self.session_path.to_string_lossy().into_owned(),
            )
            .ok_or_else(|| AgentError::Session("Current session has no header".into()))?;
        let path = new_session_path(&self.agent_dir, &self.cwd, None);
        if self.persist_session {
            write_session(&fork, &path)?;
        }
        self.session = fork;
        self.transcript = runtime::session_messages(&self.session);
        self.session_path = path;
        self.active_mcp_tools = session_active_mcp_tools(&self.session);
        self.prompt_queue.clear();
        Ok(text)
    }

    pub fn start_new_session(&mut self) -> Result<(), AgentError> {
        let parent = self.session_path.parent().ok_or_else(|| {
            AgentError::Session("current session path has no parent directory".into())
        })?;
        let timestamp = chrono::Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
            .replace([':', '.'], "-");
        let next_path = parent.join(format!("{timestamp}_{}.jsonl", uuid::Uuid::new_v4()));
        let next_session = SessionDocument::new(self.cwd.to_string_lossy().into_owned());
        if self.persist_session {
            write_session(&next_session, &next_path)?;
        }

        self.session = next_session;
        self.transcript.clear();
        self.session_path = next_path;
        self.active_mcp_tools.clear();
        self.prompt_queue.clear();
        Ok(())
    }

    /// Re-discover skills and prompt templates from disk (`/reload`); returns a one-line summary.
    ///
    /// Skill descriptions already in the system prompt stay until the next session start.
    pub fn reload_resources(&mut self) -> String {
        let resources = self.resources.clone();
        self.skills = crate::skills::SkillCatalog::discover_with_paths(
            &self.cwd,
            &self.agent_dir,
            &resources.skill_paths,
            !resources.no_skills,
        );
        self.templates = crate::prompt_templates::load_prompt_templates(
            &self.cwd,
            &self.agent_dir,
            &resources.template_paths,
            !resources.no_templates,
        );
        format!(
            "Reloaded: {} skills, {} prompt templates",
            self.skills.skills.len(),
            self.templates.len()
        )
    }

    /// Prompt templates, then `skill:<name>` commands: `(name, description, argument hint)`.
    pub fn dynamic_slash_commands(&self) -> Vec<(String, String, Option<String>)> {
        let templates = self.templates.iter().map(|template| {
            (
                template.name.clone(),
                template.description.clone(),
                template.argument_hint.clone(),
            )
        });
        let skills = skill_commands_enabled(&self.agent_dir, &self.cwd)
            .then(|| self.skills.skills.iter())
            .into_iter()
            .flatten()
            .map(|skill| {
                (
                    format!("skill:{}", skill.name),
                    skill.description.clone(),
                    None,
                )
            });
        templates.chain(skills).collect()
    }

    /// Skills discovered from Pi's global and project skill directories.
    pub fn skills(&self) -> &crate::skills::SkillCatalog {
        &self.skills
    }

    fn expand_prompt(&self, prompt: String, on_event: &mut impl FnMut(AgentEvent)) -> String {
        let prompt = crate::prompt_templates::expand_prompt_template(&prompt, &self.templates);
        if !skill_commands_enabled(&self.agent_dir, &self.cwd) {
            return prompt;
        }
        expand_skill_prompt(prompt, &self.skills, on_event)
    }

    /// The `theme` setting (project settings override agent settings), if any.
    pub fn theme_name(&self) -> Option<String> {
        if let Some(name) = &self.theme_override {
            return Some(name.clone());
        }
        configured_setting(&self.agent_dir, &self.cwd, "theme", |value| {
            value.as_str().map(str::to_owned)
        })
    }

    pub fn hide_thinking_block(&self) -> bool {
        self.hide_thinking_block
    }

    /// Run a built-in tool directly in the project directory (the TUI's `!command`).
    pub async fn run_local_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<crate::tools::ToolResult, crate::tools::ToolError> {
        crate::tools::execute_with_environment(
            &self.cwd,
            name,
            arguments,
            &self.shell_environment(),
        )
        .await
    }

    /// Execute one of Mu's Pi-compatible built-in tools in the project directory.
    pub async fn execute_tool(
        &self,
        name: &str,
        arguments: &Value,
    ) -> Result<crate::tools::ToolResult, crate::tools::ToolError> {
        if !self.enabled_tools.iter().any(|enabled| enabled == name) {
            return Err(crate::tools::ToolError::Arguments(format!(
                "tool `{name}` is not enabled"
            )));
        }
        crate::tools::execute_with_environment(
            &self.cwd,
            name,
            arguments,
            &self.shell_environment(),
        )
        .await
    }

    fn shell_environment(&self) -> HashMap<String, String> {
        let mut environment = HashMap::new();
        if let Some(header) = self.session.header() {
            environment.insert("PI_SESSION_ID".into(), header.id.as_str().to_owned());
        }
        if self.persist_session {
            environment.insert(
                "PI_SESSION_FILE".into(),
                self.session_path.to_string_lossy().into_owned(),
            );
        }
        environment.insert("PI_PROVIDER".into(), self.provider_id.clone());
        environment.insert("PI_MODEL".into(), self.provider.model.clone());
        if let Some(prefix) = shell_setting(&self.agent_dir, &self.cwd, "shellCommandPrefix") {
            environment.insert(crate::tools::SHELL_COMMAND_PREFIX_ENV.into(), prefix);
        }
        if let Some(path) = configured_bash_shell_path(&self.agent_dir, &self.cwd) {
            environment.insert(crate::tools::BASH_SHELL_PATH_ENV.into(), path);
        }
        environment
    }

    fn codemode_store(&self) -> Value {
        session_codemode_store(&self.session)
    }

    /// Provider-facing tool definitions for the current selection.
    #[cfg(test)]
    fn tool_definitions(&self, active_mcp_tools: &HashSet<String>) -> Vec<Value> {
        let plan = runtime::ToolPlan {
            enabled_tools: self.enabled_tools.clone(),
            no_tools: self.no_tools,
            selection: self.tool_selection.clone(),
            excluded: self.excluded_tools.clone(),
            codemode_inline_budget: self.codemode_inline_budget(),
            codemode_only: self.codemode_only(),
        };
        plan.definitions(&self.mcp, active_mcp_tools)
            .iter()
            .filter_map(runtime::declaration_from_definition)
            .map(|declaration| {
                runtime::request_tool(&declaration, self.provider.supports_strict_sampling)
            })
            .collect()
    }

    /// Run one user turn; queued steering and follow-up prompts continue it.
    pub async fn submit(
        &mut self,
        prompt: String,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), AgentError> {
        self.submit_with_cancel(prompt, on_event, &CancellationToken::new())
            .await
    }

    /// Continue a pending user/tool-result turn without adding a new user message.
    pub async fn continue_session(
        &mut self,
        on_event: &mut impl FnMut(AgentEvent),
    ) -> Result<(), AgentError> {
        self.continue_from_transcript(on_event, &CancellationToken::new())
            .await
    }
}

pub async fn run_prompt(prompt: String) -> Result<(), AgentError> {
    run_prompt_with_session(prompt, None).await
}

pub async fn run_prompt_with_session(
    prompt: String,
    session_file: Option<PathBuf>,
) -> Result<(), AgentError> {
    run_prompt_with_options(
        vec![prompt],
        Vec::new(),
        SessionOptions {
            path: session_file,
            ..SessionOptions::default()
        },
    )
    .await
}

/// Run one prompt and print every agent event as a JSON line (Pi's `--mode json`).
pub async fn run_prompt_json(
    prompts: Vec<String>,
    images: Vec<(String, String)>,
    session_options: SessionOptions,
) -> Result<(), AgentError> {
    let mut agent = MuAgent::connect_with_options(session_options).await?;
    agent.set_event_tap(Arc::new(|event| {
        println!("{}", crate::agent_loop::event_to_json(event));
    }));
    let mut images = Some(images);
    for prompt in prompts {
        agent
            .submit_with_images(
                prompt,
                images.take().unwrap_or_default(),
                &mut |_| {},
                &CancellationToken::new(),
            )
            .await?;
    }
    Ok(())
}

pub async fn run_prompt_with_options(
    prompts: Vec<String>,
    images: Vec<(String, String)>,
    session_options: SessionOptions,
) -> Result<(), AgentError> {
    let verbose = session_options.verbose;
    let mut agent = MuAgent::connect_with_options(session_options).await?;
    if verbose {
        eprintln!("{}", agent.startup_summary());
    }
    let mut images = Some(images);
    for prompt in prompts {
        agent
            .submit_with_images(prompt, images.take().unwrap_or_default(), &mut |event| {
                match event {
                    AgentEvent::TextDelta(text) => print!("{text}"),
                    AgentEvent::RetryScheduled {
                        attempt,
                        max_attempts,
                        delay_ms,
                        error,
                    } => eprintln!(
                        "Retry {attempt}/{max_attempts} in {:.1}s: {error} (interrupt to cancel)",
                        delay_ms as f64 / 1000.0
                    ),
                    _ => {}
                }
                let _ = io::stdout().flush();
            }, &CancellationToken::new())
            .await?;
    }
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
mod tests;
